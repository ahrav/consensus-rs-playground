//! Async I/O abstraction over platform-specific completion APIs.
//!
//! Provides a unified interface for submitting I/O operations and handling completions,
//! abstracting over Linux's `io_uring` and macOS's Grand Central Dispatch (GCD).
//!
//! # Architecture
//!
//! The design separates concerns into three layers:
//! - [`IoBackend`]: Platform-specific submission and completion mechanics
//! - [`Operation`]: Describes what I/O to perform (read, write, fsync)
//! - [`Completion`]: Tracks operation lifecycle and delivers results via callback
//!
//! # Ownership Model
//!
//! [`Completion`] instances must be pinned and outlive their in-flight operations.
//! The caller owns the buffer memory; this layer only holds raw pointers. Backends
//! return completions via opaque `user_data` identifiers (pointer cast to `u64`).

pub mod iops;

#[cfg(target_os = "linux")]
mod backend_linux;
#[cfg(target_os = "macos")]
mod backend_macos;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("This IO layer currently supports only Linux (io_uring) and macOS (GCD).");

use core::ffi::c_void;
use core::ptr::NonNull;
use std::io;
use std::os::unix::io::RawFd;
use std::time::{Duration, Instant};

use crate::stdx::{DoublyLinkedList, ListLink, ListNode};

#[cfg(target_os = "linux")]
use backend_linux::UringBackend as BackendImpl;
#[cfg(target_os = "macos")]
use backend_macos::GcdBackend as BackendImpl;

const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

/// Platform-specific I/O instance. Use this type alias for portable code.
pub type Io = IoCore<BackendImpl>;

/// Backend abstraction for platform-specific async I/O.
///
/// Implementations handle the mechanics of submitting operations to the kernel
/// and retrieving completions. The trait enforces a push-flush-drain pattern:
/// operations are queued, then flushed to the kernel, then completions are drained.
pub trait IoBackend {
    /// Initialize the backend with a fixed queue depth.
    fn new(entries: u32) -> io::Result<Self>
    where
        Self: Sized;

    /// Queue an operation for submission.
    ///
    /// # Safety
    ///
    /// - `op` must describe valid memory regions that remain valid until completion
    /// - `user_data` must be retrievable via [`drain`](Self::drain) to identify the completion
    ///
    /// Returns `Err(())` if the submission queue is full; caller should flush and retry.
    unsafe fn try_push(&mut self, op: &Operation, user_data: u64) -> Result<(), ()>;

    /// Submit queued operations to the kernel.
    ///
    /// If `wait_for_one` is true, blocks until at least one completion is available.
    fn flush(&mut self, wait_for_one: bool) -> io::Result<()>;

    /// Process all available completions without blocking.
    ///
    /// Invokes `f(user_data, result)` for each completion, where `result` is
    /// the byte count on success or a negated errno on failure.
    fn drain<F: FnMut(u64, i32)>(&mut self, f: F);
}

/// Type tag for intrusive linked list membership. Prevents a [`Completion`]
/// from being in multiple lists simultaneously.
enum IoTag {}

/// Lifecycle state of a [`Completion`].
///
/// State transitions are strictly enforced:
/// ```text
/// Idle -> Queued -> Submitted -> Completed -> Idle
///                       ^                       |
///                       +--- (via reset/complete)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionState {
    /// Available for use. Initial and terminal state.
    Idle,
    /// Queued locally, not yet submitted to kernel.
    Queued,
    /// Submitted to kernel, awaiting completion.
    Submitted,
    /// Kernel reported completion; result is available.
    Completed,
}

/// Describes an I/O operation to perform.
///
/// # Buffer Ownership
///
/// `Read` and `Write` variants hold raw pointers to caller-owned buffers.
/// The caller must ensure buffers remain valid and pinned until the operation
/// completes. The `len` field uses `u32` for wire compatibility and to catch
/// oversized requests at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// No operation. Used as a sentinel for uninitialized [`Completion`]s.
    Nop,
    /// Read from `fd` at `offset` into `buf[0..len]`.
    Read {
        fd: RawFd,
        buf: NonNull<u8>,
        len: u32,
        offset: u64,
    },
    /// Write `buf[0..len]` to `fd` at `offset`.
    Write {
        fd: RawFd,
        buf: NonNull<u8>,
        len: u32,
        offset: u64,
    },
    /// Flush all pending writes on `fd` to durable storage.
    Fsync { fd: RawFd },
}

impl Operation {
    /// Returns `true` if this is an actual I/O operation (not [`Nop`](Self::Nop)).
    #[inline]
    pub fn is_active(&self) -> bool {
        !matches!(self, Operation::Nop)
    }

    /// Performs Tiger-style runtime validation of operation parameters.
    ///
    /// # Panics
    ///
    /// Panics on invalid fd or zero-length buffer.
    pub fn validate(&self) {
        match *self {
            Operation::Read { fd, len, .. } | Operation::Write { fd, len, .. } => {
                assert!(fd >= 0, "File descriptor must be non-negative");
                assert!(len > 0, "Length must be positive");
            }
            Operation::Fsync { fd } => {
                assert!(fd >= 0, "File descriptor must be non-negative");
            }
            Operation::Nop => {}
        }
    }
}

/// Callback signature for completion notification.
///
/// # Safety
///
/// The callback receives the original context pointer and mutable access to the
/// [`Completion`]. The completion is in [`Idle`](CompletionState::Idle) state
/// when the callback runs, allowing immediate reuse.
pub type CompletionCallback = unsafe fn(*mut c_void, &mut Completion);

/// Handler trait for receiving I/O completion notifications.
///
/// Implement this trait on your context type, then use [`typed_handler_shim`]
/// as the [`CompletionCallback`] to get type-safe callbacks without allocation.
///
/// # Safety
///
/// Implementations must be prepared for `completion` to be in the [`Idle`](CompletionState::Idle)
/// state. The completion can be immediately reused or re-submitted from within the handler.
pub trait IoHandler {
    /// Called when an I/O operation completes.
    ///
    /// # Safety
    ///
    /// Caller must ensure `self` is the same instance that was stored in
    /// [`Completion::context`] when the operation was submitted.
    unsafe fn on_complete(&mut self, completion: &mut Completion);
}

/// Type-erased callback trampoline for [`IoHandler`] implementations.
///
/// Bridges the generic `IoHandler` trait to the type-erased [`CompletionCallback`]
/// signature. Each instantiation (via monomorphization) creates a specialized
/// function that knows how to cast `ctx` back to the concrete handler type.
///
/// # Usage
///
/// ```ignore
/// completion.context = handler as *mut MyHandler as *mut c_void;
/// completion.callback = Some(typed_handler_shim::<MyHandler>);
/// ```
///
/// # Safety
///
/// - `ctx` must point to a valid, properly aligned `T` that outlives the I/O operation
/// - `ctx` must be the same pointer stored in [`Completion::context`] at submission time
/// - The `T` instance must not be moved or dropped while the operation is in flight
///
/// # Why not `Box<dyn IoHandler>`?
///
/// This pattern avoids heap allocation and dynamic dispatch. The shim is a static
/// function pointer; the only indirection is the `ctx` pointer which we need anyway.
unsafe fn typed_handler_shim<T: IoHandler>(ctx: *mut c_void, completion: &mut Completion) {
    // SAFETY: Caller guarantees `ctx` is a valid `*mut T` from the original submission.
    unsafe {
        let handler = &mut *(ctx as *mut T);
        T::on_complete(handler, completion);
    }
}

/// Tracks the lifecycle of a single I/O operation.
///
/// Uses intrusive linking (via [`ListLink`]) for zero-allocation queue management.
/// Each completion can be in at most one queue at a time, enforced by the type system.
///
/// # Lifecycle
///
/// 1. Initialize via [`new`](Self::new) (state: `Idle`)
/// 2. Set `op`, `context`, and `callback` fields
/// 3. Submit to [`IoBackend`] (state: `Queued` → `Submitted`)
/// 4. Backend signals completion (state: `Completed`)
/// 5. Call [`complete`](Self::complete) to invoke callback and transition to `Idle`
/// 6. Optionally call [`reset`](Self::reset) to clear fields for reuse
pub struct Completion {
    link: ListLink<Completion, IoTag>,
    state: CompletionState,

    /// Result from the kernel: byte count on success, negated errno on failure.
    pub result: i32,
    /// The operation that was performed.
    pub op: Operation,

    /// User-provided context passed to the callback.
    pub context: *mut c_void,
    /// Callback invoked when [`complete`](Self::complete) is called.
    pub callback: Option<CompletionCallback>,

    /// Backend-specific scratch space.
    /// Used by macOS backend to stash an Arc raw pointer.
    pub backend_context: usize,
}

impl Completion {
    /// Creates a new completion in the `Idle` state.
    pub const fn new() -> Self {
        Self {
            link: ListLink::new(),
            state: CompletionState::Idle,
            result: 0,
            op: Operation::Nop,
            context: core::ptr::null_mut(),
            callback: None,
            backend_context: 0,
        }
    }

    #[inline]
    pub fn state(&self) -> CompletionState {
        self.state
    }

    #[inline]
    pub fn is_idle(&self) -> bool {
        self.state == CompletionState::Idle
    }

    /// Resets the completion for reuse.
    ///
    /// # Panics
    ///
    /// Panics if the completion is still linked in a queue or in an invalid state
    /// for reset (`Queued` or `Submitted`).
    pub fn reset(&mut self) {
        assert!(!self.link.is_linked());
        assert!(self.state == CompletionState::Idle || self.state == CompletionState::Completed);

        self.link.reset();
        self.state = CompletionState::Idle;
        self.result = 0;
        self.op = Operation::Nop;
        self.context = core::ptr::null_mut();
        self.callback = None;
        self.backend_context = 0;

        assert!(self.is_idle());
        assert!(!self.link.is_linked());
    }

    /// Finalizes the completion by invoking the callback and transitioning to `Idle`.
    ///
    /// The callback receives the stored `context` pointer and mutable access to `self`,
    /// allowing immediate reuse or re-submission from within the callback.
    ///
    /// # Panics
    ///
    /// Panics if not in `Completed` state or if still linked.
    #[inline]
    pub fn complete(&mut self) {
        assert!(self.state == CompletionState::Completed);
        assert!(!self.link.is_linked());

        let cb = self.callback.take();
        let ctx = self.context;

        self.state = CompletionState::Idle;
        assert!(self.is_idle());

        if let Some(cb) = cb {
            // SAFETY: Caller guarantees context validity; completion is now Idle.
            unsafe { cb(ctx, self) };
        }
    }

    fn set_queued(&mut self) {
        assert!(self.state == CompletionState::Idle);
        self.state = CompletionState::Queued;
    }

    fn set_submitted(&mut self) {
        assert!(self.state == CompletionState::Queued || self.state == CompletionState::Idle);
        self.state = CompletionState::Submitted;
    }

    fn set_completed(&mut self) {
        assert!(self.state == CompletionState::Submitted);
        self.state = CompletionState::Completed;
    }
}

impl Default for Completion {
    fn default() -> Self {
        Self::new()
    }
}

impl ListNode<IoTag> for Completion {
    fn list_link(&mut self) -> &mut ListLink<Self, IoTag> {
        &mut self.link
    }

    fn list_link_ref(&self) -> &ListLink<Self, IoTag> {
        &self.link
    }
}
