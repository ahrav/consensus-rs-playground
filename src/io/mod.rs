//! Async I/O over platform completion APIs.
//!
//! Targets Linux `io_uring` and macOS Grand Central Dispatch (GCD).
//!
//! The API is split into:
//! - [`IoBackend`]: submit and drain completions
//! - [`Operation`]: read/write/fsync description
//! - [`Completion`]: per-op state and callback
//!
//! # Safety / Ownership
//!
//! `Completion` values must have a stable address (pinned or otherwise immovable) while an
//! operation is in flight. Buffers are caller-owned; this layer stores raw pointers and
//! identifies completions via `user_data` (`Completion*` cast to `u64`).

pub mod iops;

#[cfg(target_os = "linux")]
mod backend_linux;
#[cfg(target_os = "macos")]
mod backend_macos;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("This I/O layer currently supports only Linux (io_uring) and macOS (GCD).");

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
/// Callers queue operations, flush them to the kernel, then drain completions.
#[allow(clippy::result_unit_err)] // Simple success/failure semantics; no error details needed.
pub trait IoBackend {
    /// Minimum supported entries for this backend.
    const ENTRIES_MIN: u32;
    /// Maximum supported entries for this backend (inclusive).
    const ENTRIES_MAX: u32;

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

/// Intrusive list tag. A [`Completion`] can only be in one list at a time.
enum IoTag {}

/// Lifecycle state of a [`Completion`].
/// State machine: `Idle -> Queued -> Submitted -> Completed -> Idle`.
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

/// Describes an I/O operation.
///
/// # Buffer Ownership
///
/// `Read` and `Write` hold raw pointers to caller-owned buffers. Buffers must remain valid and
/// immovable until completion. `len` is `u32` to cap request sizes and match on-wire types.
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

    /// Asserts basic invariants for this operation.
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
/// Called from [`Completion::complete`]. The `context` pointer must be valid for the chosen
/// callback, and the [`Completion`] is already in [`Idle`](CompletionState::Idle).
pub type CompletionCallback = unsafe fn(*mut c_void, &mut Completion);

/// Handler trait for receiving I/O completion notifications.
///
/// Implement on your context type, then use [`typed_handler_shim`] as the
/// [`CompletionCallback`] to avoid allocation and dynamic dispatch.
///
/// # Safety
///
/// `completion` is in [`Idle`](CompletionState::Idle) and may be reused or re-submitted.
pub trait IoHandler {
    /// Called when an I/O operation completes.
    ///
    /// # Safety
    ///
    /// Caller must ensure `self` is the same instance that was stored in
    /// [`Completion::context`] when the operation was submitted.
    unsafe fn on_complete(&mut self, completion: &mut Completion);
}

/// Callback trampoline for [`IoHandler`] implementations.
///
/// Casts `ctx` back to `T` and calls [`IoHandler::on_complete`].
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
unsafe fn typed_handler_shim<T: IoHandler>(ctx: *mut c_void, completion: &mut Completion) {
    // SAFETY: Caller guarantees `ctx` is a valid `*mut T` from the original submission.
    unsafe {
        let handler = &mut *(ctx as *mut T);
        T::on_complete(handler, completion);
    }
}

/// Tracks the lifecycle of a single I/O operation.
///
/// An intrusive-list node (via [`ListLink`]) used for zero-allocation queue management.
///
/// Typical flow: create in `Idle`, fill fields, submit, then call [`complete`](Self::complete)
/// after the backend reports completion.
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

    /// Backend scratch space (macOS stores an `Arc` raw pointer).
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

    /// Resets the completion for reuse (clears all fields and returns to `Idle`).
    ///
    /// # Panics
    ///
    /// Panics if the completion is linked in a queue or in `Queued`/`Submitted`.
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

    /// Invokes the callback (if any) and transitions to `Idle`.
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

/// Manages I/O submission and completion.
///
/// Wraps a platform-specific [`IoBackend`] and queues locally when the backend is full.
///
/// # Backpressure Strategy
///
/// Instead of failing when the submission queue is full, `IoCore` keeps an overflow queue.
/// This can grow without bound if submissions outpace completions.
///
/// # Invariants
///
/// - `inflight` ≤ `capacity` (enforced by overflow queuing)
/// - `total_completed` ≤ `total_submitted`
/// - Must be idle (no in-flight or queued operations) before drop
pub struct IoCore<B: IoBackend> {
    backend: B,
    /// Operations waiting for backend capacity.
    overflow: DoublyLinkedList<Completion, IoTag>,
    /// Count of operations submitted to the backend but not yet completed.
    inflight: u32,
    /// Maximum concurrent operations the backend supports.
    #[allow(dead_code)] // Reserved for future backpressure/diagnostics.
    capacity: u32,

    // Metrics.
    total_submitted: u64,
    total_completed: u64,
}

impl<B: IoBackend> IoCore<B> {
    /// Creates a new I/O core with the specified queue depth.
    ///
    /// # Panics
    ///
    /// Panics if `entries` is outside `[ENTRIES_MIN, ENTRIES_MAX]` or not a power of two.
    pub fn new(entries: u32) -> io::Result<Self> {
        assert!(entries >= B::ENTRIES_MIN);
        assert!(entries <= B::ENTRIES_MAX);
        assert!(entries.is_power_of_two());

        Ok(Self {
            backend: B::new(entries)?,
            overflow: DoublyLinkedList::init(),
            inflight: 0,
            capacity: entries,
            total_submitted: 0,
            total_completed: 0,
        })
    }

    /// Returns `true` if no operations are in-flight or queued.
    #[inline]
    pub fn is_idle(&self) -> bool {
        self.inflight == 0 && self.overflow.is_empty()
    }

    /// Performs one iteration of the I/O loop: submit pending ops, flush, and reap completions.
    ///
    /// Blocks if any operations are in-flight (waits for at least one completion).
    /// Call repeatedly in your event loop or use [`run_for_ns`](Self::run_for_ns) for time-bounded polling.
    pub fn tick(&mut self) -> io::Result<()> {
        let old_completed = self.total_completed;

        self.fill_from_overflow();
        self.backend.flush(self.inflight > 0)?;
        let reaped = self.drain_completions();

        assert!(self.total_completed == old_completed + reaped as u64);
        Ok(())
    }

    /// Runs the I/O loop until idle or the time budget is exhausted.
    ///
    ///
    /// # Spin Protection
    ///
    /// Exits after `u32::MAX` iterations to prevent infinite loops if the system is stuck.
    pub fn run_for_ns(&mut self, budget_ns: u64) -> io::Result<()> {
        let deadline = Instant::now()
            .checked_add(Duration::from_nanos(budget_ns))
            .unwrap_or_else(Instant::now);

        let mut spins: u32 = 0;
        while !self.is_idle() && Instant::now() < deadline {
            self.tick()?;
            spins = spins.wrapping_add(1);
            if spins == 0 {
                // Very unlikely unless we reallllly stuck :(
                break;
            }
        }

        Ok(())
    }

    /// Submits an I/O operation with a raw callback.
    ///
    /// Prefer the typed [`read_for`](Self::read_for)/[`write_for`](Self::write_for)/[`fsync_for`](Self::fsync_for)
    /// variants which provide type safety via [`IoHandler`].
    ///
    /// # Panics
    ///
    /// - `completion` must be idle and unlinked
    /// - `op` must be active (not [`Nop`](Operation::Nop))
    ///
    /// # Safety (Caller Obligations)
    ///
    /// - `ctx` must remain valid until the callback is invoked
    /// - Any buffers in `op` must outlive the operation
    pub fn submit(
        &mut self,
        completion: &mut Completion,
        op: Operation,
        ctx: *mut c_void,
        cb: CompletionCallback,
    ) {
        assert!(completion.is_idle());
        assert!(!completion.link.is_linked());
        assert!(op.is_active());

        op.validate();
        completion.reset();
        completion.op = op;
        completion.context = ctx;
        completion.callback = Some(cb);
        completion.result = 0;

        self.enqueue(completion);
        self.total_submitted += 1;
    }

    /// Submits a read operation with a raw callback.
    ///
    /// # Panics
    ///
    /// Panics if `buf` is null.
    ///
    /// # Safety (Caller Obligations)
    ///
    /// - `buf[0..len]` must be valid for writes and outlive the operation
    /// - `ctx` must remain valid until the callback is invoked
    #[inline]
    #[allow(clippy::too_many_arguments)] // Matches kernel API surface.
    pub fn read(
        &mut self,
        completion: &mut Completion,
        fd: RawFd,
        buf: *mut u8,
        len: u32,
        offset: u64,
        ctx: *mut c_void,
        cb: CompletionCallback,
    ) {
        let buf = NonNull::new(buf).expect("buf must not be null");
        self.submit(
            completion,
            Operation::Read {
                fd,
                buf,
                len,
                offset,
            },
            ctx,
            cb,
        );
    }

    /// Submits a write operation with a raw callback.
    ///
    /// # Panics
    ///
    /// Panics if `buf` is null.
    ///
    /// # Safety (Caller Obligations)
    ///
    /// - `buf[0..len]` must be valid for reads and outlive the operation
    /// - `ctx` must remain valid until the callback is invoked
    #[allow(clippy::too_many_arguments)] // Matches kernel API surface.
    pub fn write(
        &mut self,
        completion: &mut Completion,
        fd: RawFd,
        buf: *const u8,
        len: u32,
        offset: u64,
        ctx: *mut c_void,
        cb: CompletionCallback,
    ) {
        let buf = NonNull::new(buf as *mut u8).expect("buf must not be null");
        self.submit(
            completion,
            Operation::Write {
                fd,
                buf,
                len,
                offset,
            },
            ctx,
            cb,
        );
    }

    /// Submits an fsync operation with a raw callback.
    ///
    /// # Safety (Caller Obligations)
    ///
    /// - `ctx` must remain valid until the callback is invoked
    #[inline]
    pub fn fsync(
        &mut self,
        completion: &mut Completion,
        fd: RawFd,
        ctx: *mut c_void,
        cb: CompletionCallback,
    ) {
        self.submit(completion, Operation::Fsync { fd }, ctx, cb);
    }

    /// Submits a read with type-safe callback dispatch via [`IoHandler`].
    ///
    /// The handler receives the completion in [`IoHandler::on_complete`] when the read finishes.
    ///
    /// # Safety (Caller Obligations)
    ///
    /// - `handler` must outlive the operation and not be moved while in-flight
    /// - `buf[0..len]` must be valid for writes and outlive the operation
    #[inline]
    pub fn read_for<T: IoHandler>(
        &mut self,
        handler: &mut T,
        completion: &mut Completion,
        fd: RawFd,
        buf: *mut u8,
        len: u32,
        offset: u64,
    ) {
        let ctx_ptr = handler as *mut T as *mut c_void;
        self.read(
            completion,
            fd,
            buf,
            len,
            offset,
            ctx_ptr,
            typed_handler_shim::<T>,
        );
    }

    /// Submits a write with type-safe callback dispatch via [`IoHandler`].
    ///
    /// The handler receives the completion in [`IoHandler::on_complete`] when the write finishes.
    ///
    /// # Safety (Caller Obligations)
    ///
    /// - `handler` must outlive the operation and not be moved while in-flight
    /// - `buf[0..len]` must be valid for reads and outlive the operation
    #[inline]
    pub fn write_for<T: IoHandler>(
        &mut self,
        handler: &mut T,
        completion: &mut Completion,
        fd: RawFd,
        buf: *const u8,
        len: u32,
        offset: u64,
    ) {
        let ctx_ptr = handler as *mut T as *mut c_void;
        self.write(
            completion,
            fd,
            buf,
            len,
            offset,
            ctx_ptr,
            typed_handler_shim::<T>,
        );
    }

    /// Submits an fsync with type-safe callback dispatch via [`IoHandler`].
    ///
    /// The handler receives the completion in [`IoHandler::on_complete`] when the fsync finishes.
    ///
    /// # Safety (Caller Obligations)
    ///
    /// - `handler` must outlive the operation and not be moved while in-flight
    #[inline]
    pub fn fsync_for<T: IoHandler>(
        &mut self,
        handler: &mut T,
        completion: &mut Completion,
        fd: RawFd,
    ) {
        let ctx_ptr = handler as *mut T as *mut c_void;
        self.fsync(completion, fd, ctx_ptr, typed_handler_shim::<T>);
    }

    /// Submits an operation, queuing locally if the backend is full.
    ///
    /// The completion transitions to `Submitted` if accepted by the backend,
    /// or `Queued` if placed in the overflow queue.
    #[inline]
    fn enqueue(&mut self, completion: &mut Completion) {
        let old_inflight = self.inflight;
        if self.try_submit_one(completion).is_ok() {
            completion.set_submitted();
            self.inflight += 1;
            assert!(self.inflight == old_inflight + 1);
        } else {
            completion.set_queued();
            self.overflow.push_back(completion);
            assert!(self.inflight == old_inflight);
        }
    }

    /// Attempts to push directly to the backend (no overflow queuing).
    ///
    /// Returns `Err(())` if the backend submission queue is full. Does not update state.
    #[inline]
    #[allow(clippy::result_unit_err)] // Simple success/failure; matches IoBackend::try_push.
    pub fn try_submit_one(&mut self, completion: &mut Completion) -> Result<(), ()> {
        let user_data = completion as *mut Completion as u64;
        // SAFETY: Caller ensures completion and its buffers outlive the operation.
        unsafe { self.backend.try_push(&completion.op, user_data) }
    }

    /// Submits overflowed operations until the backend is full or the queue is empty.
    fn fill_from_overflow(&mut self) {
        while let Some(mut node) = self.overflow.pop_front() {
            // SAFETY: Node came from our overflow list, which only contains valid completions.
            let completion = unsafe { node.as_mut() };
            let old_inflight = self.inflight;

            match self.try_submit_one(completion) {
                Ok(()) => {
                    completion.set_submitted();
                    self.inflight += 1;
                    assert!(self.inflight == old_inflight + 1);
                }
                Err(()) => {
                    // Backend is full; put it back and stop.
                    // Note: completion is already in Queued state from initial enqueue.
                    debug_assert!(completion.state == CompletionState::Queued);
                    self.overflow.push_front(completion); // Push front to maintain FIFO order.
                    assert!(self.inflight == old_inflight);
                    break;
                }
            }
        }
    }

    /// Reaps all available completions from the backend.
    ///
    /// Updates each [`Completion`], calls [`Completion::complete`], and returns the count.
    fn drain_completions(&mut self) -> u32 {
        let mut reaped: u32 = 0;

        self.backend.drain(|user_data, result| {
            assert!(user_data != 0);
            // SAFETY: user_data was set to a valid Completion pointer in try_submit_one.
            let completion = unsafe { &mut *(user_data as *mut Completion) };
            assert!(completion.state() == CompletionState::Submitted);

            completion.result = result;
            completion.set_completed();
            completion.complete();
            reaped += 1;
        });

        assert!(reaped <= self.inflight);
        self.inflight -= reaped;
        self.total_completed += reaped as u64;
        reaped
    }
}

impl<B: IoBackend> Drop for IoCore<B> {
    fn drop(&mut self) {
        // It is a logic error to drop the I/O system with in-flight ops.
        assert!(self.is_idle(), "IoCore dropped with in-flight operations");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    // =========================================================================
    // Mock Backend
    // =========================================================================

    /// Mock backend that simulates submission queue behavior for testing IoCore logic.
    ///
    /// Operations are queued on `try_push`, moved to completions on `flush`,
    /// and delivered via `drain`. Supports configurable failure injection.
    pub(super) struct MockBackend {
        capacity: u32,
        /// Operations waiting to be "submitted" to kernel.
        submission_queue: VecDeque<(Operation, u64)>,
        /// Completed operations ready to be drained.
        completion_queue: VecDeque<(u64, i32)>,
        /// If set, `flush` returns this error.
        flush_error: Option<io::ErrorKind>,
        /// Result to return for each completion (default: 0 = success).
        default_result: i32,
    }

    impl MockBackend {
        pub(super) fn with_result(entries: u32, result: i32) -> io::Result<Self> {
            Ok(Self {
                capacity: entries,
                submission_queue: VecDeque::new(),
                completion_queue: VecDeque::new(),
                flush_error: None,
                default_result: result,
            })
        }
    }

    impl IoBackend for MockBackend {
        const ENTRIES_MIN: u32 = 1;
        const ENTRIES_MAX: u32 = 4096;

        fn new(entries: u32) -> io::Result<Self> {
            Self::with_result(entries, 0)
        }

        unsafe fn try_push(&mut self, op: &Operation, user_data: u64) -> Result<(), ()> {
            if self.submission_queue.len() >= self.capacity as usize {
                return Err(());
            }
            self.submission_queue.push_back((*op, user_data));
            Ok(())
        }

        fn flush(&mut self, _wait_for_one: bool) -> io::Result<()> {
            if let Some(kind) = self.flush_error {
                return Err(io::Error::new(kind, "mock flush error"));
            }
            // Move all submissions to completions (simulating instant completion).
            while let Some((_, user_data)) = self.submission_queue.pop_front() {
                self.completion_queue
                    .push_back((user_data, self.default_result));
            }
            Ok(())
        }

        fn drain<F: FnMut(u64, i32)>(&mut self, mut f: F) {
            while let Some((user_data, result)) = self.completion_queue.pop_front() {
                f(user_data, result);
            }
        }
    }

    // =========================================================================
    // Test Helpers
    // =========================================================================

    /// Creates a valid Read operation for testing.
    fn test_read_op(buf: &mut [u8]) -> Operation {
        Operation::Read {
            fd: 3, // Arbitrary valid fd
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: buf.len() as u32,
            offset: 0,
        }
    }

    /// Creates a valid Write operation for testing.
    fn test_write_op(buf: &[u8]) -> Operation {
        Operation::Write {
            fd: 3,
            buf: NonNull::new(buf.as_ptr() as *mut u8).unwrap(),
            len: buf.len() as u32,
            offset: 0,
        }
    }

    /// Test handler that tracks completion count and results.
    pub(super) struct TestHandler {
        pub(super) completed_count: u32,
        pub(super) results: Vec<i32>,
    }

    impl TestHandler {
        pub(super) fn new() -> Self {
            Self {
                completed_count: 0,
                results: Vec::new(),
            }
        }
    }

    impl IoHandler for TestHandler {
        unsafe fn on_complete(&mut self, completion: &mut Completion) {
            self.completed_count += 1;
            self.results.push(completion.result);
        }
    }

    // =========================================================================
    // Operation::validate() Tests
    // =========================================================================

    #[test]
    fn validate_accepts_read() {
        let mut buf = [0u8; 64];
        let op = test_read_op(&mut buf);
        op.validate(); // Should not panic
    }

    #[test]
    fn validate_accepts_write() {
        let buf = [0u8; 64];
        let op = test_write_op(&buf);
        op.validate(); // Should not panic
    }

    #[test]
    fn validate_accepts_fsync() {
        let op = Operation::Fsync { fd: 3 };
        op.validate(); // Should not panic
    }

    #[test]
    fn validate_accepts_nop() {
        Operation::Nop.validate(); // Should not panic
    }

    #[test]
    #[should_panic(expected = "File descriptor must be non-negative")]
    fn validate_rejects_negative_fd_read() {
        let op = Operation::Read {
            fd: -1,
            buf: NonNull::dangling(),
            len: 1,
            offset: 0,
        };
        op.validate();
    }

    #[test]
    #[should_panic(expected = "File descriptor must be non-negative")]
    fn validate_rejects_negative_fd_write() {
        let op = Operation::Write {
            fd: -1,
            buf: NonNull::dangling(),
            len: 1,
            offset: 0,
        };
        op.validate();
    }

    #[test]
    #[should_panic(expected = "File descriptor must be non-negative")]
    fn validate_rejects_negative_fd_fsync() {
        let op = Operation::Fsync { fd: -1 };
        op.validate();
    }

    #[test]
    #[should_panic(expected = "Length must be positive")]
    fn validate_rejects_zero_length_read() {
        let op = Operation::Read {
            fd: 3,
            buf: NonNull::dangling(),
            len: 0,
            offset: 0,
        };
        op.validate();
    }

    #[test]
    #[should_panic(expected = "Length must be positive")]
    fn validate_rejects_zero_length_write() {
        let op = Operation::Write {
            fd: 3,
            buf: NonNull::dangling(),
            len: 0,
            offset: 0,
        };
        op.validate();
    }

    #[test]
    fn is_active_identifies_operations() {
        assert!(!Operation::Nop.is_active());

        let mut buf = [0u8; 1];
        assert!(test_read_op(&mut buf).is_active());

        let buf = [0u8; 1];
        assert!(test_write_op(&buf).is_active());

        assert!(Operation::Fsync { fd: 0 }.is_active());
    }

    // =========================================================================
    // Completion State Machine Tests
    // =========================================================================

    #[test]
    fn completion_initial_state() {
        let comp = Completion::new();
        assert!(comp.is_idle());
        assert_eq!(comp.state(), CompletionState::Idle);
    }

    #[test]
    fn completion_default_idle() {
        let comp = Completion::default();
        assert!(comp.is_idle());
    }

    #[test]
    fn completion_reset_clears_all_fields() {
        let mut comp = Completion::new();
        // Transition to Completed state (simulating a completed operation).
        comp.state = CompletionState::Completed;
        comp.result = 42;
        comp.context = 0x1234 as *mut c_void;
        comp.backend_context = 999;

        comp.reset();

        assert!(comp.is_idle());
        assert_eq!(comp.result, 0);
        assert!(comp.context.is_null());
        assert!(comp.callback.is_none());
        assert_eq!(comp.backend_context, 0);
    }

    #[test]
    #[should_panic]
    fn completion_reset_panics_if_queued() {
        let mut comp = Completion::new();
        comp.state = CompletionState::Queued;
        comp.reset();
    }

    #[test]
    #[should_panic]
    fn completion_reset_panics_if_submitted() {
        let mut comp = Completion::new();
        comp.state = CompletionState::Submitted;
        comp.reset();
    }

    // =========================================================================
    // typed_handler_shim Tests
    // =========================================================================

    #[test]
    fn typed_handler_shim_invokes_handler() {
        let mut handler = TestHandler::new();
        let mut completion = Completion::new();
        completion.result = 42;
        // Note: complete() requires Completed state and will transition to Idle.
        // For testing the shim directly, we call it without full state machine.

        let ctx_ptr = &mut handler as *mut TestHandler as *mut c_void;

        unsafe {
            typed_handler_shim::<TestHandler>(ctx_ptr, &mut completion);
        }

        assert_eq!(handler.completed_count, 1);
        assert_eq!(handler.results, vec![42]);
    }

    #[test]
    fn typed_handler_shim_handles_multiple_calls() {
        let mut handler = TestHandler::new();
        let mut completion = Completion::new();
        let ctx_ptr = &mut handler as *mut TestHandler as *mut c_void;

        for i in 0..5 {
            completion.result = i * 10;
            unsafe {
                typed_handler_shim::<TestHandler>(ctx_ptr, &mut completion);
            }
        }

        assert_eq!(handler.completed_count, 5);
        assert_eq!(handler.results, vec![0, 10, 20, 30, 40]);
    }

    #[test]
    fn typed_handler_shim_handles_negative_results() {
        let mut handler = TestHandler::new();
        let mut completion = Completion::new();
        completion.result = -5; // Simulated I/O error (EIO = 5)
        let ctx_ptr = &mut handler as *mut TestHandler as *mut c_void;

        unsafe {
            typed_handler_shim::<TestHandler>(ctx_ptr, &mut completion);
        }

        assert_eq!(handler.completed_count, 1);
        assert_eq!(handler.results[0], -5);
    }

    // =========================================================================
    // IoCore with MockBackend Tests
    // =========================================================================

    #[test]
    fn io_core_new_checks_entries() {
        // Valid: power of two within range.
        let io: IoCore<MockBackend> = IoCore::new(16).unwrap();
        assert!(io.is_idle());
    }

    #[test]
    #[should_panic]
    fn io_core_new_rejects_non_power_of_two() {
        let _: IoCore<MockBackend> = IoCore::new(15).unwrap();
    }

    #[test]
    #[should_panic]
    fn io_core_new_rejects_zero() {
        let _: IoCore<MockBackend> = IoCore::new(0).unwrap();
    }

    #[test]
    fn io_core_submit_complete_one() {
        let mut io: IoCore<MockBackend> = IoCore::new(16).unwrap();
        let mut buf = [0u8; 64];
        let mut comp = Completion::new();

        let op = test_read_op(&mut buf);
        io.submit(&mut comp, op, core::ptr::null_mut(), |_, _| {});

        assert_eq!(io.total_submitted, 1);
        assert!(!io.is_idle());

        io.tick().unwrap();

        assert_eq!(io.total_completed, 1);
        assert!(io.is_idle());
        assert!(comp.is_idle());
    }

    #[test]
    fn io_core_state_machine_idle_to_completed() {
        let mut io: IoCore<MockBackend> = IoCore::new(16).unwrap();
        let mut buf = [0u8; 64];
        let mut comp = Completion::new();

        assert_eq!(comp.state(), CompletionState::Idle);

        let op = test_read_op(&mut buf);
        io.submit(&mut comp, op, core::ptr::null_mut(), |_, _| {});

        // After submit with capacity available, should be Submitted.
        assert_eq!(comp.state(), CompletionState::Submitted);

        io.tick().unwrap();

        // After tick completes, should be back to Idle.
        assert_eq!(comp.state(), CompletionState::Idle);
    }

    #[test]
    fn io_core_overflow_queue_backpressure() {
        // Capacity of 2, submit 5 operations.
        let mut io: IoCore<MockBackend> = IoCore::new(2).unwrap();
        let mut bufs = [[0u8; 64]; 5];
        let mut comps = [
            Completion::new(),
            Completion::new(),
            Completion::new(),
            Completion::new(),
            Completion::new(),
        ];

        for (i, comp) in comps.iter_mut().enumerate() {
            let op = test_read_op(&mut bufs[i]);
            io.submit(comp, op, core::ptr::null_mut(), |_, _| {});
        }

        // First 2 should be Submitted, remaining 3 should be Queued (overflow).
        assert_eq!(comps[0].state(), CompletionState::Submitted);
        assert_eq!(comps[1].state(), CompletionState::Submitted);
        assert_eq!(comps[2].state(), CompletionState::Queued);
        assert_eq!(comps[3].state(), CompletionState::Queued);
        assert_eq!(comps[4].state(), CompletionState::Queued);

        assert_eq!(io.total_submitted, 5);
        assert!(!io.overflow.is_empty());

        // Tick should complete the first batch and submit overflow.
        io.tick().unwrap();

        // After first tick: 2 completed, 2 more submitted from overflow, 1 queued.
        assert_eq!(io.total_completed, 2);

        // Continue ticking until all complete.
        while !io.is_idle() {
            io.tick().unwrap();
        }

        assert_eq!(io.total_completed, 5);
        assert!(io.overflow.is_empty());

        for comp in &comps {
            assert!(comp.is_idle());
        }
    }

    #[test]
    fn io_core_run_for_ns_completes_all() {
        let mut io: IoCore<MockBackend> = IoCore::new(4).unwrap();
        let mut bufs = [[0u8; 64]; 10];
        let mut comps: Vec<Completion> = (0..10).map(|_| Completion::new()).collect();

        for (i, comp) in comps.iter_mut().enumerate() {
            let op = test_read_op(&mut bufs[i]);
            io.submit(comp, op, core::ptr::null_mut(), |_, _| {});
        }

        // Run with generous timeout.
        io.run_for_ns(1_000_000_000).unwrap();

        assert!(io.is_idle());
        assert_eq!(io.total_submitted, 10);
        assert_eq!(io.total_completed, 10);
    }

    #[test]
    fn io_core_metrics_consistent() {
        let mut io: IoCore<MockBackend> = IoCore::new(8).unwrap();
        let mut bufs = [[0u8; 64]; 20];
        let mut comps: Vec<Completion> = (0..20).map(|_| Completion::new()).collect();

        // Submit all.
        for (i, comp) in comps.iter_mut().enumerate() {
            let op = test_read_op(&mut bufs[i]);
            io.submit(comp, op, core::ptr::null_mut(), |_, _| {});

            // Invariant: total_completed <= total_submitted.
            assert!(io.total_completed <= io.total_submitted);
        }

        // Process in batches.
        while !io.is_idle() {
            let old_completed = io.total_completed;
            io.tick().unwrap();

            // Invariant: completed is monotonically increasing.
            assert!(io.total_completed >= old_completed);
            // Invariant: total_completed <= total_submitted.
            assert!(io.total_completed <= io.total_submitted);
        }

        assert_eq!(io.total_submitted, 20);
        assert_eq!(io.total_completed, 20);
    }

    #[test]
    fn io_core_callback_receives_result() {
        thread_local! {
            static CALLBACK_RESULT: RefCell<Option<i32>> = const { RefCell::new(None) };
        }

        unsafe fn test_callback(_ctx: *mut c_void, comp: &mut Completion) {
            CALLBACK_RESULT.with(|r| *r.borrow_mut() = Some(comp.result));
        }

        let mut io: IoCore<MockBackend> = IoCore::new(16).unwrap();
        let mut buf = [0u8; 64];
        let mut comp = Completion::new();

        let op = test_read_op(&mut buf);
        io.submit(&mut comp, op, core::ptr::null_mut(), test_callback);
        io.tick().unwrap();

        CALLBACK_RESULT.with(|r| {
            assert_eq!(*r.borrow(), Some(0)); // MockBackend returns 0 by default.
        });
    }

    #[test]
    fn io_core_read_for_uses_typed_handler() {
        let mut io: IoCore<MockBackend> = IoCore::new(16).unwrap();
        let mut handler = TestHandler::new();
        let mut buf = [0u8; 64];
        let mut comp = Completion::new();

        io.read_for(&mut handler, &mut comp, 3, buf.as_mut_ptr(), 64, 0);
        io.tick().unwrap();

        assert_eq!(handler.completed_count, 1);
        assert_eq!(handler.results, vec![0]);
    }

    #[test]
    fn io_core_write_for_uses_typed_handler() {
        let mut io: IoCore<MockBackend> = IoCore::new(16).unwrap();
        let mut handler = TestHandler::new();
        let buf = [0u8; 64];
        let mut comp = Completion::new();

        io.write_for(&mut handler, &mut comp, 3, buf.as_ptr(), 64, 0);
        io.tick().unwrap();

        assert_eq!(handler.completed_count, 1);
    }

    #[test]
    fn io_core_fsync_for_uses_typed_handler() {
        let mut io: IoCore<MockBackend> = IoCore::new(16).unwrap();
        let mut handler = TestHandler::new();
        let mut comp = Completion::new();

        io.fsync_for(&mut handler, &mut comp, 3);
        io.tick().unwrap();

        assert_eq!(handler.completed_count, 1);
    }

    #[test]
    #[should_panic(expected = "IoCore dropped with in-flight operations")]
    fn io_core_drop_panics_with_inflight() {
        let mut io: IoCore<MockBackend> = IoCore::new(16).unwrap();
        let mut buf = [0u8; 64];
        let mut comp = Completion::new();

        let op = test_read_op(&mut buf);
        io.submit(&mut comp, op, core::ptr::null_mut(), |_, _| {});

        // Drop without completing - should panic.
        drop(io);
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn io_core_handles_completion_with_error_result() {
        thread_local! {
            static ERROR_RESULT: RefCell<Option<i32>> = const { RefCell::new(None) };
        }

        unsafe fn error_callback(_ctx: *mut c_void, comp: &mut Completion) {
            ERROR_RESULT.with(|r| *r.borrow_mut() = Some(comp.result));
        }

        // Create backend that returns error result (EIO = 5).
        let backend = MockBackend::with_result(16, -5).unwrap();
        let mut io = IoCore {
            backend,
            overflow: DoublyLinkedList::init(),
            inflight: 0,
            capacity: 16,
            total_submitted: 0,
            total_completed: 0,
        };

        let mut buf = [0u8; 64];
        let mut comp = Completion::new();
        let op = test_read_op(&mut buf);

        io.submit(&mut comp, op, core::ptr::null_mut(), error_callback);
        io.tick().unwrap();

        ERROR_RESULT.with(|r| {
            assert_eq!(*r.borrow(), Some(-5));
        });
    }
}

#[cfg(test)]
mod property_tests {
    use super::tests::*;
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Random submission/tick sequences maintain invariants.
        #[test]
        fn random_operations_invariants(
            ops in prop::collection::vec(prop::bool::ANY, 0..100)
        ) {
            let mut io: IoCore<MockBackend> = IoCore::new(8).unwrap();
            let mut bufs: Vec<[u8; 64]> = (0..32).map(|_| [0u8; 64]).collect();
            let mut comps: Vec<Completion> = (0..32).map(|_| Completion::new()).collect();
            let mut next_comp_idx = 0;

            for should_submit in ops {
                if should_submit && next_comp_idx < comps.len() {
                    // Find an idle completion to reuse or use next fresh one.
                    let comp_idx = (0..next_comp_idx)
                        .find(|&i| comps[i].is_idle())
                        .unwrap_or_else(|| {
                            let idx = next_comp_idx;
                            next_comp_idx += 1;
                            idx
                        });

                    let op = Operation::Read {
                        fd: 3,
                        buf: NonNull::new(bufs[comp_idx].as_mut_ptr()).unwrap(),
                        len: 64,
                        offset: 0,
                    };
                    io.submit(&mut comps[comp_idx], op, core::ptr::null_mut(), |_, _| {});
                }

                // Always tick to process.
                io.tick().unwrap();

                // Invariants must hold after every operation.
                prop_assert!(io.total_completed <= io.total_submitted);
                prop_assert!(io.inflight <= io.capacity);
            }

            // Drain remaining.
            while !io.is_idle() {
                io.tick().unwrap();
            }

            prop_assert_eq!(io.total_completed, io.total_submitted);
            prop_assert!(io.is_idle());
        }

        /// Handler receives correct results for various result values.
        #[test]
        fn handler_receives_arbitrary_results(result in any::<i32>()) {
            let backend = MockBackend::with_result(16, result).unwrap();
            let mut io = IoCore {
                backend,
                overflow: DoublyLinkedList::init(),
                inflight: 0,
                capacity: 16,
                total_submitted: 0,
                total_completed: 0,
            };

            let mut handler = TestHandler::new();
            let mut buf = [0u8; 64];
            let mut comp = Completion::new();

            io.read_for(&mut handler, &mut comp, 3, buf.as_mut_ptr(), 64, 0);
            io.tick().unwrap();

            prop_assert_eq!(handler.completed_count, 1);
            prop_assert_eq!(handler.results[0], result);
        }

        /// Overflow queue correctly handles varying capacities.
        #[test]
        fn overflow_queue_various_capacities(
            capacity_exp in 0u32..6,  // 1, 2, 4, 8, 16, 32
            num_ops in 1usize..50
        ) {
            let capacity = 1u32 << capacity_exp;
            let mut io: IoCore<MockBackend> = IoCore::new(capacity).unwrap();
            let mut bufs: Vec<[u8; 64]> = (0..num_ops).map(|_| [0u8; 64]).collect();
            let mut comps: Vec<Completion> = (0..num_ops).map(|_| Completion::new()).collect();

            // Submit all operations.
            for (i, comp) in comps.iter_mut().enumerate() {
                let op = Operation::Read {
                    fd: 3,
                    buf: NonNull::new(bufs[i].as_mut_ptr()).unwrap(),
                    len: 64,
                    offset: 0,
                };
                io.submit(comp, op, core::ptr::null_mut(), |_, _| {});
            }

            prop_assert_eq!(io.total_submitted, num_ops as u64);

            // Complete all.
            while !io.is_idle() {
                io.tick().unwrap();
            }

            prop_assert_eq!(io.total_completed, num_ops as u64);
            prop_assert!(io.overflow.is_empty());
        }
    }
}
