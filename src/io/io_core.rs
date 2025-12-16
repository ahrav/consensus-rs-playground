//! Core I/O types, traits, and the [`IoCore`] manager.
//!
//! This module contains the platform-agnostic abstractions for async I/O:
//! - [`IoBackend`]: trait for platform-specific submission/completion
//! - [`Operation`]: describes read/write/fsync operations
//! - [`Completion`]: tracks lifecycle of a single I/O operation
//! - [`IoCore`]: manages submission queues and completion handling

use core::ffi::c_void;
use core::ptr::NonNull;
use std::io;
use std::os::unix::io::RawFd;
use std::time::{Duration, Instant};

use crate::stdx::{DoublyLinkedList, ListLink, ListNode};

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
pub(crate) enum IoTag {}

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
                assert!(len <= i32::MAX as u32, "Length must fit in i32");
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
pub unsafe fn typed_handler_shim<T: IoHandler>(ctx: *mut c_void, completion: &mut Completion) {
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
    pub(crate) link: ListLink<Completion, IoTag>,
    pub(crate) state: CompletionState,

    /// Result from the kernel: byte count on success, negated errno on failure.
    pub result: i32,
    /// The operation that was performed.
    pub op: Operation,

    /// User-provided context passed to the callback.
    pub context: *mut c_void,
    /// Callback invoked when [`complete`](Self::complete) is called.
    pub callback: Option<CompletionCallback>,

    /// Backend scratch space (backend-specific; opaque to callers).
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

    pub(crate) fn set_queued(&mut self) {
        assert!(self.state == CompletionState::Idle);
        self.state = CompletionState::Queued;
    }

    pub(crate) fn set_submitted(&mut self) {
        assert!(self.state == CompletionState::Queued || self.state == CompletionState::Idle);
        self.state = CompletionState::Submitted;
    }

    pub(crate) fn set_completed(&mut self) {
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
    pub(crate) backend: B,
    /// Operations waiting for backend capacity.
    pub(crate) overflow: DoublyLinkedList<Completion, IoTag>,
    /// Count of operations submitted to the backend but not yet completed.
    pub(crate) inflight: u32,
    /// Maximum concurrent operations the backend supports.
    #[allow(dead_code)] // Reserved for future backpressure/diagnostics.
    pub(crate) capacity: u32,

    // Metrics.
    pub(crate) total_submitted: u64,
    pub(crate) total_completed: u64,
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
