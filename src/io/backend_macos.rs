//! macOS I/O backend using Grand Central Dispatch (GCD).
//!
//! Uses libdispatch's global concurrent queue to execute blocking I/O operations
//! off the main thread. Worker tasks park on a semaphore waiting for submissions,
//! then signal completion via a separate queue/semaphore pair.
//!
//! # Why GCD over raw threads?
//! GCD provides system-managed thread pooling with automatic load balancing.
//! This avoids thread explosion under bursty workloads and integrates with
//! macOS power management.

use core::ffi::c_void;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_queue::ArrayQueue;

use crate::io::{Completion, IoBackend, Operation};

// Raw bindings to libdispatch. We use the C function-pointer API (`dispatch_async_f`)
// rather than blocks to avoid requiring the `block` crate.
#[link(name = "dispatch")]
unsafe extern "C" {
    fn dispatch_get_global_queue(identifier: libc::c_long, flags: libc::c_long) -> *mut c_void;
    fn dispatch_async_f(queue: *mut c_void, context: *mut c_void, work: extern "C" fn(*mut c_void));

    fn dispatch_semaphore_create(value: libc::c_long) -> *mut c_void;
    fn dispatch_semaphore_signal(sema: *mut c_void) -> libc::c_long;
    fn dispatch_semaphore_wait(sema: *mut c_void, timeout: u64) -> libc::c_long;

    fn dispatch_release(obj: *mut c_void);
}

const DISPATCH_TIME_NOW: u64 = 0;
const DISPATCH_TIME_FOREVER: u64 = !0;

/// Caps completions drained per call to bound latency under heavy load.
const MAX_DRAIN_PER_CALL: u32 = 16_384;

/// EINTR retry limit. Guards against signal storms causing infinite loops.
const MAX_EINTR_RETRIES: u32 = 1_000;

/// Worker count ceiling. More workers than this provides diminishing returns
/// and wastes GCD thread-pool capacity.
const WORKER_COUNT_MAX: u32 = 64;

/// Submissions processed per worker wake before yielding.
/// Batching amortizes semaphore overhead and improves cache locality.
const MAX_SUBMISSIONS_PER_WAKE: u32 = 256;

const _: () = {
    assert!(MAX_DRAIN_PER_CALL >= 1024);
    assert!(MAX_DRAIN_PER_CALL <= u32::MAX / 2);

    assert!(MAX_EINTR_RETRIES >= 100);

    assert!(WORKER_COUNT_MAX >= 1);
    assert!(MAX_SUBMISSIONS_PER_WAKE >= 1);
    assert!(MAX_SUBMISSIONS_PER_WAKE <= 4096);
};

/// State shared between the backend and all GCD worker tasks.
///
/// Workers wait on `submit_sema`, pop from `submit`, execute the operation,
/// push to `done`, and signal `done_sema`.
struct Shared {
    submit: ArrayQueue<u64>,
    submit_sema: *mut c_void,

    done: ArrayQueue<u64>,
    done_sema: *mut c_void,

    stop: AtomicBool,

    capacity: u32,
}

// SAFETY: The raw pointers are dispatch objects which are internally synchronized.
// All mutable state is either atomic or protected by the semaphore protocol.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Drop for Shared {
    fn drop(&mut self) {
        assert!(!self.submit_sema.is_null());
        assert!(!self.done_sema.is_null());

        // SAFETY: Semaphores were created successfully in `GcdBackend::new`.
        unsafe {
            dispatch_release(self.submit_sema);
            dispatch_release(self.done_sema);
        }
    }
}

/// Context passed to each GCD worker task. Boxed and leaked to FFI,
/// ownership returns in [`worker_trampoline`].
struct WorkerCtx {
    shared: Arc<Shared>,
    worker_index: u32,
}

/// macOS I/O backend using GCD's global concurrent queue.
///
/// Operations are dispatched to worker tasks that execute blocking I/O,
/// then signal completion. See module docs for architecture details.
pub struct GcdBackend {
    workq: *mut c_void,
    shared: Arc<Shared>,

    capacity: u32,
    workers: u32,

    /// Currently in-flight operations (submitted but not yet drained).
    outstanding: u32,
}

impl IoBackend for GcdBackend {
    const ENTRIES_MIN: u32 = 1;
    const ENTRIES_MAX: u32 = 65_536;

    fn new(entries: u32) -> io::Result<Self> {
        assert!(entries >= Self::ENTRIES_MIN);
        assert!(entries <= Self::ENTRIES_MAX);

        // SAFETY: Returns the default-priority global queue, never null.
        let workq = unsafe { dispatch_get_global_queue(0, 0) };
        assert!(!workq.is_null());

        // SAFETY: Initial count of 0 means waiters block until signaled.
        let submit_sema = unsafe { dispatch_semaphore_create(0) };
        assert!(!submit_sema.is_null());

        let done_sema = unsafe { dispatch_semaphore_create(0) };
        assert!(!done_sema.is_null());

        let submit = ArrayQueue::new(entries as usize);
        let done = ArrayQueue::new(entries as usize);

        let workers = worker_count(entries);
        assert!(workers > 0);
        assert!(workers <= entries);

        let shared = Arc::new(Shared {
            submit,
            submit_sema,
            done,
            done_sema,
            stop: AtomicBool::new(false),
            capacity: entries,
        });

        for worker_index in 0..workers {
            let ctx = Box::new(WorkerCtx {
                shared: shared.clone(),
                worker_index,
            });

            // SAFETY: `Box::into_raw` transfers ownership to FFI. The worker
            // reclaims it in `worker_trampoline`.
            unsafe {
                dispatch_async_f(workq, Box::into_raw(ctx) as *mut c_void, worker_trampoline);
            }
        }

        Ok(Self {
            workq,
            shared,
            capacity: entries,
            workers,
            outstanding: 0,
        })
    }

    /// Submits an operation to a GCD worker for async execution.
    ///
    /// Returns `Err(())` if the submission queue is full (backpressure).
    /// The caller must retry after draining completions.
    ///
    /// # Safety
    /// - `user_data` must be a valid `Completion*` cast to `u64`
    /// - The `Completion` must remain valid until drained
    /// - The operation's fd and buffer must remain valid until completion
    unsafe fn try_push(&mut self, op: &Operation, user_data: u64) -> Result<(), ()> {
        // Validate operation parameters before submitting.
        // Offset check: pread/pwrite take off_t (i64), reject values that would overflow.
        match *op {
            Operation::Nop => {}

            Operation::Read {
                fd,
                buf: _,
                len,
                offset,
            } => {
                assert!(fd >= 0, "invalid file descriptor: {fd}");
                assert!(len > 0, "read length must be positive");
                assert!(
                    offset <= i64::MAX as u64,
                    "offset {offset} exceeds off_t max"
                );
            }

            Operation::Write {
                fd,
                buf: _,
                len,
                offset,
            } => {
                assert!(fd >= 0, "invalid file descriptor: {fd}");
                assert!(len > 0, "write length must be positive");
                assert!(
                    offset <= i64::MAX as u64,
                    "offset {offset} exceeds off_t max"
                );
            }

            Operation::Fsync { fd } => {
                assert!(fd >= 0, "invalid file descriptor: {fd}");
            }
        }

        assert!(user_data != 0, "null completion pointer");

        // Backpressure: reject if queue is full.
        if self.outstanding >= self.capacity {
            return Err(());
        }

        let outstanding_before = self.outstanding;
        self.outstanding += 1;

        // SAFETY: Caller guarantees user_data is a valid Completion pointer.
        // backend_context == 0 means not in-flight; we're taking ownership.
        let completion = unsafe { &mut *(user_data as *mut Completion) };
        assert!(
            completion.backend_context == 0,
            "completion already in flight"
        );

        completion.backend_context = 1; // Mark in-flight.

        let push_result = self.shared.submit.push(user_data);
        assert!(push_result.is_ok(), "queue full despite capacity check");

        // SAFETY: Semaphore valid for lifetime of Shared.
        // Signal wakes a worker to process this submission.
        unsafe { dispatch_semaphore_signal(self.shared.submit_sema) };

        assert!(self.outstanding == outstanding_before + 1);
        assert!(self.outstanding <= self.capacity);

        Ok(())
    }

    /// Ensures at least one completion is available if `wait_for_one` is true.
    ///
    /// Blocks until a worker signals the done semaphore, then immediately
    /// re-signals so `drain` can consume it. This peek-and-restore pattern
    /// avoids consuming the semaphore token before the completion is drained.
    fn flush(&mut self, wait_for_one: bool) -> io::Result<()> {
        if wait_for_one {
            assert!(self.outstanding > 0, "flush with no outstanding operations");

            // SAFETY: Semaphore valid for lifetime of Shared.
            unsafe {
                let wait_result =
                    dispatch_semaphore_wait(self.shared.done_sema, DISPATCH_TIME_FOREVER);
                assert!(wait_result == 0, "semaphore wait failed");

                // Restore the token for drain() to consume.
                dispatch_semaphore_signal(self.shared.done_sema);
            }
        }

        Ok(())
    }

    /// Drains completed operations, invoking `f(user_data, result)` for each.
    ///
    /// Blocks until at least one completion is available, then drains all
    /// pending completions up to [`MAX_DRAIN_PER_CALL`]. The callback receives
    /// the original `user_data` pointer and the operation result (bytes
    /// transferred or `-errno`).
    ///
    /// After `f` returns, the `Completion` may be reused for new submissions.
    fn drain<F: FnMut(u64, i32)>(&mut self, mut f: F) {
        let mut drained_count: u32 = 0;

        loop {
            assert!(
                drained_count < MAX_DRAIN_PER_CALL,
                "drain loop exceeded bound"
            );

            // SAFETY: Semaphore valid for lifetime of Shared.
            // Blocks until a worker signals completion.
            let rc =
                unsafe { dispatch_semaphore_wait(self.shared.done_sema, DISPATCH_TIME_FOREVER) };
            if rc != 0 {
                break;
            }

            let user_data = self
                .shared
                .done
                .pop()
                .expect("semaphore token without queue item: invariant violated");
            assert!(user_data != 0, "null completion pointer in done queue");

            // SAFETY: user_data was a valid Completion* when submitted.
            // Worker has finished; we read the result but don't modify.
            let completion = unsafe { &*(user_data as *const Completion) };
            f(user_data, completion.result);

            assert!(self.outstanding > 0, "underflow in outstanding count");
            self.outstanding -= 1;
            drained_count += 1;
        }

        assert!(drained_count <= MAX_DRAIN_PER_CALL);
    }
}

/// Determines worker count based on available parallelism.
///
/// Capped by [`WORKER_COUNT_MAX`] and the queue capacity (no point having
/// more workers than possible concurrent operations).
fn worker_count(capacity: u32) -> u32 {
    assert!(capacity > 0);

    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
        .clamp(1, WORKER_COUNT_MAX)
        .min(capacity)
}

/// FFI entry point for GCD worker tasks.
///
/// Catches panics and aborts rather than unwinding across the FFI boundary,
/// which is undefined behavior.
extern "C" fn worker_trampoline(ctx: *mut c_void) {
    let result = std::panic::catch_unwind(|| unsafe { worker_main(ctx) });
    if result.is_err() {
        std::process::abort();
    }
}

/// Worker loop. Waits for submissions, executes operations, posts completions.
///
/// # Safety
/// `ctx` must be a valid pointer from `Box::into_raw(Box<WorkerCtx>)`.
unsafe fn worker_main(ctx: *mut c_void) {
    assert!(!ctx.is_null());

    // SAFETY: Pointer was created via `Box::into_raw` in `GcdBackend::new`.
    let worker = unsafe { Box::from_raw(ctx as *mut WorkerCtx) };
    let shared = worker.shared;
    let _worker_index = worker.worker_index;

    loop {
        // Check stop flag before blocking to enable clean shutdown.
        if shared.stop.load(Ordering::Acquire) {
            break;
        }

        // SAFETY: Semaphore is valid for the lifetime of `Shared`.
        let wait_rc = unsafe { dispatch_semaphore_wait(shared.submit_sema, DISPATCH_TIME_FOREVER) };
        assert!(wait_rc == 0);

        // Re-check after wake: shutdown may have signaled to unblock us.
        if shared.stop.load(Ordering::Acquire) {
            break;
        }

        // Batch processing: handle one guaranteed item, then opportunistically
        // drain more without blocking. Each semaphore token corresponds to
        // exactly one queue item (invariant maintained by `try_push`).
        for batch_index in 0..MAX_SUBMISSIONS_PER_WAKE {
            let user_data = shared
                .submit
                .pop()
                .expect("semaphore token without queue item: invariant violated");
            assert!(user_data != 0);

            // SAFETY: `user_data` is a `Completion*` cast to u64 by the submitter.
            // Exclusive access guaranteed: completion is in-flight (backend_context == 1)
            // and only one worker processes each submission.
            let completion = unsafe { &mut *(user_data as *mut Completion) };
            assert!(completion.backend_context == 1);

            // SAFETY: Operation contains valid fd/buffer from caller.
            let res = unsafe { perform_io(&completion.op) };
            completion.result = res;
            completion.backend_context = 0; // Mark complete; safe to reuse after drain.

            let push_result = shared.done.push(user_data);
            assert!(push_result.is_ok());

            // SAFETY: Semaphore is valid for the lifetime of `Shared`.
            unsafe { dispatch_semaphore_signal(shared.done_sema) };

            // Reached batch limit; yield back to outer loop.
            if batch_index + 1 == MAX_SUBMISSIONS_PER_WAKE {
                break;
            }

            // Non-blocking attempt to grab another item. Returns non-zero if
            // no token available, meaning queue is empty or contended.
            let next_rc = unsafe { dispatch_semaphore_wait(shared.submit_sema, DISPATCH_TIME_NOW) };
            if next_rc != 0 {
                break;
            }

            if shared.stop.load(Ordering::Acquire) {
                return;
            }
        }
    }
}

/// Executes a blocking I/O operation, returning bytes transferred or `-errno`.
///
/// # Safety
/// The `Operation` must contain valid file descriptors and buffer pointers.
unsafe fn perform_io(op: &Operation) -> i32 {
    match *op {
        Operation::Nop => 0,

        Operation::Read {
            fd,
            buf,
            len,
            offset,
        } => {
            assert!(fd >= 0);

            // SAFETY: Caller guarantees fd is valid and buf points to len bytes.
            unsafe {
                syscall_ret_isize(|| {
                    libc::pread(
                        fd,
                        buf.as_ptr() as *mut c_void,
                        len as usize,
                        offset as libc::off_t,
                    )
                })
            }
        }

        Operation::Write {
            fd,
            buf,
            len,
            offset,
        } => {
            assert!(fd >= 0);

            // SAFETY: Caller guarantees fd is valid and buf points to len bytes.
            unsafe {
                syscall_ret_isize(|| {
                    libc::pwrite(
                        fd,
                        buf.as_ptr() as *const c_void,
                        len as usize,
                        offset as libc::off_t,
                    )
                })
            }
        }

        Operation::Fsync { fd } => {
            assert!(fd >= 0);
            // SAFETY: Caller guarantees fd is valid.
            unsafe { fsync_full(fd) }
        }
    }
}

/// Returns the current thread-local errno value.
///
/// # Safety
/// Must be called immediately after a failed syscall, before any other
/// library calls that might clobber errno.
#[inline]
unsafe fn errno() -> i32 {
    // SAFETY: `__error()` returns a pointer to the thread-local errno.
    unsafe { *libc::__error() }
}

/// Retries a syscall on EINTR, returning bytes transferred or `-errno`.
///
/// Converts `isize` return values (pread/pwrite) to `i32`. Values exceeding
/// `i32::MAX` would indicate a bug in the caller's length parameter.
///
/// # Safety
/// The closure must be a valid syscall that sets errno on failure.
#[inline]
unsafe fn syscall_ret_isize(mut f: impl FnMut() -> isize) -> i32 {
    for _ in 0..MAX_EINTR_RETRIES {
        let rc = f();
        if rc >= 0 {
            return rc as i32;
        }
        // SAFETY: Called immediately after failed syscall.
        let e = unsafe { errno() };
        if e != libc::EINTR {
            return -e;
        }
    }

    unreachable!("EINTR retry loop escaped bounds");
}

/// Retries a syscall on EINTR, returning the result or `-errno`.
///
/// For syscalls returning `i32` directly (fcntl, fsync).
///
/// # Safety
/// The closure must be a valid syscall that sets errno on failure.
#[inline]
unsafe fn syscall_ret_i32(mut f: impl FnMut() -> i32) -> i32 {
    for _ in 0..MAX_EINTR_RETRIES {
        let rc = f();
        if rc >= 0 {
            return rc;
        }
        // SAFETY: Called immediately after failed syscall.
        let e = unsafe { errno() };
        if e != libc::EINTR {
            return -e;
        }
    }

    unreachable!("EINTR retry loop escaped bounds");
}

/// Durable fsync using F_FULLFSYNC, with fallback to regular fsync.
///
/// macOS fsync() only flushes to the drive's cache, not to platters.
/// F_FULLFSYNC issues a cache flush command to the drive firmware,
/// providing actual durability guarantees. Falls back to fsync() on
/// filesystems that don't support F_FULLFSYNC (e.g., some network mounts).
///
/// # Safety
/// `fd` must be a valid, open file descriptor.
#[inline]
unsafe fn fsync_full(fd: i32) -> i32 {
    assert!(fd >= 0);

    // SAFETY: fd is valid per caller contract.
    let rc = unsafe { libc::fcntl(fd, libc::F_FULLFSYNC) };
    if rc == 0 {
        return 0;
    }

    // SAFETY: Called immediately after failed fcntl.
    let e = unsafe { errno() };
    assert!(e != 0);

    // F_FULLFSYNC unsupported on this filesystem; fall back to best-effort fsync.
    if e == libc::EINVAL || e == libc::ENOTTY || e == libc::ENOTSUP {
        // SAFETY: fd is valid per caller contract.
        return unsafe { syscall_ret_i32(|| libc::fsync(fd)) };
    }

    -e
}
