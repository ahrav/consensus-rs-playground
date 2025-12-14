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
// Note: libdispatch is part of libSystem on macOS, no explicit linking needed.
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
                assert!(len <= i32::MAX as u32, "read length {len} exceeds i32::MAX");
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
                    len <= i32::MAX as u32,
                    "write length {len} exceeds i32::MAX"
                );
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
    /// Drains all currently-available completions (non-blocking) up to
    /// [`MAX_DRAIN_PER_CALL`]. If the caller needs to block for at least one
    /// completion, call [`flush`](Self::flush) with `wait_for_one = true` first.
    ///
    /// The callback receives the original `user_data` pointer and the operation
    /// result (bytes transferred or `-errno`).
    ///
    /// After `f` returns, the `Completion` may be reused for new submissions.
    fn drain<F: FnMut(u64, i32)>(&mut self, mut f: F) {
        let mut drained_count: u32 = 0;

        while drained_count < MAX_DRAIN_PER_CALL {
            // SAFETY: Semaphore valid for lifetime of Shared.
            // Non-blocking: returns non-zero when no completion token is available.
            let rc = unsafe { dispatch_semaphore_wait(self.shared.done_sema, DISPATCH_TIME_NOW) };
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

impl Drop for GcdBackend {
    fn drop(&mut self) {
        // It is a logic error to drop the backend with in-flight operations.
        assert!(
            self.outstanding == 0,
            "GcdBackend dropped with outstanding operations"
        );

        self.shared.stop.store(true, Ordering::Release);

        // Wake any parked workers so they can observe `stop` and exit.
        for _ in 0..self.workers {
            // SAFETY: Semaphore valid for lifetime of Shared.
            unsafe {
                dispatch_semaphore_signal(self.shared.submit_sema);
            }
        }
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
                let res = syscall_ret_isize(|| {
                    libc::pread(
                        fd,
                        buf.as_ptr() as *mut c_void,
                        len as usize,
                        offset as libc::off_t,
                    )
                });

                if res == -libc::ESPIPE {
                    syscall_ret_isize(|| libc::read(fd, buf.as_ptr() as *mut c_void, len as usize))
                } else {
                    res
                }
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
                let res = syscall_ret_isize(|| {
                    libc::pwrite(
                        fd,
                        buf.as_ptr() as *const c_void,
                        len as usize,
                        offset as libc::off_t,
                    )
                });

                if res == -libc::ESPIPE {
                    syscall_ret_isize(|| {
                        libc::write(fd, buf.as_ptr() as *const c_void, len as usize)
                    })
                } else {
                    res
                }
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
            assert!(
                rc <= i32::MAX as isize,
                "syscall returned {rc} which exceeds i32::MAX"
            );
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
    if e == libc::EINVAL || e == libc::ENOTTY || e == libc::ENOTSUP || e == libc::ENODEV {
        // SAFETY: fd is valid per caller contract.
        return unsafe { syscall_ret_i32(|| libc::fsync(fd)) };
    }

    -e
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    // Helper: create pinned completion.
    fn completion_with_op(op: Operation) -> Box<Completion> {
        let mut comp = Box::new(Completion::new());
        comp.op = op;
        comp
    }

    #[test]
    #[should_panic]
    fn new_zero() {
        let _ = GcdBackend::new(0);
    }

    #[test]
    fn drain_max() {
        let count = MAX_DRAIN_PER_CALL as usize + 1;
        assert!(count <= GcdBackend::ENTRIES_MAX as usize);

        let mut backend = GcdBackend::new(count as u32).unwrap();

        let mut completions: Vec<Box<Completion>> = (0..count)
            .map(|_| completion_with_op(Operation::Nop))
            .collect();

        // Manually inject completed items into the done queue to deterministically
        // exercise the drain cap without relying on worker scheduling.
        backend.outstanding = count as u32;
        for comp in completions.iter_mut() {
            let ptr = &mut **comp as *mut Completion as u64;
            backend.shared.done.push(ptr).unwrap();
            unsafe {
                dispatch_semaphore_signal(backend.shared.done_sema);
            }
        }

        let mut first = 0usize;
        backend.drain(|_, _| first += 1);

        // It must not exceed the limit.
        assert!(first <= MAX_DRAIN_PER_CALL as usize);
        // We injected `count` completions, so it must hit the cap.
        assert_eq!(first, MAX_DRAIN_PER_CALL as usize);

        let mut second = 0usize;
        backend.drain(|_, _| second += 1);
        assert_eq!(second, 1);
    }

    #[test]
    fn drain_all() {
        let mut backend = GcdBackend::new(GcdBackend::ENTRIES_MAX).unwrap();

        let count = (MAX_DRAIN_PER_CALL + 128) as usize;
        assert!(count <= GcdBackend::ENTRIES_MAX as usize);

        let mut completions: Vec<Box<Completion>> = (0..count)
            .map(|_| {
                let mut c = completion_with_op(Operation::Nop);
                c.backend_context = 0;
                c
            })
            .collect();

        let completion_ptrs: Vec<u64> = completions
            .iter_mut()
            .map(|c| &mut **c as *mut Completion as u64)
            .collect();

        for &ptr in &completion_ptrs {
            unsafe {
                backend.try_push(&Operation::Nop, ptr).unwrap();
            }
        }

        backend.flush(false).unwrap();

        let deadline = Instant::now() + Duration::from_secs(10);
        let mut seen = std::collections::HashSet::new();
        let mut remaining = count;

        while remaining > 0 && Instant::now() < deadline {
            backend.drain(|user_data, result| {
                assert_eq!(result, 0);
                assert!(seen.insert(user_data), "Double completion for {user_data}");
                remaining -= 1;
            });
            if remaining > 0 {
                backend.flush(true).ok();
            }
        }

        assert_eq!(remaining, 0);
        assert_eq!(seen.len(), count);
        for ptr in completion_ptrs {
            assert!(seen.contains(&ptr));
        }
    }

    #[test]
    fn flush_wait() {
        let mut backend = GcdBackend::new(8).unwrap();
        let mut comp = completion_with_op(Operation::Nop);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe {
            backend.try_push(&Operation::Nop, ptr).unwrap();
        }

        // Flush async.
        backend.flush(false).unwrap();

        // Flush wait.
        backend.flush(true).unwrap();

        let mut count = 0;
        backend.drain(|_, _| count += 1);
        assert_eq!(count, 1);
    }

    #[test]
    fn flush_wait_peek() {
        let mut backend = GcdBackend::new(8).unwrap();

        let mut comp = completion_with_op(Operation::Nop);
        comp.result = 123;
        let ptr = &mut *comp as *mut Completion as u64;

        backend.outstanding = 1;
        backend.shared.done.push(ptr).unwrap();
        unsafe {
            dispatch_semaphore_signal(backend.shared.done_sema);
        }

        backend.flush(true).unwrap();

        assert_eq!(backend.shared.done.len(), 1);

        // Non-blocking check: the completion token should still be available.
        let rc = unsafe { dispatch_semaphore_wait(backend.shared.done_sema, DISPATCH_TIME_NOW) };
        assert_eq!(rc, 0, "flush(true) consumed the completion token");
        unsafe {
            dispatch_semaphore_signal(backend.shared.done_sema);
        }

        let mut got = None;
        backend.drain(|user_data, result| got = Some((user_data, result)));
        assert_eq!(got, Some((ptr, 123)));
    }

    #[test]
    fn new_min() {
        let backend = GcdBackend::new(GcdBackend::ENTRIES_MIN).unwrap();
        assert_eq!(backend.capacity, GcdBackend::ENTRIES_MIN);
    }

    #[test]
    fn new_sizes() {
        for &size in &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let backend = GcdBackend::new(size).unwrap();
            assert_eq!(backend.capacity, size);
        }
    }

    #[test]
    fn new_max() {
        let backend = GcdBackend::new(GcdBackend::ENTRIES_MAX).unwrap();
        assert_eq!(backend.capacity, GcdBackend::ENTRIES_MAX);
    }

    #[test]
    #[should_panic]
    fn new_overflow() {
        let _ = GcdBackend::new(GcdBackend::ENTRIES_MAX + 1);
    }

    #[test]
    fn push_empty() {
        let mut backend = GcdBackend::new(8).unwrap();

        let mut comp = completion_with_op(Operation::Nop);

        let result =
            unsafe { backend.try_push(&Operation::Nop, &mut *comp as *mut Completion as u64) };

        assert!(result.is_ok());

        backend.flush(true).unwrap();

        backend.drain(|_, _| {});
    }

    #[test]
    fn push_capacity() {
        let mut backend = GcdBackend::new(4).unwrap();

        let mut comps: Vec<Box<Completion>> =
            (0..5).map(|_| completion_with_op(Operation::Nop)).collect();

        for comp in comps.iter_mut().take(4) {
            let ptr = &mut **comp as *mut Completion as u64;

            let result = unsafe { backend.try_push(&Operation::Nop, ptr) };

            assert!(result.is_ok());
        }

        let ptr = &mut *comps[4] as *mut Completion as u64;

        let result = unsafe { backend.try_push(&Operation::Nop, ptr) };

        assert!(result.is_err());

        let mut remaining = 4usize;
        while remaining > 0 {
            backend.flush(true).unwrap();
            backend.drain(|_, _| remaining -= 1);
        }
    }

    #[test]
    fn push_full() {
        let mut backend = GcdBackend::new(2).unwrap();
        let mut comps: Vec<Box<Completion>> =
            (0..3).map(|_| completion_with_op(Operation::Nop)).collect();

        unsafe {
            backend
                .try_push(&Operation::Nop, &mut *comps[0] as *mut Completion as u64)
                .unwrap();
            backend
                .try_push(&Operation::Nop, &mut *comps[1] as *mut Completion as u64)
                .unwrap();
        }

        let result =
            unsafe { backend.try_push(&Operation::Nop, &mut *comps[2] as *mut Completion as u64) };
        assert!(result.is_err());

        let mut remaining = 2usize;
        while remaining > 0 {
            backend.flush(true).unwrap();
            backend.drain(|_, _| remaining -= 1);
        }
    }

    #[test]
    fn push_recycle() {
        let mut backend = GcdBackend::new(2).unwrap();
        let mut comps: Vec<Box<Completion>> =
            (0..3).map(|_| completion_with_op(Operation::Nop)).collect();

        unsafe {
            backend
                .try_push(&Operation::Nop, &mut *comps[0] as *mut Completion as u64)
                .unwrap();
            backend
                .try_push(&Operation::Nop, &mut *comps[1] as *mut Completion as u64)
                .unwrap();
        }

        // Wait for them to complete
        let mut remaining = 2;
        while remaining > 0 {
            backend.flush(true).unwrap();
            backend.drain(|_, _| remaining -= 1);
        }

        let result =
            unsafe { backend.try_push(&Operation::Nop, &mut *comps[2] as *mut Completion as u64) };

        assert!(result.is_ok());

        backend.flush(true).unwrap();

        backend.drain(|_, _| {});
    }

    #[test]
    fn flush_no_wait() {
        let mut backend = GcdBackend::new(8).unwrap();
        let mut comp = completion_with_op(Operation::Nop);
        unsafe {
            backend
                .try_push(&Operation::Nop, &mut *comp as *mut Completion as u64)
                .unwrap()
        };
        backend.flush(false).unwrap();

        backend.flush(true).unwrap();
        backend.drain(|_, _| {});
    }

    #[test]
    fn flush_empty() {
        let mut backend = GcdBackend::new(8).unwrap();
        backend.flush(false).unwrap();
    }

    #[test]
    fn drain_empty() {
        let mut backend = GcdBackend::new(8).unwrap();
        let mut count = 0;
        backend.drain(|_, _| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn nop() {
        let mut backend = GcdBackend::new(8).unwrap();
        let mut comp = completion_with_op(Operation::Nop);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&Operation::Nop, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut completed = false;
        backend.drain(|user_data, result| {
            assert_eq!(user_data, ptr);
            assert_eq!(result, 0);
            completed = true;
        });
        assert!(completed);
    }

    #[test]
    #[should_panic(expected = "invalid file descriptor")]
    fn read_neg_fd() {
        let mut backend = GcdBackend::new(8).unwrap();
        let op = Operation::Read {
            fd: -1,
            buf: core::ptr::NonNull::dangling(),
            len: 1,
            offset: 0,
        };
        let mut comp = completion_with_op(op);
        unsafe {
            let _ = backend.try_push(&op, &mut *comp as *mut Completion as u64);
        }
    }

    #[test]
    #[should_panic(expected = "invalid file descriptor")]
    fn write_neg_fd() {
        let mut backend = GcdBackend::new(8).unwrap();
        let op = Operation::Write {
            fd: -1,
            buf: core::ptr::NonNull::dangling(),
            len: 1,
            offset: 0,
        };
        let mut comp = completion_with_op(op);
        unsafe {
            let _ = backend.try_push(&op, &mut *comp as *mut Completion as u64);
        }
    }

    #[test]
    #[should_panic(expected = "invalid file descriptor")]
    fn fsync_neg_fd() {
        let mut backend = GcdBackend::new(8).unwrap();
        let op = Operation::Fsync { fd: -1 };
        let mut comp = completion_with_op(op);
        unsafe {
            let _ = backend.try_push(&op, &mut *comp as *mut Completion as u64);
        }
    }

    #[test]
    #[should_panic(expected = "read length must be positive")]
    fn read_zero_len() {
        let mut backend = GcdBackend::new(8).unwrap();
        let op = Operation::Read {
            fd: 0,
            buf: core::ptr::NonNull::dangling(),
            len: 0,
            offset: 0,
        };
        let mut comp = completion_with_op(op);
        unsafe {
            let _ = backend.try_push(&op, &mut *comp as *mut Completion as u64);
        }
    }

    #[test]
    #[should_panic(expected = "write length must be positive")]
    fn write_zero_len() {
        let mut backend = GcdBackend::new(8).unwrap();
        let op = Operation::Write {
            fd: 0,
            buf: core::ptr::NonNull::dangling(),
            len: 0,
            offset: 0,
        };
        let mut comp = completion_with_op(op);
        unsafe {
            let _ = backend.try_push(&op, &mut *comp as *mut Completion as u64);
        }
    }

    #[test]
    #[should_panic(expected = "exceeds off_t max")]
    fn read_offset_overflow() {
        let mut backend = GcdBackend::new(8).unwrap();
        let op = Operation::Read {
            fd: 0,
            buf: core::ptr::NonNull::dangling(),
            len: 1,
            offset: u64::MAX,
        };
        let mut comp = completion_with_op(op);
        unsafe {
            let _ = backend.try_push(&op, &mut *comp as *mut Completion as u64);
        }
    }

    #[test]
    #[should_panic(expected = "exceeds off_t max")]
    fn write_offset_overflow() {
        let mut backend = GcdBackend::new(8).unwrap();
        let op = Operation::Write {
            fd: 0,
            buf: core::ptr::NonNull::dangling(),
            len: 1,
            offset: u64::MAX,
        };
        let mut comp = completion_with_op(op);
        unsafe {
            let _ = backend.try_push(&op, &mut *comp as *mut Completion as u64);
        }
    }

    #[test]
    #[should_panic(expected = "exceeds i32::MAX")]
    fn read_len_max() {
        let mut backend = GcdBackend::new(8).unwrap();
        let op = Operation::Read {
            fd: 0,
            buf: core::ptr::NonNull::dangling(),
            len: i32::MAX as u32 + 1,
            offset: 0,
        };
        let mut comp = completion_with_op(op);
        unsafe {
            let _ = backend.try_push(&op, &mut *comp as *mut Completion as u64);
        }
    }

    #[test]
    #[should_panic(expected = "exceeds i32::MAX")]
    fn write_len_max() {
        let mut backend = GcdBackend::new(8).unwrap();
        let op = Operation::Write {
            fd: 0,
            buf: core::ptr::NonNull::dangling(),
            len: i32::MAX as u32 + 1,
            offset: 0,
        };
        let mut comp = completion_with_op(op);
        unsafe {
            let _ = backend.try_push(&op, &mut *comp as *mut Completion as u64);
        }
    }

    #[test]
    #[should_panic(expected = "null completion pointer")]
    fn push_null() {
        let mut backend = GcdBackend::new(8).unwrap();
        unsafe {
            let _ = backend.try_push(&Operation::Nop, 0);
        }
    }

    #[test]
    fn syscall_retry() {
        let mut attempts: u32 = 0;
        let rc = unsafe {
            syscall_ret_isize(|| {
                attempts += 1;
                if attempts < 3 {
                    *libc::__error() = libc::EINTR;
                    -1
                } else {
                    7
                }
            })
        };
        assert_eq!(rc, 7);
        assert_eq!(attempts, 3);

        unsafe {
            *libc::__error() = 0;
        }
    }

    #[test]
    fn syscall_errno() {
        let rc = unsafe {
            syscall_ret_isize(|| {
                *libc::__error() = libc::EBADF;
                -1
            })
        };
        assert_eq!(rc, -libc::EBADF);
        unsafe {
            *libc::__error() = 0;
        }
    }

    #[test]
    fn syscall_i32_retry() {
        let mut attempts: u32 = 0;
        let rc = unsafe {
            syscall_ret_i32(|| {
                attempts += 1;
                if attempts < 4 {
                    *libc::__error() = libc::EINTR;
                    -1
                } else {
                    0
                }
            })
        };
        assert_eq!(rc, 0);
        assert_eq!(attempts, 4);

        unsafe {
            *libc::__error() = 0;
        }
    }

    #[test]
    fn syscall_i32_errno() {
        let rc = unsafe {
            syscall_ret_i32(|| {
                *libc::__error() = libc::EINVAL;
                -1
            })
        };
        assert_eq!(rc, -libc::EINVAL);
        unsafe {
            *libc::__error() = 0;
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use core::ptr::NonNull;
    use libc;
    use std::fs::{File, OpenOptions};
    use std::io::{Read, Write};
    use std::os::unix::io::AsRawFd;
    use std::os::unix::net::UnixStream;
    use std::time::{Duration, Instant};
    use tempfile::tempdir;

    fn completion_with_op(op: Operation) -> Box<Completion> {
        let mut comp = Box::new(Completion::new());
        comp.op = op;
        comp
    }

    #[test]
    fn read_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("read_test.txt");
        let test_data = b"Hello, GCD!";

        File::create(&path).unwrap().write_all(test_data).unwrap();

        let file = File::open(&path).unwrap();
        let fd = file.as_raw_fd();
        let mut backend = GcdBackend::new(8).unwrap();

        let mut buf = vec![0u8; test_data.len()];
        let op = Operation::Read {
            fd,
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: buf.len() as u32,
            offset: 0,
        };

        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut completed = false;
        backend.drain(|data, result| {
            assert_eq!(data, ptr);
            assert_eq!(result, test_data.len() as i32);
            completed = true;
        });

        assert!(completed);
        assert_eq!(&buf, test_data);
    }

    #[test]
    fn read_at_offset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("offset_read.txt");

        File::create(&path)
            .unwrap()
            .write_all(b"0123456789ABCDEF")
            .unwrap();

        let file = File::open(&path).unwrap();
        let fd = file.as_raw_fd();
        let mut backend = GcdBackend::new(8).unwrap();

        let mut buf = vec![0u8; 4];
        let op = Operation::Read {
            fd,
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: 4,
            offset: 10,
        };

        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut bytes_read = 0;
        backend.drain(|_, result| bytes_read = result);

        assert_eq!(bytes_read, 4);
        assert_eq!(&buf, b"ABCD");
    }

    #[test]
    fn write_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("write_test.txt");

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        let fd = file.as_raw_fd();
        let mut backend = GcdBackend::new(8).unwrap();

        let data = b"GCD write test";
        let op = Operation::Write {
            fd,
            buf: NonNull::new(data.as_ptr() as *mut u8).unwrap(),
            len: data.len() as u32,
            offset: 0,
        };

        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut bytes_written = 0;
        backend.drain(|_, result| bytes_written = result);

        assert_eq!(bytes_written, data.len() as i32);

        drop(file);
        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents, "GCD write test");
    }

    #[test]
    fn write_at_offset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("write_offset_test.txt");

        File::create(&path)
            .unwrap()
            .write_all(b"0123456789ABCDEF")
            .unwrap();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let fd = file.as_raw_fd();
        let mut backend = GcdBackend::new(8).unwrap();

        let data = b"WXYZ";
        let op = Operation::Write {
            fd,
            buf: NonNull::new(data.as_ptr() as *mut u8).unwrap(),
            len: data.len() as u32,
            offset: 4,
        };

        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut bytes_written = None;
        backend.drain(|user_data, result| {
            assert_eq!(user_data, ptr);
            bytes_written = Some(result);
        });
        assert_eq!(bytes_written, Some(data.len() as i32));

        drop(file);
        let mut contents = Vec::new();
        File::open(&path)
            .unwrap()
            .read_to_end(&mut contents)
            .unwrap();
        assert_eq!(contents, b"0123WXYZ89ABCDEF");
    }

    #[test]
    fn write_fsync() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("fsync_test.txt");

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        let fd = file.as_raw_fd();
        let mut backend = GcdBackend::new(8).unwrap();

        let data = b"durable data";
        let write_op = Operation::Write {
            fd,
            buf: NonNull::new(data.as_ptr() as *mut u8).unwrap(),
            len: data.len() as u32,
            offset: 0,
        };
        let mut write_comp = completion_with_op(write_op);
        let write_ptr = &mut *write_comp as *mut Completion as u64;

        let fsync_op = Operation::Fsync { fd };
        let mut fsync_comp = completion_with_op(fsync_op);
        let fsync_ptr = &mut *fsync_comp as *mut Completion as u64;

        unsafe { backend.try_push(&write_op, write_ptr).unwrap() };
        unsafe { backend.try_push(&fsync_op, fsync_ptr).unwrap() };

        let mut write_result = None;
        let mut fsync_result = None;

        let start = Instant::now();
        while (write_result.is_none() || fsync_result.is_none())
            && start.elapsed() < Duration::from_secs(5)
        {
            backend.flush(true).unwrap();
            backend.drain(|user_data, result| {
                if user_data == write_ptr {
                    write_result = Some(result);
                } else if user_data == fsync_ptr {
                    fsync_result = Some(result);
                } else {
                    panic!("unexpected user_data");
                }
            });
        }

        assert_eq!(write_result, Some(data.len() as i32));
        assert_eq!(fsync_result, Some(0));

        drop(file);
        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents, "durable data");
    }

    #[test]
    fn batch_ops() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("batch_test.txt");

        File::create(&path)
            .unwrap()
            .write_all(b"initial content here")
            .unwrap();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let fd = file.as_raw_fd();
        let mut backend = GcdBackend::new(16).unwrap();

        let mut read_buf = vec![0u8; 7];
        let write_data = b"REPLACED";

        let read_op = Operation::Read {
            fd,
            buf: NonNull::new(read_buf.as_mut_ptr()).unwrap(),
            len: 7,
            offset: 0,
        };
        let write_op = Operation::Write {
            fd,
            buf: NonNull::new(write_data.as_ptr() as *mut u8).unwrap(),
            len: write_data.len() as u32,
            offset: 0,
        };

        let mut read_comp = completion_with_op(read_op);
        let mut write_comp = completion_with_op(write_op);
        let mut nop_comp = completion_with_op(Operation::Nop);

        let read_ptr = &mut *read_comp as *mut Completion as u64;
        let write_ptr = &mut *write_comp as *mut Completion as u64;
        let nop_ptr = &mut *nop_comp as *mut Completion as u64;

        unsafe {
            backend.try_push(&read_op, read_ptr).unwrap();
            backend.try_push(&write_op, write_ptr).unwrap();
            backend.try_push(&Operation::Nop, nop_ptr).unwrap();
        }

        let mut completions = vec![];
        let start = Instant::now();
        while completions.len() < 3 && start.elapsed() < Duration::from_secs(5) {
            backend.flush(true).unwrap();
            backend.drain(|user_data, result| completions.push((user_data, result)));
        }

        assert_eq!(completions.len(), 3);
        completions.sort_by_key(|(ud, _)| *ud);

        // Sort our pointers to match
        let mut expected = vec![
            (read_ptr, 7),
            (write_ptr, write_data.len() as i32),
            (nop_ptr, 0),
        ];
        expected.sort_by_key(|(ud, _)| *ud);

        assert_eq!(completions, expected);
    }

    #[test]
    fn read_bad_fd() {
        let mut backend = GcdBackend::new(8).unwrap();
        let fd = i32::MAX;

        let mut buf = vec![0u8; 64];
        let op = Operation::Read {
            fd,
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: 64,
            offset: 0,
        };
        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut got_error = false;
        backend.drain(|user_data, result| {
            assert_eq!(user_data, ptr);
            assert_eq!(result, -libc::EBADF);
            got_error = true;
        });
        assert!(got_error);
    }

    #[test]
    fn fsync_bad_fd() {
        let mut backend = GcdBackend::new(8).unwrap();
        let fd = i32::MAX;

        let op = Operation::Fsync { fd };
        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut got_error = false;
        backend.drain(|user_data, result| {
            assert_eq!(user_data, ptr);
            assert_eq!(result, -libc::EBADF);
            got_error = true;
        });
        assert!(got_error);
    }

    #[test]
    fn write_bad_fd() {
        let mut backend = GcdBackend::new(8).unwrap();
        let fd = i32::MAX;

        let data = b"invalid fd";
        let op = Operation::Write {
            fd,
            buf: NonNull::new(data.as_ptr() as *mut u8).unwrap(),
            len: data.len() as u32,
            offset: 0,
        };
        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut got_error = false;
        backend.drain(|user_data, result| {
            assert_eq!(user_data, ptr);
            assert_eq!(result, -libc::EBADF);
            got_error = true;
        });
        assert!(got_error);
    }

    #[test]
    fn write_readonly_fd() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("readonly_test.txt");

        File::create(&path).unwrap();
        let file = File::open(&path).unwrap();
        let fd = file.as_raw_fd();
        let mut backend = GcdBackend::new(8).unwrap();

        let data = b"should fail";
        let op = Operation::Write {
            fd,
            buf: NonNull::new(data.as_ptr() as *mut u8).unwrap(),
            len: data.len() as u32,
            offset: 0,
        };

        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut got_error = false;
        backend.drain(|user_data, result| {
            assert_eq!(user_data, ptr);
            assert_eq!(result, -libc::EBADF);
            got_error = true;
        });
        assert!(got_error);
    }

    #[test]
    fn fsync_fallback() {
        let file = OpenOptions::new().write(true).open("/dev/null").unwrap();
        let fd = file.as_raw_fd();

        let mut backend = GcdBackend::new(8).unwrap();
        let op = Operation::Fsync { fd };
        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut got = None;
        backend.drain(|user_data, result| {
            assert_eq!(user_data, ptr);
            got = Some(result);
        });

        assert_eq!(got, Some(0));
    }

    #[test]
    fn read_unix_stream() {
        let (mut left, right) = UnixStream::pair().unwrap();
        let fd = right.as_raw_fd();

        let expected = b"hello from socket";
        left.write_all(expected).unwrap();

        let mut backend = GcdBackend::new(8).unwrap();
        let mut buf = vec![0u8; expected.len()];
        let op = Operation::Read {
            fd,
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: buf.len() as u32,
            offset: 0,
        };

        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut result = None;
        backend.drain(|user_data, res| {
            assert_eq!(user_data, ptr);
            result = Some(res);
        });
        assert_eq!(result, Some(expected.len() as i32));
        assert_eq!(&buf, expected);
    }

    #[test]
    fn write_unix_stream() {
        let (left, mut right) = UnixStream::pair().unwrap();
        let fd = left.as_raw_fd();

        let data = b"hello to socket";
        let op = Operation::Write {
            fd,
            buf: NonNull::new(data.as_ptr() as *mut u8).unwrap(),
            len: data.len() as u32,
            offset: 0,
        };

        let mut backend = GcdBackend::new(8).unwrap();
        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut result = None;
        backend.drain(|user_data, res| {
            assert_eq!(user_data, ptr);
            result = Some(res);
        });
        assert_eq!(result, Some(data.len() as i32));

        let mut buf = vec![0u8; data.len()];
        right.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, data);
    }

    #[test]
    fn short_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("short_read.txt");

        File::create(&path).unwrap().write_all(b"hello").unwrap();

        let file = File::open(&path).unwrap();
        let fd = file.as_raw_fd();
        let mut backend = GcdBackend::new(8).unwrap();

        let mut buf = vec![0u8; 100];
        let op = Operation::Read {
            fd,
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: 100,
            offset: 0,
        };

        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut bytes_read = 0;
        backend.drain(|_, result| bytes_read = result);

        assert_eq!(bytes_read, 5);
        assert_eq!(&buf[..5], b"hello");
    }

    #[test]
    fn read_eof() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("eof_test.txt");

        File::create(&path)
            .unwrap()
            .write_all(b"0123456789")
            .unwrap();

        let file = File::open(&path).unwrap();
        let fd = file.as_raw_fd();
        let mut backend = GcdBackend::new(8).unwrap();

        let mut buf = vec![0u8; 10];
        let op = Operation::Read {
            fd,
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: 10,
            offset: 10,
        };

        let mut comp = completion_with_op(op);
        let ptr = &mut *comp as *mut Completion as u64;

        unsafe { backend.try_push(&op, ptr).unwrap() };
        backend.flush(true).unwrap();

        let mut bytes_read = -1;
        backend.drain(|_, result| bytes_read = result);

        assert_eq!(bytes_read, 0);
    }
}
