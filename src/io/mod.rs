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

mod io_core;

#[cfg(target_os = "linux")]
mod backend_linux;
#[cfg(target_os = "macos")]
mod backend_macos;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("This I/O layer currently supports only Linux (io_uring) and macOS (GCD).");

#[cfg(target_os = "linux")]
use backend_linux::UringBackend as BackendImpl;
#[cfg(target_os = "macos")]
use backend_macos::GcdBackend as BackendImpl;

const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

// Re-export core types for public API.
pub use self::io_core::{
    Completion, CompletionCallback, CompletionState, IoBackend, IoCore, IoHandler, Operation,
    typed_handler_shim,
};

/// Platform-specific I/O instance. Use this type alias for portable code.
pub type Io = IoCore<BackendImpl>;

#[cfg(test)]
mod tests {
    use super::*;
    use core::ffi::c_void;
    use core::ptr::NonNull;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::io;

    use crate::stdx::DoublyLinkedList;

    use self::io_core::IoTag;

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
    #[should_panic(expected = "Length must fit in i32")]
    fn validate_rejects_oversized_length_read() {
        let op = Operation::Read {
            fd: 3,
            buf: NonNull::dangling(),
            len: i32::MAX as u32 + 1,
            offset: 0,
        };
        op.validate();
    }

    #[test]
    #[should_panic(expected = "Length must fit in i32")]
    fn validate_rejects_oversized_length_write() {
        let op = Operation::Write {
            fd: 3,
            buf: NonNull::dangling(),
            len: i32::MAX as u32 + 1,
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

    #[test]
    #[should_panic]
    fn completion_reset_panics_if_linked() {
        let mut list: DoublyLinkedList<Completion, IoTag> = DoublyLinkedList::init();
        let mut comp = Completion::new();
        list.push_back(&mut comp);
        comp.reset();
    }

    #[test]
    fn completion_complete_transitions_to_idle_and_invokes_callback() {
        unsafe fn callback(ctx: *mut c_void, completion: &mut Completion) {
            let called = unsafe { &mut *(ctx as *mut bool) };
            *called = true;
            assert!(completion.is_idle());
        }

        let mut called = false;
        let mut comp = Completion::new();
        comp.state = CompletionState::Completed;
        comp.result = 123;
        comp.context = (&mut called as *mut bool).cast::<c_void>();
        comp.callback = Some(callback);

        comp.complete();

        assert!(called);
        assert!(comp.is_idle());
        assert!(comp.callback.is_none());
        assert_eq!(comp.result, 123);
    }

    #[test]
    #[should_panic]
    fn completion_complete_panics_if_not_completed() {
        let mut comp = Completion::new();
        comp.complete();
    }

    #[test]
    #[should_panic]
    fn completion_complete_panics_if_linked() {
        let mut list: DoublyLinkedList<Completion, IoTag> = DoublyLinkedList::init();
        let mut comp = Completion::new();
        comp.state = CompletionState::Completed;
        list.push_back(&mut comp);
        comp.complete();
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

    // =========================================================================
    // IoCore Preconditions Tests
    // =========================================================================

    #[test]
    #[should_panic(expected = "buf must not be null")]
    fn io_core_read_panics_on_null_buffer() {
        unsafe fn callback(_ctx: *mut c_void, _completion: &mut Completion) {}

        let mut io: IoCore<MockBackend> = IoCore::new(1).unwrap();
        let mut comp = Completion::new();

        io.read(
            &mut comp,
            3,
            core::ptr::null_mut(),
            1,
            0,
            core::ptr::null_mut(),
            callback,
        );
    }

    #[test]
    #[should_panic(expected = "buf must not be null")]
    fn io_core_write_panics_on_null_buffer() {
        unsafe fn callback(_ctx: *mut c_void, _completion: &mut Completion) {}

        let mut io: IoCore<MockBackend> = IoCore::new(1).unwrap();
        let mut comp = Completion::new();

        io.write(
            &mut comp,
            3,
            core::ptr::null(),
            1,
            0,
            core::ptr::null_mut(),
            callback,
        );
    }

    #[test]
    #[should_panic]
    fn io_core_submit_panics_on_nop_operation() {
        unsafe fn callback(_ctx: *mut c_void, _completion: &mut Completion) {}

        let mut io: IoCore<MockBackend> = IoCore::new(1).unwrap();
        let mut comp = Completion::new();

        io.submit(&mut comp, Operation::Nop, core::ptr::null_mut(), callback);
    }

    #[test]
    #[should_panic]
    fn io_core_submit_panics_if_completion_not_idle() {
        unsafe fn callback(_ctx: *mut c_void, _completion: &mut Completion) {}

        let mut io: IoCore<MockBackend> = IoCore::new(1).unwrap();
        let mut buf = [0u8; 1];
        let mut comp = Completion::new();
        comp.state = CompletionState::Completed;

        let op = test_read_op(&mut buf);
        io.submit(&mut comp, op, core::ptr::null_mut(), callback);
    }

    #[test]
    #[should_panic]
    fn io_core_submit_panics_if_completion_linked() {
        unsafe fn callback(_ctx: *mut c_void, _completion: &mut Completion) {}

        let mut io: IoCore<MockBackend> = IoCore::new(1).unwrap();
        let mut buf = [0u8; 1];
        let mut comp = Completion::new();

        let mut list: DoublyLinkedList<Completion, IoTag> = DoublyLinkedList::init();
        list.push_back(&mut comp);

        let op = test_read_op(&mut buf);
        io.submit(&mut comp, op, core::ptr::null_mut(), callback);
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
    fn io_core_overflow_queue_is_fifo_ordered() {
        thread_local! {
            static ORDER: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
        }

        unsafe fn record_order(_ctx: *mut c_void, completion: &mut Completion) {
            let offset = match completion.op {
                Operation::Read { offset, .. } => offset,
                _ => panic!("unexpected operation"),
            };
            ORDER.with(|o| o.borrow_mut().push(offset));
        }

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
            let op = Operation::Read {
                fd: 3,
                buf: NonNull::new(bufs[i].as_mut_ptr()).unwrap(),
                len: 64,
                offset: i as u64,
            };
            io.submit(comp, op, core::ptr::null_mut(), record_order);
        }

        while !io.is_idle() {
            io.tick().unwrap();
        }

        ORDER.with(|o| {
            assert_eq!(*o.borrow(), vec![0, 1, 2, 3, 4]);
        });
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
    fn io_core_tick_propagates_flush_error_and_recovers() {
        unsafe fn callback(ctx: *mut c_void, _completion: &mut Completion) {
            let called = unsafe { &mut *(ctx as *mut bool) };
            *called = true;
        }

        let mut io: IoCore<MockBackend> = IoCore::new(1).unwrap();
        io.backend.flush_error = Some(io::ErrorKind::Other);

        let mut called = false;
        let mut buf = [0u8; 64];
        let mut comp = Completion::new();
        let op = test_read_op(&mut buf);
        io.submit(
            &mut comp,
            op,
            (&mut called as *mut bool).cast::<c_void>(),
            callback,
        );

        assert!(io.tick().is_err());
        assert!(!called);
        assert_eq!(io.total_completed, 0);
        assert_eq!(comp.state(), CompletionState::Submitted);

        io.backend.flush_error = None;
        io.tick().unwrap();

        assert!(called);
        assert!(io.is_idle());
        assert_eq!(io.total_completed, 1);
        assert!(comp.is_idle());
    }

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
    use core::ptr::NonNull;
    use proptest::prelude::*;

    use crate::stdx::DoublyLinkedList;

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
