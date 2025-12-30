mod tests {
    use super::super::*;
    use crate::vsr::{Command, command::PrepareCmd, header::ProtoHeader};

    #[test]
    fn pool_zero_capacity() {
        let pool = MessagePool::new(0);
        // Pool with 0 capacity should immediately return None
        assert!(pool.try_get::<PrepareCmd>().is_none());
    }

    #[test]
    fn pool_minimum_capacity() {
        let pool = MessagePool::new(1);
        let msg = pool.try_get::<PrepareCmd>();
        assert!(msg.is_some());
        // Second acquire should fail
        assert!(pool.try_get::<PrepareCmd>().is_none());
    }

    #[test]
    fn buffer_zero_initialized_on_acquire() {
        let pool = MessagePool::new(1);
        let msg: Message<PrepareCmd> = pool.get();
        // The header region should be zeroed except for the fields set by try_get
        // We can't easily check all bytes, but we can verify size/command/protocol are set correctly
        assert_eq!(msg.header().size(), constants::HEADER_SIZE);
        assert_eq!(msg.header().command(), Command::Prepare);
        assert_eq!(msg.header().protocol(), constants::VSR_VERSION);
    }

    #[test]
    fn pool_clone_shares_storage() {
        let pool1 = MessagePool::new(2);
        let pool2 = pool1.clone();

        // Acquire from pool1
        let msg1: Message<PrepareCmd> = pool1.get();
        // pool2 should see reduced availability
        let _msg2: Message<PrepareCmd> = pool2.get();

        // Both pools should now be exhausted
        assert!(pool1.try_get::<PrepareCmd>().is_none());
        assert!(pool2.try_get::<PrepareCmd>().is_none());

        // Drop one message
        drop(msg1);

        // Both pools should now have one available
        assert!(pool1.try_get::<PrepareCmd>().is_some());
    }

    #[test]
    fn acquire_sets_refcount_to_one() {
        let pool = MessagePool::new(1);
        let msg: Message<PrepareCmd> = pool.get();
        assert!(
            msg.is_unique(),
            "Newly acquired message should have refcount 1"
        );
    }

    #[test]
    fn clone_increments_refcount() {
        let pool = MessagePool::new(1);
        let msg1: Message<PrepareCmd> = pool.get();
        assert!(msg1.is_unique());

        let msg2 = msg1.clone();
        assert!(
            !msg1.is_unique(),
            "After clone, original should not be unique"
        );
        assert!(!msg2.is_unique(), "After clone, clone should not be unique");

        drop(msg2);
        assert!(
            msg1.is_unique(),
            "After dropping clone, original should be unique again"
        );
    }

    #[test]
    fn drop_returns_to_pool_at_zero() {
        let pool = MessagePool::new(1);

        let msg: Message<PrepareCmd> = pool.get();
        assert!(pool.try_get::<PrepareCmd>().is_none());

        drop(msg);
        // After drop, buffer should be back in pool
        assert!(pool.try_get::<PrepareCmd>().is_some());
    }

    #[test]
    fn multiple_clones_all_dropped_recycled() {
        let pool = MessagePool::new(1);
        let msg1: Message<PrepareCmd> = pool.get();
        let msg2 = msg1.clone();
        let msg3 = msg1.clone();
        let msg4 = msg2.clone();

        assert!(pool.try_get::<PrepareCmd>().is_none());

        drop(msg1);
        assert!(pool.try_get::<PrepareCmd>().is_none()); // Still 3 refs

        drop(msg2);
        assert!(pool.try_get::<PrepareCmd>().is_none()); // Still 2 refs

        drop(msg3);
        assert!(pool.try_get::<PrepareCmd>().is_none()); // Still 1 ref

        drop(msg4);
        // All refs dropped, buffer should be recycled
        assert!(pool.try_get::<PrepareCmd>().is_some());
    }

    #[test]
    fn exhausted_pool_try_get_returns_none() {
        let pool = MessagePool::new(2);
        let _m1: Message<PrepareCmd> = pool.get();
        let _m2: Message<PrepareCmd> = pool.get();

        assert!(pool.try_get::<PrepareCmd>().is_none());
    }

    #[test]
    fn buffer_reacquirable_after_release() {
        let pool = MessagePool::new(1);

        for _ in 0..100 {
            let msg: Message<PrepareCmd> = pool.get();
            drop(msg);
        }
        // Should still work after many cycles
        assert!(pool.try_get::<PrepareCmd>().is_some());
    }

    #[test]
    #[should_panic(expected = "message pool is empty")]
    fn exhausted_pool_get_panics() {
        let pool = MessagePool::new(1);
        let _m1: Message<PrepareCmd> = pool.get();
        let _m2: Message<PrepareCmd> = pool.get(); // Should panic
    }

    #[test]
    #[should_panic(expected = "message refs overflow")]
    fn refcount_overflow_panics() {
        let pool = MessagePool::new(1);
        let msg: Message<PrepareCmd> = pool.get();

        // Artificially set refcount to MAX
        unsafe {
            msg.inner.as_ref().refs.set(u32::MAX);
        }

        // Clone should panic
        let _clone = msg.clone();
    }

    #[test]
    #[should_panic(expected = "message is shared")]
    fn header_mut_panics_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();

        // Should panic because message is shared
        let _ = msg1.header_mut();
    }

    #[test]
    #[should_panic(expected = "message is shared")]
    fn as_bytes_mut_panics_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();

        let _ = msg1.as_bytes_mut();
    }

    #[test]
    #[should_panic(expected = "message is shared")]
    fn body_used_mut_panics_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();

        let _ = msg1.body_used_mut();
    }

    #[test]
    #[should_panic(expected = "message is shared; cannot mutably borrow header")]
    fn set_used_len_panics_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();

        msg1.set_used_len(constants::HEADER_SIZE_USIZE);
    }

    #[test]
    #[should_panic(expected = "message is shared; cannot reset header")]
    fn reset_header_panics_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();

        // Should panic because message is shared
        msg1.reset_header();
    }

    #[test]
    #[should_panic]
    fn used_len_panics_below_header_size() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        // Force an invalid size
        msg.header_mut().set_size(constants::HEADER_SIZE - 1);

        // used_len should panic
        let _ = msg.used_len();
    }

    #[test]
    #[should_panic]
    fn used_len_panics_above_max_size() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        // Force an invalid size
        msg.header_mut().set_size(constants::MESSAGE_SIZE_MAX + 1);

        // used_len should panic
        let _ = msg.used_len();
    }

    #[test]
    #[should_panic]
    fn set_used_len_panics_below_header_size() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        msg.set_used_len(constants::HEADER_SIZE_USIZE - 1);
    }

    #[test]
    #[should_panic]
    fn set_used_len_panics_above_max_size() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        msg.set_used_len(constants::MESSAGE_SIZE_MAX_USIZE + 1);
    }

    #[test]
    fn body_used_returns_correct_slice() {
        let pool = MessagePool::new(1);
        let msg: Message<PrepareCmd> = pool.get();

        // Default: header only, no body
        assert_eq!(msg.body_used().len(), 0);
    }

    #[test]
    fn try_header_mut_returns_some_when_unique() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();
        assert!(msg.try_header_mut().is_some());
    }

    #[test]
    fn try_header_mut_returns_none_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();
        assert!(msg1.try_header_mut().is_none());
    }

    #[test]
    fn try_as_bytes_mut_returns_some_when_unique() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();
        let bytes = msg.try_as_bytes_mut();
        assert!(bytes.is_some());
        assert_eq!(bytes.unwrap().len(), constants::HEADER_SIZE_USIZE);
    }

    #[test]
    fn try_as_bytes_mut_returns_none_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();
        assert!(msg1.try_as_bytes_mut().is_none());
    }

    #[test]
    fn try_body_used_mut_returns_some_when_unique() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();
        msg.set_used_len(constants::HEADER_SIZE_USIZE + 4);
        let body = msg.try_body_used_mut();
        assert!(body.is_some());
        assert_eq!(body.unwrap().len(), 4);
    }

    #[test]
    fn try_body_used_mut_returns_none_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        msg1.set_used_len(constants::HEADER_SIZE_USIZE + 4);
        let _msg2 = msg1.clone();
        assert!(msg1.try_body_used_mut().is_none());
    }

    #[test]
    fn body_used_returns_correct_slice_after_resize() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        let body_len = 256;
        msg.set_used_len(constants::HEADER_SIZE_USIZE + body_len);

        let body = msg.body_used();
        assert_eq!(body.len(), body_len);

        // Verify body starts at the right offset
        let full = msg.as_bytes();
        assert_eq!(&full[constants::HEADER_SIZE_USIZE..], body);
    }

    #[test]
    fn stress_acquire_release_cycles() {
        let pool = MessagePool::new(8);

        for _ in 0..100 {
            let mut handles: Vec<Message<PrepareCmd>> = Vec::new();

            // Acquire all
            while let Some(m) = pool.try_get::<PrepareCmd>() {
                handles.push(m);
            }

            // Release all (drop)
            handles.clear();
        }

        // Pool should be in clean state
        assert!(pool.try_get::<PrepareCmd>().is_some());
    }

    #[test]
    fn stress_clone_drop_interleaved() {
        let pool = MessagePool::new(4);

        for _ in 0..10 {
            let msg1: Message<PrepareCmd> = pool.get();
            let msg2 = msg1.clone();
            let msg3 = msg2.clone();

            drop(msg1);
            let msg4 = msg3.clone();
            drop(msg2);
            drop(msg4);
            drop(msg3);
        }

        // All messages should be returned
        let mut handles = Vec::new();
        while let Some(m) = pool.try_get::<PrepareCmd>() {
            handles.push(m);
        }
        assert_eq!(handles.len(), 4);
    }

    #[test]
    fn pool_drop_after_messages_dropped() {
        let msg: Message<PrepareCmd>;
        {
            let pool = MessagePool::new(1);
            msg = pool.get();
            // Pool dropped here, but msg holds Rc to pool inner
        }
        // Message should still be usable because Rc keeps pool alive
        assert!(msg.is_unique());
        assert_eq!(msg.header().command(), Command::Prepare);
    }

    #[test]
    fn message_with_header_prepare_full_lifecycle() {
        let pool = MessagePool::new(2);

        // Acquire
        let mut msg: Message<PrepareCmd> = pool.get();

        // Verify initialization
        assert_eq!(msg.header().command(), Command::Prepare);
        assert_eq!(msg.header().size(), constants::HEADER_SIZE);
        assert_eq!(msg.header().protocol(), constants::VSR_VERSION);

        // Modify header
        msg.header_mut().set_size(constants::HEADER_SIZE + 64);

        // Clone and verify sharing
        let msg2 = msg.clone();
        assert!(!msg.is_unique());
        assert_eq!(msg.header().size(), msg2.header().size());

        // Drop clone, verify original is unique again
        drop(msg2);
        assert!(msg.is_unique());

        // Reset header
        msg.reset_header();
        assert_eq!(msg.header().size(), constants::HEADER_SIZE);

        // Drop and reacquire
        let ptr = msg.buffer_ptr();
        drop(msg);

        let msg3: Message<PrepareCmd> = pool.get();
        // Due to LIFO, should get same buffer back
        assert_eq!(msg3.buffer_ptr(), ptr);
    }

    #[test]
    fn pool_clone_message_lifecycle() {
        let pool1 = MessagePool::new(4);
        let pool2 = pool1.clone();

        // Acquire from pool1
        let msg1: Message<PrepareCmd> = pool1.get();
        let msg2: Message<PrepareCmd> = pool2.get();

        // Both should be unique initially
        assert!(msg1.is_unique());
        assert!(msg2.is_unique());

        // Different buffers
        assert_ne!(msg1.buffer_ptr(), msg2.buffer_ptr());

        // Release both
        drop(msg1);
        drop(msg2);

        // Both pools see the freed buffers
        let msg3 = pool1.try_get::<PrepareCmd>();
        let msg4 = pool2.try_get::<PrepareCmd>();
        assert!(msg3.is_some());
        assert!(msg4.is_some());
    }

    struct RequestCmd;
    impl crate::vsr::command::CommandMarker for RequestCmd {
        const COMMAND: Command = Command::Request;
        type Header = crate::vsr::HeaderPrepare; // Reuse layout for test
    }

    #[test]
    fn reuse_with_different_command_type() {
        let pool = MessagePool::new(1);

        // 1. Get as Prepare
        let mut msg1: Message<PrepareCmd> = pool.get();
        msg1.header_mut().set_size(100);
        drop(msg1);

        // 2. Get as Request
        let msg2: Message<RequestCmd> = pool.get();

        // Header should be re-initialized correctly
        assert_eq!(msg2.header().command(), Command::Request);
        assert_eq!(msg2.header().size(), constants::HEADER_SIZE);
        assert_eq!(msg2.header().protocol(), constants::VSR_VERSION);
    }

    #[test]
    fn body_persistence_dirty_reuse() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        // Write to body
        let body_len = 32;
        msg.set_used_len(constants::HEADER_SIZE_USIZE + body_len);
        msg.body_used_mut().fill(0xAA);

        // Verify write
        assert_eq!(msg.body_used()[0], 0xAA);

        drop(msg);

        // Reacquire
        let mut msg2: Message<PrepareCmd> = pool.get();

        // Header is reset (size == HEADER_SIZE)
        assert_eq!(msg2.used_len(), constants::HEADER_SIZE_USIZE);

        // But underlying buffer still has data if we look
        // We need to extend size to see it safely via API
        msg2.set_used_len(constants::HEADER_SIZE_USIZE + body_len);

        // Data should still be there (Zero-copy / dirty reuse)
        assert_eq!(msg2.body_used()[0], 0xAA);
    }

    #[test]
    fn message_debug_impl() {
        let pool = MessagePool::new(1);
        let msg: Message<PrepareCmd> = pool.get();
        let s = format!("{:?}", msg);
        assert!(s.contains("Message"));
        assert!(s.contains("Prepare"));
        assert!(s.contains("used_len"));
    }
}

mod proptests {
    use super::super::*;
    use crate::vsr::command::PrepareCmd;
    use proptest::prelude::*;

    // Keep proptests small: each buffer is 1 MiB, so allocations scale quickly.
    const PROPTEST_CASES: u32 = 8;
    const PROPTEST_CAPACITY_MAX: usize = 4;
    const PROPTEST_HANDLES_MAX: usize = 8;
    const PROPTEST_OPS_MAX: usize = 16;

    /// Operations that can be performed on a pool
    #[derive(Debug, Clone)]
    enum PoolOp {
        Acquire,
        Release(usize),     // Index into held handles
        CloneHandle(usize), // Clone handle at index
    }

    fn pool_op_strategy(max_handles: usize) -> impl Strategy<Value = PoolOp> {
        prop_oneof![
            3 => Just(PoolOp::Acquire),
            2 => (0..max_handles).prop_map(PoolOp::Release),
            1 => (0..max_handles).prop_map(PoolOp::CloneHandle),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        /// Property: After any sequence of acquire/release/clone operations,
        /// the pool maintains the invariant that all buffers are either in-use or available.
        #[test]
        fn prop_pool_conservation(
            capacity in 1usize..=PROPTEST_CAPACITY_MAX,
            ops in prop::collection::vec(pool_op_strategy(PROPTEST_HANDLES_MAX), 0..=PROPTEST_OPS_MAX)
        ) {
            let pool = MessagePool::new(capacity);
            let mut handles: Vec<Option<Message<PrepareCmd>>> = Vec::new();

            for op in ops {
                match op {
                    PoolOp::Acquire => {
                        if let Some(m) = pool.try_get::<PrepareCmd>() {
                            handles.push(Some(m));
                        }
                    }
                    PoolOp::Release(idx) => {
                        if idx < handles.len() {
                            handles[idx] = None;
                        }
                    }
                    PoolOp::CloneHandle(idx) => {
                        if idx < handles.len()
                            && let Some(ref h) = handles[idx]
                        {
                            handles.push(Some(h.clone()));
                        }
                    }
                }
            }

            // Drop all and verify conservation
            handles.clear();

            // All buffers should be available now
            let mut available = Vec::new();
            while let Some(m) = pool.try_get::<PrepareCmd>() {
                available.push(m);
            }
            prop_assert_eq!(available.len(), capacity, "Pool capacity not conserved");
        }

        /// Property: Header is always correctly initialized after try_get
        #[test]
        fn prop_header_initialized_correctly(capacity in 1usize..=PROPTEST_CAPACITY_MAX) {
            let pool = MessagePool::new(capacity);

            for _ in 0..capacity {
                let msg: Message<PrepareCmd> = pool.get();
                prop_assert_eq!(
                    msg.header().command(),
                    crate::vsr::Command::Prepare,
                    "Command not set correctly"
                );
                prop_assert_eq!(
                    msg.header().size(),
                    constants::HEADER_SIZE,
                    "Size not set correctly"
                );
                prop_assert_eq!(
                    msg.header().protocol(),
                    constants::VSR_VERSION,
                    "Protocol not set correctly"
                );
            }
        }

        /// Property: set_used_len and used_len are inverse operations
        #[test]
        fn prop_used_len_roundtrip(
            len in constants::HEADER_SIZE_USIZE..=constants::MESSAGE_SIZE_MAX_USIZE
        ) {
            let pool = MessagePool::new(1);
            let mut msg: Message<PrepareCmd> = pool.get();

            msg.set_used_len(len);
            prop_assert_eq!(msg.used_len(), len);
            prop_assert_eq!(msg.as_bytes().len(), len);
        }

        /// Property: Free-list is LIFO (last released = first acquired)
        #[test]
        fn prop_free_list_lifo(_seed in 0u32..100) {
            let pool = MessagePool::new(4);

            // Acquire all 4
            let msg0: Message<PrepareCmd> = pool.get();
            let msg1: Message<PrepareCmd> = pool.get();
            let msg2: Message<PrepareCmd> = pool.get();
            let msg3: Message<PrepareCmd> = pool.get();

            let ptr0 = msg0.buffer_ptr();
            let ptr1 = msg1.buffer_ptr();
            let ptr2 = msg2.buffer_ptr();
            let ptr3 = msg3.buffer_ptr();

            // Release in order: 0, 1, 2, 3
            drop(msg0);
            drop(msg1);
            drop(msg2);
            drop(msg3);

            // LIFO: should get back in reverse order: 3, 2, 1, 0
            let re0: Message<PrepareCmd> = pool.get();
            let re1: Message<PrepareCmd> = pool.get();
            let re2: Message<PrepareCmd> = pool.get();
            let re3: Message<PrepareCmd> = pool.get();

            prop_assert_eq!(re0.buffer_ptr(), ptr3, "First reacquired should be last released");
            prop_assert_eq!(re1.buffer_ptr(), ptr2);
            prop_assert_eq!(re2.buffer_ptr(), ptr1);
            prop_assert_eq!(re3.buffer_ptr(), ptr0, "Last reacquired should be first released");
        }

        /// Property: Buffer alignment is always SECTOR_SIZE
        #[test]
        fn prop_buffer_alignment(capacity in 1usize..=PROPTEST_CAPACITY_MAX) {
            let pool = MessagePool::new(capacity);
            let mut handles: Vec<Message<PrepareCmd>> = Vec::new();

            for _ in 0..capacity {
                let msg = pool.get();
                let ptr = msg.buffer_ptr() as usize;
                prop_assert_eq!(
                    ptr % constants::SECTOR_SIZE,
                    0,
                    "Buffer not sector-aligned: {:#x}",
                    ptr
                );
                handles.push(msg);
            }
        }

        /// Property: Clone preserves buffer pointer identity
        #[test]
        fn prop_clone_preserves_pointer(_seed in 0u32..1000) {
            let pool = MessagePool::new(4);
            let msg1: Message<PrepareCmd> = pool.get();
            let ptr1 = msg1.buffer_ptr();

            let msg2 = msg1.clone();
            let msg3 = msg2.clone();

            prop_assert_eq!(msg1.buffer_ptr(), ptr1);
            prop_assert_eq!(msg2.buffer_ptr(), ptr1);
            prop_assert_eq!(msg3.buffer_ptr(), ptr1);
        }

        /// Property: After reset_header, header is in initial state
        #[test]
        fn prop_reset_header_restores_initial(
            new_size in constants::HEADER_SIZE_USIZE..=constants::MESSAGE_SIZE_MAX_USIZE
        ) {
            let pool = MessagePool::new(1);
            let mut msg: Message<PrepareCmd> = pool.get();

            // Modify header
            msg.header_mut().set_size(new_size as u32);

            // Reset
            msg.reset_header();

            // Verify initial state
            prop_assert_eq!(msg.header().size(), constants::HEADER_SIZE);
            prop_assert_eq!(msg.header().command(), crate::vsr::Command::Prepare);
            prop_assert_eq!(msg.header().protocol(), constants::VSR_VERSION);
        }
    }
}
