//! Fixed-capacity message pool with reference-counted handles.
//!
//! Provides pre-allocated [`Message`] buffers to avoid allocation in hot paths.
//! Messages are acquired from the pool, passed around via [`MessageHandle`],
//! and returned when the reference count drops to zero.

use super::Message;
use crate::stdx::RingBuffer;
use core::ptr::NonNull;

const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

/// One-word, `Copy` handle to a pooled [`Message`].
///
/// Callers must manually manage reference counts via [`acquire`](Self::acquire)
/// and [`MessagePool::release`].
#[derive(Copy, Clone)]
pub struct MessageHandle {
    ptr: NonNull<Message>,
}

const _: () = assert!(size_of::<MessageHandle>() == size_of::<*const Message>());

impl MessageHandle {
    /// # Safety
    ///
    /// `ptr` must point to a valid, properly aligned [`Message`] that outlives
    /// this handle. Caller must ensure the message's reference count is managed.
    #[inline]
    unsafe fn new(ptr: NonNull<Message>) -> Self {
        assert!((ptr.as_ptr() as usize).is_multiple_of(align_of::<Message>()));

        Self { ptr }
    }

    #[inline]
    pub fn as_ptr(&self) -> NonNull<Message> {
        self.ptr
    }

    /// # Safety
    ///
    /// The handle must point to a valid message with refcount > 0.
    /// No mutable references may exist to the same message.
    #[inline]
    pub unsafe fn as_ref(&self) -> &Message {
        assert!((self.ptr.as_ptr() as usize).is_multiple_of(align_of::<Message>()));

        unsafe { self.ptr.as_ref() }
    }

    /// # Safety
    ///
    /// The handle must point to a valid message with refcount > 0.
    /// No other references (mutable or shared) may exist to the same message.
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut Message {
        assert!((self.ptr.as_ptr() as usize).is_multiple_of(align_of::<Message>()));

        unsafe { self.ptr.as_mut() }
    }

    /// Increments the reference count and returns `self`.
    ///
    /// # Safety
    ///
    /// The handle must point to a valid message with refcount > 0.
    ///
    /// # Panics
    ///
    /// Panics if reference count would overflow `u32::MAX`.
    #[inline]
    pub unsafe fn acquire(self) -> Self {
        unsafe {
            let msg = self.ptr.as_ref();
            let old_count = msg.references.get();

            assert!(old_count > 0, "acquire called on unreferenced message");
            assert!(old_count < u32::MAX, "reference count overflowed");

            let new_count = old_count + 1;
            msg.references.set(new_count);

            assert!(msg.references.get() == new_count);
            assert!(msg.references.get() == old_count + 1);

            self
        }
    }
}

/// Capacity
///
/// `N` must be in range `1..=1024` (enforced at compile time).
pub struct MessagePool<const N: usize> {
    /// Owns the actual allocations; pointers in `free` reference these.
    ///
    /// We use `Box<[Message]>` (a boxed slice) instead of `Vec<Message>` to ensure
    /// pointer stability. Once the slice is boxed, the `Message` structs inside
    /// stay at fixed memory addresses, even if the `MessagePool` struct itself moves.
    ///
    /// # Invariants
    ///
    /// Although not strictly `Pin`, the implementation guarantees that elements
    /// within this slice are **never moved or swapped**. This manual "pinning" is
    /// required because `free` and `MessageHandle` hold raw pointers to these addresses.
    owned: Box<[Message]>,
    free: RingBuffer<NonNull<Message>, N>,
}

impl<const N: usize> MessagePool<N> {
    const CAPACITY: u32 = {
        assert!(N > 0, "pool capacity must be > 0");
        assert!(N <= u32::MAX as usize, "Pool capacity must fit in u32");
        N as u32
    };

    const _MIN_CAP_CHECK: () = assert!(N >= 1);
    const _MAX_CAP_CHECK: () = assert!(N <= 1024);

    /// Creates a pool with `N` pre-allocated messages.
    pub fn new() -> Self {
        // 1. Allocate and initialize messages in a temporary Vec.
        //    We cannot use `free` yet because the addresses aren't stable.
        let mut messages = Vec::with_capacity(N);
        for _ in 0..N {
            messages.push(Message::new_zeroed());
        }

        // 2. Convert to Box<[Message]>. This pins the messages in memory.
        //    The slice pointer and length are fixed; the data is on the heap.
        let mut owned = messages.into_boxed_slice();
        assert!(owned.len() == N);

        // 3. Populate the free list with stable pointers to the boxed messages.
        let mut free: RingBuffer<NonNull<Message>, N> = RingBuffer::new();

        for msg in owned.iter_mut() {
            // Verify initial state
            assert!(msg.references.get() == 0);
            assert!(msg.len() == Message::LEN_MIN);

            let ptr = NonNull::from(msg);
            assert!((ptr.as_ptr() as usize).is_multiple_of(align_of::<Message>()));

            free.push_back_assume_capacity(ptr);
        }

        let pool = Self { owned, free };

        assert!(pool.owned.len() == N);
        assert!(pool.free.len() == Self::CAPACITY);
        assert!(pool.free.is_full());

        pool
    }

    #[inline]
    pub fn available(&self) -> u32 {
        self.free.len()
    }

    #[inline]
    pub fn capacity(&self) -> u32 {
        Self::CAPACITY
    }

    #[inline]
    pub fn in_use(&self) -> u32 {
        let in_use = Self::CAPACITY - self.free.len();

        assert!(in_use + self.free.len() == Self::CAPACITY);
        in_use
    }

    /// Acquires a message from the pool with refcount initialized to 1.
    ///
    /// # Panics
    ///
    /// Panics if the pool is empty. Use [`try_acquire`](Self::try_acquire) for
    /// fallible acquisition.
    pub fn acquire(&mut self) -> MessageHandle {
        assert!(!self.free.is_empty(), "message pool exhausted");

        let old_available = self.free.len();
        let ptr = self.free.pop_front().expect("free list unexpectedly empty");

        assert!((ptr.as_ptr() as usize).is_multiple_of(align_of::<Message>()));

        let msg = unsafe { ptr.as_ref() };
        assert!(msg.references.get() == 0);

        msg.references.set(1);

        assert!(msg.references.get() == 1);
        assert!(self.free.len() == old_available - 1);

        unsafe { MessageHandle::new(ptr) }
    }

    /// Attempts to acquire a message, returning `None` if the pool is exhausted.
    pub fn try_acquire(&mut self) -> Option<MessageHandle> {
        if self.free.is_empty() {
            return None;
        }
        Some(self.acquire())
    }

    /// Decrements the reference count, returning the message to the pool if it reaches zero.
    ///
    /// # Panics
    ///
    /// Panics if the message has refcount 0 (double-release).
    pub fn release(&mut self, handle: MessageHandle) {
        assert!((handle.ptr.as_ptr() as usize).is_multiple_of(align_of::<Message>()));

        let msg = unsafe { handle.ptr.as_ref() };
        let old_count = msg.references.get();

        assert!(old_count >= 1, "release called on unreferenced message");

        if old_count == 1 {
            let old_available = self.free.len();

            msg.references.set(0);

            assert!(!self.free.is_full());

            self.free.push_back_assume_capacity(handle.ptr);

            assert!(msg.references.get() == 0);
            assert!(self.free.len() == old_available + 1);
        } else {
            let new_count = old_count - 1;
            msg.references.set(new_count);

            assert!(msg.references.get() == new_count);
            assert!(msg.references.get() == old_count - 1);
            assert!(msg.references.get() > 0);
        }
    }

    /// Returns `true` if `handle` points to a message owned by this pool.
    pub fn owns(&self, handle: MessageHandle) -> bool {
        let ptr_addr = handle.ptr.as_ptr() as usize;
        let start_addr = self.owned.as_ptr() as usize;
        let end_addr = start_addr + (self.owned.len() * size_of::<Message>());

        ptr_addr >= start_addr && ptr_addr < end_addr
    }
}

impl<const N: usize> Default for MessagePool<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsr::wire::Command;

    #[test]
    fn pool_initialization() {
        let pool: MessagePool<8> = MessagePool::new();

        assert!(pool.capacity() == 8);
        assert!(pool.available() == 8);
        assert!(pool.in_use() == 0);
    }

    #[test]
    fn message_can_be_used() {
        let mut pool: MessagePool<4> = MessagePool::new();

        let mut handle = pool.acquire();

        // SAFETY: We have exclusive access via mut handle
        unsafe {
            let msg = handle.as_mut();
            msg.reset(Command::Ping, 42, 1);

            assert!(msg.header().command == Command::Ping);
            assert!(msg.header().cluster == 42);
            assert!(msg.header().replica == 1);
        }

        pool.release(handle);
    }

    #[test]
    #[should_panic(expected = "message pool exhausted")]
    fn acquire_panics_when_exhausted() {
        let mut pool: MessagePool<2> = MessagePool::new();

        let _h1 = pool.acquire();
        let _h2 = pool.acquire();
        let _h3 = pool.acquire(); // Should panic
    }

    #[test]
    #[should_panic(expected = "release called on unreferenced message")]
    fn double_release_panics() {
        let mut pool: MessagePool<4> = MessagePool::new();

        let handle = pool.acquire();
        pool.release(handle);
        pool.release(handle); // Should panic
    }

    #[test]
    fn try_acquire_success() {
        let mut pool: MessagePool<4> = MessagePool::new();
        let handle = pool.try_acquire();
        assert!(handle.is_some());
        assert!(pool.available() == 3);
        pool.release(handle.unwrap());
        assert!(pool.available() == 4);
    }

    #[test]
    fn handle_aliasing_and_refcounts() {
        let mut pool: MessagePool<4> = MessagePool::new();

        let h1 = pool.acquire(); // Refcount = 1
        let h2 = h1; // Copy pointer, Refcount still 1

        // SAFETY: Handle is valid
        unsafe {
            // Increment refcount via alias
            let _ = h2.acquire();
            assert!(h1.as_ref().references.get() == 2);
            assert!(h2.as_ref().references.get() == 2);
        }

        // Release via h1
        pool.release(h1); // Refcount = 1
        assert!(pool.in_use() == 1);

        // h2 is still valid
        // SAFETY: h2 is valid
        unsafe {
            assert!(h2.as_ref().references.get() == 1);
        }

        // Release via h2
        pool.release(h2); // Refcount = 0
        assert!(pool.in_use() == 0);
        assert!(pool.available() == 4);
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn pool_minimum_capacity() {
        let mut pool: MessagePool<1> = MessagePool::new();

        assert!(pool.capacity() == 1);
        assert!(pool.available() == 1);

        let handle = pool.acquire();
        assert!(pool.available() == 0);
        assert!(pool.in_use() == 1);
        assert!(pool.try_acquire().is_none());

        pool.release(handle);
        assert!(pool.available() == 1);
    }

    #[test]
    fn pool_default_trait() {
        let pool: MessagePool<4> = MessagePool::default();

        assert!(pool.capacity() == 4);
        assert!(pool.available() == 4);
        assert!(pool.in_use() == 0);
    }

    #[test]
    fn handle_as_ptr_returns_valid_pointer() {
        let mut pool: MessagePool<4> = MessagePool::new();

        let handle = pool.acquire();
        let ptr = handle.as_ptr();

        // Verify pointer is aligned
        assert!((ptr.as_ptr() as usize).is_multiple_of(align_of::<Message>()));

        // Verify we can safely dereference through as_ptr
        // SAFETY: Handle is valid
        unsafe {
            let msg_via_ptr = ptr.as_ref();
            let msg_via_handle = handle.as_ref();
            assert!(core::ptr::eq(msg_via_ptr, msg_via_handle));
        }

        pool.release(handle);
    }

    #[test]
    fn owns_returns_true_for_released_handles() {
        let mut pool: MessagePool<4> = MessagePool::new();

        let handle = pool.acquire();
        assert!(pool.owns(handle));

        pool.release(handle);

        // Handle still points to pool-owned memory even after release
        assert!(pool.owns(handle));
    }

    #[test]
    fn high_refcount_acquire_release() {
        let mut pool: MessagePool<4> = MessagePool::new();

        let handle = pool.acquire();

        // Build up refcount
        // SAFETY: Handle is valid
        unsafe {
            for i in 1..100 {
                let _ = handle.acquire();
                assert!(handle.as_ref().references.get() == i + 1);
            }

            assert!(handle.as_ref().references.get() == 100);

            // Release all but one
            for i in (1..100).rev() {
                pool.release(handle);
                assert!(handle.as_ref().references.get() == i);
                assert!(pool.in_use() == 1); // Still in use
            }

            // Final release
            pool.release(handle);
        }

        assert!(pool.in_use() == 0);
        assert!(pool.available() == 4);
    }

    #[test]
    #[should_panic(expected = "acquire called on unreferenced message")]
    fn handle_acquire_panics_on_zero_refcount() {
        let mut pool: MessagePool<4> = MessagePool::new();

        let handle = pool.acquire();
        pool.release(handle);

        // SAFETY: This is intentionally testing the panic condition
        // The handle's message now has refcount 0
        unsafe {
            let _ = handle.acquire(); // Should panic
        }
    }

    #[test]
    fn message_reset_between_uses() {
        let mut pool: MessagePool<2> = MessagePool::new();

        // First use
        let mut handle = pool.acquire();
        // SAFETY: Handle is valid
        unsafe {
            let msg = handle.as_mut();
            msg.reset(Command::Ping, 100, 5);
            assert!(msg.header().command == Command::Ping);
            assert!(msg.header().cluster == 100);
        }
        pool.release(handle);

        // Second use - same message, different data
        let mut handle = pool.acquire();
        // SAFETY: Handle is valid
        unsafe {
            let msg = handle.as_mut();
            // Message still has old data until reset
            msg.reset(Command::Pong, 200, 10);
            assert!(msg.header().command == Command::Pong);
            assert!(msg.header().cluster == 200);
        }
        pool.release(handle);
    }

    #[test]
    fn refcount_at_max_minus_one_can_acquire() {
        let mut pool: MessagePool<4> = MessagePool::new();

        let handle = pool.acquire();

        // SAFETY: Handle is valid, we're directly setting refcount for testing
        unsafe {
            // Set refcount to u32::MAX - 1 (one below overflow threshold)
            handle.as_ref().references.set(u32::MAX - 1);
            assert!(handle.as_ref().references.get() == u32::MAX - 1);

            // This acquire should succeed, bringing refcount to u32::MAX
            let _ = handle.acquire();
            assert!(handle.as_ref().references.get() == u32::MAX);
        }

        // Cleanup: reset refcount to allow proper release
        unsafe {
            handle.as_ref().references.set(1);
        }
        pool.release(handle);
    }

    #[test]
    #[should_panic(expected = "reference count overflowed")]
    fn refcount_overflow_panics_at_max() {
        let mut pool: MessagePool<4> = MessagePool::new();

        let handle = pool.acquire();

        // SAFETY: Handle is valid, we're directly setting refcount for testing
        unsafe {
            // Set refcount to u32::MAX (at overflow threshold)
            handle.as_ref().references.set(u32::MAX);

            // This acquire should panic because we're already at MAX
            let _ = handle.acquire(); // Should panic
        }
    }

    // NOTE: This test documents a known safety invariant, not a bug.
    // MessageHandle is Copy and holds raw pointers. If the pool is dropped
    // while handles exist, those handles become dangling. This is documented
    // in the safety requirements but cannot be enforced at compile time.
    //
    // Potential mitigations (not implemented):
    // - Use indices instead of raw pointers
    // - Add generation counters to detect stale handles
    // - Make MessageHandle non-Copy with lifetime bounds
    // - Use Arc<Message> instead of raw pointers
    #[test]
    fn documents_handle_lifetime_safety_requirement() {
        // This test verifies the documented invariant:
        // "ptr must point to a valid Message that outlives this handle"
        //
        // We cannot test UAF directly (undefined behavior), but we document
        // that dropping the pool invalidates all outstanding handles.

        let mut pool: MessagePool<4> = MessagePool::new();
        let handle = pool.acquire();

        // Handle is valid while pool exists
        assert!(pool.owns(handle));

        // After this point, if pool were dropped:
        // - handle.ptr would be dangling
        // - handle.as_ref() would be UAF (undefined behavior)
        //
        // Callers MUST ensure pool outlives all handles.

        pool.release(handle);
        drop(pool);

        // handle still exists here (it's Copy), but using it would be UB
        // We cannot safely test this - it would be undefined behavior
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Operations that can be performed on a pool
    #[derive(Debug, Clone)]
    enum PoolOp {
        Acquire,
        Release(usize),    // Index into held handles
        AcquireRef(usize), // Acquire additional ref on handle at index
    }

    fn pool_op_strategy(max_handles: usize) -> impl Strategy<Value = PoolOp> {
        prop_oneof![
            3 => Just(PoolOp::Acquire),
            2 => (0..max_handles).prop_map(PoolOp::Release),
            1 => (0..max_handles).prop_map(PoolOp::AcquireRef),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn invariant_available_plus_in_use_equals_capacity(
            acquire_count in 0usize..=8,
            release_indices in prop::collection::vec(0usize..8, 0..8)
        ) {
            let mut pool: MessagePool<8> = MessagePool::new();
            let mut handles: Vec<Option<MessageHandle>> = Vec::new();

            // Acquire some handles
            for _ in 0..acquire_count {
                if let Some(h) = pool.try_acquire() {
                    handles.push(Some(h));
                }
            }

            // Invariant must hold after acquires
            prop_assert_eq!(pool.available() + pool.in_use(), pool.capacity());

            // Release some handles
            for idx in release_indices {
                if idx < handles.len()
                    && let Some(h) = handles[idx].take()
                {
                    pool.release(h);
                }
            }

            // Invariant must still hold after releases
            prop_assert_eq!(pool.available() + pool.in_use(), pool.capacity());
        }

        #[test]
        fn messages_never_lost_after_full_cycle(iterations in 1usize..50) {
            let mut pool: MessagePool<8> = MessagePool::new();

            for _ in 0..iterations {
                let initial_available = pool.available();

                // Acquire all available
                let mut handles = Vec::new();
                while let Some(h) = pool.try_acquire() {
                    handles.push(h);
                }

                // Release all
                for h in handles {
                    pool.release(h);
                }

                // Must return to initial state
                prop_assert_eq!(pool.available(), initial_available);
            }
        }

        #[test]
        fn reference_counting_correctness(ops in prop::collection::vec(pool_op_strategy(4), 0..50)) {
            let mut pool: MessagePool<4> = MessagePool::new();
            let mut handles: Vec<Option<(MessageHandle, u32)>> = Vec::new(); // (handle, expected_refcount)

            for op in ops {
                match op {
                    PoolOp::Acquire => {
                        if let Some(h) = pool.try_acquire() {
                            handles.push(Some((h, 1)));
                        }
                    }
                    PoolOp::Release(idx) => {
                        if idx < handles.len()
                            && let Some((h, refcount)) = handles[idx].take()
                        {
                            pool.release(h);
                            if refcount > 1 {
                                // Still has refs, put back with decremented count
                                handles[idx] = Some((h, refcount - 1));
                            }
                        }
                    }
                    PoolOp::AcquireRef(idx) => {
                        if idx < handles.len()
                            && let Some((h, refcount)) = &mut handles[idx]
                            && *refcount < 100
                        {
                            // SAFETY: Handle is valid with refcount > 0
                            unsafe {
                                let _ = h.acquire();
                            }
                            *refcount += 1;
                        }
                    }
                }

                // Invariant check
                prop_assert_eq!(pool.available() + pool.in_use(), pool.capacity());
            }

            // Cleanup: release all remaining handles
            for entry in &mut handles {
                if let Some((h, refcount)) = entry.take() {
                    for _ in 0..refcount {
                        pool.release(h);
                    }
                }
            }

            prop_assert_eq!(pool.available(), pool.capacity());
        }

        #[test]
        fn pools_are_isolated(
            ops1 in prop::collection::vec(0u8..2, 0..20),
            ops2 in prop::collection::vec(0u8..2, 0..20)
        ) {
            let mut pool1: MessagePool<4> = MessagePool::new();
            let mut pool2: MessagePool<4> = MessagePool::new();
            let mut handles1: Vec<MessageHandle> = Vec::new();
            let mut handles2: Vec<MessageHandle> = Vec::new();

            // Interleave operations on both pools
            let max_len = ops1.len().max(ops2.len());
            for i in 0..max_len {
                // Pool 1 operation
                if let Some(&op) = ops1.get(i) {
                    if op == 0 {
                        if let Some(h) = pool1.try_acquire() {
                            // Verify pool2 doesn't own it
                            prop_assert!(!pool2.owns(h));
                            handles1.push(h);
                        }
                    } else if let Some(h) = handles1.pop() {
                        pool1.release(h);
                    }
                }

                // Pool 2 operation
                if let Some(&op) = ops2.get(i) {
                    if op == 0 {
                        if let Some(h) = pool2.try_acquire() {
                            // Verify pool1 doesn't own it
                            prop_assert!(!pool1.owns(h));
                            handles2.push(h);
                        }
                    } else if let Some(h) = handles2.pop() {
                        pool2.release(h);
                    }
                }

                // Both pools maintain their invariants independently
                prop_assert_eq!(pool1.available() + pool1.in_use(), pool1.capacity());
                prop_assert_eq!(pool2.available() + pool2.in_use(), pool2.capacity());
            }

            // Cleanup
            for h in handles1 {
                pool1.release(h);
            }
            for h in handles2 {
                pool2.release(h);
            }
        }

        #[test]
        fn fifo_ordering_preserved(_seed in 0u32..1000) {
            // Use a pool that we fully drain to test FIFO ordering
            let mut pool: MessagePool<4> = MessagePool::new();

            // Acquire ALL handles to fully drain the pool
            let mut handles = Vec::new();
            let mut first_ptrs = Vec::new();
            while let Some(h) = pool.try_acquire() {
                first_ptrs.push(h.as_ptr());
                handles.push(h);
            }

            prop_assert_eq!(first_ptrs.len(), 4);

            // Release all (they go to back of queue in release order)
            for h in handles {
                pool.release(h);
            }

            // Acquire again - should get same pointers in same order (FIFO)
            // because the queue was empty before releases, so release order = acquire order
            let mut handles = Vec::new();
            let mut second_ptrs = Vec::new();
            while let Some(h) = pool.try_acquire() {
                second_ptrs.push(h.as_ptr());
                handles.push(h);
            }

            // Release for cleanup
            for h in handles {
                pool.release(h);
            }

            prop_assert_eq!(first_ptrs, second_ptrs);
        }
    }
}
