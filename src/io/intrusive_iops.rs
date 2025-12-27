//! Type-safe object pool for I/O operation slots.
//!
//! Provides fixed-capacity pools where slots are acquired by pointer and released back
//! when complete. Uses an intrusive linked list internally for O(1) acquire/release
//! without additional allocations.
//!
//! # Design Decisions
//!
//! **Pointer-based API**: Returns [`NonNull<Iop<T>>`] rather than RAII guards. This avoids
//! lifetime complexity when slots must outlive the borrow of the pool (e.g., I/O callbacks).
//! The tradeoff is that callers must manually release slots and ensure pointer validity.
//!
//! **FIFO ordering**: Released slots are reused in FIFO order for cache locality.
//!
//! **Not thread-safe**: The pool requires `&mut self` for all state-mutating operations.
//! External synchronization is needed for concurrent access.
//!
//! # Example
//!
//! ```
//! use consensus::io::intrusive_iops::Iops;
//!
//! let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);
//!
//! // Acquire a slot
//! let ptr = pool.acquire().expect("pool not exhausted");
//!
//! // Access the value
//! unsafe {
//!     pool.get_mut(ptr).value = 42;
//!     assert_eq!(pool.get(ptr).value, 42);
//!
//!     // Release when done
//!     pool.release(ptr);
//! }
//! ```

use crate::stdx::queue::{Queue, QueueLink, QueueNode};
use core::ptr::NonNull;

// Compile-time validations
const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

/// Phantom tag for the free-list intrusive queue.
enum FreeTag {}

/// A single slot in an [`Iops`] pool.
///
/// Wraps user data (`value`) with pool bookkeeping. The `value` field is public
/// for direct access; use [`Iops::get`] or [`Iops::get_mut`] to obtain references.
pub struct Iop<T> {
    /// User-provided data stored in this slot.
    pub value: T,
    free_link: QueueLink<Iop<T>, FreeTag>,
    acquired: bool,
}

impl<T> Iop<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            free_link: QueueLink::new(),
            acquired: false,
        }
    }

    #[inline]
    pub fn is_acquired(&self) -> bool {
        self.acquired
    }
}

impl<T> QueueNode<FreeTag> for Iop<T> {
    fn queue_link(&mut self) -> &mut QueueLink<Iop<T>, FreeTag> {
        &mut self.free_link
    }

    fn queue_link_ref(&self) -> &QueueLink<Iop<T>, FreeTag> {
        &self.free_link
    }
}

/// Fixed-capacity pool of [`Iop<T>`] slots.
///
/// Allocates `N` slots on construction. Slots are acquired and released by pointer,
/// enabling zero-allocation I/O dispatch patterns where operations must outlive
/// borrows of the pool.
///
/// # Capacity
///
/// The const generic `N` must be in `1..=u32::MAX`. Internally uses `u32` counters
/// for portability across 32/64-bit platforms.
pub struct Iops<T, const N: usize> {
    slots: Box<[Iop<T>]>,
    free: Queue<Iop<T>, FreeTag>,
    in_use: u32,
}

impl<T, const N: usize> Iops<T, N> {
    /// Maximum pool capacity, limited by `u32` counters.
    pub const MAX_CAPACITY: usize = u32::MAX as usize;

    /// Creates a pool with `N` slots, initializing each with `init(index)`.
    ///
    /// # Panics
    ///
    /// Compile-time panic if `N == 0` or `N > u32::MAX`.
    pub fn new(mut init: impl FnMut(usize) -> T) -> Self {
        // Compile-time validations
        const { assert!(N > 0, "Pool capacity must be > 0") };
        const { assert!(N <= u32::MAX as usize, "Pool capacity exceeds u32::MAX") };

        let mut v = Vec::with_capacity(N);
        for i in 0..N {
            v.push(Iop::new(init(i)));
        }

        assert!(v.len() == N);
        assert!(v.capacity() == N);

        let mut slots = v.into_boxed_slice();
        assert!(slots.len() == N);

        let mut free: Queue<Iop<T>, FreeTag> = Queue::init();
        slots.iter_mut().for_each(|slot| free.push(slot));

        assert!(free.len() == N as u32);

        let pool = Self {
            slots,
            free,
            in_use: 0,
        };

        assert!(pool.capacity() == N as u32);
        assert!(pool.in_use() == 0);
        assert!(pool.available() == N as u32);

        pool
    }

    #[inline]
    pub fn capacity(&self) -> u32 {
        let cap = self.slots.len() as u32;
        assert!(cap == N as u32);
        cap
    }

    #[inline]
    pub fn in_use(&self) -> u32 {
        assert!(self.in_use + self.free.len() == N as u32);
        self.in_use
    }

    #[inline]
    pub fn available(&self) -> u32 {
        let avail = self.free.len();
        assert!(self.in_use + avail == N as u32);
        avail
    }

    /// Returns `true` if all slots are currently acquired.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        let exhausted = self.free.is_empty();
        assert!(exhausted == (self.in_use == self.capacity()));
        exhausted
    }

    /// Acquires a slot from the pool.
    ///
    /// Returns `None` if the pool is exhausted. The returned pointer remains valid
    /// until [`release`](Self::release) is called with it. Slots are reused in FIFO
    /// order.
    ///
    /// # Slot Lifecycle
    ///
    /// 1. Call `acquire()` to obtain a pointer
    /// 2. Use [`get`](Self::get)/[`get_mut`](Self::get_mut) to access the slot
    /// 3. Call [`release`](Self::release) when done (required, not automatic)
    pub fn acquire(&mut self) -> Option<NonNull<Iop<T>>> {
        let old_in_use = self.in_use();
        let old_avail = self.available();

        let mut ptr = self.free.pop()?;
        assert!(old_in_use < self.capacity());

        unsafe {
            let iop = ptr.as_mut();
            assert!(!iop.is_acquired());

            iop.acquired = true;
            assert!(iop.is_acquired());
        }

        self.in_use += 1;

        assert!(self.in_use == old_in_use + 1);
        assert!(self.available() == old_avail - 1);
        assert!(self.in_use + self.available() == self.capacity());

        assert!(self.contains_ptr(ptr));

        Some(ptr)
    }

    /// Get mutable access to an acquired slot.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by a prior call to `acquire()` on this pool
    /// - `ptr` must not have been released since acquisition
    /// - The caller must ensure no other references to this slot exist
    pub unsafe fn get_mut(&mut self, ptr: NonNull<Iop<T>>) -> &mut Iop<T> {
        assert!(self.contains_ptr(ptr));
        let iop = unsafe { &mut *ptr.as_ptr() };
        assert!(iop.is_acquired(), "get_mut on non-acquired slot");
        iop
    }

    /// Get shared access to an acquired slot.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by a prior call to `acquire()` on this pool
    /// - `ptr` must not have been released since acquisition
    pub unsafe fn get(&self, ptr: NonNull<Iop<T>>) -> &Iop<T> {
        assert!(self.contains_ptr(ptr));
        let iop = unsafe { &*ptr.as_ptr() };
        assert!(iop.is_acquired(), "get on non-acquired slot");
        iop
    }

    /// Release an acquired slot back to the pool.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by a prior call to `acquire()` on this pool
    /// - `ptr` must not have been released since acquisition
    /// - The caller must ensure no references to this slot exist after release
    pub unsafe fn release(&mut self, ptr: NonNull<Iop<T>>) {
        assert!(self.contains_ptr(ptr));

        let iop = unsafe { &mut *ptr.as_ptr() };
        assert!(iop.is_acquired(), "double release detected");

        let old_in_use = self.in_use;
        let old_avail = self.available();

        assert!(old_in_use > 0);

        iop.acquired = false;

        assert!(!iop.is_acquired());

        iop.free_link.reset();
        self.free.push(iop);
        self.in_use -= 1;

        assert!(self.in_use == old_in_use - 1);
        assert!(self.available() == old_avail + 1);
        assert!(self.in_use + self.available() == self.capacity());
    }

    /// Checks if `ptr` points to a valid slot within this pool.
    ///
    /// Uses checked arithmetic to guard against overflow on platforms where
    /// `len * size_of::<Iop<T>>()` could theoretically wrap.
    #[inline]
    fn contains_ptr(&self, ptr: NonNull<Iop<T>>) -> bool {
        let ptr_addr = ptr.as_ptr() as usize;
        let slots_start = self.slots.as_ptr() as usize;

        let Some(byte_len) = self.slots.len().checked_mul(size_of::<Iop<T>>()) else {
            return false;
        };
        let Some(slots_end) = slots_start.checked_add(byte_len) else {
            return false;
        };

        if ptr_addr < slots_start || ptr_addr >= slots_end {
            return false;
        }

        let offset = ptr_addr - slots_start;
        offset.is_multiple_of(size_of::<Iop<T>>())
    }

    /// Compute slot index from pointer. For debugging.
    #[inline]
    #[allow(dead_code)]
    fn ptr_to_index(&self, ptr: NonNull<Iop<T>>) -> u32 {
        // Precondition: pointer must be valid
        assert!(self.contains_ptr(ptr));

        let ptr_addr = ptr.as_ptr() as usize;
        let slots_start = self.slots.as_ptr() as usize;
        let offset = ptr_addr - slots_start;
        let index = offset / size_of::<Iop<T>>();

        // Postcondition: index is in bounds
        assert!(index < N);

        index as u32
    }

    /// Verify pool invariants. For debugging/testing.
    #[cfg(debug_assertions)]
    pub fn check_invariants(&self) {
        // Capacity is N
        assert!(self.slots.len() == N);

        // in_use + free == capacity
        assert!(self.in_use + self.free.len() == self.capacity());

        // Count acquired slots
        let mut acquired_count: u32 = 0;
        for slot in self.slots.iter() {
            if slot.is_acquired() {
                acquired_count += 1;

                // Acquired slots must not be linked
                assert!(
                    slot.free_link.is_unlinked(),
                    "acquired slot is in free list"
                );
            } else {
                // Non-acquired slots must be linked (in free list)
                assert!(
                    !slot.free_link.is_unlinked(),
                    "non-acquired slot is not in free list"
                );
            }
        }

        // Acquired count matches in_use
        assert!(
            acquired_count == self.in_use,
            "acquired count {} != in_use {}",
            acquired_count,
            self.in_use
        );

        // Free list length matches available
        assert!(self.free.len() == self.capacity() - self.in_use);
    }
}

#[cfg(debug_assertions)]
impl<T, const N: usize> Drop for Iops<T, N> {
    fn drop(&mut self) {
        // Skip check if already panicking to avoid double-panic abort
        if !std::thread::panicking() {
            assert!(
                self.in_use == 0,
                "Iops dropped with {} slots still acquired - caller holds dangling pointers",
                self.in_use
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Basic Operations ====================

    #[test]
    fn new_pool_is_empty() {
        let pool: Iops<u32, 4> = Iops::new(|i| i as u32);

        assert!(pool.capacity() == 4);
        assert!(pool.in_use() == 0);
        assert!(pool.available() == 4);
        assert!(!pool.is_exhausted());
    }

    #[test]
    fn acquire_release_single() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);

        let ptr = pool.acquire().unwrap();

        assert!(pool.in_use() == 1);
        assert!(pool.available() == 3);

        unsafe {
            assert!(pool.get(ptr).is_acquired());
            pool.release(ptr);
        }

        assert!(pool.in_use() == 0);
        assert!(pool.available() == 4);
    }

    #[test]
    fn acquire_all_slots() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);
        let mut ptrs = Vec::new();

        for _ in 0..4 {
            ptrs.push(pool.acquire().unwrap());
        }

        assert!(pool.in_use() == 4);
        assert!(pool.available() == 0);
        assert!(pool.is_exhausted());
        assert!(pool.acquire().is_none());

        // Release all
        for ptr in ptrs {
            unsafe { pool.release(ptr) };
        }

        assert!(pool.in_use() == 0);
        assert!(!pool.is_exhausted());
    }

    #[test]
    fn acquired_slots_have_distinct_values() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);

        let a = pool.acquire().unwrap();
        let b = pool.acquire().unwrap();

        unsafe {
            assert!(pool.get(a).value != pool.get(b).value);
            pool.release(a);
            pool.release(b);
        }
    }

    #[test]
    fn reacquire_released_slot() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);

        let a = pool.acquire().unwrap();
        let a_value = unsafe { pool.get(a).value };

        unsafe { pool.release(a) };

        // Acquire again - should get different slot (FIFO)
        let b = pool.acquire().unwrap();
        let b_value = unsafe { pool.get(b).value };

        assert!(a_value != b_value);

        unsafe { pool.release(b) };
    }

    #[test]
    fn mutate_acquired_slot() {
        let mut pool: Iops<u32, 4> = Iops::new(|_| 0);

        let ptr = pool.acquire().unwrap();

        unsafe {
            pool.get_mut(ptr).value = 42;
            assert!(pool.get(ptr).value == 42);
            pool.release(ptr);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    fn check_invariants_throughout_lifecycle() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);
        pool.check_invariants();

        let a = pool.acquire().unwrap();
        pool.check_invariants();

        let b = pool.acquire().unwrap();
        pool.check_invariants();

        unsafe { pool.release(a) };
        pool.check_invariants();

        unsafe { pool.release(b) };
        pool.check_invariants();
    }

    #[test]
    #[should_panic(expected = "double release detected")]
    fn double_release_panics() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);

        let ptr = pool.acquire().unwrap();

        unsafe {
            pool.release(ptr);
            pool.release(ptr); // Should panic
        }
    }

    #[test]
    #[should_panic(expected = "get_mut on non-acquired slot")]
    fn get_mut_on_released_panics() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);

        let ptr = pool.acquire().unwrap();

        unsafe {
            pool.release(ptr);
            pool.get_mut(ptr); // Should panic
        }
    }

    #[test]
    #[should_panic(expected = "get on non-acquired slot")]
    fn get_on_released_panics() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);

        let ptr = pool.acquire().unwrap();

        unsafe {
            pool.release(ptr);
            pool.get(ptr); // Should panic
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn release_from_different_pool_panics() {
        let mut pool_a: Iops<u32, 4> = Iops::new(|i| i as u32);
        let mut pool_b: Iops<u32, 4> = Iops::new(|i| i as u32);

        let ptr_a = pool_a.acquire().unwrap();

        // Attempt to release ptr_a into pool_b - should panic on contains_ptr check
        unsafe { pool_b.release(ptr_a) };
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn get_from_different_pool_panics() {
        let pool_a: Iops<u32, 4> = Iops::new(|i| i as u32);
        let mut pool_b: Iops<u32, 4> = Iops::new(|i| i as u32);

        let ptr_b = pool_b.acquire().unwrap();

        // Attempt to get ptr_b from pool_a - should panic on contains_ptr check
        unsafe { pool_a.get(ptr_b) };
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn fabricated_pointer_panics() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);

        // Create a fabricated pointer that was never acquired
        let fake_ptr = NonNull::dangling();

        // Attempt to release fabricated pointer - should panic on contains_ptr check
        unsafe { pool.release(fake_ptr) };
    }

    #[test]
    #[should_panic(expected = "Iops dropped with")]
    #[cfg(debug_assertions)]
    fn drop_with_acquired_slots_panics() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);
        let _ptr = pool.acquire().unwrap();
        // pool drops here with in_use > 0
    }

    // ==================== Boundary Condition Tests ====================

    #[test]
    fn single_slot_pool() {
        let mut pool: Iops<u32, 1> = Iops::new(|i| i as u32);

        assert!(pool.capacity() == 1);
        assert!(pool.available() == 1);
        assert!(!pool.is_exhausted());

        let ptr = pool.acquire().unwrap();
        assert!(pool.is_exhausted());
        assert!(pool.acquire().is_none());

        unsafe {
            assert!(pool.get(ptr).value == 0);
            pool.release(ptr);
        }

        assert!(!pool.is_exhausted());
        assert!(pool.available() == 1);

        // Acquire again
        let ptr2 = pool.acquire().unwrap();
        assert!(pool.is_exhausted());
        unsafe { pool.release(ptr2) };
    }

    #[test]
    #[cfg(debug_assertions)]
    fn single_slot_invariants() {
        let mut pool: Iops<u32, 1> = Iops::new(|_| 42);
        pool.check_invariants();

        let ptr = pool.acquire().unwrap();
        pool.check_invariants();

        unsafe { pool.release(ptr) };
        pool.check_invariants();
    }

    // ==================== ZST (Zero-Sized Type) Tests ====================

    #[test]
    fn zst_pool() {
        // Zero-sized type - Iop<()> still has non-zero size due to QueueLink and bool
        let mut pool: Iops<(), 4> = Iops::new(|_| ());

        assert!(pool.capacity() == 4);
        assert!(pool.available() == 4);

        // Verify Iop<()> is NOT zero-sized (has link and acquired fields)
        assert!(size_of::<Iop<()>>() > 0);

        let ptr = pool.acquire().unwrap();
        assert!(pool.in_use() == 1);

        unsafe { pool.release(ptr) };
        assert!(pool.in_use() == 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn zst_invariants() {
        let mut pool: Iops<(), 8> = Iops::new(|_| ());
        pool.check_invariants();

        let mut ptrs = Vec::new();
        for _ in 0..8 {
            ptrs.push(pool.acquire().unwrap());
            pool.check_invariants();
        }

        for ptr in ptrs {
            unsafe { pool.release(ptr) };
            pool.check_invariants();
        }
    }

    // ==================== Pointer Validation Tests ====================

    #[test]
    fn contains_ptr_alignment() {
        let pool: Iops<u64, 4> = Iops::new(|i| i as u64);

        // Get a valid pointer's address
        let valid_ptr = NonNull::from(&pool.slots[0]);
        assert!(pool.contains_ptr(valid_ptr));

        // Create a misaligned pointer (offset by 1 byte)
        let misaligned_addr = valid_ptr.as_ptr() as usize + 1;
        let misaligned_ptr = NonNull::new(misaligned_addr as *mut Iop<u64>).unwrap();
        assert!(!pool.contains_ptr(misaligned_ptr));
    }

    #[test]
    fn contains_ptr_bounds() {
        let pool: Iops<u32, 4> = Iops::new(|i| i as u32);

        // Valid pointers
        for i in 0..4 {
            let ptr = NonNull::from(&pool.slots[i]);
            assert!(pool.contains_ptr(ptr));
        }

        // Pointer before slots
        let before_addr = pool.slots.as_ptr() as usize - size_of::<Iop<u32>>();
        if let Some(before_ptr) = NonNull::new(before_addr as *mut Iop<u32>) {
            assert!(!pool.contains_ptr(before_ptr));
        }

        // Pointer after slots
        let after_addr = pool.slots.as_ptr() as usize + (4 * size_of::<Iop<u32>>());
        if let Some(after_ptr) = NonNull::new(after_addr as *mut Iop<u32>) {
            assert!(!pool.contains_ptr(after_ptr));
        }
    }

    // ==================== ptr_to_index Tests ====================

    #[test]
    fn ptr_to_index_mapping() {
        let mut pool: Iops<u32, 8> = Iops::new(|i| i as u32);

        // Acquire all slots and verify ptr_to_index returns correct indices
        let mut ptrs = Vec::new();
        for _ in 0..8 {
            ptrs.push(pool.acquire().unwrap());
        }

        // The indices should cover 0..8 (order depends on free list)
        let mut indices: Vec<u32> = ptrs.iter().map(|&ptr| pool.ptr_to_index(ptr)).collect();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);

        for ptr in ptrs {
            unsafe { pool.release(ptr) };
        }
    }

    #[test]
    fn ptr_to_index_matches_value() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);

        // Values are initialized with their index, so we can verify
        for _ in 0..4 {
            let ptr = pool.acquire().unwrap();
            let idx = pool.ptr_to_index(ptr);
            let value = unsafe { pool.get(ptr).value };
            assert_eq!(idx, value);
            unsafe { pool.release(ptr) };
        }
    }

    // ==================== FIFO Ordering Tests ====================

    #[test]
    fn fifo_ordering() {
        let mut pool: Iops<u32, 8> = Iops::new(|i| i as u32);

        // Acquire all 8 slots - should get them in order 0,1,2,3,4,5,6,7
        let mut first_round: Vec<NonNull<Iop<u32>>> = Vec::new();
        let mut first_values: Vec<u32> = Vec::new();
        for _ in 0..8 {
            let ptr = pool.acquire().unwrap();
            first_values.push(unsafe { pool.get(ptr).value });
            first_round.push(ptr);
        }

        // Release in reverse order: 7,6,5,4,3,2,1,0
        for ptr in first_round.into_iter().rev() {
            unsafe { pool.release(ptr) };
        }

        // Now acquire again - FIFO means we should get them in the order they were released
        // Released order was 7,6,5,4,3,2,1,0, so acquire should return 7,6,5,4,3,2,1,0
        let mut second_values: Vec<u32> = Vec::new();
        for _ in 0..8 {
            let ptr = pool.acquire().unwrap();
            second_values.push(unsafe { pool.get(ptr).value });
            unsafe { pool.release(ptr) };
        }

        // The second round should be reverse of first round (FIFO after reverse release)
        let expected: Vec<u32> = first_values.iter().rev().copied().collect();
        assert_eq!(second_values, expected);
    }

    #[test]
    fn interleaved_acquire_release() {
        let mut pool: Iops<u32, 4> = Iops::new(|i| i as u32);

        // Acquire slots 0, 1
        let p0 = pool.acquire().unwrap();
        let p1 = pool.acquire().unwrap();

        // Release slot 0
        unsafe { pool.release(p0) };

        // Acquire slot 2 (should be slot 2, not 0 - FIFO)
        let p2 = pool.acquire().unwrap();

        // Release slot 1
        unsafe { pool.release(p1) };

        // Acquire next (should be slot 3 - FIFO)
        let p3 = pool.acquire().unwrap();

        // Now acquire again (should be slot 0 - first released)
        let p0_again = pool.acquire().unwrap();

        // Verify we got distinct slots
        let v2 = unsafe { pool.get(p2).value };
        let v3 = unsafe { pool.get(p3).value };
        let v0_again = unsafe { pool.get(p0_again).value };

        // Values should be 2, 3, 0 respectively (FIFO ordering)
        assert_eq!(v2, 2);
        assert_eq!(v3, 3);
        assert_eq!(v0_again, 0);

        // Cleanup
        unsafe {
            pool.release(p2);
            pool.release(p3);
            pool.release(p0_again);
        }
    }

    // ==================== Large Pool Tests ====================

    #[test]
    fn large_pool() {
        const SIZE: usize = 1000;
        let mut pool: Iops<u32, SIZE> = Iops::new(|i| i as u32);

        assert!(pool.capacity() == SIZE as u32);

        let mut ptrs = Vec::with_capacity(SIZE);
        for _ in 0..SIZE {
            ptrs.push(pool.acquire().unwrap());
        }

        assert!(pool.is_exhausted());
        assert!(pool.acquire().is_none());

        for ptr in ptrs {
            unsafe { pool.release(ptr) };
        }

        assert!(!pool.is_exhausted());
        assert!(pool.available() == SIZE as u32);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn large_pool_invariants() {
        const SIZE: usize = 100;
        let mut pool: Iops<u32, SIZE> = Iops::new(|i| i as u32);
        pool.check_invariants();

        // Acquire half
        let mut ptrs = Vec::new();
        for _ in 0..SIZE / 2 {
            ptrs.push(pool.acquire().unwrap());
        }
        pool.check_invariants();

        // Release all
        for ptr in ptrs {
            unsafe { pool.release(ptr) };
        }
        pool.check_invariants();
    }

    // ==================== Complex Value Type Tests ====================

    #[test]
    fn complex_value_type() {
        #[derive(Debug, PartialEq)]
        struct ComplexValue {
            id: u64,
            data: [u8; 32],
            flag: bool,
        }

        let mut pool: Iops<ComplexValue, 4> = Iops::new(|i| ComplexValue {
            id: i as u64,
            data: [i as u8; 32],
            flag: i % 2 == 0,
        });

        let ptr = pool.acquire().unwrap();

        unsafe {
            let val = &pool.get(ptr).value;
            assert_eq!(val.data, [val.id as u8; 32]);

            pool.get_mut(ptr).value.flag = true;
            assert!(pool.get(ptr).value.flag);

            pool.release(ptr);
        }
    }

    // ==================== Stress Tests ====================

    #[test]
    fn repeated_exhaust_and_refill() {
        let mut pool: Iops<u32, 8> = Iops::new(|i| i as u32);

        for iteration in 0..100 {
            // Exhaust pool
            let mut ptrs = Vec::new();
            for _ in 0..8 {
                ptrs.push(pool.acquire().unwrap());
            }
            assert!(pool.is_exhausted(), "iteration {}", iteration);

            // Refill
            for ptr in ptrs {
                unsafe { pool.release(ptr) };
            }
            assert!(!pool.is_exhausted(), "iteration {}", iteration);
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    proptest! {
        /// Random acquire/release operations maintain invariants
        #[test]
        fn random_ops(ops in prop::collection::vec((prop::bool::ANY, any::<usize>()), 0..200)) {
            let mut pool: Iops<u32, 16> = Iops::new(|i| i as u32);
            let mut acquired: Vec<NonNull<Iop<u32>>> = Vec::new();

            for (is_acquire, idx) in ops {
                if is_acquire {
                    if let Some(ptr) = pool.acquire() {
                        // Validate the pointer is part of the pool
                        assert!(pool.contains_ptr(ptr));
                        acquired.push(ptr);
                    } else {
                        assert!(pool.is_exhausted());
                    }
                } else if !acquired.is_empty() {
                    let remove_idx = idx % acquired.len();
                    let ptr = acquired.swap_remove(remove_idx);

                    unsafe {
                        // Validate before release
                        assert!(pool.get(ptr).is_acquired());
                        pool.release(ptr)
                    };
                }

                // Check invariants after every operation
                pool.check_invariants();

                // External consistency checks
                assert_eq!(pool.in_use(), acquired.len() as u32);
                assert_eq!(pool.available(), 16 - acquired.len() as u32);
            }

            // Clean up everything at the end
            for ptr in acquired {
                unsafe { pool.release(ptr) };
            }
            pool.check_invariants();
            assert_eq!(pool.in_use(), 0);
            assert_eq!(pool.available(), 16);
        }

        /// Acquired pointers are always unique (no double-allocation)
        #[test]
        fn acquired_pointers_unique(acquire_count in 1..=16usize) {
            let mut pool: Iops<u32, 16> = Iops::new(|i| i as u32);
            let mut ptrs: Vec<NonNull<Iop<u32>>> = Vec::new();
            let mut addrs: HashSet<usize> = HashSet::new();

            for _ in 0..acquire_count {
                let ptr = pool.acquire().unwrap();
                let addr = ptr.as_ptr() as usize;

                // Each pointer must be unique
                prop_assert!(!addrs.contains(&addr), "duplicate pointer allocated!");
                addrs.insert(addr);
                ptrs.push(ptr);
            }

            prop_assert_eq!(addrs.len(), acquire_count);

            // Cleanup
            for ptr in ptrs {
                unsafe { pool.release(ptr) };
            }
        }

        /// Pool never allocates beyond capacity
        #[test]
        fn never_exceeds_capacity(extra_attempts in 0..100usize) {
            let mut pool: Iops<u32, 8> = Iops::new(|i| i as u32);
            let mut acquired = Vec::new();

            // Fill to capacity
            for _ in 0..8 {
                acquired.push(pool.acquire().unwrap());
            }

            prop_assert!(pool.is_exhausted());
            prop_assert_eq!(pool.in_use(), 8);

            // Additional acquires should all fail
            for _ in 0..extra_attempts {
                prop_assert!(pool.acquire().is_none());
                prop_assert!(pool.is_exhausted());
            }

            // Cleanup
            for ptr in acquired {
                unsafe { pool.release(ptr) };
            }
        }

        /// Round-trip: acquire all, release all, acquire all again works correctly
        #[test]
        fn round_trip(iterations in 1..10usize) {
            let mut pool: Iops<u32, 8> = Iops::new(|i| i as u32);

            for _ in 0..iterations {
                // Acquire all
                let mut ptrs = Vec::new();
                for _ in 0..8 {
                    ptrs.push(pool.acquire().unwrap());
                }
                prop_assert!(pool.is_exhausted());

                // Release all
                for ptr in ptrs {
                    unsafe { pool.release(ptr) };
                }
                prop_assert!(!pool.is_exhausted());
                prop_assert_eq!(pool.available(), 8);
            }
        }

        /// Values are preserved while slots are acquired (not zeroed on acquire)
        #[test]
        fn values_preserved_while_acquired(modifications in prop::collection::vec(any::<u32>(), 1..16)) {
            let mut pool: Iops<u32, 16> = Iops::new(|i| i as u32);
            let mut ptrs = Vec::new();

            // Acquire slots and modify values
            for &new_val in &modifications {
                let ptr = pool.acquire().unwrap();
                unsafe { pool.get_mut(ptr).value = new_val };
                ptrs.push((ptr, new_val));
            }

            // Verify values are preserved while still acquired
            for &(ptr, expected) in &ptrs {
                let actual = unsafe { pool.get(ptr).value };
                prop_assert_eq!(actual, expected);
            }

            // Read values multiple times - should be stable
            for &(ptr, expected) in &ptrs {
                let actual1 = unsafe { pool.get(ptr).value };
                let actual2 = unsafe { pool.get(ptr).value };
                prop_assert_eq!(actual1, expected);
                prop_assert_eq!(actual2, expected);
            }

            // Cleanup
            for (ptr, _) in ptrs {
                unsafe { pool.release(ptr) };
            }
        }

        /// Values in slots are not zeroed on release (for same-slot re-acquire)
        #[test]
        fn values_not_zeroed_on_release(_unused in Just(())) {
            // Use a pool of size 1 to guarantee we get the same slot back
            let mut pool: Iops<u32, 1> = Iops::new(|_| 0);

            let ptr = pool.acquire().unwrap();
            unsafe { pool.get_mut(ptr).value = 42 };
            unsafe { pool.release(ptr) };

            // Re-acquire same slot (only slot in pool)
            let ptr2 = pool.acquire().unwrap();
            let value = unsafe { pool.get(ptr2).value };

            // Value should still be 42 (not zeroed)
            prop_assert_eq!(value, 42);

            unsafe { pool.release(ptr2) };
        }

        /// ptr_to_index is consistent and within bounds
        #[test]
        fn ptr_to_index_bounds(acquire_count in 1..=16usize) {
            let mut pool: Iops<u32, 16> = Iops::new(|i| i as u32);
            let mut ptrs = Vec::new();

            for _ in 0..acquire_count {
                ptrs.push(pool.acquire().unwrap());
            }

            // All indices should be valid and unique
            let indices: Vec<u32> = ptrs.iter().map(|&ptr| pool.ptr_to_index(ptr)).collect();
            let unique_count = indices.iter().collect::<HashSet<_>>().len();
            prop_assert_eq!(unique_count, acquire_count);

            // All indices should be in bounds
            for &idx in &indices {
                prop_assert!(idx < 16);
            }

            // Cleanup
            for ptr in ptrs {
                unsafe { pool.release(ptr) };
            }
        }

        /// Different pool sizes work correctly
        #[test]
        fn various_sizes(ops in prop::collection::vec(prop::bool::ANY, 0..50)) {
            // Test with size 4
            let mut pool4: Iops<u32, 4> = Iops::new(|i| i as u32);
            let mut acquired4 = Vec::new();

            for &should_acquire in &ops {
                if should_acquire {
                    if let Some(ptr) = pool4.acquire() {
                        acquired4.push(ptr);
                    }
                } else if let Some(ptr) = acquired4.pop() {
                    unsafe { pool4.release(ptr) };
                }
                pool4.check_invariants();
            }

            for ptr in acquired4 {
                unsafe { pool4.release(ptr) };
            }
        }

        /// ZST pools work correctly under random operations
        #[test]
        fn zst_random_ops(ops in prop::collection::vec(prop::bool::ANY, 0..100)) {
            let mut pool: Iops<(), 8> = Iops::new(|_| ());
            let mut acquired = Vec::new();

            for should_acquire in ops {
                if should_acquire {
                    if let Some(ptr) = pool.acquire() {
                        acquired.push(ptr);
                    }
                } else if let Some(ptr) = acquired.pop() {
                    unsafe { pool.release(ptr) };
                }

                pool.check_invariants();
                prop_assert_eq!(pool.in_use(), acquired.len() as u32);
            }

            // Cleanup
            for ptr in acquired {
                unsafe { pool.release(ptr) };
            }
        }
    }
}
