//! Fixed-size, stack-allocated object pool for in-flight I/O operations.
//!
//! Callers acquire raw pointers to slots, initialize them, and later release them.
//! Raw pointers serve as handles that can be stored independently (e.g., in kernel
//! completion queues) without borrowing the pool.

use core::mem::MaybeUninit;
use std::marker::PhantomData;

use crate::stdx::{BitSetIterator, bitset::BitSet};

/// Fixed-capacity pool of `SIZE` slots, tracked by a [`BitSet`].
///
/// Acquired slots return pointers to uninitialized memory. The caller must
/// initialize before use and release exactly once.
pub struct IOPSType<T, const SIZE: usize, const WORDS: usize> {
    pub items: [MaybeUninit<T>; SIZE],
    busy: BitSet<SIZE, WORDS>,
}

impl<T, const SIZE: usize, const WORDS: usize> Default for IOPSType<T, SIZE, WORDS> {
    fn default() -> Self {
        // SAFETY: An array of `MaybeUninit<T>` is valid uninitialized.
        let items = unsafe { MaybeUninit::<[MaybeUninit<T>; SIZE]>::uninit().assume_init() };
        Self {
            items,
            busy: BitSet::empty(),
        }
    }
}

impl<T, const SIZE: usize, const WORDS: usize> IOPSType<T, SIZE, WORDS> {
    const _ASSERT_SIZE_FITS_U8: () = { assert!(SIZE <= u8::MAX as usize) };

    /// Returns a pointer to **zero-initialized** memory, or `None` if full.
    ///
    /// The slot is zeroed before return to ensure safe defaults for any fields
    /// that might be read before the caller completes initialization (e.g., in
    /// iteration patterns where acquired-but-not-submitted entries are scanned).
    ///
    /// Caller must write a valid `T` before reading and call [`release`](Self::release) when done.
    #[inline]
    pub fn acquire(&mut self) -> Option<*mut T> {
        let index = self.busy.first_unset()?;
        self.busy.set(index);

        let ptr = self.items[index].as_mut_ptr();
        // Zero-initialize to ensure safe defaults for any fields that may be
        // read before the caller fully initializes the slot.
        // SAFETY: ptr is valid, aligned, and we're writing SIZE_OF::<T> zeros.
        unsafe { ptr.write_bytes(0, 1) };
        Some(ptr)
    }

    /// Releases a slot. Caller must have already dropped `T` if needed.
    ///
    /// # Panics
    ///
    /// Panics if `item` wasn't acquired from this pool or is misaligned.
    /// Panics if the slot is not currently executing (double free).
    #[inline]
    pub fn release(&mut self, item: *mut T) {
        let index = self.index(item) as usize;
        assert!(self.busy.is_set(index), "double free");
        self.items[index] = MaybeUninit::uninit();
        self.busy.unset(index);
    }

    #[inline]
    fn index(&self, item: *const T) -> u8 {
        assert!(SIZE <= u8::MAX as usize);
        let base = self.items.as_ptr() as usize;
        let item = item as usize;

        let elem_size = size_of::<T>();
        assert!(elem_size != 0, "ZSTs not supported");
        assert!(item >= base, "pointer below pool base");

        let diff = item - base;
        assert!(diff.is_multiple_of(elem_size), "misaligned pointer");

        let i = diff / elem_size;
        assert!(i < SIZE, "pointer beyond pool bounds");

        i as u8
    }

    #[inline]
    pub fn available(&self) -> usize {
        SIZE - self.busy.count()
    }

    #[inline]
    pub const fn total(&self) -> u8 {
        SIZE as u8
    }

    #[inline]
    pub fn executing(&self) -> u8 {
        self.busy.count() as u8
    }

    /// Iterator over pointers to all acquired slots.
    #[inline]
    pub fn iterate(&mut self) -> IopsIterMut<'_, T, SIZE, WORDS> {
        IopsIterMut {
            iops: self as *mut _,
            bitset_iter: self.busy.iter(),
            _marker: PhantomData,
        }
    }

    /// Iterator over pointers to all acquired slots.
    #[inline]
    pub fn iterate_const(&self) -> IopsIter<'_, T, SIZE, WORDS> {
        IopsIter {
            iops: self as *const _,
            bitset_iter: self.busy.iter(),
            _marker: PhantomData,
        }
    }
}

pub struct IopsIterMut<'a, T, const SIZE: usize, const WORDS: usize> {
    iops: *mut IOPSType<T, SIZE, WORDS>,
    bitset_iter: BitSetIterator<SIZE, WORDS>,
    _marker: PhantomData<&'a mut IOPSType<T, SIZE, WORDS>>,
}

impl<'a, T, const SIZE: usize, const WORDS: usize> Iterator for IopsIterMut<'a, T, SIZE, WORDS> {
    type Item = *mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.bitset_iter.next()?;
        // SAFETY: PhantomData ensures pool outlives iterator; bitset yields only busy indices.
        let iops = unsafe { &mut *self.iops };
        Some(iops.items[index].as_mut_ptr())
    }
}

pub struct IopsIter<'a, T, const SIZE: usize, const WORDS: usize> {
    iops: *const IOPSType<T, SIZE, WORDS>,
    bitset_iter: BitSetIterator<SIZE, WORDS>,
    _marker: PhantomData<&'a IOPSType<T, SIZE, WORDS>>,
}

impl<'a, T, const SIZE: usize, const WORDS: usize> Iterator for IopsIter<'a, T, SIZE, WORDS> {
    type Item = *const T;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.bitset_iter.next()?;
        // SAFETY: PhantomData ensures pool outlives iterator; bitset yields only busy indices.
        let iops = unsafe { &*self.iops };
        Some(iops.items[index].as_ptr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::ptr;
    use std::rc::Rc;

    use crate::stdx::bitset::words_for_bits;

    // Type aliases for common pool configurations
    type SmallPool = IOPSType<u32, 4, { words_for_bits(4) }>;
    type MediumPool = IOPSType<u64, 16, { words_for_bits(16) }>;

    // ==================== Basic Operations ====================

    #[test]
    fn new_pool_is_empty() {
        let pool: SmallPool = IOPSType::default();

        assert_eq!(pool.available(), 4);
        assert_eq!(pool.executing(), 0);
        assert_eq!(pool.total(), 4);
    }

    #[test]
    fn acquire_single_slot() {
        let mut pool: SmallPool = IOPSType::default();

        let ptr = pool.acquire().expect("pool not empty");

        assert!(!ptr.is_null());
        assert_eq!(pool.executing(), 1);
        assert_eq!(pool.available(), 3);

        // Cleanup
        pool.release(ptr);
    }

    #[test]
    fn release_acquired_slot() {
        let mut pool: SmallPool = IOPSType::default();

        let ptr = pool.acquire().unwrap();
        unsafe { ptr::write(ptr, 42) };

        assert_eq!(pool.executing(), 1);

        unsafe { ptr::drop_in_place(ptr) };
        pool.release(ptr);

        assert_eq!(pool.executing(), 0);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn acquire_all_slots_returns_none() {
        let mut pool: SmallPool = IOPSType::default();
        let mut ptrs = Vec::new();

        for _ in 0..4 {
            ptrs.push(pool.acquire().unwrap());
        }

        assert_eq!(pool.executing(), 4);
        assert_eq!(pool.available(), 0);
        assert!(pool.acquire().is_none());

        // Cleanup
        for ptr in ptrs {
            pool.release(ptr);
        }
    }

    #[test]
    fn acquire_release_cycle() {
        let mut pool: SmallPool = IOPSType::default();

        // First cycle: acquire all
        let mut ptrs = Vec::new();
        for i in 0..4 {
            let ptr = pool.acquire().unwrap();
            unsafe { ptr::write(ptr, i as u32) };
            ptrs.push(ptr);
        }
        assert!(pool.acquire().is_none());

        // Release all
        for ptr in ptrs {
            unsafe { ptr::drop_in_place(ptr) };
            pool.release(ptr);
        }
        assert_eq!(pool.available(), 4);

        // Second cycle: should be able to acquire all again
        let mut ptrs2 = Vec::new();
        for _ in 0..4 {
            ptrs2.push(pool.acquire().unwrap());
        }
        assert_eq!(pool.executing(), 4);

        // Cleanup
        for ptr in ptrs2 {
            pool.release(ptr);
        }
    }

    #[test]
    fn mutate_acquired_slot() {
        let mut pool: MediumPool = IOPSType::default();

        let ptr = pool.acquire().unwrap();
        unsafe { ptr::write(ptr, 0) };

        // Mutate
        unsafe { *ptr = 42 };
        assert_eq!(unsafe { *ptr }, 42);

        // Mutate again
        unsafe { *ptr = 99 };
        assert_eq!(unsafe { *ptr }, 99);

        // Cleanup
        unsafe { ptr::drop_in_place(ptr) };
        pool.release(ptr);
    }

    // ==================== Boundary Conditions ====================

    #[test]
    fn single_slot_pool() {
        let mut pool: IOPSType<u32, 1, { words_for_bits(1) }> = IOPSType::default();

        assert_eq!(pool.total(), 1);
        assert_eq!(pool.available(), 1);

        let ptr = pool.acquire().unwrap();
        assert!(pool.acquire().is_none());
        assert_eq!(pool.executing(), 1);

        pool.release(ptr);
        assert_eq!(pool.available(), 1);

        // Can acquire again
        let ptr2 = pool.acquire().unwrap();
        assert!(pool.acquire().is_none());
        pool.release(ptr2);
    }

    #[test]
    fn pool_with_multiple_words() {
        // SIZE=128 requires 2 words (128/64 = 2)
        let mut pool: IOPSType<u32, 128, { words_for_bits(128) }> = IOPSType::default();

        assert_eq!(pool.total(), 128);

        // Acquire slots that span word boundaries
        let mut ptrs = Vec::new();
        for _ in 0..128 {
            ptrs.push(pool.acquire().unwrap());
        }

        assert_eq!(pool.executing(), 128);
        assert!(pool.acquire().is_none());

        // Release all
        for ptr in ptrs {
            pool.release(ptr);
        }
        assert_eq!(pool.available(), 128);
    }

    #[test]
    fn pool_at_size_limit_255() {
        // Maximum SIZE due to u8 index constraint
        let mut pool: IOPSType<u8, 255, { words_for_bits(255) }> = IOPSType::default();

        assert_eq!(pool.total(), 255);

        // Verify all 255 slots can be acquired
        let mut ptrs = Vec::new();
        for _ in 0..255 {
            ptrs.push(pool.acquire().unwrap());
        }

        assert_eq!(pool.executing(), 255);
        assert!(pool.acquire().is_none());

        // Cleanup
        for ptr in ptrs {
            pool.release(ptr);
        }
    }

    #[test]
    fn complex_type_pool() {
        #[derive(Debug)]
        struct DropTracker {
            #[allow(dead_code)]
            id: u32,
            drops: Rc<Cell<usize>>,
        }

        impl Drop for DropTracker {
            fn drop(&mut self) {
                self.drops.set(self.drops.get() + 1);
            }
        }

        let drops = Rc::new(Cell::new(0));
        let mut pool: IOPSType<DropTracker, 4, { words_for_bits(4) }> = IOPSType::default();

        // Acquire and initialize slots
        let mut ptrs = Vec::new();
        for i in 0..4 {
            let ptr = pool.acquire().unwrap();
            unsafe {
                ptr::write(
                    ptr,
                    DropTracker {
                        id: i,
                        drops: Rc::clone(&drops),
                    },
                );
            }
            ptrs.push(ptr);
        }

        assert_eq!(drops.get(), 0);

        // Drop and release
        for ptr in ptrs {
            unsafe { ptr::drop_in_place(ptr) };
            pool.release(ptr);
        }

        assert_eq!(drops.get(), 4);
    }

    // ==================== Pointer Validation (Panic Tests) ====================

    #[test]
    #[should_panic(expected = "ZSTs not supported")]
    fn zst_panics_correctly() {
        let mut pool: IOPSType<(), 4, { words_for_bits(4) }> = IOPSType::default();
        let ptr = pool.acquire().unwrap();
        // This will panic in index() when we try to release
        pool.release(ptr);
    }

    #[test]
    #[should_panic(expected = "misaligned pointer")]
    fn release_misaligned_pointer_panics() {
        let mut pool: IOPSType<u64, 4, { words_for_bits(4) }> = IOPSType::default();
        let ptr = pool.acquire().unwrap();

        // Create a misaligned pointer (offset by 1 byte)
        let misaligned = (ptr as usize + 1) as *mut u64;

        pool.release(misaligned);
    }

    #[test]
    #[should_panic(expected = "pointer below pool base")]
    fn release_below_bounds_panics() {
        let mut pool: IOPSType<u64, 4, { words_for_bits(4) }> = IOPSType::default();
        let ptr = pool.acquire().unwrap();

        // Create a pointer below the pool base
        let below = (ptr as usize - 1024) as *mut u64;

        pool.release(below);
    }

    #[test]
    #[should_panic(expected = "pointer beyond pool bounds")]
    fn release_beyond_bounds_panics() {
        let mut pool: IOPSType<u64, 4, { words_for_bits(4) }> = IOPSType::default();
        let _ptr = pool.acquire().unwrap();

        // Create a pointer well beyond the pool bounds
        let beyond = (pool.items.as_ptr() as usize + 1000 * size_of::<u64>()) as *mut u64;

        pool.release(beyond);
    }

    #[test]
    #[should_panic]
    fn release_from_different_pool_panics() {
        let mut pool_a: SmallPool = IOPSType::default();
        let mut pool_b: SmallPool = IOPSType::default();

        let ptr_a = pool_a.acquire().unwrap();

        // Try to release ptr_a into pool_b - should panic
        pool_b.release(ptr_a);
    }

    #[test]
    #[should_panic(expected = "double free")]
    fn release_double_free_panics() {
        let mut pool: SmallPool = IOPSType::default();
        let ptr = pool.acquire().unwrap();
        pool.release(ptr);
        pool.release(ptr);
    }

    // ==================== Iterator Tests ====================

    #[test]
    fn iterate_empty_pool() {
        let mut pool: SmallPool = IOPSType::default();

        assert_eq!(pool.iterate().count(), 0);
    }

    #[test]
    fn iterate_acquired_slots() {
        let mut pool: SmallPool = IOPSType::default();

        // Acquire 3 slots
        let ptr1 = pool.acquire().unwrap();
        let ptr2 = pool.acquire().unwrap();
        let ptr3 = pool.acquire().unwrap();

        unsafe {
            ptr::write(ptr1, 10);
            ptr::write(ptr2, 20);
            ptr::write(ptr3, 30);
        }

        // Collect iterator results
        let iter_ptrs: Vec<*mut u32> = pool.iterate().collect();

        assert_eq!(iter_ptrs.len(), 3);

        // Sum the values to verify we got the right pointers
        let sum: u32 = iter_ptrs.iter().map(|&p| unsafe { *p }).sum();
        assert_eq!(sum, 60);

        // Cleanup
        pool.release(ptr1);
        pool.release(ptr2);
        pool.release(ptr3);
    }

    #[test]
    fn iterate_const_acquired_slots() {
        let mut pool: SmallPool = IOPSType::default();

        let ptr1 = pool.acquire().unwrap();
        let ptr2 = pool.acquire().unwrap();

        unsafe {
            ptr::write(ptr1, 100);
            ptr::write(ptr2, 200);
        }

        // Use const iterator
        let iter_ptrs: Vec<*const u32> = pool.iterate_const().collect();

        assert_eq!(iter_ptrs.len(), 2);

        let sum: u32 = iter_ptrs.iter().map(|&p| unsafe { *p }).sum();
        assert_eq!(sum, 300);

        // Cleanup
        pool.release(ptr1);
        pool.release(ptr2);
    }

    #[test]
    fn iterate_after_release() {
        let mut pool: SmallPool = IOPSType::default();

        // Acquire 4 slots
        let ptr1 = pool.acquire().unwrap();
        let ptr2 = pool.acquire().unwrap();
        let ptr3 = pool.acquire().unwrap();
        let ptr4 = pool.acquire().unwrap();

        unsafe {
            ptr::write(ptr1, 1);
            ptr::write(ptr2, 2);
            ptr::write(ptr3, 3);
            ptr::write(ptr4, 4);
        }

        // Release 2 of them
        pool.release(ptr2);
        pool.release(ptr4);

        // Iterator should yield only 2 slots
        let iter_ptrs: Vec<*mut u32> = pool.iterate().collect();
        assert_eq!(iter_ptrs.len(), 2);

        // Values should be 1 and 3 (the ones we didn't release)
        let mut values: Vec<u32> = iter_ptrs.iter().map(|&p| unsafe { *p }).collect();
        values.sort();
        assert_eq!(values, vec![1, 3]);

        // Cleanup
        pool.release(ptr1);
        pool.release(ptr3);
    }

    #[test]
    fn iterate_full_pool() {
        let mut pool: MediumPool = IOPSType::default();

        // Acquire all 16 slots
        let mut ptrs = Vec::new();
        for i in 0..16 {
            let ptr = pool.acquire().unwrap();
            unsafe { ptr::write(ptr, i as u64) };
            ptrs.push(ptr);
        }

        // Iterator should yield all 16
        assert_eq!(pool.iterate().count(), 16);

        // Cleanup
        for ptr in ptrs {
            pool.release(ptr);
        }
    }

    #[test]
    fn iterate_yields_valid_pointers() {
        let mut pool: SmallPool = IOPSType::default();

        let ptr1 = pool.acquire().unwrap();
        let ptr2 = pool.acquire().unwrap();

        let base = pool.items.as_ptr() as usize;
        let end = base + 4 * size_of::<MaybeUninit<u32>>();

        // All iterated pointers should be within pool bounds
        for ptr in pool.iterate() {
            let addr = ptr as usize;
            assert!(addr >= base, "pointer below pool base");
            assert!(addr < end, "pointer beyond pool bounds");
        }

        // Cleanup
        pool.release(ptr1);
        pool.release(ptr2);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;
    use std::ptr;

    use crate::stdx::bitset::words_for_bits;

    const PROPTEST_CASES: u32 = 16;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(
            crate::test_utils::proptest_cases(PROPTEST_CASES)
        ))]

        /// Random acquire/release operations maintain invariants
        #[test]
        fn random_ops(ops in prop::collection::vec((prop::bool::ANY, any::<usize>()), 0..200)) {
            let mut pool: IOPSType<u64, 16, { words_for_bits(16) }> = IOPSType::default();
            let mut acquired: Vec<*mut u64> = Vec::new();

            for (is_acquire, idx) in ops {
                if is_acquire {
                    if let Some(ptr) = pool.acquire() {
                        unsafe { ptr::write(ptr, 0) };
                        acquired.push(ptr);
                    } else {
                        prop_assert_eq!(pool.available(), 0);
                    }
                } else if !acquired.is_empty() {
                    let remove_idx = idx % acquired.len();
                    let ptr = acquired.swap_remove(remove_idx);
                    unsafe { ptr::drop_in_place(ptr) };
                    pool.release(ptr);
                }

                // Invariants
                prop_assert_eq!(pool.executing() as usize, acquired.len());
                prop_assert_eq!(pool.available() + acquired.len(), 16);
            }

            // Cleanup
            for ptr in acquired {
                unsafe { ptr::drop_in_place(ptr) };
                pool.release(ptr);
            }
        }

        /// Acquired pointers are always unique (no double-allocation)
        #[test]
        fn acquired_pointers_unique(acquire_count in 1..=16usize) {
            let mut pool: IOPSType<u64, 16, { words_for_bits(16) }> = IOPSType::default();
            let mut ptrs = Vec::new();
            let mut addrs = HashSet::new();

            for _ in 0..acquire_count {
                let ptr = pool.acquire().unwrap();
                let addr = ptr as usize;

                // Each pointer must be unique
                prop_assert!(!addrs.contains(&addr), "duplicate pointer allocated");
                addrs.insert(addr);
                ptrs.push(ptr);
            }

            prop_assert_eq!(addrs.len(), acquire_count);

            // Cleanup
            for ptr in ptrs {
                pool.release(ptr);
            }
        }

        /// Pool never exceeds capacity
        #[test]
        fn never_exceeds_capacity(extra_attempts in 0..100usize) {
            let mut pool: IOPSType<u32, 8, { words_for_bits(8) }> = IOPSType::default();
            let mut acquired = Vec::new();

            // Fill to capacity
            for _ in 0..8 {
                acquired.push(pool.acquire().unwrap());
            }

            prop_assert_eq!(pool.executing(), 8);
            prop_assert_eq!(pool.available(), 0);

            // Additional acquires should all fail
            for _ in 0..extra_attempts {
                prop_assert!(pool.acquire().is_none());
            }

            // Cleanup
            for ptr in acquired {
                pool.release(ptr);
            }
        }

        /// Iterator count matches executing count
        #[test]
        fn iterator_count_matches_executing(acquire_count in 0..=16usize) {
            let mut pool: IOPSType<u64, 16, { words_for_bits(16) }> = IOPSType::default();
            let mut ptrs = Vec::new();

            for _ in 0..acquire_count {
                ptrs.push(pool.acquire().unwrap());
            }

            let iter_count = pool.iterate().count();
            let executing = pool.executing() as usize;
            prop_assert_eq!(iter_count, executing);
            prop_assert_eq!(iter_count, acquire_count);

            // Cleanup
            for ptr in ptrs {
                pool.release(ptr);
            }
        }

        /// Round-trip acquire/release/acquire works
        #[test]
        fn round_trip(iterations in 1..10usize) {
            let mut pool: IOPSType<u32, 8, { words_for_bits(8) }> = IOPSType::default();

            for _ in 0..iterations {
                // Acquire all
                let mut ptrs = Vec::new();
                for _ in 0..8 {
                    ptrs.push(pool.acquire().unwrap());
                }
                prop_assert_eq!(pool.available(), 0);

                // Release all
                for ptr in ptrs {
                    pool.release(ptr);
                }
                prop_assert_eq!(pool.available(), 8);
            }
        }

        /// available + executing always equals total
        #[test]
        fn available_plus_executing_equals_total(ops in prop::collection::vec(prop::bool::ANY, 0..100)) {
            let mut pool: IOPSType<u32, 8, { words_for_bits(8) }> = IOPSType::default();
            let mut acquired = Vec::new();

            for should_acquire in ops {
                if should_acquire {
                    if let Some(ptr) = pool.acquire() {
                        acquired.push(ptr);
                    }
                } else if let Some(ptr) = acquired.pop() {
                    pool.release(ptr);
                }

                // Invariant must always hold
                prop_assert_eq!(
                    pool.available() + pool.executing() as usize,
                    pool.total() as usize
                );
            }

            // Cleanup
            for ptr in acquired {
                pool.release(ptr);
            }
        }
    }
}
