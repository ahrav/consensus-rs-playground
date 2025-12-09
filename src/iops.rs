use crate::stdx::queue::{Queue, QueueLink, QueueNode};
use core::ptr::NonNull;

// Compile-time validations
const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

enum FreeTag {}

pub struct Iop<T> {
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

pub struct Iops<T, const N: usize> {
    slots: Box<[Iop<T>]>,
    free: Queue<Iop<T>, FreeTag>, // intrusive queue
    in_use: u32,
}

impl<T, const N: usize> Iops<T, N> {
    pub const MAX_CAPACITY: usize = u32::MAX as usize;

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

    #[inline]
    pub fn is_exhausted(&self) -> bool {
        let exhausted = self.free.is_empty();
        assert!(exhausted == (self.in_use == self.capacity()));
        exhausted
    }

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

    pub unsafe fn get_mut(&mut self, ptr: NonNull<Iop<T>>) -> &mut Iop<T> {
        assert!(self.contains_ptr(ptr));
        let iop = unsafe { &mut *ptr.as_ptr() };
        assert!(iop.is_acquired(), "get_mut on non-acquired slot");
        iop
    }

    pub unsafe fn get(&self, ptr: NonNull<Iop<T>>) -> &Iop<T> {
        assert!(self.contains_ptr(ptr));
        let iop = unsafe { &*ptr.as_ptr() };
        assert!(iop.is_acquired(), "get on non-acquired slot");
        iop
    }

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

    #[inline]
    fn contains_ptr(&self, ptr: NonNull<Iop<T>>) -> bool {
        let ptr_addr = ptr.as_ptr() as usize;
        let slots_start = self.slots.as_ptr() as usize;
        let slots_end = slots_start + (self.slots.len() * size_of::<Iop<T>>());

        if ptr_addr < slots_start || ptr_addr >= slots_end {
            return false;
        }

        let offset = ptr_addr - slots_start;
        offset % size_of::<Iop<T>>() == 0
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

#[cfg(test)]
mod tests {
    use super::*;

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
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
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
                } else {
                    if !acquired.is_empty() {
                        let remove_idx = idx % acquired.len();
                        let ptr = acquired.swap_remove(remove_idx);

                        unsafe {
                            // Validate before release
                            assert!(pool.get(ptr).is_acquired());
                            pool.release(ptr)
                        };
                    }
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
    }
}
