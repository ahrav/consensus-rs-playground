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
    items: [MaybeUninit<T>; SIZE],
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

    /// Returns a pointer to uninitialized memory, or `None` if full.
    ///
    /// Caller must write a valid `T` before reading and call [`release`](Self::release) when done.
    #[inline]
    pub fn acquire(&mut self) -> Option<*mut T> {
        let index = self.busy.first_unset()?;
        self.busy.set(index);

        let ptr = self.items[index as usize].as_mut_ptr();
        Some(ptr)
    }

    /// Releases a slot. Caller must have already dropped `T` if needed.
    ///
    /// # Panics
    ///
    /// Panics if `item` wasn't acquired from this pool or is misaligned.
    #[inline]
    pub fn release(&mut self, item: *mut T) {
        let index = self.index(item) as usize;
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
        Some(iops.items[index as usize].as_mut_ptr())
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
        Some(iops.items[index as usize].as_ptr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iops_basic() {
        let mut iops: IOPSType<u64, 8, 2> = IOPSType::default();
        assert_eq!(iops.executing(), 0);
        assert_eq!(iops.available(), 8);

        let mut items: [*mut u64; 8] = [core::ptr::null_mut(); 8];

        for i in 0..8 {
            let item = iops.acquire().expect("should have slot");
            unsafe { *item = (i as u64) + 1 };
            items[i] = item;
        }

        assert_eq!(iops.executing(), 8);
        assert_eq!(iops.available(), 0);
        assert!(iops.acquire().is_none());

        let mut sum: u64 = 0;
        for item in iops.iterate() {
            unsafe { sum += *item };
        }
        assert_eq!(sum, 36);

        for item in items {
            iops.release(item);
        }
        assert_eq!(iops.executing(), 0);
        assert_eq!(iops.available(), 8);

        // Re-acquire/release sanity.
        for _ in 0..8 {
            let item = iops.acquire().unwrap();
            iops.release(item);
        }
        assert_eq!(iops.executing(), 0);
    }
}
