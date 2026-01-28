//! Stack-allocated array with runtime length tracking.
//!
//! Provides a fixed-capacity, variable-length array that lives entirely on the stack.
//! Unlike `Vec`, requires no heap allocation. Unlike `[T; N]`, tracks actual length.
//!
//! # Design Constraints
//!
//! - `T: Copy` required: enables bitwise copies of `MaybeUninit<T>` without Drop concerns
//! - Capacity `N` validated at compile time (must be in `1..=isize::MAX` and
//!   `N * size_of::<T>() <= isize::MAX` for non-ZSTs)
//! - Panics on capacity overflow rather than returning errors (fail-fast philosophy)
//!
//! # Invariant
//!
//! Elements `[0..len)` are always initialized; `[len..N)` are uninitialized.

use core::{mem::MaybeUninit, slice};

const fn assert_valid_capacity<T, const N: usize>() {
    assert!(N > 0);
    assert!(N <= isize::MAX as usize);
    let elem_size = core::mem::size_of::<T>();
    if elem_size != 0 {
        let max = isize::MAX as usize;
        let bytes = match N.checked_mul(elem_size) {
            Some(bytes) => bytes,
            None => panic!("capacity overflow"),
        };
        assert!(bytes <= max);
    }
}

/// Fixed-capacity array with tracked length, allocated on the stack.
///
/// Behaves like a `Vec<T>` but with compile-time capacity and no heap allocation.
/// The `Copy` bound on `T` ensures safe handling of uninitialized memory.
#[derive(Clone, Copy)]
pub struct BoundedArray<T: Copy, const N: usize> {
    len: usize,
    data: [MaybeUninit<T>; N],
}

impl<T: Copy, const N: usize> BoundedArray<T, N> {
    const CAPACITY_CHECK: () = assert_valid_capacity::<T, N>();

    #[inline]
    fn assert_invariants(&self) {
        let () = Self::CAPACITY_CHECK;
        assert!(self.len <= N);
    }

    /// Creates an empty array. Usable in `const` contexts for static initialization.
    #[inline]
    pub const fn new() -> Self {
        let () = Self::CAPACITY_CHECK;
        Self {
            len: 0,
            data: [MaybeUninit::uninit(); N],
        }
    }

    /// Creates an array with `count` copies of `value`.
    ///
    /// # Panics
    ///
    /// Panics if `count > N`.
    #[inline]
    pub fn from_value(value: T, count: usize) -> Self {
        assert!(count <= N);

        let mut arr = Self::new();
        for i in 0..count {
            assert!(i < N);
            arr.data[i].write(value);
        }

        arr.len = count;
        arr.assert_invariants();
        assert!(arr.len == count);

        arr
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Alias for [`len`](Self::len).
    #[inline]
    pub fn count(&self) -> usize {
        assert!(self.len <= N);
        self.len
    }

    #[inline]
    pub fn len(&self) -> usize {
        assert!(self.len <= N);
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == N
    }

    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        assert!(self.len <= N);
        let remain = N - self.len;
        debug_assert!(remain <= N);
        assert_eq!(self.len + remain, N);

        remain
    }

    /// Returns initialized elements as a slice.
    ///
    /// Named `const_slice` (not `as_slice`) to parallel [`slice`](Self::slice) for mutable access.
    #[inline]
    pub fn const_slice(&self) -> &[T] {
        assert!(self.len <= N);

        let ptr = self.data.as_ptr().cast::<T>();
        let len = self.len;

        // SAFETY: Elements [0..len) are initialized per the struct invariant.
        // MaybeUninit<T> has the same layout as T.
        let result = unsafe { slice::from_raw_parts(ptr, len) };
        assert_eq!(result.as_ptr(), self.data.as_ptr().cast::<T>());

        result
    }

    /// Returns initialized elements as a mutable slice.
    #[inline]
    pub fn slice(&mut self) -> &mut [T] {
        assert!(self.len <= N);

        let ptr = self.data.as_mut_ptr().cast::<T>();
        let len = self.len;

        // SAFETY: Elements [0..len) are initialized per the struct invariant.
        // MaybeUninit<T> has the same layout as T.
        let result = unsafe { slice::from_raw_parts_mut(ptr, len) };
        assert_eq!(result.as_ptr(), self.data.as_mut_ptr().cast::<T>());

        result
    }

    /// Returns the entire backing array including uninitialized elements.
    ///
    /// # Safety Note
    ///
    /// Only elements `[0..len)` are initialized. Calling `assume_init` on
    /// elements `[len..N)` is undefined behavior.
    #[inline]
    pub fn as_uninit_slice(&self) -> &[MaybeUninit<T>] {
        &self.data
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.const_slice().iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.slice().iter_mut()
    }

    /// Returns element by value. Panics if out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> T {
        assert!(
            index < self.len,
            "index out of bounds: {} >= {}",
            index,
            self.len
        );
        assert!(self.len <= N);

        // SAFETY: Bounds checked above, element is initialized.
        unsafe { self.data[index].assume_init() }
    }

    /// Returns element reference, or `None` if out of bounds.
    #[inline]
    pub fn try_get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }

        // SAFETY: index < len, element is initialized.
        let value = unsafe { self.data[index].assume_init_ref() };
        Some(value)
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        assert!(
            index < self.len,
            "index out of bounds: {} >= {}",
            index,
            self.len
        );
        assert!(self.len <= N);

        // SAFETY: Bounds checked above, element is initialized.
        unsafe { self.data[index].assume_init_mut() }
    }

    #[inline]
    pub fn first(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        Some(self.get(0))
    }

    #[inline]
    pub fn last(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        Some(self.get(self.len - 1))
    }

    /// Appends an element. Panics if full.
    ///
    /// Use [`try_push`](Self::try_push) for fallible insertion.
    #[inline]
    pub fn push(&mut self, value: T) {
        let old_len = self.len;
        assert!(
            old_len < N,
            "push on full array: len {} == capacity {}",
            old_len,
            N
        );

        self.data[old_len].write(value);
        self.len = old_len + 1;

        self.assert_invariants();
    }

    /// Appends an element, returning `Err(value)` if full.
    #[inline]
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.is_full() {
            return Err(value);
        }
        self.push(value);
        Ok(())
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let old_len = self.len;
        self.len = old_len - 1;

        // SAFETY: old_len > 0, element at old_len-1 is initialized.
        let value = unsafe { self.data[old_len - 1].assume_init() };

        self.assert_invariants();
        Some(value)
    }

    /// Appends all elements from the slice. Panics if insufficient capacity.
    #[inline]
    pub fn extend_from_slice(&mut self, src: &[T]) {
        let old_len = self.len;
        let src_len = src.len();
        assert!(
            N - old_len >= src_len,
            "extend exceeds capacity: need {} more slots, have {}",
            src_len,
            N - old_len
        );

        for (i, &v) in src.iter().enumerate() {
            self.data[old_len + i].write(v);
        }
        self.len = old_len + src_len;

        self.assert_invariants();
    }

    /// Reduces length to `new_len`. Panics if `new_len > len`.
    ///
    /// Since `T: Copy`, no destructors run. Truncated elements become
    /// uninitialized and their slots can be reused.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        let old_len = self.len;
        assert!(
            new_len <= old_len,
            "truncate length exceeds current len: {} > {}",
            new_len,
            old_len
        );

        self.len = new_len;
        self.assert_invariants()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
        assert!(self.is_empty());
        assert_eq!(self.remaining_capacity(), N);
        self.assert_invariants()
    }

    /// Overwrites element at `index`. Panics if out of bounds.
    ///
    /// Does not read the previous value (just overwrites via `MaybeUninit::write`).
    #[inline]
    pub fn set(&mut self, index: usize, value: T) {
        assert!(
            index < self.len,
            "index out of bounds: {} >= {}",
            index,
            self.len
        );
        self.data[index].write(value);
        self.assert_invariants()
    }

    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        assert!(a < self.len);
        assert!(b < self.len);
        self.data.swap(a, b);
        self.assert_invariants()
    }
}

impl<T: Copy, const N: usize> Default for BoundedArray<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + PartialEq, const N: usize> PartialEq for BoundedArray<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.const_slice() == other.const_slice()
    }
}

impl<T: Copy + Eq, const N: usize> Eq for BoundedArray<T, N> {}

impl<T: Copy + PartialOrd, const N: usize> PartialOrd for BoundedArray<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.const_slice().partial_cmp(other.const_slice())
    }
}

impl<T: Copy + Ord, const N: usize> Ord for BoundedArray<T, N> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.const_slice().cmp(other.const_slice())
    }
}

impl<T: Copy + core::hash::Hash, const N: usize> core::hash::Hash for BoundedArray<T, N> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.const_slice().hash(state);
    }
}

impl<T: Copy + core::fmt::Debug, const N: usize> core::fmt::Debug for BoundedArray<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BoundedArray")
            .field("len", &self.len)
            .field("capacity", &N)
            .field("data", &self.const_slice())
            .finish()
    }
}

impl<T: Copy, const N: usize> core::ops::Index<usize> for BoundedArray<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            index < self.len,
            "index {} out of bounds for length {}",
            index,
            self.len
        );

        // SAFETY: index < len, elements [0..len) are initialized
        unsafe { self.data[index].assume_init_ref() }
    }
}

impl<T: Copy, const N: usize> core::ops::IndexMut<usize> for BoundedArray<T, N> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
    }
}

/// Owning iterator over [`BoundedArray`] elements.
///
/// Created by [`BoundedArray::into_iter`]. Yields elements by value since `T: Copy`.
pub struct BoundedArrayIter<T: Copy, const N: usize> {
    array: BoundedArray<T, N>,
    index: usize,
}

impl<T: Copy, const N: usize> IntoIterator for BoundedArray<T, N> {
    type Item = T;
    type IntoIter = BoundedArrayIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        BoundedArrayIter {
            array: self,
            index: 0,
        }
    }
}

impl<T: Copy, const N: usize> Iterator for BoundedArrayIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.array.len {
            return None;
        }

        let item = self.array.get(self.index);
        self.index += 1;

        assert!(self.index <= self.array.len);

        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remain = self.array.len - self.index;
        (remain, Some(remain))
    }
}

impl<T: Copy, const N: usize> ExactSizeIterator for BoundedArrayIter<T, N> {}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time capacity test
    const _: () = {
        // Verify BoundedArray can be created with various sizes
        let _: BoundedArray<u8, 1> = BoundedArray::new();
        let _: BoundedArray<u64, 100> = BoundedArray::new();
    };

    #[test]
    #[should_panic]
    fn capacity_check_rejects_len_overflow() {
        const TOO_LARGE: usize = (isize::MAX as usize / 2) + 1;
        assert_valid_capacity::<[u8; 2], TOO_LARGE>();
    }

    #[test]
    fn new_is_empty() {
        let arr: BoundedArray<u32, 10> = BoundedArray::new();

        assert_eq!(arr.len(), 0);
        assert_eq!(arr.count(), 0);
        assert!(arr.is_empty());
        assert!(!arr.is_full());
        assert_eq!(arr.capacity(), 10);
        assert_eq!(arr.remaining_capacity(), 10);
    }

    #[test]
    fn push_to_capacity() {
        let mut arr: BoundedArray<u32, 3> = BoundedArray::new();

        arr.push(1);
        arr.push(2);
        arr.push(3);

        assert!(arr.is_full());
        assert_eq!(arr.remaining_capacity(), 0);
    }

    #[test]
    #[should_panic(expected = "push on full array")]
    fn push_overflow() {
        let mut arr: BoundedArray<u32, 2> = BoundedArray::new();
        arr.push(1);
        arr.push(2);
        arr.push(3); // Should panic
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn get_out_of_bounds() {
        let arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.get(0); // Empty array, should panic
    }

    #[test]
    #[should_panic(expected = "extend exceeds capacity")]
    fn extend_overflow() {
        let mut arr: BoundedArray<u32, 3> = BoundedArray::new();
        arr.push(1);
        arr.extend_from_slice(&[2, 3, 4]); // Would need 4 total, capacity is 3
    }

    #[test]
    fn truncate_then_extend_overwrites_tail() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.extend_from_slice(&[1, 2, 3, 4]);

        arr.truncate(2);
        arr.extend_from_slice(&[9, 8]);

        assert_eq!(arr.len(), 4);
        assert_eq!(arr.const_slice(), &[1, 2, 9, 8]);
    }

    #[test]
    #[should_panic(expected = "truncate length exceeds current len")]
    fn truncate_out_of_bounds() {
        let mut arr: BoundedArray<u32, 10> = BoundedArray::new();
        arr.push(1);
        arr.truncate(5); // Current len is 1, should panic
    }

    #[test]
    fn try_push_err_does_not_mutate() {
        let mut arr: BoundedArray<u32, 2> = BoundedArray::new();
        arr.push(1);
        arr.push(2);

        let before = arr.const_slice().to_vec();
        let len_before = arr.len();

        assert_eq!(arr.try_push(3), Err(3));
        assert_eq!(arr.len(), len_before);
        assert_eq!(arr.remaining_capacity(), 0);
        assert_eq!(arr.const_slice(), before.as_slice());
    }

    #[test]
    fn iterator_size_hint() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.extend_from_slice(&[1, 2, 3]);

        let iter = arr.into_iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn equality() {
        let mut a: BoundedArray<u32, 10> = BoundedArray::new();
        let mut b: BoundedArray<u32, 10> = BoundedArray::new();

        a.extend_from_slice(&[1, 2, 3]);
        b.extend_from_slice(&[1, 2, 3]);

        assert_eq!(a, b);

        b.push(4);
        assert_ne!(a, b);
    }

    #[test]
    fn debug_format() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.push(1);
        arr.push(2);

        let debug_str = format!("{:?}", arr);
        assert!(debug_str.contains("len: 2"));
        assert!(debug_str.contains("capacity: 4"));
    }

    #[test]
    fn capacity_min() {
        let mut arr: BoundedArray<u32, 1> = BoundedArray::new();

        assert!(arr.is_empty());
        assert!(!arr.is_full());

        arr.push(42);

        assert!(!arr.is_empty());
        assert!(arr.is_full());
        assert_eq!(arr.get(0), 42);

        assert_eq!(arr.pop(), Some(42));
        assert!(arr.is_empty());
    }

    #[test]
    fn default_trait() {
        let arr: BoundedArray<u32, 10> = BoundedArray::default();

        assert!(arr.is_empty());
        assert_eq!(arr.capacity(), 10);
    }

    #[test]
    fn as_uninit_slice() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.push(1);
        arr.push(2);

        let uninit_slice = arr.as_uninit_slice();

        // Should return the full backing array
        assert_eq!(uninit_slice.len(), 4);

        // First two elements should be initialized
        assert_eq!(unsafe { uninit_slice[0].assume_init() }, 1);
        assert_eq!(unsafe { uninit_slice[1].assume_init() }, 2);
    }

    #[test]
    fn get_mut_direct() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.extend_from_slice(&[10, 20, 30]);

        // Direct get_mut test
        *arr.get_mut(1) = 200;

        assert_eq!(arr.get(1), 200);
        assert_eq!(arr.const_slice(), &[10, 200, 30]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn get_mut_out_of_bounds() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.push(1);
        let _ = arr.get_mut(1); // Should panic
    }

    #[test]
    fn index_mut_trait() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.extend_from_slice(&[1, 2, 3]);

        // Test IndexMut trait
        arr[0] = 100;
        arr[2] = 300;

        assert_eq!(arr[0], 100);
        assert_eq!(arr[1], 2);
        assert_eq!(arr[2], 300);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn index_mut_out_of_bounds() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.push(1);
        arr[5] = 10; // Should panic
    }

    #[test]
    fn from_value_full_capacity() {
        let arr: BoundedArray<u32, 5> = BoundedArray::from_value(7, 5);

        assert!(arr.is_full());
        assert_eq!(arr.len(), 5);
        assert_eq!(arr.const_slice(), &[7, 7, 7, 7, 7]);
    }

    #[test]
    #[should_panic]
    fn from_value_overflow() {
        let _arr: BoundedArray<u32, 3> = BoundedArray::from_value(1, 5);
    }

    #[test]
    fn empty_slice_operations() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();

        // Empty const_slice
        assert_eq!(arr.const_slice(), &[]);
        assert!(arr.const_slice().is_empty());

        // Empty mutable slice
        assert_eq!(arr.slice(), &mut []);
        assert!(arr.slice().is_empty());
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn set_out_of_bounds() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.push(1);
        arr.set(5, 10); // Should panic
    }

    #[test]
    #[should_panic]
    fn swap_out_of_bounds_first() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.push(1);
        arr.swap(5, 0); // First index out of bounds
    }

    #[test]
    #[should_panic]
    fn swap_out_of_bounds_second() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.push(1);
        arr.swap(0, 5); // Second index out of bounds
    }

    #[test]
    fn swap_same_index() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.extend_from_slice(&[1, 2, 3]);

        // Swapping same index should be a no-op
        arr.swap(1, 1);

        assert_eq!(arr.const_slice(), &[1, 2, 3]);
    }

    #[test]
    fn iterator_partial_consumption() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.extend_from_slice(&[1, 2, 3, 4]);

        let mut iter = arr.into_iter();

        // Consume some elements
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));

        // Check size_hint after partial consumption
        assert_eq!(iter.size_hint(), (2, Some(2)));
        assert_eq!(iter.len(), 2); // ExactSizeIterator

        // Consume rest
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), None);

        // After exhaustion
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn iterator_empty_array() {
        let arr: BoundedArray<u32, 4> = BoundedArray::new();

        let mut iter = arr.into_iter();

        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn extend_from_empty_slice() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.push(1);

        arr.extend_from_slice(&[]);

        assert_eq!(arr.len(), 1);
        assert_eq!(arr.const_slice(), &[1]);
    }

    #[test]
    fn slot_reuse() {
        // Verify slots can be reused after pop
        let mut arr: BoundedArray<u32, 2> = BoundedArray::new();

        arr.push(1);
        arr.push(2);
        assert!(arr.is_full());

        arr.pop();
        assert!(!arr.is_full());

        arr.push(3);
        assert!(arr.is_full());
        assert_eq!(arr.const_slice(), &[1, 3]);
    }

    #[test]
    fn equality_same_values_different_order() {
        let mut a: BoundedArray<u32, 10> = BoundedArray::new();
        let mut b: BoundedArray<u32, 10> = BoundedArray::new();

        a.extend_from_slice(&[1, 2, 3]);
        b.extend_from_slice(&[3, 2, 1]);

        assert_ne!(a, b);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_out_of_bounds() {
        let arr: BoundedArray<u32, 4> = BoundedArray::new();
        let _ = arr[0]; // Empty array, should panic
    }

    #[test]
    fn try_get_at_boundary() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.extend_from_slice(&[10, 20, 30]);

        // Last valid index
        assert_eq!(arr.try_get(2), Some(&30));

        // First invalid index
        assert_eq!(arr.try_get(3), None);

        // Way out of bounds
        assert_eq!(arr.try_get(100), None);
    }

    #[test]
    fn clone_independence() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.extend_from_slice(&[1, 2, 3]);

        // Clone (uses derived Clone)
        let mut arr2 = arr;

        // Modify original
        arr.set(0, 100);
        arr.push(4);

        // Clone should be unaffected (Copy semantics)
        assert_eq!(arr2.const_slice(), &[1, 2, 3]);
        assert_eq!(arr.const_slice(), &[100, 2, 3, 4]);

        // Modify clone
        arr2.pop();
        assert_eq!(arr2.const_slice(), &[1, 2]);

        // Original unaffected
        assert_eq!(arr.const_slice(), &[100, 2, 3, 4]);
    }

    #[test]
    fn large_capacity() {
        // Test with a larger capacity
        let mut arr: BoundedArray<u8, 1000> = BoundedArray::new();

        for i in 0..1000u16 {
            arr.push(i as u8);
        }

        assert!(arr.is_full());
        assert_eq!(arr.len(), 1000);
        assert_eq!(arr.get(0), 0);
        assert_eq!(arr.get(999), 231); // 999 % 256 = 231
    }

    #[test]
    fn iter_and_iter_mut() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.extend_from_slice(&[1, 2, 3]);

        // iter
        let collected: Vec<&u32> = arr.iter().collect();
        assert_eq!(collected, vec![&1, &2, &3]);

        // iter_mut
        for v in arr.iter_mut() {
            *v *= 10;
        }
        assert_eq!(arr.const_slice(), &[10, 20, 30]);
    }

    #[test]
    fn ordering_and_hashing() {
        use core::cmp::Ordering;
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut a: BoundedArray<u32, 4> = BoundedArray::new();
        a.extend_from_slice(&[1, 2, 3]);

        let mut b: BoundedArray<u32, 4> = BoundedArray::new();
        b.extend_from_slice(&[1, 2, 3]);

        // Ordering
        assert_eq!(a.cmp(&b), Ordering::Equal);
        b.push(4);
        assert_eq!(a.cmp(&b), Ordering::Less); // [1,2,3] < [1,2,3,4]

        // Hashing
        let mut hasher_a = DefaultHasher::new();
        a.hash(&mut hasher_a);

        let mut c: BoundedArray<u32, 4> = BoundedArray::new();
        c.extend_from_slice(&[1, 2, 3]);
        let mut hasher_c = DefaultHasher::new();
        c.hash(&mut hasher_c);

        assert_eq!(hasher_a.finish(), hasher_c.finish());
    }

    #[test]
    fn truncate_noop() {
        let mut arr: BoundedArray<u32, 4> = BoundedArray::new();
        arr.push(1);

        // Truncate to same length -> no-op
        arr.truncate(1);
        assert_eq!(arr.len(), 1);
        assert_eq!(arr.const_slice(), &[1]);

        // Truncate empty -> no-op
        arr.clear();
        arr.truncate(0);
        assert!(arr.is_empty());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    const PROPTEST_CASES: u32 = 16;
    const TEST_CAPACITY: usize = 64;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(
            crate::test_utils::proptest_cases(PROPTEST_CASES)
        ))]

        /// Property: push increases len by 1 until capacity
        #[test]
        fn push_increments_len(values in prop::collection::vec(any::<u32>(), 0..TEST_CAPACITY)) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();

            for (i, &v) in values.iter().enumerate() {
                prop_assert_eq!(arr.len(), i);
                arr.push(v);
                prop_assert_eq!(arr.len(), i + 1);
            }
        }

        /// Property: pop decrements len by 1 and returns LIFO order
        #[test]
        fn pop_decrements_len_lifo(values in prop::collection::vec(any::<u32>(), 1..TEST_CAPACITY)) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();

            for &v in &values {
                arr.push(v);
            }

            for i in (0..values.len()).rev() {
                prop_assert_eq!(arr.len(), i + 1);
                let popped = arr.pop();
                prop_assert_eq!(popped, Some(values[i]));
                prop_assert_eq!(arr.len(), i);
            }

            prop_assert!(arr.is_empty());
            prop_assert_eq!(arr.pop(), None);
        }

        /// Property: const_slice returns exactly the pushed values in order
        #[test]
        fn const_slice_matches_pushed_values(values in prop::collection::vec(any::<u32>(), 0..TEST_CAPACITY)) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();

            for &v in &values {
                arr.push(v);
            }

            prop_assert_eq!(arr.const_slice(), values.as_slice());
        }

        /// Property: get(i) returns the i-th pushed value
        #[test]
        fn get_returns_correct_value(values in prop::collection::vec(any::<u32>(), 1..TEST_CAPACITY)) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();

            for &v in &values {
                arr.push(v);
            }

            for (i, &expected) in values.iter().enumerate() {
                prop_assert_eq!(arr.get(i), expected);
                prop_assert_eq!(arr[i], expected);
                prop_assert_eq!(arr.try_get(i), Some(&expected));
            }
        }

        /// Property: remaining_capacity + len == capacity
        #[test]
        fn remaining_capacity_invariant(values in prop::collection::vec(any::<u32>(), 0..TEST_CAPACITY)) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();

            for &v in &values {
                arr.push(v);
                prop_assert_eq!(arr.len() + arr.remaining_capacity(), TEST_CAPACITY);
            }
        }

        /// Property: extend_from_slice equivalent to multiple pushes
        #[test]
        fn extend_equivalent_to_push(values in prop::collection::vec(any::<u32>(), 0..TEST_CAPACITY)) {
            let mut arr1: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();
            let mut arr2: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();

            // Method 1: push each value
            for &v in &values {
                arr1.push(v);
            }

            // Method 2: extend_from_slice
            arr2.extend_from_slice(&values);

            prop_assert_eq!(arr1, arr2);
        }

        /// Property: truncate preserves prefix
        #[test]
        fn truncate_preserves_prefix(
            values in prop::collection::vec(any::<u32>(), 1..TEST_CAPACITY),
            truncate_to in 0..TEST_CAPACITY
        ) {
            let truncate_to = truncate_to.min(values.len());

            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();
            arr.extend_from_slice(&values);

            arr.truncate(truncate_to);

            prop_assert_eq!(arr.len(), truncate_to);
            prop_assert_eq!(arr.const_slice(), &values[..truncate_to]);
        }

        /// Property: set modifies only the target index
        #[test]
        fn set_modifies_only_target(
            values in prop::collection::vec(any::<u32>(), 1..TEST_CAPACITY),
            new_value: u32
        ) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();
            arr.extend_from_slice(&values);

            let target_idx = values.len() / 2;
            arr.set(target_idx, new_value);

            for (i, &expected) in values.iter().enumerate() {
                if i == target_idx {
                    prop_assert_eq!(arr.get(i), new_value);
                } else {
                    prop_assert_eq!(arr.get(i), expected);
                }
            }
        }

        /// Property: swap exchanges exactly two elements
        #[test]
        fn swap_exchanges_elements(
            values in prop::collection::vec(any::<u32>(), 2..TEST_CAPACITY)
        ) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();
            arr.extend_from_slice(&values);

            let a = 0;
            let b = values.len() - 1;
            let val_a = arr.get(a);
            let val_b = arr.get(b);

            arr.swap(a, b);

            prop_assert_eq!(arr.get(a), val_b);
            prop_assert_eq!(arr.get(b), val_a);

            // Other elements unchanged
            for (i, &expected) in values.iter().enumerate().take(values.len() - 1).skip(1) {
                prop_assert_eq!(arr.get(i), expected);
            }
        }

        /// Property: from_value creates array with all same values
        #[test]
        fn from_value_all_same(value: u32, count in 0..TEST_CAPACITY) {
            let arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::from_value(value, count);

            prop_assert_eq!(arr.len(), count);
            for i in 0..count {
                prop_assert_eq!(arr.get(i), value);
            }
        }

        /// Property: iterator yields same values as const_slice
        #[test]
        fn iterator_matches_slice(values in prop::collection::vec(any::<u32>(), 0..TEST_CAPACITY)) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();
            arr.extend_from_slice(&values);

            let slice_values: Vec<u32> = arr.const_slice().to_vec();
            let iter_values: Vec<u32> = arr.into_iter().collect();

            prop_assert_eq!(slice_values, iter_values);
        }

        /// Property: clear resets to empty state
        #[test]
        fn clear_resets_state(values in prop::collection::vec(any::<u32>(), 0..TEST_CAPACITY)) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();
            arr.extend_from_slice(&values);

            arr.clear();

            prop_assert!(arr.is_empty());
            prop_assert_eq!(arr.len(), 0);
            prop_assert_eq!(arr.remaining_capacity(), TEST_CAPACITY);
        }

        /// Property: first/last return correct values
        #[test]
        fn first_last_correct(values in prop::collection::vec(any::<u32>(), 1..TEST_CAPACITY)) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();
            arr.extend_from_slice(&values);

            prop_assert_eq!(arr.first(), Some(values[0]));
            prop_assert_eq!(arr.last(), Some(values[values.len() - 1]));
        }

        /// Property: try_push succeeds until full, then fails
        #[test]
        fn try_push_behavior(values in prop::collection::vec(any::<u32>(), 0..TEST_CAPACITY + 10)) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();

            for (i, &v) in values.iter().enumerate() {
                let result = arr.try_push(v);
                if i < TEST_CAPACITY {
                    prop_assert!(result.is_ok());
                } else {
                    prop_assert_eq!(result, Err(v));
                }
            }
        }

        /// Property: equality is reflexive
        #[test]
        fn equality_reflexive(values in prop::collection::vec(any::<u32>(), 0..TEST_CAPACITY)) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();
            arr.extend_from_slice(&values);

            prop_assert_eq!(arr, arr);
        }

        /// Property: copy produces equal but independent arrays
        #[test]
        fn copy_produces_equal_independent(values in prop::collection::vec(any::<u32>(), 0..TEST_CAPACITY)) {
            let mut arr1: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();
            arr1.extend_from_slice(&values);

            let arr2 = arr1;

            prop_assert_eq!(arr1, arr2);
            prop_assert_eq!(arr1.const_slice(), arr2.const_slice());
        }

        /// Property: mutable slice modifications are visible via get
        #[test]
        fn mutable_slice_modifications_visible(
            values in prop::collection::vec(1u32..1000, 1..TEST_CAPACITY)
        ) {
            let mut arr: BoundedArray<u32, TEST_CAPACITY> = BoundedArray::new();
            arr.extend_from_slice(&values);

            // Double each value via mutable slice
            for v in arr.slice().iter_mut() {
                *v *= 2;
            }

            // Verify via get
            for (i, &original) in values.iter().enumerate() {
                prop_assert_eq!(arr.get(i), original * 2);
            }
        }
    }
}
