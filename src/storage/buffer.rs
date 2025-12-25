//! Memory allocation primitives for direct I/O.

use core::ptr::NonNull;
use std::alloc;

use super::constants::SECTOR_SIZE_MAX;

/// Heap-allocated buffer with guaranteed alignment for direct I/O.
///
/// Direct I/O (O_DIRECT) requires buffers aligned to the storage device's
/// sector size. Standard allocators don't guarantee this, so `AlignedBuf`
/// uses [`std::alloc`] directly to enforce alignment.
///
/// # Invariants
///
/// - `len > 0` (empty buffers disallowed)
/// - `align` is a power of two and ≤ [`SECTOR_SIZE_MAX`]
/// - `ptr` is aligned to `align`
///
/// # Example
///
/// ```ignore
/// let buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
/// assert_eq!(buf.as_ptr() as usize % SECTOR_SIZE_DEFAULT, 0);
/// ```
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
    align: usize,
}

impl AlignedBuf {
    /// Allocates a zero-initialized buffer with the specified alignment.
    ///
    /// # Panics
    ///
    /// - `len == 0`
    /// - `align == 0` or not a power of two
    /// - `align > SECTOR_SIZE_MAX`
    /// - `len > isize::MAX` (Rust allocation limit)
    /// - Allocation failure
    pub fn new_zeroed(len: usize, align: usize) -> Self {
        assert!(len > 0);
        assert!(align > 0);
        assert!(align.is_power_of_two());
        assert!(align <= SECTOR_SIZE_MAX);
        assert!(len <= isize::MAX as usize);

        let layout = alloc::Layout::from_size_align(len, align).expect("bad layout");

        // SAFETY: Layout is valid (non-zero size, power-of-two alignment).
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).expect("alloc failed");

        let result = Self { ptr, len, align };
        result.assert_invariants();

        result
    }

    /// Validates structural invariants. Called on all public operations.
    #[inline]
    fn assert_invariants(&self) {
        assert!(self.len > 0);
        assert!(self.align > 0);
        assert!(self.align.is_power_of_two());
        assert!((self.ptr.as_ptr() as usize).is_multiple_of(self.align));
    }

    /// Returns the buffer as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.assert_invariants();
        // SAFETY: We own the allocation exclusively, it's valid for `len` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the buffer as an immutable byte slice.
    pub fn as_slice(&self) -> &[u8] {
        self.assert_invariants();
        // SAFETY: Valid allocation of `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a raw pointer to the buffer start.
    pub fn as_ptr(&self) -> *const u8 {
        self.assert_invariants();
        self.ptr.as_ptr()
    }

    /// Returns a mutable raw pointer to the buffer start.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.assert_invariants();
        self.ptr.as_ptr()
    }

    /// Returns the buffer length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always returns `false` (empty buffers are disallowed by construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the alignment in bytes.
    pub fn align(&self) -> usize {
        self.align
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        self.assert_invariants();

        let layout =
            alloc::Layout::from_size_align(self.len, self.align).expect("bad layout in drop");
        assert!(layout.size() == self.len);
        assert!(layout.align() == self.align);

        // SAFETY: `ptr` was allocated with this exact layout in `new_zeroed`.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) }
    }
}

// SAFETY: The buffer owns its allocation exclusively. No shared mutable state.
unsafe impl Send for AlignedBuf {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::constants::{SECTOR_SIZE_DEFAULT, SECTOR_SIZE_MIN};

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn buf_panic_len_0() {
        let _ = AlignedBuf::new_zeroed(0, SECTOR_SIZE_DEFAULT);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn buf_panic_align_0() {
        let _ = AlignedBuf::new_zeroed(4096, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn buf_panic_align_not_pow2() {
        let _ = AlignedBuf::new_zeroed(4096, 100);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn buf_panic_align_max() {
        let _ = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_MAX * 2);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn buf_panic_len_over_isize_max() {
        let _ = AlignedBuf::new_zeroed((isize::MAX as usize) + 1, SECTOR_SIZE_DEFAULT);
    }

    #[test]
    fn buf_slice_lengths() {
        let mut buf = AlignedBuf::new_zeroed(1234, SECTOR_SIZE_DEFAULT);
        assert_eq!(buf.as_slice().len(), 1234);
        assert_eq!(buf.as_mut_slice().len(), 1234);
    }

    #[test]
    fn buf_drop() {
        for _ in 0..100 {
            let mut buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
            buf.as_mut_slice().fill(0xAB);
        }
    }

    #[test]
    fn buf_send() {
        let mut buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
        buf.as_mut_slice()[0] = 42;
        let handle = std::thread::spawn(move || {
            assert_eq!(buf.as_slice()[0], 42);
            buf.len()
        });
        assert_eq!(handle.join().unwrap(), 4096);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::storage::constants::SECTOR_SIZE_MIN;
    use proptest::prelude::*;

    fn align_strategy() -> impl Strategy<Value = usize> {
        prop::sample::select(vec![512usize, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    }

    fn len_strategy() -> impl Strategy<Value = usize> {
        1usize..=(1024 * 1024)
    }

    proptest! {
        #[test]
        fn prop_aligned_buf_alignment(
            len in len_strategy(),
            align in align_strategy(),
        ) {
            let buf = AlignedBuf::new_zeroed(len, align);
            prop_assert_eq!(buf.as_ptr() as usize % align, 0);
            prop_assert_eq!(buf.len(), len);
            prop_assert_eq!(buf.align(), align);
        }

        #[test]
        fn prop_aligned_buf_zero_init(len in 1usize..10000) {
            let buf = AlignedBuf::new_zeroed(len, SECTOR_SIZE_MIN);
            prop_assert!(buf.as_slice().iter().all(|&b| b == 0));
        }

        #[test]
        fn prop_aligned_buf_is_empty_always_false(
            len in len_strategy(),
            align in align_strategy(),
        ) {
            let buf = AlignedBuf::new_zeroed(len, align);
            prop_assert!(!buf.is_empty());
        }

        #[test]
        fn prop_aligned_buf_ptr_consistency(
            len in len_strategy(),
            align in align_strategy(),
        ) {
            let mut buf = AlignedBuf::new_zeroed(len, align);
            let ptr = buf.as_ptr();
            prop_assert_eq!(ptr, buf.as_mut_ptr() as *const u8);
            prop_assert_eq!(ptr, buf.as_slice().as_ptr());
        }

        #[test]
        fn prop_aligned_buf_mutability(
            len in 1usize..10000,
            pattern in any::<u8>(),
        ) {
            let mut buf = AlignedBuf::new_zeroed(len, SECTOR_SIZE_MIN);
            buf.as_mut_slice().fill(pattern);
            prop_assert!(buf.as_slice().iter().all(|&b| b == pattern));
        }
    }
}
