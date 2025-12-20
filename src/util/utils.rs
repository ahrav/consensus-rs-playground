//! Low-level byte manipulation and alignment utilities.

use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use core::slice;
use std::alloc::{Layout, alloc_zeroed, dealloc};

/// Reinterprets a reference as a byte slice, including padding.
///
/// # Panics
///
/// Panics if `T` is a ZST or exceeds `isize::MAX` bytes.
#[inline]
pub fn as_bytes<T: Sized>(v: &T) -> &[u8] {
    const { assert!(size_of::<T>() > 0) };
    assert!(size_of::<T>() <= isize::MAX as usize);

    let ptr = v as *const T;
    assert!(ptr.is_aligned());

    let byte_ptr = ptr.cast::<u8>();
    let len = size_of::<T>();
    let result = unsafe { slice::from_raw_parts(byte_ptr, len) };

    assert_eq!(result.len(), size_of::<T>());
    assert_eq!(result.as_ptr() as usize, ptr as usize);

    result
}

/// Mutable version of [`as_bytes`].
///
/// # Panics
///
/// Panics if `T` is a ZST or exceeds `isize::MAX` bytes.
#[inline]
pub fn as_bytes_mut<T: Sized>(v: &mut T) -> &mut [u8] {
    const { assert!(size_of::<T>() > 0) };
    assert!(size_of::<T>() <= isize::MAX as usize);

    let ptr = v as *mut T;
    assert!(ptr.is_aligned());

    let byte_ptr = ptr.cast::<u8>();
    let len = size_of::<T>();
    let result = unsafe { slice::from_raw_parts_mut(byte_ptr, len) };

    assert_eq!(result.len(), size_of::<T>());
    assert_eq!(result.as_ptr() as usize, ptr as usize);

    result
}

/// Compares two values byte-for-byte, including padding.
///
/// This checks for exact memory equality, unlike `PartialEq`.
///
/// # Panics
///
/// Panics if `a` and `b` overlap partially.
#[inline]
pub fn equal_bytes<T: Sized>(a: &T, b: &T) -> bool {
    let a_addr = a as *const T as usize;
    let b_addr = b as *const T as usize;
    let size = size_of::<T>();

    let disjoint = a_addr.wrapping_add(size) <= b_addr || b_addr.wrapping_add(size) <= a_addr;
    let identical = a_addr == b_addr;
    assert!(disjoint || identical);

    let result = as_bytes(a) == as_bytes(b);
    if identical {
        assert!(result);
    }

    result
}

/// Rounds `value` up to the next multiple of `alignment`.
///
/// # Panics
///
/// Panics if `alignment` is not a power of two or if the result overflows.
#[inline]
pub fn align_up(value: usize, alignment: usize) -> usize {
    assert!(alignment > 0);
    assert!(alignment.is_power_of_two());
    assert!(alignment <= isize::MAX as usize);

    let max_alignable = usize::MAX & !(alignment - 1);
    assert!(value <= max_alignable);

    let mask = alignment - 1;
    let result = (value + mask) & !mask;

    assert!(result >= value);
    assert!(result.is_multiple_of(alignment));
    assert!(result - value < alignment);

    if value.is_multiple_of(alignment) {
        assert_eq!(result, value);
    }

    result
}

/// Heap-allocated, zero-initialized value with guaranteed alignment.
pub struct AlignedBox<T> {
    ptr: NonNull<T>,
    layout: Layout,
}

const _: () = {
    assert!(size_of::<AlignedBox<u8>>() <= 32);
    assert!(size_of::<NonNull<u8>>() == size_of::<*const u8>());
};

impl<T> AlignedBox<T> {
    /// Allocates a zero-initialized `T` on the heap.
    ///
    /// # Panics
    ///
    /// Panics if `T` is a ZST, too large, or if allocation fails.
    pub fn new_zeroed() -> Self {
        const { assert!(size_of::<T>() > 0) };

        assert!(size_of::<T>() <= isize::MAX as usize);
        assert!(align_of::<T>() <= isize::MAX as usize);

        let layout = Layout::new::<T>();
        assert!(layout.size() > 0);
        assert!(layout.align().is_power_of_two());

        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw.cast()).expect("allocation failed");

        assert!((ptr.as_ptr() as usize).is_multiple_of(align_of::<T>()));

        Self { ptr, layout }
    }

    /// Returns a raw pointer to the allocated value.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        let ptr = self.ptr.as_ptr();
        assert!((ptr as usize).is_multiple_of(align_of::<T>()));
        ptr
    }

    /// Returns a mutable raw pointer to the allocated value.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        let ptr = self.ptr.as_ptr();
        assert!((ptr as usize).is_multiple_of(align_of::<T>()));
        ptr
    }

    /// Returns the memory layout used for this allocation.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout
    }
}

impl<T> core::ops::Deref for AlignedBox<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        assert!((self.ptr.as_ptr() as usize).is_multiple_of(align_of::<T>()));
        unsafe { &*self.ptr.as_ptr() }
    }
}

impl<T> core::ops::DerefMut for AlignedBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        assert!((self.ptr.as_ptr() as usize).is_multiple_of(align_of::<T>()));
        unsafe { &mut *self.ptr.as_ptr() }
    }
}

impl<T> core::ops::Drop for AlignedBox<T> {
    fn drop(&mut self) {
        let ptr = self.ptr.as_ptr();
        assert!((ptr as usize).is_multiple_of(align_of::<T>()));
        assert!(self.layout().size() == size_of::<T>());
        assert!(self.layout().align() == align_of::<T>());

        unsafe {
            core::ptr::drop_in_place(ptr);
            dealloc(ptr.cast::<u8>(), self.layout);
        }
    }
}

// SAFETY: AlignedBox owns its data exclusively and behaves like Box<T>.
// Send/Sync bounds follow the same rules as Box: if T can be sent/shared
// across threads, so can AlignedBox<T>.
unsafe impl<T: Send> Send for AlignedBox<T> {}
unsafe impl<T: Sync> Sync for AlignedBox<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
    }

    #[test]
    fn test_align_up_pow2() {
        for power in 0..16 {
            let alignment = 1usize << power;
            // Already aligned values stay unchanged
            assert_eq!(align_up(alignment, alignment), alignment);
            // One less gets rounded up
            if alignment > 1 {
                assert_eq!(align_up(alignment - 1, alignment), alignment);
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_align_up_panic_non_pow2() {
        align_up(10, 3);
    }

    #[test]
    #[should_panic]
    fn test_align_up_panic_zero() {
        align_up(10, 0);
    }

    #[test]
    #[should_panic]
    fn test_align_up_panic_overflow() {
        align_up(usize::MAX, 2);
    }

    #[test]
    #[should_panic]
    fn test_align_up_panic_large_align() {
        let alignment = 1usize << (usize::BITS - 1);
        align_up(0, alignment);
    }

    #[test]
    fn test_aligned_box() {
        let boxed: AlignedBox<u64> = AlignedBox::new_zeroed();
        assert_eq!(*boxed, 0u64);
        assert!((boxed.as_ptr() as usize).is_multiple_of(align_of::<u64>()));
    }

    #[test]
    fn test_aligned_box_mut() {
        let mut boxed: AlignedBox<u64> = AlignedBox::new_zeroed();
        *boxed = 42;
        assert_eq!(*boxed, 42);
    }

    #[test]
    fn test_as_bytes() {
        let value: u32 = 0xDEADBEEF;
        let bytes = as_bytes(&value);
        assert_eq!(bytes.len(), 4);

        // Verify we can read the bytes back
        let reconstructed = u32::from_ne_bytes(bytes.try_into().unwrap());
        assert_eq!(reconstructed, value);
    }

    #[test]
    fn test_equal_bytes() {
        let a: u64 = 12345;
        let b: u64 = 12345;
        let c: u64 = 54321;

        assert!(equal_bytes(&a, &b));
        assert!(!equal_bytes(&a, &c));
        assert!(equal_bytes(&a, &a)); // Same address
    }

    #[repr(C)]
    struct TestStruct {
        a: u32,
        b: u32,
    }

    #[test]
    fn test_aligned_box_struct() {
        let mut boxed: AlignedBox<TestStruct> = AlignedBox::new_zeroed();
        assert_eq!(boxed.a, 0);
        assert_eq!(boxed.b, 0);

        boxed.a = 1;
        boxed.b = 2;

        assert_eq!(boxed.a, 1);
        assert_eq!(boxed.b, 2);
    }

    #[test]
    #[should_panic]
    fn test_align_up_panic_near_max() {
        // usize::MAX is ...1111
        // align 4 mask is ...1100
        // max_alignable is ...1100
        // passing ...1101 should panic
        align_up(usize::MAX - 2, 4);
    }

    #[test]
    fn test_align_up_max() {
        let alignment = 8usize;
        let max_alignable = usize::MAX & !(alignment - 1);

        assert_eq!(align_up(max_alignable, alignment), max_alignable);
        assert_eq!(
            align_up(max_alignable - (alignment - 1), alignment),
            max_alignable
        );
    }

    #[test]
    fn test_equal_bytes_padding() {
        #[repr(C, align(8))]
        #[derive(Clone, Copy)]
        struct Padded {
            a: u8,
            // 7 bytes padding implied here to align `b` to 8
            b: u64,
        }

        // Ensure we actually have padding
        assert!(size_of::<Padded>() > size_of::<u8>() + size_of::<u64>());

        let mut x = Padded { a: 1, b: 2 };
        let mut y = Padded { a: 1, b: 2 };

        // We can't easily safely write to padding in safe Rust,
        // but let's try to dirty the stack or use unsafe to prove the point.
        unsafe {
            // Write non-zero to x's padding
            let ptr = &mut x as *mut Padded as *mut u8;
            ptr.add(1).write_bytes(0xAA, 7);

            // Write zero to y's padding
            let ptr = &mut y as *mut Padded as *mut u8;
            ptr.add(1).write_bytes(0x00, 7);
        }

        // Semantically equal
        assert_eq!(x.a, y.a);
        assert_eq!(x.b, y.b);

        // Bytes are not equal
        assert!(!equal_bytes(&x, &y));
    }

    #[test]
    fn test_aligned_box_drop() {
        use std::sync::atomic::{AtomicU32, Ordering};

        static DROP_COUNT: AtomicU32 = AtomicU32::new(0);

        struct Dropper {
            _padding: u8,
        }
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        {
            let _boxed: AlignedBox<Dropper> = AlignedBox::new_zeroed();
        }

        // Expect 1 because AlignedBox now calls drop_in_place
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_as_bytes_mut() {
        let mut value: u32 = 0;
        let bytes = as_bytes_mut(&mut value);

        // Write bytes in native endian
        bytes.copy_from_slice(&0xDEADBEEFu32.to_ne_bytes());

        assert_eq!(value, 0xDEADBEEF);
    }

    #[test]
    fn test_as_bytes_aligned() {
        #[repr(C, align(64))]
        struct CacheAligned {
            value: u64,
        }

        let val = CacheAligned { value: 42 };
        let bytes = as_bytes(&val);

        // Size includes padding to alignment
        assert_eq!(bytes.len(), size_of::<CacheAligned>());
        assert!((bytes.as_ptr() as usize).is_multiple_of(64));

        // First 8 bytes contain the value
        let extracted = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
        assert_eq!(extracted, 42);
    }

    #[test]
    fn test_aligned_box_aligned() {
        #[repr(C, align(128))]
        struct HighlyAligned {
            data: [u8; 64],
        }

        let boxed: AlignedBox<HighlyAligned> = AlignedBox::new_zeroed();
        assert!((boxed.as_ptr() as usize).is_multiple_of(128));
        assert_eq!(boxed.data, [0u8; 64]);
    }

    #[test]
    fn test_equal_bytes_distinct() {
        let a: u64 = 0xDEADBEEFCAFEBABE;
        let b: u64 = 0xDEADBEEFCAFEBABE;

        // Verify they're at different addresses
        assert_ne!(&a as *const u64, &b as *const u64);

        // But byte-wise equal
        assert!(equal_bytes(&a, &b));
    }

    #[test]
    fn test_align_up_boundaries() {
        // Test alignment of 1 (no-op)
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(1, 1), 1);
        assert_eq!(align_up(usize::MAX, 1), usize::MAX);

        // Test large alignment
        let large_align = 1usize << 20; // 1 MiB
        assert_eq!(align_up(0, large_align), 0);
        assert_eq!(align_up(1, large_align), large_align);
        assert_eq!(align_up(large_align - 1, large_align), large_align);
        assert_eq!(align_up(large_align, large_align), large_align);
        assert_eq!(align_up(large_align + 1, large_align), 2 * large_align);
    }

    #[test]
    fn test_aligned_box_layout() {
        let boxed: AlignedBox<u128> = AlignedBox::new_zeroed();
        let layout = boxed.layout();

        assert_eq!(layout.size(), size_of::<u128>());
        assert_eq!(layout.align(), align_of::<u128>());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_align_up_idempotent(value in 0usize..1_000_000, power in 0u32..20u32) {
            let alignment = 1usize << power;
            let aligned = align_up(value, alignment);
            let double_aligned = align_up(aligned, alignment);
            prop_assert_eq!(aligned, double_aligned);
        }

        #[test]
        fn prop_align_up_always_aligned(value in 0usize..1_000_000, power in 0u32..20u32) {
            let alignment = 1usize << power;
            let result = align_up(value, alignment);
            prop_assert!(result >= value);
            prop_assert_eq!(result % alignment, 0);
        }

        #[test]
        fn prop_align_up_minimal(value in 0usize..1_000_000, power in 0u32..20u32) {
            let alignment = 1usize << power;
            let result = align_up(value, alignment);
            // Result is the smallest aligned value >= input
            prop_assert!(result - value < alignment);
        }

        #[test]
        fn prop_as_bytes_roundtrip_u64(value: u64) {
            let bytes = as_bytes(&value);
            let restored = u64::from_ne_bytes(bytes.try_into().unwrap());
            prop_assert_eq!(value, restored);
        }

        #[test]
        fn prop_as_bytes_roundtrip_u128(value: u128) {
            let bytes = as_bytes(&value);
            let restored = u128::from_ne_bytes(bytes.try_into().unwrap());
            prop_assert_eq!(value, restored);
        }

        #[test]
        fn prop_as_bytes_mut_roundtrip(value: u64) {
            let mut v = value;
            let bytes = as_bytes_mut(&mut v);
            let restored = u64::from_ne_bytes(bytes.try_into().unwrap());
            prop_assert_eq!(value, restored);
        }

        #[test]
        fn prop_equal_bytes_reflexive(value: u64) {
            prop_assert!(equal_bytes(&value, &value));
        }

        #[test]
        fn prop_equal_bytes_symmetric(a: u64, b: u64) {
            prop_assert_eq!(equal_bytes(&a, &b), equal_bytes(&b, &a));
        }

        #[test]
        fn prop_equal_bytes_consistent_with_eq(a: u64, b: u64) {
            // For types without padding, equal_bytes should match ==
            prop_assert_eq!(equal_bytes(&a, &b), a == b);
        }

        #[test]
        fn prop_aligned_box_zeroed(value in any::<[u8; 32]>()) {
            // Ignore input, just verify zeroing works for various runs
            let _ = value;
            let boxed: AlignedBox<[u64; 4]> = AlignedBox::new_zeroed();
            prop_assert_eq!(*boxed, [0u64; 4]);
        }
    }
}
