//! Low-level byte manipulation and alignment utilities.

use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use core::slice;
use std::alloc::{Layout, alloc_zeroed, dealloc};

/// # Safety
///
/// Implementers guarantee the type has no padding bytes and all bytes are
/// initialized for any valid value.
pub unsafe trait Pod: Copy {}

/// # Safety
///
/// Implementers guarantee the all-zero byte pattern is a valid value.
pub unsafe trait Zeroable {}

unsafe impl Pod for bool {}
unsafe impl Pod for u8 {}
unsafe impl Pod for u16 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for u64 {}
unsafe impl Pod for u128 {}
unsafe impl Pod for usize {}
unsafe impl Pod for i8 {}
unsafe impl Pod for i16 {}
unsafe impl Pod for i32 {}
unsafe impl Pod for i64 {}
unsafe impl Pod for i128 {}
unsafe impl Pod for isize {}
unsafe impl Pod for f32 {}
unsafe impl Pod for f64 {}
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

unsafe impl Zeroable for bool {}
unsafe impl Zeroable for u8 {}
unsafe impl Zeroable for u16 {}
unsafe impl Zeroable for u32 {}
unsafe impl Zeroable for u64 {}
unsafe impl Zeroable for u128 {}
unsafe impl Zeroable for usize {}
unsafe impl Zeroable for i8 {}
unsafe impl Zeroable for i16 {}
unsafe impl Zeroable for i32 {}
unsafe impl Zeroable for i64 {}
unsafe impl Zeroable for i128 {}
unsafe impl Zeroable for isize {}
unsafe impl Zeroable for f32 {}
unsafe impl Zeroable for f64 {}
unsafe impl<T: Zeroable, const N: usize> Zeroable for [T; N] {}

/// Reinterprets a `Pod` reference as a byte slice.
///
/// # Safety
///
/// `T` must be `Pod`, and callers must not mutate `v` through other means while
/// the returned slice is alive.
///
/// # Panics
///
/// Panics if `T` is a ZST or exceeds `isize::MAX` bytes.
#[inline]
pub unsafe fn as_bytes<T: Pod>(v: &T) -> &[u8] {
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

/// Reinterprets a reference as a byte slice without requiring `Pod`.
///
/// # Safety
///
/// Caller must ensure all bytes of `v` are initialized (e.g., via `mem::zeroed()`
/// or by reading from storage). Reading uninitialized bytes is undefined behavior.
///
/// # Panics
///
/// Panics if `T` is a ZST or exceeds `isize::MAX` bytes.
#[inline]
pub unsafe fn as_bytes_unchecked<T>(v: &T) -> &[u8] {
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

/// Mutable version of [`as_bytes_unchecked`].
///
/// Reinterprets a mutable reference as a mutable byte slice without requiring `Pod`.
/// This is useful for types that need to be filled by I/O operations (e.g., reading
/// from storage into a struct buffer).
///
/// # Safety
///
/// - Caller must ensure all bytes of `v` are initialized before reading them.
/// - Any writes through the returned slice must leave `v` in a valid state.
/// - The caller must not create overlapping mutable references.
///
/// # Panics
///
/// Panics if `T` is a ZST or exceeds `isize::MAX` bytes.
#[inline]
pub unsafe fn as_bytes_unchecked_mut<T>(v: &mut T) -> &mut [u8] {
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

/// Mutable version of [`as_bytes`].
///
/// # Safety
///
/// `T` must be `Pod`, and any writes through the returned slice must leave `v`
/// in a valid state.
///
/// # Panics
///
/// Panics if `T` is a ZST or exceeds `isize::MAX` bytes.
#[inline]
pub unsafe fn as_bytes_mut<T: Pod>(v: &mut T) -> &mut [u8] {
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

/// Compares two `Pod` values byte-for-byte.
///
/// This checks for exact memory equality, unlike `PartialEq`.
///
/// # Safety
///
/// `T` must be `Pod`, and `a`/`b` must not partially overlap.
///
/// # Panics
///
/// Panics if `a` and `b` overlap partially.
#[inline]
pub unsafe fn equal_bytes<T: Pod>(a: &T, b: &T) -> bool {
    let a_addr = a as *const T as usize;
    let b_addr = b as *const T as usize;
    let size = size_of::<T>();

    let disjoint = a_addr.wrapping_add(size) <= b_addr || b_addr.wrapping_add(size) <= a_addr;
    let identical = a_addr == b_addr;
    assert!(disjoint || identical);

    // SAFETY: Caller guarantees `T` is Pod; overlap is checked above.
    let result = unsafe { as_bytes(a) } == unsafe { as_bytes(b) };
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
pub const fn align_up(value: usize, alignment: usize) -> usize {
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
    /// # Safety
    ///
    /// The caller must guarantee the all-zero byte pattern is a valid `T`.
    ///
    /// # Panics
    ///
    /// Panics if `T` is a ZST, too large, or if allocation fails.
    pub unsafe fn new_zeroed() -> Self
    where
        T: Zeroable,
    {
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
        let boxed: AlignedBox<u64> = unsafe { AlignedBox::new_zeroed() };
        assert_eq!(*boxed, 0u64);
        assert!((boxed.as_ptr() as usize).is_multiple_of(align_of::<u64>()));
    }

    #[test]
    fn test_aligned_box_mut() {
        let mut boxed: AlignedBox<u64> = unsafe { AlignedBox::new_zeroed() };
        *boxed = 42;
        assert_eq!(*boxed, 42);
    }

    #[test]
    fn test_as_bytes() {
        let value: u32 = 0xDEADBEEF;
        let bytes = unsafe { as_bytes(&value) };
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

        unsafe {
            assert!(equal_bytes(&a, &b));
            assert!(!equal_bytes(&a, &c));
            assert!(equal_bytes(&a, &a)); // Same address
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TestStruct {
        a: u32,
        b: u32,
    }

    unsafe impl Zeroable for TestStruct {}

    #[test]
    fn test_aligned_box_struct() {
        let mut boxed: AlignedBox<TestStruct> = unsafe { AlignedBox::new_zeroed() };
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
    fn test_equal_bytes_explicit_padding() {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct Padded {
            a: u8,
            padding: [u8; 7],
            b: u64,
        }

        unsafe impl Pod for Padded {}

        assert_eq!(size_of::<Padded>(), 16);

        let x = Padded {
            a: 1,
            padding: [0xAA; 7],
            b: 2,
        };
        let y = Padded {
            a: 1,
            padding: [0xAA; 7],
            b: 2,
        };
        let z = Padded {
            a: 1,
            padding: [0x00; 7],
            b: 2,
        };

        assert_eq!(x, y);
        assert_ne!(x, z);

        unsafe {
            assert!(equal_bytes(&x, &y));
            assert!(!equal_bytes(&x, &z));
        }
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

        unsafe impl Zeroable for Dropper {}

        {
            let _boxed: AlignedBox<Dropper> = unsafe { AlignedBox::new_zeroed() };
        }

        // Expect 1 because AlignedBox now calls drop_in_place
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_as_bytes_mut() {
        let mut value: u32 = 0;
        let bytes = unsafe { as_bytes_mut(&mut value) };

        // Write bytes in native endian
        bytes.copy_from_slice(&0xDEADBEEFu32.to_ne_bytes());

        assert_eq!(value, 0xDEADBEEF);
    }

    #[test]
    fn test_as_bytes_unchecked_mut() {
        let mut value: u32 = 0;
        let bytes = unsafe { as_bytes_unchecked_mut(&mut value) };

        // Write bytes in native endian
        bytes.copy_from_slice(&0xCAFEBABEu32.to_ne_bytes());

        assert_eq!(value, 0xCAFEBABE);
    }

    #[test]
    fn test_as_bytes_unchecked_mut_struct() {
        // Test with a non-Pod struct (no Pod impl required)
        #[repr(C)]
        struct NonPodStruct {
            a: u32,
            b: u32,
        }

        let mut value = NonPodStruct { a: 0, b: 0 };
        let bytes = unsafe { as_bytes_unchecked_mut(&mut value) };

        assert_eq!(bytes.len(), size_of::<NonPodStruct>());

        // Write values through byte slice
        bytes[..4].copy_from_slice(&42u32.to_ne_bytes());
        bytes[4..].copy_from_slice(&100u32.to_ne_bytes());

        assert_eq!(value.a, 42);
        assert_eq!(value.b, 100);
    }

    #[test]
    fn test_as_bytes_unchecked_roundtrip() {
        #[repr(C)]
        struct Header {
            magic: u32,
            version: u16,
            flags: u16,
        }

        let mut header = Header {
            magic: 0,
            version: 0,
            flags: 0,
        };

        // Simulate I/O read filling the buffer
        let bytes = unsafe { as_bytes_unchecked_mut(&mut header) };
        bytes[..4].copy_from_slice(&0xDEADBEEFu32.to_ne_bytes());
        bytes[4..6].copy_from_slice(&1u16.to_ne_bytes());
        bytes[6..8].copy_from_slice(&0x8000u16.to_ne_bytes());

        assert_eq!(header.magic, 0xDEADBEEF);
        assert_eq!(header.version, 1);
        assert_eq!(header.flags, 0x8000);

        // Now read back using immutable version
        let read_bytes = unsafe { as_bytes_unchecked(&header) };
        assert_eq!(read_bytes.len(), 8);
        assert_eq!(&read_bytes[..4], &0xDEADBEEFu32.to_ne_bytes());
    }

    #[test]
    fn test_as_bytes_aligned() {
        #[repr(C, align(64))]
        #[derive(Clone, Copy)]
        struct CacheAligned {
            value: u64,
            padding: [u8; 56],
        }

        unsafe impl Pod for CacheAligned {}

        let val = CacheAligned {
            value: 42,
            padding: [0u8; 56],
        };
        let bytes = unsafe { as_bytes(&val) };

        // Size includes explicit padding to alignment.
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

        unsafe impl Zeroable for HighlyAligned {}

        let boxed: AlignedBox<HighlyAligned> = unsafe { AlignedBox::new_zeroed() };
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
        unsafe {
            assert!(equal_bytes(&a, &b));
        }
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
        let boxed: AlignedBox<u128> = unsafe { AlignedBox::new_zeroed() };
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
            let bytes = unsafe { as_bytes(&value) };
            let restored = u64::from_ne_bytes(bytes.try_into().unwrap());
            prop_assert_eq!(value, restored);
        }

        #[test]
        fn prop_as_bytes_roundtrip_u128(value: u128) {
            let bytes = unsafe { as_bytes(&value) };
            let restored = u128::from_ne_bytes(bytes.try_into().unwrap());
            prop_assert_eq!(value, restored);
        }

        #[test]
        fn prop_as_bytes_mut_roundtrip(value: u64) {
            let mut v = value;
            let bytes = unsafe { as_bytes_mut(&mut v) };
            let restored = u64::from_ne_bytes(bytes.try_into().unwrap());
            prop_assert_eq!(value, restored);
        }

        #[test]
        fn prop_as_bytes_unchecked_mut_roundtrip(value: u64) {
            let mut v = value;
            let bytes = unsafe { as_bytes_unchecked_mut(&mut v) };
            let restored = u64::from_ne_bytes(bytes.try_into().unwrap());
            prop_assert_eq!(value, restored);
        }

        #[test]
        fn prop_as_bytes_unchecked_mut_write_read(value: u64) {
            let mut v: u64 = 0;
            let bytes = unsafe { as_bytes_unchecked_mut(&mut v) };
            bytes.copy_from_slice(&value.to_ne_bytes());
            prop_assert_eq!(v, value);
        }

        #[test]
        fn prop_equal_bytes_reflexive(value: u64) {
            let result = unsafe { equal_bytes(&value, &value) };
            prop_assert!(result);
        }

        #[test]
        fn prop_equal_bytes_symmetric(a: u64, b: u64) {
            let ab = unsafe { equal_bytes(&a, &b) };
            let ba = unsafe { equal_bytes(&b, &a) };
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_equal_bytes_consistent_with_eq(a: u64, b: u64) {
            // For types without padding, equal_bytes should match ==
            let ab = unsafe { equal_bytes(&a, &b) };
            prop_assert_eq!(ab, a == b);
        }

        #[test]
        fn prop_aligned_box_zeroed(value in any::<[u8; 32]>()) {
            // Ignore input, just verify zeroing works for various runs
            let _ = value;
            let boxed: AlignedBox<[u64; 4]> = unsafe { AlignedBox::new_zeroed() };
            prop_assert_eq!(*boxed, [0u64; 4]);
        }
    }
}
