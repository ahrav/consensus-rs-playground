//! Storage primitives for direct I/O with sector-aligned buffers.
//!
//! Provides [`AlignedBuf`] for allocating memory that satisfies direct I/O
//! alignment requirements, enabling O_DIRECT reads/writes without kernel
//! buffering.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::alloc;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::Path;

use crate::io::{Completion as IoCompletion, Io, Operation};
use crate::stdx::queue::{Queue, QueueLink, QueueNode};

pub use super::layout::{Layout, Zone, ZoneSpec};

/// Default sector size for most modern storage devices.
pub const SECTOR_SIZE_DEFAULT: usize = 4096;

/// Minimum sector size (legacy 512-byte sectors).
pub const SECTOR_SIZE_MIN: usize = 512;

/// Maximum supported sector size (64 KiB for large-block devices).
pub const SECTOR_SIZE_MAX: usize = 65536;

/// Bounds next-tick callback processing to prevent runaway loops.
const MAX_NEXT_TICK_ITERATIONS: usize = 10_000;

const _: () = {
    assert!(SECTOR_SIZE_DEFAULT.is_power_of_two());
    assert!(SECTOR_SIZE_MIN.is_power_of_two());
    assert!(SECTOR_SIZE_MAX.is_power_of_two());
    assert!(SECTOR_SIZE_MIN <= SECTOR_SIZE_DEFAULT);
    assert!(SECTOR_SIZE_DEFAULT <= SECTOR_SIZE_MAX);

    assert!(std::mem::size_of::<usize>() >= 8);

    assert!(MAX_NEXT_TICK_ITERATIONS > 0);
    assert!(MAX_NEXT_TICK_ITERATIONS <= 100_000);
};

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
        assert!(self.ptr.as_ptr() as usize % self.align == 0);
    }

    /// Returns the buffer as a mutable byte slice.
    ///
    /// # Safety
    ///
    /// Interior mutability through `&self` is sound because `AlignedBuf` owns
    /// its allocation exclusively and is `!Sync`.
    pub fn as_mut_slice(&self) -> &mut [u8] {
        self.assert_invariants();
        // SAFETY: We own the allocation, it's valid for `len` bytes,
        // and `AlignedBuf` is `!Sync` so no data races.
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
    pub fn as_mut_ptr(&self) -> *mut u8 {
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
