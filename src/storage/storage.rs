//! Storage primitives for direct I/O with sector-aligned buffers.
//!
//! Provides [`AlignedBuf`] for allocating memory that satisfies direct I/O
//! alignment requirements, enabling O_DIRECT reads/writes without kernel
//! buffering.

#![allow(dead_code, unused_imports)]

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

/// Controls whether I/O operations complete synchronously or asynchronously.
///
/// Used to configure storage behavior for testing (synchronous) vs production
/// (asynchronous with io_uring/epoll).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Synchronicity {
    /// Operations block until completion. Useful for deterministic testing.
    AlwaysSynchronous,
    /// Operations return immediately; completion notified via callback.
    AlwaysAsynchronous,
}

/// Identifies which subsystem a deferred callback belongs to.
///
/// Callbacks are partitioned to allow priority-based processing: VSR protocol
/// operations typically take precedence over LSM tree maintenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextTickQueue {
    /// Viewstamped Replication protocol callbacks (high priority).
    Vsr,
    /// Log-Structured Merge tree callbacks (background maintenance).
    Lsm,
}

// ─────────────────────────────────────────────────────────────────────────────
// I/O Operation Structs
// ─────────────────────────────────────────────────────────────────────────────
//
// `Read` and `Write` are self-contained cursors for asynchronous I/O. They
// embed a [`ZoneSpec`] snapshot from the [`Layout`], enabling absolute offset
// calculation without holding a reference to the global layout.
//
// # Address Translation
//
// The storage file is divided into [`Zone`]s with fixed positions. These structs
// translate logical (zone-relative) offsets to physical (absolute) file offsets:
//
// ```text
// Logical:   Read/Write { zone: WalPrepares, offset: 100 }
//                                              │
//            ┌─────────────────────────────────┘
//            ▼
// Physical:  zone_spec.base + offset = 8192 + 100 = 8292
// ```
//
// The embedded `zone_spec` is copied from [`Layout`] at creation time,
// making operations stateless relative to global layout—they can be passed
// to isolated I/O threads with all math self-contained.
//
// # Layout
//
// `#[repr(C)]` ensures predictable field ordering for cache efficiency and
// potential FFI with the kernel I/O subsystem.
// ─────────────────────────────────────────────────────────────────────────────

/// A pending read operation. See module comment above for address translation.
#[repr(C)]
pub struct Read {
    /// I/O completion state (result, status flags).
    pub io: IoCompletion,
    /// Invoked when I/O completes; `None` if not pending.
    callback: Option<fn(&mut Read)>,
    /// Expected bytes for validation (short reads are errors in direct I/O).
    expected_len: usize,

    /// Which zone this read targets.
    pub zone: Zone,
    /// Snapshot of zone position/size from [`Layout`] at submission time.
    pub zone_spec: ZoneSpec,
    /// Logical offset within the zone (0 = zone start).
    pub offset: u64,
}

impl Read {
    /// Creates a zeroed `Read` targeting the SuperBlock at offset 0.
    ///
    /// Call site must populate `zone`, `zone_spec`, and `offset` before use.
    pub const fn new() -> Self {
        Self {
            io: IoCompletion::new(),
            callback: None,
            expected_len: 0,
            zone: Zone::SuperBlock,
            zone_spec: ZoneSpec { base: 0, size: 0 },
            offset: 0,
        }
    }

    /// Returns `true` if this read is awaiting I/O completion.
    #[inline]
    pub fn is_pending(&self) -> bool {
        self.callback.is_some()
    }

    /// Expected read length; short reads indicate I/O errors with O_DIRECT.
    #[inline]
    pub fn expected_len(&self) -> usize {
        self.expected_len
    }

    /// Computes the absolute file offset from zone-relative addressing.
    ///
    /// # Panics
    ///
    /// - `offset > zone_spec.size` (would read past zone boundary)
    #[inline]
    pub fn absolute_offset(&self) -> u64 {
        assert!(self.offset <= self.zone_spec.size);

        let abs = self.zone_spec.offset(self.offset);
        assert!(abs >= self.zone_spec.base);

        abs
    }
}

impl Default for Read {
    fn default() -> Self {
        Self::new()
    }
}

/// A pending write operation. See module comment above for address translation.
#[repr(C)]
pub struct Write {
    /// I/O completion state (result, status flags).
    pub io: IoCompletion,
    /// Invoked when I/O completes; `None` if not pending.
    callback: Option<fn(&mut Write)>,
    /// Expected bytes for validation (short writes are errors in direct I/O).
    expected_len: usize,

    /// Which zone this write targets.
    pub zone: Zone,
    /// Snapshot of zone position/size from [`Layout`] at submission time.
    pub zone_spec: ZoneSpec,
    /// Logical offset within the zone (0 = zone start).
    pub offset: u64,
}

impl Write {
    /// Creates a zeroed `Write` targeting the SuperBlock at offset 0.
    ///
    /// Call site must populate `zone`, `zone_spec`, and `offset` before use.
    pub const fn new() -> Self {
        Self {
            io: IoCompletion::new(),
            callback: None,
            expected_len: 0,
            zone: Zone::SuperBlock,
            zone_spec: ZoneSpec { base: 0, size: 0 },
            offset: 0,
        }
    }

    /// Returns `true` if this write is awaiting I/O completion.
    #[inline]
    pub fn is_pending(&self) -> bool {
        self.callback.is_some()
    }

    /// Expected write length; short writes indicate I/O errors with O_DIRECT.
    #[inline]
    pub fn expected_len(&self) -> usize {
        self.expected_len
    }

    /// Computes the absolute file offset from zone-relative addressing.
    ///
    /// # Panics
    ///
    /// - `offset > zone_spec.size` (would write past zone boundary)
    #[inline]
    pub fn absolute_offset(&self) -> u64 {
        assert!(self.offset <= self.zone_spec.size);

        let abs = self.zone_spec.offset(self.offset);
        assert!(abs >= self.zone_spec.base);

        abs
    }
}

impl Default for Write {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Constants Tests ====================

    #[test]
    fn sector_size_constants_are_power_of_two() {
        assert!(SECTOR_SIZE_MIN.is_power_of_two());
        assert!(SECTOR_SIZE_DEFAULT.is_power_of_two());
        assert!(SECTOR_SIZE_MAX.is_power_of_two());
    }

    #[test]
    fn sector_size_ordering() {
        // Redundant with compile-time assertions, but kept for test coverage visibility
        const _: () = assert!(SECTOR_SIZE_MIN <= SECTOR_SIZE_DEFAULT);
        const _: () = assert!(SECTOR_SIZE_DEFAULT <= SECTOR_SIZE_MAX);
    }

    #[test]
    fn sector_size_values() {
        assert_eq!(SECTOR_SIZE_MIN, 512);
        assert_eq!(SECTOR_SIZE_DEFAULT, 4096);
        assert_eq!(SECTOR_SIZE_MAX, 65536);
    }

    // ==================== AlignedBuf Construction Tests ====================

    #[test]
    fn aligned_buf_with_default_sector() {
        let buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
        assert_eq!(buf.len(), 4096);
        assert_eq!(buf.align(), SECTOR_SIZE_DEFAULT);
        assert_eq!(buf.as_ptr() as usize % SECTOR_SIZE_DEFAULT, 0);
    }

    #[test]
    fn aligned_buf_with_min_sector() {
        let buf = AlignedBuf::new_zeroed(512, SECTOR_SIZE_MIN);
        assert_eq!(buf.len(), 512);
        assert_eq!(buf.align(), SECTOR_SIZE_MIN);
        assert_eq!(buf.as_ptr() as usize % SECTOR_SIZE_MIN, 0);
    }

    #[test]
    fn aligned_buf_with_max_sector() {
        let buf = AlignedBuf::new_zeroed(65536, SECTOR_SIZE_MAX);
        assert_eq!(buf.len(), 65536);
        assert_eq!(buf.align(), SECTOR_SIZE_MAX);
        assert_eq!(buf.as_ptr() as usize % SECTOR_SIZE_MAX, 0);
    }

    #[test]
    fn aligned_buf_small_len_large_align() {
        let buf = AlignedBuf::new_zeroed(1, SECTOR_SIZE_MAX);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.align(), SECTOR_SIZE_MAX);
        assert_eq!(buf.as_ptr() as usize % SECTOR_SIZE_MAX, 0);
    }

    #[test]
    fn aligned_buf_large_len_small_align() {
        let buf = AlignedBuf::new_zeroed(1024 * 1024, SECTOR_SIZE_MIN);
        assert_eq!(buf.len(), 1024 * 1024);
        assert_eq!(buf.align(), SECTOR_SIZE_MIN);
        assert_eq!(buf.as_ptr() as usize % SECTOR_SIZE_MIN, 0);
    }

    #[test]
    fn aligned_buf_various_alignments() {
        for align in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] {
            let buf = AlignedBuf::new_zeroed(align, align);
            assert_eq!(buf.as_ptr() as usize % align, 0);
        }
    }

    // ==================== AlignedBuf Panic Tests ====================

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn aligned_buf_zero_len_panics() {
        let _ = AlignedBuf::new_zeroed(0, SECTOR_SIZE_DEFAULT);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn aligned_buf_zero_align_panics() {
        let _ = AlignedBuf::new_zeroed(4096, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn aligned_buf_non_power_of_two_align_panics() {
        let _ = AlignedBuf::new_zeroed(4096, 100);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn aligned_buf_align_exceeds_max_panics() {
        let _ = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_MAX * 2);
    }

    // ==================== AlignedBuf Method Tests ====================

    #[test]
    fn aligned_buf_zero_initialized() {
        let buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
        assert!(buf.as_slice().iter().all(|&b| b == 0));
    }

    #[test]
    fn aligned_buf_as_slice_len() {
        let buf = AlignedBuf::new_zeroed(1234, SECTOR_SIZE_MIN);
        assert_eq!(buf.as_slice().len(), 1234);
    }

    #[test]
    fn aligned_buf_as_mut_slice_write() {
        let buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
        let slice = buf.as_mut_slice();
        slice[0] = 0xAB;
        slice[1] = 0xCD;
        slice[4095] = 0xEF;

        assert_eq!(buf.as_slice()[0], 0xAB);
        assert_eq!(buf.as_slice()[1], 0xCD);
        assert_eq!(buf.as_slice()[4095], 0xEF);
    }

    #[test]
    fn aligned_buf_as_mut_slice_fill() {
        let buf = AlignedBuf::new_zeroed(512, SECTOR_SIZE_MIN);
        buf.as_mut_slice().fill(0xFF);
        assert!(buf.as_slice().iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn aligned_buf_ptr_consistency() {
        let buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
        assert_eq!(buf.as_ptr(), buf.as_mut_ptr() as *const u8);
        assert_eq!(buf.as_ptr(), buf.as_slice().as_ptr());
    }

    #[test]
    fn aligned_buf_is_empty_always_false() {
        let buf = AlignedBuf::new_zeroed(1, SECTOR_SIZE_MIN);
        assert!(!buf.is_empty());

        let buf2 = AlignedBuf::new_zeroed(1024 * 1024, SECTOR_SIZE_DEFAULT);
        assert!(!buf2.is_empty());
    }

    #[test]
    fn aligned_buf_len_matches_requested() {
        for len in [1, 100, 512, 4096, 65536, 1024 * 1024] {
            let buf = AlignedBuf::new_zeroed(len, SECTOR_SIZE_MIN);
            assert_eq!(buf.len(), len);
        }
    }

    #[test]
    fn aligned_buf_align_matches_requested() {
        for align in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] {
            let buf = AlignedBuf::new_zeroed(4096, align);
            assert_eq!(buf.align(), align);
        }
    }

    // ==================== AlignedBuf Drop & Send Tests ====================

    #[test]
    fn aligned_buf_drop_multiple() {
        for _ in 0..100 {
            let buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
            buf.as_mut_slice().fill(0xAB);
            // buf dropped here
        }
    }

    #[test]
    fn aligned_buf_send() {
        let buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
        buf.as_mut_slice()[0] = 42;

        let handle = std::thread::spawn(move || {
            assert_eq!(buf.as_slice()[0], 42);
            buf.len()
        });

        assert_eq!(handle.join().unwrap(), 4096);
    }

    // ==================== Synchronicity & NextTickQueue Tests ====================

    #[test]
    fn synchronicity_variants() {
        let sync = Synchronicity::AlwaysSynchronous;
        let async_var = Synchronicity::AlwaysAsynchronous;

        assert_eq!(sync, Synchronicity::AlwaysSynchronous);
        assert_eq!(async_var, Synchronicity::AlwaysAsynchronous);
        assert_ne!(sync, async_var);
    }

    #[test]
    fn synchronicity_clone_eq() {
        let sync = Synchronicity::AlwaysSynchronous;
        let cloned = sync;
        assert_eq!(sync, cloned);
    }

    #[test]
    fn next_tick_queue_variants() {
        let vsr = NextTickQueue::Vsr;
        let lsm = NextTickQueue::Lsm;

        assert_eq!(vsr, NextTickQueue::Vsr);
        assert_eq!(lsm, NextTickQueue::Lsm);
        assert_ne!(vsr, lsm);
    }

    // ==================== Read Tests ====================

    #[test]
    fn read_new_defaults() {
        let read = Read::new();
        assert!(!read.is_pending());
        assert_eq!(read.expected_len(), 0);
        assert_eq!(read.zone, Zone::SuperBlock);
        assert_eq!(read.zone_spec.base, 0);
        assert_eq!(read.zone_spec.size, 0);
        assert_eq!(read.offset, 0);
    }

    #[test]
    fn read_default_eq_new() {
        let read_new = Read::new();
        let read_default = Read::default();

        assert_eq!(read_new.zone, read_default.zone);
        assert_eq!(read_new.offset, read_default.offset);
        assert_eq!(read_new.expected_len(), read_default.expected_len());
    }

    #[test]
    fn read_is_pending_without_callback() {
        let read = Read::new();
        assert!(!read.is_pending());
    }

    #[test]
    fn read_absolute_offset_zero() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec {
            base: 1000,
            size: 500,
        };
        read.offset = 0;

        assert_eq!(read.absolute_offset(), 1000);
    }

    #[test]
    fn read_absolute_offset_middle() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec {
            base: 1000,
            size: 500,
        };
        read.offset = 250;

        assert_eq!(read.absolute_offset(), 1250);
    }

    #[test]
    fn read_absolute_offset_at_boundary() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec {
            base: 1000,
            size: 500,
        };
        read.offset = 500; // At boundary (valid)

        assert_eq!(read.absolute_offset(), 1500);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_absolute_offset_beyond_zone_panics() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec {
            base: 1000,
            size: 500,
        };
        read.offset = 501; // Beyond zone

        let _ = read.absolute_offset();
    }

    #[test]
    fn read_with_layout_integration() {
        let layout = Layout::new(512, 4096, 512, 512, 512, 512, 4096);

        let mut read = Read::new();
        read.zone = Zone::WalPrepares;
        read.zone_spec = layout.zone(Zone::WalPrepares);
        read.offset = 100;

        let abs = read.absolute_offset();
        assert_eq!(abs, layout.start(Zone::WalPrepares) + 100);
    }

    // ==================== Write Tests ====================

    #[test]
    fn write_new_defaults() {
        let write = Write::new();
        assert!(!write.is_pending());
        assert_eq!(write.expected_len(), 0);
        assert_eq!(write.zone, Zone::SuperBlock);
        assert_eq!(write.zone_spec.base, 0);
        assert_eq!(write.zone_spec.size, 0);
        assert_eq!(write.offset, 0);
    }

    #[test]
    fn write_default_eq_new() {
        let write_new = Write::new();
        let write_default = Write::default();

        assert_eq!(write_new.zone, write_default.zone);
        assert_eq!(write_new.offset, write_default.offset);
        assert_eq!(write_new.expected_len(), write_default.expected_len());
    }

    #[test]
    fn write_is_pending_without_callback() {
        let write = Write::new();
        assert!(!write.is_pending());
    }

    #[test]
    fn write_absolute_offset_zero() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec {
            base: 2000,
            size: 1000,
        };
        write.offset = 0;

        assert_eq!(write.absolute_offset(), 2000);
    }

    #[test]
    fn write_absolute_offset_middle() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec {
            base: 2000,
            size: 1000,
        };
        write.offset = 500;

        assert_eq!(write.absolute_offset(), 2500);
    }

    #[test]
    fn write_absolute_offset_at_boundary() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec {
            base: 2000,
            size: 1000,
        };
        write.offset = 1000;

        assert_eq!(write.absolute_offset(), 3000);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_absolute_offset_beyond_zone_panics() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec {
            base: 2000,
            size: 1000,
        };
        write.offset = 1001;

        let _ = write.absolute_offset();
    }

    #[test]
    fn write_with_layout_integration() {
        let layout = Layout::new(512, 4096, 512, 512, 512, 512, 4096);

        let mut write = Write::new();
        write.zone = Zone::ClientReplies;
        write.zone_spec = layout.zone(Zone::ClientReplies);
        write.offset = 200;

        let abs = write.absolute_offset();
        assert_eq!(abs, layout.start(Zone::ClientReplies) + 200);
    }

    // ==================== Read/Write Consistency Tests ====================

    #[test]
    fn read_write_same_offset_same_result() {
        let zone_spec = ZoneSpec {
            base: 8192,
            size: 4096,
        };

        let mut read = Read::new();
        read.zone_spec = zone_spec;
        read.offset = 100;

        let mut write = Write::new();
        write.zone_spec = zone_spec;
        write.offset = 100;

        assert_eq!(read.absolute_offset(), write.absolute_offset());
    }

    #[test]
    fn read_write_all_zones() {
        let layout = Layout::new(512, 4096, 4096, 8192, 16384, 8192, 65536);

        for zone in Zone::ALL {
            let zone_spec = layout.zone(zone);
            if zone_spec.size > 0 {
                let offset = zone_spec.size / 2;

                let mut read = Read::new();
                read.zone = zone;
                read.zone_spec = zone_spec;
                read.offset = offset;

                let mut write = Write::new();
                write.zone = zone;
                write.zone_spec = zone_spec;
                write.offset = offset;

                assert_eq!(read.absolute_offset(), write.absolute_offset());
                assert_eq!(read.absolute_offset(), zone_spec.base + offset);
            }
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
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
            let buf = AlignedBuf::new_zeroed(len, align);
            prop_assert_eq!(buf.as_ptr(), buf.as_mut_ptr() as *const u8);
            prop_assert_eq!(buf.as_ptr(), buf.as_slice().as_ptr());
        }

        #[test]
        fn prop_aligned_buf_mutability(
            len in 1usize..10000,
            pattern in any::<u8>(),
        ) {
            let buf = AlignedBuf::new_zeroed(len, SECTOR_SIZE_MIN);
            buf.as_mut_slice().fill(pattern);
            prop_assert!(buf.as_slice().iter().all(|&b| b == pattern));
        }

        #[test]
        fn prop_read_absolute_offset(
            base in 0u64..1_000_000,
            size in 1u64..100_000,
            offset_factor in 0.0f64..1.0,
        ) {
            let mut read = Read::new();
            read.zone_spec = ZoneSpec { base, size };
            read.offset = (size as f64 * offset_factor) as u64;

            let abs = read.absolute_offset();
            prop_assert_eq!(abs, base + read.offset);
            prop_assert!(abs >= base);
        }

        #[test]
        fn prop_write_absolute_offset(
            base in 0u64..1_000_000,
            size in 1u64..100_000,
            offset_factor in 0.0f64..1.0,
        ) {
            let mut write = Write::new();
            write.zone_spec = ZoneSpec { base, size };
            write.offset = (size as f64 * offset_factor) as u64;

            let abs = write.absolute_offset();
            prop_assert_eq!(abs, base + write.offset);
            prop_assert!(abs >= base);
        }

        #[test]
        fn prop_read_write_consistency(
            base in 0u64..1_000_000,
            size in 1u64..100_000,
            offset_factor in 0.0f64..1.0,
        ) {
            let zone_spec = ZoneSpec { base, size };
            let offset = (size as f64 * offset_factor) as u64;

            let mut read = Read::new();
            read.zone_spec = zone_spec;
            read.offset = offset;

            let mut write = Write::new();
            write.zone_spec = zone_spec;
            write.offset = offset;

            prop_assert_eq!(read.absolute_offset(), write.absolute_offset());
        }

        #[test]
        fn prop_read_offset_within_zone(
            base in 0u64..1_000_000,
            size in 1u64..100_000,
            offset_factor in 0.0f64..1.0,
        ) {
            let zone_spec = ZoneSpec { base, size };
            let offset = (size as f64 * offset_factor) as u64;

            let mut read = Read::new();
            read.zone_spec = zone_spec;
            read.offset = offset;

            let abs = read.absolute_offset();
            prop_assert!(abs >= zone_spec.base);
            prop_assert!(abs <= zone_spec.end());
        }

        #[test]
        fn prop_layout_integration(
            sector_exp in 9u32..13u32,
            block_exp in 0u32..4u32,
            zone_idx in 0usize..Zone::COUNT,
        ) {
            let sector_size = 1u64 << sector_exp;
            let block_size = sector_size << block_exp;

            let layout = Layout::new(
                sector_size,
                block_size,
                sector_size * 2,
                sector_size * 4,
                sector_size * 8,
                sector_size * 2,
                block_size * 16,
            );

            let zone = Zone::ALL[zone_idx];
            let zone_spec = layout.zone(zone);

            if zone_spec.size > 0 {
                let offset = zone_spec.size / 2;

                let mut read = Read::new();
                read.zone = zone;
                read.zone_spec = zone_spec;
                read.offset = offset;

                let abs = read.absolute_offset();
                prop_assert_eq!(abs, layout.offset(zone, offset));
            }
        }
    }
}

#[cfg(test)]
mod security_tests {
    use super::*;

    #[test]
    #[should_panic(expected = "ZoneSpec::offset overflow")]
    fn test_read_absolute_offset_overflow() {
        // Scenario: ZoneSpec is corrupted or maliciously crafted such that
        // base + size overflows u64, but we try to read within "size".
        let mut read = Read::new();
        read.zone_spec = ZoneSpec {
            base: u64::MAX - 10,
            size: 20, // base + size overflows
        };
        // offset is within "size" (15 <= 20), so the first check passes.
        // But base + offset (MAX - 10 + 15) = MAX + 5 -> Overflows.
        read.offset = 15;

        let _ = read.absolute_offset();
    }

    #[test]
    #[should_panic(expected = "ZoneSpec::offset overflow")]
    fn test_write_absolute_offset_overflow() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec {
            base: u64::MAX - 100,
            size: 200,
        };
        write.offset = 150;
        let _ = write.absolute_offset();
    }

    #[test]
    fn test_struct_sizes() {
        // Ensure structs don't accidentally grow or shrink in ways that might affect FFI or cache lines.
        // These assertions are platform-dependent (likely 64-bit), but valuable for tracking changes.
        // Read/Write:
        // IoCompletion (approx 88 bytes, includes Operation(32), ListLink(16), etc.)
        // + Option<fn> (8)
        // + usize (8)
        // + Zone (1+7pad = 8)
        // + ZoneSpec (16)
        // + u64 (8)
        // = 88 + 8 + 8 + 8 + 16 + 8 = 136 bytes.
        assert_eq!(std::mem::size_of::<Read>(), 136);
        assert_eq!(std::mem::size_of::<Write>(), 136);

        assert_eq!(std::mem::align_of::<Read>(), 8);
        assert_eq!(std::mem::align_of::<Write>(), 8);
    }
}
