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

use libc;

use crate::io::{Completion as IoCompletion, Io, Operation};
use crate::stdx::queue::{Queue, QueueLink, QueueNode};

pub use super::layout::{Layout, Zone, ZoneSpec};

/// Default sector size for most modern storage devices.
pub const SECTOR_SIZE_DEFAULT: usize = 4096;

/// Minimum sector size (legacy 512-byte sectors).
pub const SECTOR_SIZE_MIN: usize = 512;

/// Maximum supported sector size (64 KiB for large-block devices).
pub const SECTOR_SIZE_MAX: usize = 65536;

/// Default number of concurrent I/O operations for the storage engine.
/// Must be a power of two. Provides good concurrency for async I/O without excessive memory overhead.
const IO_ENTRIES_DEFAULT: u32 = 256;

/// Bounds next-tick callback processing to prevent runaway loops.
const MAX_NEXT_TICK_ITERATIONS: usize = 10_000;

const _: () = {
    assert!(SECTOR_SIZE_DEFAULT.is_power_of_two());
    assert!(SECTOR_SIZE_MIN.is_power_of_two());
    assert!(SECTOR_SIZE_MAX.is_power_of_two());
    assert!(SECTOR_SIZE_MIN <= SECTOR_SIZE_DEFAULT);
    assert!(SECTOR_SIZE_DEFAULT <= SECTOR_SIZE_MAX);

    assert!(std::mem::size_of::<usize>() >= 8);

    assert!(IO_ENTRIES_DEFAULT > 0);
    assert!(IO_ENTRIES_DEFAULT.is_power_of_two());

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
// I/O Control Blocks
// ─────────────────────────────────────────────────────────────────────────────
//
// `Read` and `Write` are **I/O Control Blocks (IOCBs)**—each represents a slot
// for an in-flight async operation. The embedded [`IoCompletion`] is bound to
// a fixed memory address where the kernel/io_uring writes results.
//
// # Address Translation
//
// IOCBs translate logical (zone-relative) offsets to physical (absolute) offsets:
//
// ```text
// Logical:   Read { zone: WalPrepares, offset: 100 }
//                                        │
//            ┌───────────────────────────┘
//            ▼
// Physical:  zone_spec.base + offset = 8192 + 100 = 8292
// ```
//
// The embedded `zone_spec` is copied from [`Layout`], making IOCBs self-contained.
// Once prepared, an IOCB doesn't need to access the global Layout.
//
// The storage engine only sees `absolute_offset()`—no knowledge of Zones. The
// `zone` field exists for **completion context**: callbacks inspect it to
// determine how to process results.
//
// # Memory Layout
//
// `#[repr(C)]` ensures predictable field ordering. The `io` field is first so
// the IOCB address equals the completion address.
// ─────────────────────────────────────────────────────────────────────────────

/// I/O Control Block for a pending read. See module comment for lifecycle.
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

/// I/O Control Block for a pending write. See module comment for lifecycle.
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

/// Phantom type tag for [`NextTick`] intrusive lists.
enum NextTickTag {}

/// Intrusive linked list node for deferred callbacks ("next tick" semantics).
///
/// Uses an intrusive design to avoid heap allocation during scheduling—nodes
/// live in caller-owned memory. The `callback` is `None` when idle, `Some`
/// when queued. This approach guarantees deterministic memory usage and
/// eliminates allocation failure paths during I/O completion.
#[repr(C)]
pub struct NextTick {
    link: QueueLink<NextTick, NextTickTag>,
    callback: Option<fn(&mut NextTick)>,
}

impl NextTick {
    pub const fn new() -> Self {
        Self {
            link: QueueLink::new(),
            callback: None,
        }
    }

    /// Returns `true` if this callback is queued and awaiting execution.
    #[inline]
    pub fn is_pending(&self) -> bool {
        self.callback.is_some()
    }
}

impl Default for NextTick {
    fn default() -> Self {
        Self::new()
    }
}

impl QueueNode<NextTickTag> for NextTick {
    fn queue_link(&mut self) -> &mut QueueLink<Self, NextTickTag> {
        &mut self.link
    }

    fn queue_link_ref(&self) -> &QueueLink<Self, NextTickTag> {
        &self.link
    }
}

/// Direct I/O storage engine with async operation queues and deferred callbacks.
///
/// Manages a single preallocated file divided into zones (see [`Layout`]).
/// Provides two priority-separated "next tick" queues: VSR protocol work
/// executes before LSM background maintenance, ensuring consensus operations
/// aren't starved by compaction.
///
/// # Invariants
///
/// - `fd >= 0` (valid file descriptor)
/// - `sector_size` is a power of two in `[SECTOR_SIZE_MIN, SECTOR_SIZE_MAX]`
/// - File size matches `layout.total_size()` and is sector-aligned
pub struct Storage {
    io: Io,
    file: File,
    fd: RawFd,

    layout: Layout,
    sector_size: usize,

    /// High-priority callbacks (VSR protocol operations).
    next_tick_vsr: Queue<NextTick, NextTickTag>,
    /// Low-priority callbacks (LSM tree maintenance).
    next_tick_lsm: Queue<NextTick, NextTickTag>,
}

/// Configuration for opening a storage file.
pub struct Options<'a> {
    pub path: &'a Path,
    pub layout: Layout,
    pub direct_io: bool,
    /// Sector size in bytes. If 0, defaults to [`SECTOR_SIZE_DEFAULT`].
    pub sector_size: usize,
}

impl Storage {
    pub const SYNCHRONICITY: Synchronicity = Synchronicity::AlwaysAsynchronous;

    /// Opens or creates a storage file with the specified layout.
    ///
    /// Preallocates the file to `layout.total_size()` and enables direct I/O
    /// if requested (O_DIRECT on Linux, F_NOCACHE on macOS). Direct I/O
    /// bypasses the kernel page cache for predictable latency and durability.
    ///
    /// # Panics
    ///
    /// - `path` is empty
    /// - `sector_size` is invalid (not power-of-two or out of range)
    /// - `layout.sector_size` doesn't match resolved `sector_size`
    /// - `layout.total_size()` isn't sector-aligned
    ///
    /// # Errors
    ///
    /// Returns I/O errors from file creation, preallocation, or I/O queue setup.
    pub fn open(opts: Options<'_>) -> std::io::Result<Self> {
        assert!(!opts.path.as_os_str().is_empty());

        let sector_size = if opts.sector_size == 0 {
            SECTOR_SIZE_DEFAULT
        } else {
            opts.sector_size
        };

        assert!(sector_size >= SECTOR_SIZE_MIN);
        assert!(sector_size <= SECTOR_SIZE_MAX);
        assert!(sector_size.is_power_of_two());
        assert_eq!(opts.layout.sector_size as usize, sector_size);

        let total_size = opts.layout.total_size();
        assert!(total_size > 0);
        assert!(total_size.is_multiple_of(sector_size as u64));

        let mut oo = OpenOptions::new();
        oo.read(true).write(true).create(true);

        #[cfg(target_os = "linux")]
        if opts.direct_io {
            oo.custom_flags(libc::O_DIRECT);
        }

        let file = oo.open(opts.path)?;
        file.set_len(total_size)?;

        // macOS doesn't support O_DIRECT; use F_NOCACHE for similar semantics.
        #[cfg(target_os = "macos")]
        if opts.direct_io {
            let fd = file.as_raw_fd();
            // SAFETY: `fd` is a valid file descriptor owned by `file`.
            // F_NOCACHE disables kernel page cache without data corruption risk.
            let result = unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
            if result == -1 {
                return Err(std::io::Error::last_os_error());
            }
        }

        let fd = file.as_raw_fd();
        assert!(fd >= 0);

        let storage = Self {
            io: Io::new(IO_ENTRIES_DEFAULT)?,
            file,
            fd,
            layout: opts.layout,
            sector_size,
            next_tick_vsr: Queue::init(),
            next_tick_lsm: Queue::init(),
        };
        storage.assert_invariants();

        Ok(storage)
    }

    /// Validates structural invariants. Called in public methods for defense-in-depth.
    #[inline]
    fn assert_invariants(&self) {
        assert!(self.fd >= 0);
        assert!(self.sector_size >= SECTOR_SIZE_MIN);
        assert!(self.sector_size <= SECTOR_SIZE_MAX);
        assert!(self.sector_size.is_power_of_two());
        assert!(self.layout.total_size() > 0);
    }

    /// Processes I/O completions and drains deferred callbacks.
    ///
    /// Execution order:
    /// 1. Drain VSR and LSM next-tick queues (queued before this call)
    /// 2. Process I/O completions via `io.tick()` (may schedule new next-ticks)
    /// 3. Execute drained callbacks (VSR first for priority)
    ///
    /// This ordering prevents recursion: callbacks scheduled during I/O completions
    /// execute on the *next* `run()` call, preserving "next tick" semantics.
    /// VSR callbacks run before LSM to prioritize consensus over compaction.
    ///
    /// # Panics
    ///
    /// - I/O queue tick fails
    /// - Either queue exceeds [`MAX_NEXT_TICK_ITERATIONS`] (indicates runaway loop)
    pub fn run(&mut self) {
        self.assert_invariants();

        // Snapshot queues before I/O processing to avoid recursion.
        let mut vsr_ticks = self.next_tick_vsr.take_all();
        let mut lsm_ticks = self.next_tick_lsm.take_all();

        self.io.tick().expect("storage engine I/O tick failed");

        // Process VSR callbacks first (higher priority).
        let mut vsr_processed: usize = 0;
        for _ in 0..MAX_NEXT_TICK_ITERATIONS {
            let Some(mut nt_ptr) = vsr_ticks.pop() else {
                break;
            };
            // SAFETY: Pointer obtained from intrusive queue; node lifetime
            // guaranteed by caller (nodes are typically stack-allocated or
            // embedded in long-lived structures).
            let nt = unsafe { nt_ptr.as_mut() };
            if let Some(cb) = nt.callback.take() {
                cb(nt);
            }
            vsr_processed += 1;
        }
        assert!(
            vsr_ticks.is_empty(),
            "VSR next-tick queue not drained: processed {vsr_processed}, limit {MAX_NEXT_TICK_ITERATIONS}"
        );

        // Process LSM callbacks second (background priority).
        let mut lsm_processed: usize = 0;
        for _ in 0..MAX_NEXT_TICK_ITERATIONS {
            let Some(mut nt_ptr) = lsm_ticks.pop() else {
                break;
            };
            // SAFETY: Same as VSR loop above.
            let nt = unsafe { nt_ptr.as_mut() };
            if let Some(cb) = nt.callback.take() {
                cb(nt);
            }
            lsm_processed += 1;
        }
        assert!(
            lsm_ticks.is_empty(),
            "LSM next-tick queue not drained: processed {lsm_processed}, limit {MAX_NEXT_TICK_ITERATIONS}"
        );
    }

    /// Alias for [`Self::run`]. Advances the event loop by one tick.
    #[inline]
    pub fn tick(&mut self) {
        self.run();
    }

    #[inline]
    pub fn sector_size(&self) -> usize {
        self.sector_size
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    #[inline]
    pub fn size(&self) -> u64 {
        self.layout.total_size()
    }

    /// Schedules a callback to execute on the next [`Self::run`] call.
    ///
    /// The callback runs after I/O completions but before returning from `run()`.
    /// Choose `queue` based on priority: [`NextTickQueue::Vsr`] for protocol work,
    /// [`NextTickQueue::Lsm`] for background maintenance.
    ///
    /// # Panics
    ///
    /// - `next_tick.is_pending()` is already true (double-scheduling)
    pub fn on_next_tick(
        &mut self,
        queue: NextTickQueue,
        callback: fn(&mut NextTick),
        next_tick: &mut NextTick,
    ) {
        assert!(!next_tick.is_pending());

        next_tick.callback = Some(callback);
        match queue {
            NextTickQueue::Vsr => self.next_tick_vsr.push(next_tick),
            NextTickQueue::Lsm => self.next_tick_lsm.push(next_tick),
        }

        assert!(next_tick.is_pending());
    }

    /// Cancels all pending LSM callbacks without executing them.
    ///
    /// Used during LSM tree resets (e.g., compaction abort) to discard stale
    /// maintenance work. VSR callbacks are never cancelled—consensus operations
    /// must complete.
    ///
    /// # Panics
    ///
    /// - Queue exceeds [`MAX_NEXT_TICK_ITERATIONS`] (indicates memory corruption)
    pub fn reset_next_tick_lsm(&mut self) {
        let mut ticks = self.next_tick_lsm.take_all();
        let mut cleared: usize = 0;
        for _ in 0..MAX_NEXT_TICK_ITERATIONS {
            let Some(mut nt_ptr) = ticks.pop() else { break };
            // SAFETY: Pointer from intrusive queue; lifetime managed by caller.
            let nt = unsafe { nt_ptr.as_mut() };
            nt.callback = None;
            cleared += 1;
        }

        assert!(
            ticks.is_empty(),
            "LSM reset incomplete: cleared {cleared}, limit {MAX_NEXT_TICK_ITERATIONS}"
        );
    }

    /// Returns a mutable reference to the underlying I/O subsystem.
    ///
    /// Used to submit I/O operations ([`Read`], [`Write`]) which are then
    /// driven to completion by [`Storage::tick`].
    pub fn io(&mut self) -> &mut Io {
        &mut self.io
    }

    /// Returns a reference to the underlying file.
    pub fn file(&self) -> &File {
        &self.file
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Constants ====================

    #[test]
    fn const_pow2() {
        assert!(SECTOR_SIZE_MIN.is_power_of_two());
        assert!(SECTOR_SIZE_DEFAULT.is_power_of_two());
        assert!(SECTOR_SIZE_MAX.is_power_of_two());
    }

    #[test]
    fn const_values() {
        assert_eq!(SECTOR_SIZE_MIN, 512);
        assert_eq!(SECTOR_SIZE_DEFAULT, 4096);
        assert_eq!(SECTOR_SIZE_MAX, 65536);
        const { assert!(SECTOR_SIZE_MIN <= SECTOR_SIZE_DEFAULT) };
        const { assert!(SECTOR_SIZE_DEFAULT <= SECTOR_SIZE_MAX) };
    }

    // ==================== AlignedBuf ====================

    #[test]
    fn buf_alloc_default() {
        let buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
        assert_eq!(buf.len(), 4096);
        assert_eq!(buf.align(), SECTOR_SIZE_DEFAULT);
        assert_eq!(buf.as_ptr() as usize % SECTOR_SIZE_DEFAULT, 0);
    }

    #[test]
    fn buf_alloc_min() {
        let buf = AlignedBuf::new_zeroed(512, SECTOR_SIZE_MIN);
        assert_eq!(buf.len(), 512);
        assert_eq!(buf.align(), SECTOR_SIZE_MIN);
        assert_eq!(buf.as_ptr() as usize % SECTOR_SIZE_MIN, 0);
    }

    #[test]
    fn buf_alloc_max() {
        let buf = AlignedBuf::new_zeroed(65536, SECTOR_SIZE_MAX);
        assert_eq!(buf.len(), 65536);
        assert_eq!(buf.align(), SECTOR_SIZE_MAX);
        assert_eq!(buf.as_ptr() as usize % SECTOR_SIZE_MAX, 0);
    }

    #[test]
    fn buf_alloc_edges() {
        // Small len, large align
        let buf = AlignedBuf::new_zeroed(1, SECTOR_SIZE_MAX);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.align(), SECTOR_SIZE_MAX);
        assert_eq!(buf.as_ptr() as usize % SECTOR_SIZE_MAX, 0);

        // Large len, small align
        let buf2 = AlignedBuf::new_zeroed(1024 * 1024, SECTOR_SIZE_MIN);
        assert_eq!(buf2.len(), 1024 * 1024);
        assert_eq!(buf2.align(), SECTOR_SIZE_MIN);
        assert_eq!(buf2.as_ptr() as usize % SECTOR_SIZE_MIN, 0);
    }

    #[test]
    fn buf_alloc_alignments() {
        for align in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] {
            let buf = AlignedBuf::new_zeroed(align, align);
            assert_eq!(buf.as_ptr() as usize % align, 0);
        }
    }

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
    fn buf_is_zeroed() {
        let buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
        assert!(buf.as_slice().iter().all(|&b| b == 0));
    }

    #[test]
    fn buf_mut_write() {
        let mut buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
        let slice = buf.as_mut_slice();
        slice[0] = 0xAB;
        slice[1] = 0xCD;
        slice[4095] = 0xEF;

        assert_eq!(buf.as_slice()[0], 0xAB);
        assert_eq!(buf.as_slice()[1], 0xCD);
        assert_eq!(buf.as_slice()[4095], 0xEF);
    }

    #[test]
    fn buf_mut_fill() {
        let mut buf = AlignedBuf::new_zeroed(512, SECTOR_SIZE_MIN);
        buf.as_mut_slice().fill(0xFF);
        assert!(buf.as_slice().iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn buf_ptrs() {
        let mut buf = AlignedBuf::new_zeroed(4096, SECTOR_SIZE_DEFAULT);
        let ptr = buf.as_ptr();
        assert_eq!(ptr, buf.as_mut_ptr() as *const u8);
        assert_eq!(ptr, buf.as_slice().as_ptr());
    }

    #[test]
    fn buf_slice_lengths() {
        let mut buf = AlignedBuf::new_zeroed(1234, SECTOR_SIZE_DEFAULT);
        assert_eq!(buf.as_slice().len(), 1234);
        assert_eq!(buf.as_mut_slice().len(), 1234);
    }

    #[test]
    fn buf_not_empty() {
        let buf = AlignedBuf::new_zeroed(1, SECTOR_SIZE_MIN);
        assert!(!buf.is_empty());
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

    // ==================== Read ====================

    #[test]
    fn read_init() {
        let read = Read::new();
        assert!(!read.is_pending());
        assert_eq!(read.expected_len(), 0);
        assert_eq!(read.zone, Zone::SuperBlock);
        assert_eq!(read.zone_spec.base, 0);
        assert_eq!(read.offset, 0);
    }

    #[test]
    fn read_default() {
        let r1 = Read::new();
        let r2 = Read::default();
        assert_eq!(r1.zone, r2.zone);
        assert_eq!(r1.offset, r2.offset);
    }

    #[test]
    fn read_pending() {
        let read = Read::new();
        assert!(!read.is_pending());
    }

    #[test]
    fn read_pending_true_when_callback_set() {
        fn cb(_: &mut Read) {}
        let mut read = Read::new();
        assert!(!read.is_pending());
        read.callback = Some(cb);
        assert!(read.is_pending());
        read.callback = None;
        assert!(!read.is_pending());
    }

    #[test]
    fn read_expected_len_accessors() {
        let mut read = Read::new();
        assert_eq!(read.expected_len(), 0);
        read.expected_len = 123;
        assert_eq!(read.expected_len(), 123);
    }

    #[test]
    fn read_io_starts_idle() {
        let read = Read::new();
        assert!(read.io.is_idle());
    }

    #[test]
    fn read_offset_logic() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec {
            base: 1000,
            size: 500,
        };

        // Start
        read.offset = 0;
        assert_eq!(read.absolute_offset(), 1000);

        // Middle
        read.offset = 250;
        assert_eq!(read.absolute_offset(), 1250);

        // Limit (inclusive of size for calculation, EOF check is separate)
        read.offset = 500;
        assert_eq!(read.absolute_offset(), 1500);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_offset_panic() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec {
            base: 1000,
            size: 500,
        };
        read.offset = 501;
        let _ = read.absolute_offset();
    }

    #[test]
    fn read_zero_size_zone_allows_only_zero_offset() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec { base: 42, size: 0 };
        read.offset = 0;
        assert_eq!(read.absolute_offset(), 42);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_zero_size_zone_panics_on_nonzero_offset() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec { base: 42, size: 0 };
        read.offset = 1;
        let _ = read.absolute_offset();
    }

    #[test]
    fn read_layout() {
        let layout = Layout::new(512, 4096, 512, 512, 512, 512, 4096);
        let mut read = Read::new();
        read.zone = Zone::WalPrepares;
        read.zone_spec = layout.zone(Zone::WalPrepares);
        read.offset = 100;
        assert_eq!(
            read.absolute_offset(),
            layout.start(Zone::WalPrepares) + 100
        );
    }

    // ==================== Write ====================

    #[test]
    fn write_init() {
        let write = Write::new();
        assert!(!write.is_pending());
        assert_eq!(write.expected_len(), 0);
        assert_eq!(write.zone, Zone::SuperBlock);
        assert_eq!(write.zone_spec.base, 0);
        assert_eq!(write.offset, 0);
    }

    #[test]
    fn write_default() {
        let w1 = Write::new();
        let w2 = Write::default();
        assert_eq!(w1.zone, w2.zone);
        assert_eq!(w1.offset, w2.offset);
    }

    #[test]
    fn write_pending() {
        let write = Write::new();
        assert!(!write.is_pending());
    }

    #[test]
    fn write_pending_true_when_callback_set() {
        fn cb(_: &mut Write) {}
        let mut write = Write::new();
        assert!(!write.is_pending());
        write.callback = Some(cb);
        assert!(write.is_pending());
        write.callback = None;
        assert!(!write.is_pending());
    }

    #[test]
    fn write_expected_len_accessors() {
        let mut write = Write::new();
        assert_eq!(write.expected_len(), 0);
        write.expected_len = 456;
        assert_eq!(write.expected_len(), 456);
    }

    #[test]
    fn write_io_starts_idle() {
        let write = Write::new();
        assert!(write.io.is_idle());
    }

    #[test]
    fn write_offset_logic() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec {
            base: 2000,
            size: 1000,
        };

        write.offset = 0;
        assert_eq!(write.absolute_offset(), 2000);

        write.offset = 500;
        assert_eq!(write.absolute_offset(), 2500);

        write.offset = 1000;
        assert_eq!(write.absolute_offset(), 3000);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_offset_panic() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec {
            base: 2000,
            size: 1000,
        };
        write.offset = 1001;
        let _ = write.absolute_offset();
    }

    #[test]
    fn write_zero_size_zone_allows_only_zero_offset() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec { base: 99, size: 0 };
        write.offset = 0;
        assert_eq!(write.absolute_offset(), 99);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_zero_size_zone_panics_on_nonzero_offset() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec { base: 99, size: 0 };
        write.offset = 1;
        let _ = write.absolute_offset();
    }

    #[test]
    fn write_layout() {
        let layout = Layout::new(512, 4096, 512, 512, 512, 512, 4096);
        let mut write = Write::new();
        write.zone = Zone::ClientReplies;
        write.zone_spec = layout.zone(Zone::ClientReplies);
        write.offset = 200;
        assert_eq!(
            write.absolute_offset(),
            layout.start(Zone::ClientReplies) + 200
        );
    }

    // ==================== Consistency ====================

    #[test]
    fn rw_consistency() {
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
    fn rw_all_zones() {
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

    #[test]
    fn read_io_field_is_first() {
        let offset = unsafe {
            let uninit = core::mem::MaybeUninit::<Read>::uninit();
            let base = uninit.as_ptr();
            let io_ptr = core::ptr::addr_of!((*base).io);
            (io_ptr as usize) - (base as usize)
        };
        assert_eq!(offset, 0);
    }

    #[test]
    fn write_io_field_is_first() {
        let offset = unsafe {
            let uninit = core::mem::MaybeUninit::<Write>::uninit();
            let base = uninit.as_ptr();
            let io_ptr = core::ptr::addr_of!((*base).io);
            (io_ptr as usize) - (base as usize)
        };
        assert_eq!(offset, 0);
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

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::io::{Completion, Operation};
    use std::cell::{Cell, RefCell};
    use std::fs::File;
    use std::io::{Read as IoRead, Seek, Write as IoWrite};
    use tempfile::tempdir;

    fn default_layout() -> Layout {
        Layout::new(512, 4096, 4096, 4096, 4096, 4096, 4096)
    }

    fn default_layout_4k() -> Layout {
        Layout::new(
            SECTOR_SIZE_DEFAULT as u64,
            SECTOR_SIZE_DEFAULT as u64,
            SECTOR_SIZE_DEFAULT as u64,
            SECTOR_SIZE_DEFAULT as u64,
            SECTOR_SIZE_DEFAULT as u64,
            SECTOR_SIZE_DEFAULT as u64,
            SECTOR_SIZE_DEFAULT as u64,
        )
    }

    #[test]
    fn open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_basic");
        let layout = default_layout();

        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };

        let storage = Storage::open(opts).unwrap();
        assert_eq!(storage.sector_size(), 512);
        assert_eq!(storage.size(), layout.total_size());
        assert_eq!(
            storage.file().metadata().unwrap().len(),
            layout.total_size()
        );
    }

    #[test]
    fn open_fd_zero() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_fd_zero");

        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };

        let mut storage = Storage::open(opts).unwrap();
        storage.fd = 0;
        storage.assert_invariants();
    }

    #[test]
    fn open_defaults() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_default_sector_size");
        let layout = default_layout_4k();

        let opts = Options {
            path: &path,
            layout,
            direct_io: false,
            sector_size: 0,
        };

        let storage = Storage::open(opts).unwrap();
        assert_eq!(storage.sector_size(), SECTOR_SIZE_DEFAULT);
    }

    #[test]
    fn priority_vsr_over_lsm() {
        thread_local! {
            static EXECUTION_ORDER: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
        }

        fn cb_vsr(_: &mut NextTick) {
            EXECUTION_ORDER.with(|o| o.borrow_mut().push("VSR"));
        }

        fn cb_lsm(_: &mut NextTick) {
            EXECUTION_ORDER.with(|o| o.borrow_mut().push("LSM"));
        }

        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_tick_order");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        let mut nt_vsr = NextTick::new();
        let mut nt_lsm = NextTick::new();

        // Queue LSM first, then VSR.
        storage.on_next_tick(NextTickQueue::Lsm, cb_lsm, &mut nt_lsm);
        storage.on_next_tick(NextTickQueue::Vsr, cb_vsr, &mut nt_vsr);

        storage.run();

        EXECUTION_ORDER.with(|o| {
            assert_eq!(*o.borrow(), vec!["VSR", "LSM"]);
            o.borrow_mut().clear();
        });
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn double_schedule_panic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_double_schedule");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        fn cb(_: &mut NextTick) {}

        let mut next_tick = NextTick::new();
        storage.on_next_tick(NextTickQueue::Vsr, cb, &mut next_tick);
        storage.on_next_tick(NextTickQueue::Vsr, cb, &mut next_tick);
    }

    #[test]
    fn reset_lsm() {
        thread_local! {
            static EXECUTION_ORDER: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
        }

        fn cb_vsr(_: &mut NextTick) {
            EXECUTION_ORDER.with(|o| o.borrow_mut().push("VSR"));
        }

        fn cb_lsm(_: &mut NextTick) {
            EXECUTION_ORDER.with(|o| o.borrow_mut().push("LSM"));
        }

        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_reset_lsm");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        let mut nt_vsr = NextTick::new();
        let mut nt_lsm = NextTick::new();
        storage.on_next_tick(NextTickQueue::Vsr, cb_vsr, &mut nt_vsr);
        storage.on_next_tick(NextTickQueue::Lsm, cb_lsm, &mut nt_lsm);

        storage.reset_next_tick_lsm();
        assert!(!nt_lsm.is_pending());

        storage.run();

        EXECUTION_ORDER.with(|o| {
            assert_eq!(*o.borrow(), vec!["VSR"]);
            o.borrow_mut().clear();
        });

        assert!(!nt_vsr.is_pending());
        assert!(!nt_lsm.is_pending());
    }

    #[test]
    fn requeue_next_tick() {
        thread_local! {
             static EXECUTION_COUNT: RefCell<u32> = const { RefCell::new(0) };
             static STORAGE_PTR: RefCell<Option<*mut Storage>> = const { RefCell::new(None) };
        }

        fn cb_requeue(nt: &mut NextTick) {
            EXECUTION_COUNT.with(|c| *c.borrow_mut() += 1);

            STORAGE_PTR.with(|s| {
                if let Some(storage_ptr) = *s.borrow() {
                    let storage = unsafe { &mut *storage_ptr };
                    // Safe because nt is detached from queue during callback
                    storage.on_next_tick(NextTickQueue::Vsr, cb_requeue, nt);
                }
            });
        }

        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_requeue");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut nt = NextTick::new();

        STORAGE_PTR.with(|s| *s.borrow_mut() = Some(&mut storage as *mut Storage));

        storage.on_next_tick(NextTickQueue::Vsr, cb_requeue, &mut nt);

        // First run: executes once, re-queues.
        storage.run();
        EXECUTION_COUNT.with(|c| assert_eq!(*c.borrow(), 1));

        // Second run: executes again.
        storage.run();
        EXECUTION_COUNT.with(|c| assert_eq!(*c.borrow(), 2));

        STORAGE_PTR.with(|s| *s.borrow_mut() = None);
        EXECUTION_COUNT.with(|c| *c.borrow_mut() = 0);
    }

    #[test]
    fn io_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_io_read");
        let layout = default_layout();

        {
            let mut f = File::create(&path).unwrap();
            f.set_len(layout.total_size()).unwrap();
            // Write something at WalPrepares (zone 2).
            // Zone 0: 4096, 1: 4096. Zone 2 starts at 8192.
            f.seek(std::io::SeekFrom::Start(8192)).unwrap();
            f.write_all(&[0x42; 512]).unwrap();
        }

        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        let mut read = Read::new();
        read.zone = Zone::WalPrepares;
        read.zone_spec = layout.zone(Zone::WalPrepares);
        read.offset = 0;
        read.expected_len = 512;

        let mut buf = AlignedBuf::new_zeroed(512, 512);

        thread_local! {
            static READ_DONE: RefCell<bool> = const { RefCell::new(false) };
        }

        fn cb(_: &mut Read) {
            READ_DONE.with(|b| *b.borrow_mut() = true);
        }
        read.callback = Some(cb);

        let abs_offset = read.absolute_offset();

        // Manual shim to bridge IoCompletion -> Read callback
        unsafe fn read_shim(ctx: *mut core::ffi::c_void, _comp: &mut Completion) {
            // SAFETY: ctx is *mut Read
            unsafe {
                let read = &mut *(ctx as *mut Read);
                if let Some(cb) = read.callback.take() {
                    cb(read);
                }
            }
        }

        let ctx = &mut read as *mut Read as *mut core::ffi::c_void;
        let fd = storage.fd;

        storage.io().read(
            &mut read.io,
            fd,
            buf.as_mut_ptr(),
            512,
            abs_offset,
            ctx,
            read_shim,
        );

        // Run tick to drive IO
        // Note: Io::tick waits for at least one completion if inflight > 0.
        storage.tick();

        READ_DONE.with(|b| assert!(*b.borrow()));
        assert_eq!(buf.as_slice()[0], 0x42);

        assert_eq!(read.io.result, 512);
        assert!(read.io.is_idle());
        assert!(!read.is_pending());

        READ_DONE.with(|b| *b.borrow_mut() = false);
    }

    #[test]
    fn io_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_io_write");
        let layout = default_layout();

        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        let mut write = Write::new();
        write.zone = Zone::WalPrepares;
        write.zone_spec = layout.zone(Zone::WalPrepares);
        write.offset = 0;
        write.expected_len = 512;

        let mut buf = AlignedBuf::new_zeroed(512, 512);
        buf.as_mut_slice().fill(0xA5);

        thread_local! {
            static WRITE_DONE: Cell<bool> = const { Cell::new(false) };
        }

        fn cb(_: &mut Write) {
            WRITE_DONE.with(|b| b.set(true));
        }
        write.callback = Some(cb);

        let abs_offset = write.absolute_offset();

        unsafe fn write_shim(ctx: *mut core::ffi::c_void, _comp: &mut Completion) {
            // SAFETY: ctx is *mut Write
            unsafe {
                let write = &mut *(ctx as *mut Write);
                if let Some(cb) = write.callback.take() {
                    cb(write);
                }
            }
        }

        let ctx = &mut write as *mut Write as *mut core::ffi::c_void;
        let fd = storage.fd;

        storage.io().write(
            &mut write.io,
            fd,
            buf.as_ptr(),
            512,
            abs_offset,
            ctx,
            write_shim,
        );
        storage.tick();

        WRITE_DONE.with(|b| assert!(b.get()));

        assert_eq!(write.io.result, 512);
        assert!(write.io.is_idle());
        assert!(!write.is_pending());

        let mut f = File::open(&path).unwrap();
        f.seek(std::io::SeekFrom::Start(abs_offset)).unwrap();
        let mut read_back = vec![0u8; 512];
        f.read_exact(&mut read_back).unwrap();
        assert!(read_back.iter().all(|&b| b == 0xA5));

        WRITE_DONE.with(|b| b.set(false));
    }

    #[test]
    fn io_defers_next_tick() {
        thread_local! {
            static IO_DONE: Cell<bool> = const { Cell::new(false) };
            static NEXT_TICK_DONE: Cell<bool> = const { Cell::new(false) };
        }

        fn next_tick_cb(_: &mut NextTick) {
            NEXT_TICK_DONE.with(|b| b.set(true));
        }

        struct IoCtx {
            queue: *mut Queue<NextTick, NextTickTag>,
            next_tick: *mut NextTick,
        }

        unsafe fn io_cb(ctx: *mut core::ffi::c_void, _comp: &mut Completion) {
            IO_DONE.with(|b| b.set(true));

            // SAFETY: ctx is a valid `IoCtx*` for the duration of the I/O op.
            unsafe {
                let ctx = &mut *(ctx as *mut IoCtx);
                let next_tick = &mut *ctx.next_tick;
                assert!(!next_tick.is_pending());

                next_tick.callback = Some(next_tick_cb);
                let queue = &mut *ctx.queue;
                queue.push(next_tick);
            }
        }

        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_io_schedules_next_tick");
        let layout = default_layout();

        {
            let mut f = File::create(&path).unwrap();
            f.set_len(layout.total_size()).unwrap();
            f.seek(std::io::SeekFrom::Start(layout.start(Zone::WalPrepares)))
                .unwrap();
            f.write_all(&[0x11; 512]).unwrap();
        }

        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        let mut nt = NextTick::new();
        let queue_ptr: *mut Queue<NextTick, NextTickTag> = &mut storage.next_tick_vsr;
        let mut ctx = IoCtx {
            queue: queue_ptr,
            next_tick: &mut nt,
        };

        let mut read = Read::new();
        read.zone = Zone::WalPrepares;
        read.zone_spec = layout.zone(Zone::WalPrepares);
        read.offset = 0;
        read.expected_len = 512;

        let mut buf = AlignedBuf::new_zeroed(512, 512);
        let abs_offset = read.absolute_offset();

        let fd = storage.fd;
        storage.io().read(
            &mut read.io,
            fd,
            buf.as_mut_ptr(),
            512,
            abs_offset,
            &mut ctx as *mut IoCtx as *mut core::ffi::c_void,
            io_cb,
        );

        // First run: completes the I/O and queues the next-tick, but must not run it.
        storage.run();
        IO_DONE.with(|b| assert!(b.get()));
        NEXT_TICK_DONE.with(|b| assert!(!b.get()));
        assert!(nt.is_pending());

        // Second run: drains the queued next-tick.
        storage.run();
        NEXT_TICK_DONE.with(|b| assert!(b.get()));
        assert!(!nt.is_pending());

        IO_DONE.with(|b| b.set(false));
        NEXT_TICK_DONE.with(|b| b.set(false));
    }
}
