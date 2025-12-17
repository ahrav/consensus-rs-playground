//! Storage primitives for direct I/O with sector-aligned buffers.
//!
//! Provides [`AlignedBuf`] for allocating memory that satisfies direct I/O
//! alignment requirements, enabling O_DIRECT reads/writes without kernel
//! buffering.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::fs::{File, OpenOptions};
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::Path;

use libc;

use crate::io::{Completion as IoCompletion, Io, Operation};
use crate::stdx::queue::Queue;

use super::constants::{
    IO_ENTRIES_DEFAULT, MAX_NEXT_TICK_ITERATIONS, SECTOR_SIZE_DEFAULT, SECTOR_SIZE_MAX,
    SECTOR_SIZE_MIN,
};
use super::iocb::{NextTick, NextTickQueue, NextTickTag, Read, Synchronicity, Write};
pub use super::layout::{Layout, Zone, ZoneSpec};

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

    /// Submits async read from `zone` at zone-relative `offset`.
    ///
    /// Translates to absolute file offset and validates sector alignment for O_DIRECT.
    /// `read` must outlive the operation; `callback` invoked on completion.
    ///
    /// # Panics
    ///
    /// - `buffer.is_empty()` or `read.is_pending()`
    /// - Buffer pointer, length, or offset not sector-aligned
    /// - Read exceeds zone boundary
    pub fn read_sectors(
        &mut self,
        callback: fn(&mut Read),
        read: &mut Read,
        buffer: &mut [u8],
        zone: Zone,
        offset: u64,
    ) {
        self.assert_invariants();
        assert!(!buffer.is_empty());
        assert!(!read.is_pending());

        self.assert_buffer_aligned(buffer.as_ptr() as usize, buffer.len(), offset);

        let zone_spec = self.layout.zone(zone);
        assert!(zone_spec.size > 0);
        assert!(offset < zone_spec.size);
        assert!(buffer.len() as u64 <= zone_spec.size - offset);

        let abs = zone_spec.base.checked_add(offset).expect("offset overflow");
        assert!(abs >= zone_spec.base);
        assert!(abs < zone_spec.base + zone_spec.size);

        read.callback = Some(callback);
        read.expected_len = buffer.len();
        read.zone = zone;
        read.zone_spec = zone_spec;
        read.offset = offset;

        assert!(read.is_pending());
        assert_eq!(read.expected_len, buffer.len());
        assert_eq!(read.absolute_offset(), abs);

        let ctx = (read as *mut Read).cast::<c_void>();
        let buf = NonNull::new(buffer.as_mut_ptr()).expect("buffer pointer is null");
        self.io.submit(
            &mut read.io,
            Operation::Read {
                fd: self.fd,
                buf,
                len: buffer.len() as u32,
                offset: abs,
            },
            ctx,
            read_complete_trampoline,
        );
    }

    /// Submits async write to `zone` at zone-relative `offset`.
    ///
    /// Translates to absolute file offset and validates sector alignment for O_DIRECT.
    /// `write` must outlive the operation; `callback` invoked on completion.
    /// `buffer` is const but cast to `*mut u8` for io_uring API (kernel doesn't modify).
    ///
    /// # Panics
    ///
    /// - `buffer.is_empty()` or `write.is_pending()`
    /// - Buffer pointer, length, or offset not sector-aligned
    /// - Write exceeds zone boundary
    pub fn write_sectors(
        &mut self,
        callback: fn(&mut Write),
        write: &mut Write,
        buffer: &[u8],
        zone: Zone,
        offset: u64,
    ) {
        self.assert_invariants();
        assert!(!buffer.is_empty());
        assert!(!write.is_pending());

        self.assert_buffer_aligned(buffer.as_ptr() as usize, buffer.len(), offset);

        let zone_spec = self.layout.zone(zone);
        assert!(zone_spec.size > 0);
        assert!(offset < zone_spec.size);
        assert!(buffer.len() as u64 <= zone_spec.size - offset);

        let abs = zone_spec.base.checked_add(offset).expect("offset overflow");
        assert!(abs >= zone_spec.base);
        assert!(abs < zone_spec.base + zone_spec.size);

        write.callback = Some(callback);
        write.expected_len = buffer.len();
        write.zone = zone;
        write.zone_spec = zone_spec;
        write.offset = offset;

        assert!(write.is_pending());
        assert_eq!(write.expected_len, buffer.len());
        assert_eq!(write.absolute_offset(), abs);

        let ctx = (write as *mut Write).cast::<c_void>();
        let buf = NonNull::new(buffer.as_ptr() as *mut u8).expect("buffer pointer is null");
        self.io.submit(
            &mut write.io,
            Operation::Write {
                fd: self.fd,
                buf,
                len: buffer.len() as u32,
                offset: abs,
            },
            ctx,
            write_complete_trampoline,
        );
    }

    /// Flushes all writes to stable storage (durability barrier).
    ///
    /// Generic `ctx`/`cb` pattern allows caller-defined completion context.
    pub fn fsync(
        &mut self,
        completion: &mut IoCompletion,
        ctx: *mut c_void,
        cb: unsafe fn(*mut c_void, &mut IoCompletion),
    ) {
        self.assert_invariants();
        assert!(self.fd >= 0);

        self.io
            .submit(completion, Operation::Fsync { fd: self.fd }, ctx, cb);
    }

    /// Validates alignment for O_DIRECT: buffer pointer, length, and offset must
    /// be sector-aligned. `isize::MAX` check prevents kernel pointer arithmetic overflow.
    fn assert_buffer_aligned(&self, buf_ptr: usize, len: usize, offset: u64) {
        assert!(len.is_multiple_of(self.sector_size()));
        assert!(offset.is_multiple_of(self.sector_size() as u64));
        assert!(buf_ptr.is_multiple_of(self.sector_size()));

        assert!(len > 0);
        assert!(len <= isize::MAX as usize);
    }
}

/// C callback trampoline: casts `ctx` to [`Read`], validates result, invokes typed callback.
///
/// Panics on I/O errors or short reads (fail-stop: storage corruption is unrecoverable).
///
/// # Safety
///
/// `ctx` must point to valid [`Read`] that outlives the operation.
unsafe fn read_complete_trampoline(ctx: *mut c_void, completion: &mut IoCompletion) {
    assert!(!ctx.is_null());

    let read = unsafe { &mut *(ctx as *mut Read) };
    if completion.result < 0 {
        let errno = -completion.result;
        panic!(
            "storage read failed: errno={errno} (zone={:?}, offset={}, expected_len={})",
            read.zone, read.offset, read.expected_len
        );
    }

    let n = completion.result as usize;
    if n != read.expected_len {
        panic!(
            "storage read failed: short read {n} != {} (zone={:?}, offset={}, expected_len={})",
            read.expected_len, read.zone, read.offset, read.expected_len
        );
    }

    if let Some(cb) = read.callback.take() {
        assert!(!read.is_pending());
        cb(read);
    }
}

/// C callback trampoline: casts `ctx` to [`Write`], validates result, invokes typed callback.
///
/// Panics on I/O errors or short writes (fail-stop: storage corruption is unrecoverable).
///
/// # Safety
///
/// `ctx` must point to valid [`Write`] that outlives the operation.
unsafe fn write_complete_trampoline(ctx: *mut c_void, completion: &mut IoCompletion) {
    assert!(!ctx.is_null());

    let write = unsafe { &mut *(ctx as *mut Write) };
    if completion.result < 0 {
        let errno = -completion.result;
        panic!(
            "storage write failed: errno={errno} (zone={:?}, offset={}, expected_len={})",
            write.zone, write.offset, write.expected_len
        );
    }

    let n = completion.result as usize;
    if n != write.expected_len {
        panic!(
            "storage write failed: short write {n} != {} (zone={:?}, offset={}, expected_len={})",
            write.expected_len, write.zone, write.offset, write.expected_len
        );
    }

    if let Some(cb) = write.callback.take() {
        assert!(!write.is_pending());
        cb(write);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::io::Completion;
    use crate::storage::buffer::AlignedBuf;
    use std::cell::{Cell, RefCell};
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

    #[test]
    fn read_sectors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_read_sectors");
        let layout = default_layout();

        // Setup file content
        {
            let mut f = File::create(&path).unwrap();
            f.set_len(layout.total_size()).unwrap();
            f.seek(std::io::SeekFrom::Start(layout.start(Zone::WalPrepares)))
                .unwrap();
            f.write_all(&[0xBB; 512]).unwrap();
        }

        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        let mut read = Read::new();
        let mut buf = AlignedBuf::new_zeroed(512, 512);

        thread_local! {
            static DONE: Cell<bool> = const { Cell::new(false) };
        }
        fn cb(_: &mut Read) {
            DONE.with(|b| b.set(true));
        }

        storage.read_sectors(cb, &mut read, buf.as_mut_slice(), Zone::WalPrepares, 0);
        storage.tick();

        DONE.with(|b| assert!(b.get()));
        assert_eq!(buf.as_slice()[0], 0xBB);
        assert!(!read.is_pending());
    }

    #[test]
    fn write_sectors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_write_sectors");
        let layout = default_layout();

        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        let mut write = Write::new();
        let mut buf = AlignedBuf::new_zeroed(512, 512);
        buf.as_mut_slice().fill(0xCC);

        thread_local! {
            static DONE: Cell<bool> = const { Cell::new(false) };
        }
        fn cb(_: &mut Write) {
            DONE.with(|b| b.set(true));
        }

        storage.write_sectors(cb, &mut write, buf.as_slice(), Zone::WalPrepares, 0);
        storage.tick();

        DONE.with(|b| assert!(b.get()));
        assert!(!write.is_pending());

        // Verify content
        let mut f = File::open(&path).unwrap();
        f.seek(std::io::SeekFrom::Start(layout.start(Zone::WalPrepares)))
            .unwrap();
        let mut check = [0u8; 512];
        f.read_exact(&mut check).unwrap();
        assert!(check.iter().all(|&b| b == 0xCC));
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_sectors_unaligned_buffer() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_read_unaligned");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut read = Read::new();

        // Unaligned slice (offset by 1)
        let mut raw = vec![0u8; 512 + 1];
        let slice = &mut raw[1..];

        storage.read_sectors(|_| {}, &mut read, slice, Zone::WalPrepares, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_sectors_unaligned_buffer() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_write_unaligned");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut write = Write::new();

        let raw = vec![0u8; 512 + 1];
        let slice = &raw[1..];

        storage.write_sectors(|_| {}, &mut write, slice, Zone::WalPrepares, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_sectors_unaligned_len() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_read_len");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut read = Read::new();
        let mut buf = AlignedBuf::new_zeroed(1024, 512); // aligned

        // Valid ptr, invalid len
        storage.read_sectors(
            |_| {},
            &mut read,
            &mut buf.as_mut_slice()[..511],
            Zone::WalPrepares,
            0,
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_sectors_unaligned_offset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_read_offset");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut read = Read::new();
        let mut buf = AlignedBuf::new_zeroed(512, 512);

        // Invalid offset
        storage.read_sectors(|_| {}, &mut read, buf.as_mut_slice(), Zone::WalPrepares, 1);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_sectors_out_of_bounds() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_read_oob");
        let layout = default_layout(); // WalPrepares is 4096 bytes
        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut read = Read::new();
        let mut buf = AlignedBuf::new_zeroed(512, 512);

        // Offset 4096 is exactly at end, but len 512 pushes it over
        storage.read_sectors(
            |_| {},
            &mut read,
            buf.as_mut_slice(),
            Zone::WalPrepares,
            4096,
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_sectors_len_overflow() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_read_len_overflow");
        let layout = default_layout();
        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut read = Read::new();
        // 1024 bytes buffer
        let mut buf = AlignedBuf::new_zeroed(1024, 512);

        // Offset 3584 (4096 - 512). Reading 1024 bytes goes past end (4096 + 512).
        storage.read_sectors(
            |_| {},
            &mut read,
            buf.as_mut_slice(),
            Zone::WalPrepares,
            3584,
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_sectors_out_of_bounds() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_write_oob");
        let layout = default_layout();
        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut write = Write::new();
        let buf = AlignedBuf::new_zeroed(512, 512);

        storage.write_sectors(|_| {}, &mut write, buf.as_slice(), Zone::WalPrepares, 4096);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_sectors_empty_buffer_panics() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_read_empty");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut read = Read::new();
        let empty: &mut [u8] = &mut [];

        storage.read_sectors(|_| {}, &mut read, empty, Zone::WalPrepares, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_sectors_empty_buffer_panics() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_write_empty");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut write = Write::new();
        let empty: &[u8] = &[];

        storage.write_sectors(|_| {}, &mut write, empty, Zone::WalPrepares, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_sectors_pending_panics() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_read_pending");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut read = Read::new();
        // Force pending state
        read.callback = Some(|_: &mut Read| {});
        assert!(read.is_pending());

        let mut buf = AlignedBuf::new_zeroed(512, 512);
        storage.read_sectors(|_| {}, &mut read, buf.as_mut_slice(), Zone::WalPrepares, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_sectors_pending_panics() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_write_pending");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut write = Write::new();
        // Force pending state
        write.callback = Some(|_: &mut Write| {});
        assert!(write.is_pending());

        let buf = AlignedBuf::new_zeroed(512, 512);
        storage.write_sectors(|_| {}, &mut write, buf.as_slice(), Zone::WalPrepares, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_sectors_unaligned_len() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_write_len");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut write = Write::new();
        let buf = AlignedBuf::new_zeroed(1024, 512);

        // Valid ptr, invalid len (511 not multiple of 512)
        storage.write_sectors(
            |_| {},
            &mut write,
            &buf.as_slice()[..511],
            Zone::WalPrepares,
            0,
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_sectors_unaligned_offset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_write_offset");
        let opts = Options {
            path: &path,
            layout: default_layout(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut write = Write::new();
        let buf = AlignedBuf::new_zeroed(512, 512);

        // Invalid offset (1 not multiple of 512)
        storage.write_sectors(|_| {}, &mut write, buf.as_slice(), Zone::WalPrepares, 1);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_sectors_len_overflow() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_write_len_overflow");
        let layout = default_layout();
        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();
        let mut write = Write::new();
        // 1024 bytes buffer
        let buf = AlignedBuf::new_zeroed(1024, 512);

        // Offset 3584 (4096 - 512). Writing 1024 bytes goes past end.
        storage.write_sectors(|_| {}, &mut write, buf.as_slice(), Zone::WalPrepares, 3584);
    }

    #[test]
    fn read_write_roundtrip_data_integrity() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_roundtrip");
        let layout = default_layout();
        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        // Write known pattern
        let mut write = Write::new();
        let mut write_buf = AlignedBuf::new_zeroed(512, 512);
        for (i, b) in write_buf.as_mut_slice().iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }

        thread_local! {
            static WRITE_DONE: Cell<bool> = const { Cell::new(false) };
        }
        fn write_cb(_: &mut Write) {
            WRITE_DONE.with(|b| b.set(true));
        }

        storage.write_sectors(
            write_cb,
            &mut write,
            write_buf.as_slice(),
            Zone::WalPrepares,
            512,
        );
        storage.tick();
        WRITE_DONE.with(|b| assert!(b.get()));

        // Read it back
        let mut read = Read::new();
        let mut read_buf = AlignedBuf::new_zeroed(512, 512);

        thread_local! {
            static READ_DONE: Cell<bool> = const { Cell::new(false) };
        }
        fn read_cb(_: &mut Read) {
            READ_DONE.with(|b| b.set(true));
        }

        storage.read_sectors(
            read_cb,
            &mut read,
            read_buf.as_mut_slice(),
            Zone::WalPrepares,
            512,
        );
        storage.tick();
        READ_DONE.with(|b| assert!(b.get()));

        // Verify data integrity
        for (i, &b) in read_buf.as_slice().iter().enumerate() {
            assert_eq!(b, (i % 256) as u8, "data mismatch at byte {i}");
        }
    }

    #[test]
    fn read_sectors_exact_zone_fill() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_exact_fill");
        let layout = default_layout();

        // Pre-fill the zone
        {
            let mut f = File::create(&path).unwrap();
            f.set_len(layout.total_size()).unwrap();
            f.seek(std::io::SeekFrom::Start(layout.start(Zone::WalPrepares)))
                .unwrap();
            let zone_size = layout.zone(Zone::WalPrepares).size as usize;
            f.write_all(&vec![0xEE; zone_size]).unwrap();
        }

        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        let zone_size = layout.zone(Zone::WalPrepares).size as usize;
        let mut read = Read::new();
        let mut buf = AlignedBuf::new_zeroed(zone_size, 512);

        thread_local! {
            static DONE: Cell<bool> = const { Cell::new(false) };
        }
        fn cb(_: &mut Read) {
            DONE.with(|b| b.set(true));
        }

        // Read exactly zone.size bytes starting at offset 0
        storage.read_sectors(cb, &mut read, buf.as_mut_slice(), Zone::WalPrepares, 0);
        storage.tick();

        DONE.with(|b| assert!(b.get()));
        assert!(buf.as_slice().iter().all(|&b| b == 0xEE));
    }

    #[test]
    fn read_sectors_last_sector() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_last_sector");
        let layout = default_layout();
        let zone_size = layout.zone(Zone::WalPrepares).size;
        let last_sector_offset = zone_size - 512;

        // Pre-fill last sector
        {
            let mut f = File::create(&path).unwrap();
            f.set_len(layout.total_size()).unwrap();
            let abs_offset = layout.start(Zone::WalPrepares) + last_sector_offset;
            f.seek(std::io::SeekFrom::Start(abs_offset)).unwrap();
            f.write_all(&[0xFF; 512]).unwrap();
        }

        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        let mut read = Read::new();
        let mut buf = AlignedBuf::new_zeroed(512, 512);

        thread_local! {
            static DONE: Cell<bool> = const { Cell::new(false) };
        }
        fn cb(_: &mut Read) {
            DONE.with(|b| b.set(true));
        }

        // Read last sector of zone
        storage.read_sectors(
            cb,
            &mut read,
            buf.as_mut_slice(),
            Zone::WalPrepares,
            last_sector_offset,
        );
        storage.tick();

        DONE.with(|b| assert!(b.get()));
        assert!(buf.as_slice().iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn write_sectors_last_sector() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_write_last");
        let layout = default_layout();
        let zone_size = layout.zone(Zone::WalPrepares).size;
        let last_sector_offset = zone_size - 512;

        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        let mut write = Write::new();
        let mut buf = AlignedBuf::new_zeroed(512, 512);
        buf.as_mut_slice().fill(0xAA);

        thread_local! {
            static DONE: Cell<bool> = const { Cell::new(false) };
        }
        fn cb(_: &mut Write) {
            DONE.with(|b| b.set(true));
        }

        // Write last sector of zone
        storage.write_sectors(
            cb,
            &mut write,
            buf.as_slice(),
            Zone::WalPrepares,
            last_sector_offset,
        );
        storage.tick();

        DONE.with(|b| assert!(b.get()));

        // Verify
        let mut f = File::open(&path).unwrap();
        let abs_offset = layout.start(Zone::WalPrepares) + last_sector_offset;
        f.seek(std::io::SeekFrom::Start(abs_offset)).unwrap();
        let mut check = [0u8; 512];
        f.read_exact(&mut check).unwrap();
        assert!(check.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn read_write_multiple_zones() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_multi_zone");
        let layout = default_layout();

        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        // Test zones with non-zero size
        let test_zones = [
            Zone::SuperBlock,
            Zone::WalHeaders,
            Zone::WalPrepares,
            Zone::ClientReplies,
            Zone::Grid,
        ];

        for (idx, &zone) in test_zones.iter().enumerate() {
            let zone_spec = layout.zone(zone);
            if zone_spec.size == 0 {
                continue;
            }

            // Write unique pattern per zone
            let pattern = ((idx + 1) * 0x11) as u8;
            let mut write = Write::new();
            let mut buf = AlignedBuf::new_zeroed(512, 512);
            buf.as_mut_slice().fill(pattern);

            thread_local! {
                static DONE: Cell<bool> = const { Cell::new(false) };
            }
            fn cb_w(_: &mut Write) {
                DONE.with(|b| b.set(true));
            }

            DONE.with(|b| b.set(false));
            storage.write_sectors(cb_w, &mut write, buf.as_slice(), zone, 0);
            storage.tick();
            DONE.with(|b| assert!(b.get(), "write failed for zone {:?}", zone));

            // Read back and verify
            let mut read = Read::new();
            let mut read_buf = AlignedBuf::new_zeroed(512, 512);

            fn cb_r(_: &mut Read) {
                DONE.with(|b| b.set(true));
            }

            DONE.with(|b| b.set(false));
            storage.read_sectors(cb_r, &mut read, read_buf.as_mut_slice(), zone, 0);
            storage.tick();
            DONE.with(|b| assert!(b.get(), "read failed for zone {:?}", zone));

            assert!(
                read_buf.as_slice().iter().all(|&b| b == pattern),
                "data mismatch for zone {:?}",
                zone
            );
        }
    }

    #[test]
    fn zone_isolation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("storage_isolation");
        let layout = default_layout();

        // Pre-fill WalHeaders with 0xAA
        {
            let mut f = File::create(&path).unwrap();
            f.set_len(layout.total_size()).unwrap();
            f.seek(std::io::SeekFrom::Start(layout.start(Zone::WalHeaders)))
                .unwrap();
            f.write_all(&[0xAA; 512]).unwrap();
        }

        let opts = Options {
            path: &path,
            layout: layout.clone(),
            direct_io: false,
            sector_size: 512,
        };
        let mut storage = Storage::open(opts).unwrap();

        // Write 0xBB to WalPrepares
        let mut write = Write::new();
        let mut buf = AlignedBuf::new_zeroed(512, 512);
        buf.as_mut_slice().fill(0xBB);

        thread_local! {
            static DONE: Cell<bool> = const { Cell::new(false) };
        }
        fn cb_w(_: &mut Write) {
            DONE.with(|b| b.set(true));
        }

        storage.write_sectors(cb_w, &mut write, buf.as_slice(), Zone::WalPrepares, 0);
        storage.tick();
        DONE.with(|b| assert!(b.get()));

        // Verify WalHeaders is still 0xAA (not corrupted by WalPrepares write)
        let mut read = Read::new();
        let mut read_buf = AlignedBuf::new_zeroed(512, 512);

        fn cb_r(_: &mut Read) {
            DONE.with(|b| b.set(true));
        }

        DONE.with(|b| b.set(false));
        storage.read_sectors(
            cb_r,
            &mut read,
            read_buf.as_mut_slice(),
            Zone::WalHeaders,
            0,
        );
        storage.tick();
        DONE.with(|b| assert!(b.get()));

        assert!(
            read_buf.as_slice().iter().all(|&b| b == 0xAA),
            "WalHeaders was corrupted by WalPrepares write"
        );
    }
}
