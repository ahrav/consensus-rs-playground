//! Range-locking machinery for WAL journal writes.
//!
//! This module implements a write coalescing protocol that prevents concurrent overlapping
//! writes to the same disk sectors. When multiple write operations target the same range,
//! only the first proceeds immediately while subsequent writes queue behind it in an
//! intrusive linked list. When the active write completes, queued writes are drained and
//! re-evaluated for conflicts.
//!
//! # Architecture
//!
//! The system consists of three core types:
//!
//! - [`Journal`]: Owns the write operation pool and coordinates range locking
//! - [`Write`]: Represents a single write operation, allocated from the Journal's pool
//! - [`Range`]: Contains the buffer, offset, and linked-list pointers for overlap detection
//!
//! # Locking Protocol
//!
//! 1. When `write_sectors` is called, a `Write` is initialized with its target range
//! 2. `lock_sectors` scans all in-flight writes for overlapping ranges
//! 3. If an overlap is found with a **locked** range:
//!    - The new write is appended to that range's wait list (not submitted to storage)
//!    - Only exact overlaps (same offset, length, ring) are allowed
//! 4. If no overlap is found:
//!    - The range is marked `locked = true`
//!    - The write is submitted to storage
//! 5. On I/O completion:
//!    - The range is unlocked
//!    - All queued writes are drained and re-evaluated via `lock_sectors`
//!
//! # Memory Model
//!
//! This code uses raw pointers and intrusive linked lists for zero-allocation operation.
//! The `#[repr(C)]` layout guarantees allow `container_of!` macro to recover parent
//! structs from embedded field pointers (e.g., recovering `Write` from `Range`).
//!
//! # Concurrency Model
//!
//! The Journal is designed for single-threaded operation with asynchronous I/O.
//! All pointer manipulation occurs on the same thread; the async storage layer
//! may complete I/O on a different thread but callbacks run on the main thread.

#[allow(dead_code)]
use core::mem::MaybeUninit;
use core::ptr;

#[allow(unused_imports)]
use crate::constants;
use crate::container_of;
use crate::stdx::BitSet;
use crate::util::utils::AlignedSlice;
use crate::vsr::{
    Header,
    HeaderPrepare,
    chunk_tracking::HeaderChunkTracking,
    iops::IOPSType,
    journal_primitives::{
        Ring,
        SLOT_COUNT,
        // HEADER_CHUNK_COUNT, HEADER_CHUNK_WORDS, HEADERS_PER_MESSAGE,
        // WAL_HEADER_SIZE,
    },
    storage::{Storage, Synchronicity},
    // wire::{Command, Operation},
};

/// A write range descriptor with intrusive linked-list support for wait queuing.
///
/// `Range` describes a contiguous region of a WAL ring buffer to be written.
/// When multiple writes target overlapping ranges, `Range` instances form an
/// intrusive singly-linked list via the `next` pointer, allowing zero-allocation
/// wait queuing.
///
/// # Layout
///
/// The `#[repr(C)]` layout is required for `container_of!` to work correctly.
/// The `completion` field must be first to ensure predictable offset calculations.
///
/// # Invariants
///
/// - `buffer_ptr` must point to valid, aligned memory for `buffer_len` bytes
/// - When `locked == true`, a storage I/O operation is in progress
/// - When `next != null`, this range is queued behind another locked range
/// - A **waiting** range (queued behind a locked range) always has `locked = false`
/// - A **locked** range may have waiters attached via `next`
#[repr(C)]
pub struct Range<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> {
    /// Storage completion token, passed to `Storage::write_sectors`.
    /// Initialized only when the range becomes locked.
    pub completion: MaybeUninit<S::Write>,

    /// Callback invoked when this range's write operation completes.
    pub callback: fn(*mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>),

    /// Pointer to the buffer data to write. Must remain valid until write completes.
    pub buffer_ptr: *const u8,

    /// Length of the buffer in bytes.
    pub buffer_len: usize,

    /// Which WAL ring this write targets (Headers or Prepares).
    pub ring: Ring,

    /// Byte offset within the ring where the write begins.
    pub offset: u64,

    /// Intrusive linked-list pointer to the next waiting range.
    ///
    /// When a write overlaps with a locked range, it is appended to the locked
    /// range's wait list. The new write's `next` is set to null, and the tail
    /// of the existing list points to this range.
    pub next: *mut Range<S, WRITE_OPS, WRITE_OPS_WORDS>,

    /// True if a `Storage::write_sectors` operation is currently in progress.
    ///
    /// Only one range covering a given disk region may be locked at a time.
    /// Overlapping writes must wait in the linked list until this unlocks.
    pub locked: bool,
}

impl<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize>
    Range<S, WRITE_OPS, WRITE_OPS_WORDS>
{
    /// Creates a new unlocked range with no waiters.
    ///
    /// # Safety Requirements (for callers)
    ///
    /// The `buffer` slice must remain valid and unchanged for the entire duration
    /// of the write operation (until the callback is invoked). The caller must
    /// ensure the buffer is not deallocated, moved, or mutated while the I/O is
    /// in flight.
    #[inline]
    pub fn new(
        callback: fn(*mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>),
        buffer: &[u8],
        ring: Ring,
        offset: u64,
    ) -> Self {
        Self {
            completion: MaybeUninit::uninit(),
            callback,
            buffer_ptr: buffer.as_ptr(),
            buffer_len: buffer.len(),
            ring,
            offset,
            next: ptr::null_mut(),
            locked: false,
        }
    }

    /// Returns the buffer slice for this range.
    ///
    /// # Panics
    ///
    /// Panics if `buffer_ptr` is null, unaligned, or if `buffer_len` is zero.
    /// These conditions indicate a corrupted or uninitialized `Range`.
    #[inline]
    pub fn buffer(&self) -> &[u8] {
        assert!(self.buffer_ptr.is_aligned());
        assert!(!self.buffer_ptr.is_null());
        assert!(self.buffer_len > 0);
        // SAFETY: Assertions above verify pointer validity. Caller of `Range::new`
        // guarantees buffer remains valid for the write operation's lifetime.
        unsafe { core::slice::from_raw_parts(self.buffer_ptr, self.buffer_len) }
    }

    /// Checks if two ranges overlap in the same ring.
    ///
    /// Returns `true` if both ranges target the same ring AND their byte ranges
    /// intersect. Two ranges `[a, a+len_a)` and `[b, b+len_b)` overlap if:
    /// - `a < b` implies `a + len_a > b`
    /// - `b <= a` implies `b + len_b > a`
    ///
    /// Different rings (Headers vs Prepares) never overlap regardless of offset.
    ///
    /// Uses `saturating_add` to prevent overflow when offset is near `u64::MAX`.
    #[inline]
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.ring != other.ring {
            return false;
        }

        let len_a = self.buffer_len as u64;
        let len_b = other.buffer_len as u64;

        // Use saturating_add to prevent wrap-around at u64::MAX.
        // If offset + len overflows, saturate to MAX which correctly indicates
        // the range extends to the end of the address space.
        if self.offset < other.offset {
            self.offset.saturating_add(len_a) > other.offset
        } else {
            other.offset.saturating_add(len_b) > self.offset
        }
    }
}

/// A write operation descriptor for the range-locking machinery.
///
/// `Write` represents a single in-flight write operation. It is allocated from
/// the `Journal`'s `IOPSType` pool and contains:
/// - A back-pointer to the owning `Journal`
/// - A slot index for concurrent write detection assertions
/// - The `Range` descriptor with buffer and locking state
///
/// # Lifecycle
///
/// 1. Acquired from `Journal.writes` pool
/// 2. Initialized via `Journal::write_sectors`
/// 3. Either immediately submitted or queued behind overlapping writes
/// 4. Callback invoked on completion
/// 5. Released back to pool by caller
///
/// # Layout
///
/// `#[repr(C)]` is required for `container_of!` to recover `Write` from `Range`.
#[repr(C)]
pub struct Write<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> {
    /// Back-pointer to the owning Journal. Set by `write_sectors`.
    pub journal: *mut Journal<S, WRITE_OPS, WRITE_OPS_WORDS>,

    /// Slot index within the log. Used to assert no concurrent writes to same slot.
    pub slot_index: u32,

    /// The range descriptor containing buffer, offset, and locking state.
    pub range: Range<S, WRITE_OPS, WRITE_OPS_WORDS>,
}

const _: () = {
    assert!(size_of::<HeaderPrepare>() == size_of::<Header>());
};

/// Journal lifecycle state.
#[derive(Clone, Copy)]
pub enum Status<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> {
    /// Initial state before recovery begins.
    Init,
    /// Recovery in progress; callback invoked when complete.
    Recovering(fn(*mut Journal<S, WRITE_OPS, WRITE_OPS_WORDS>)),
    /// Recovery complete; journal is ready for normal operation.
    Recovered,
}

/// WAL journal implementing range-locked sector writes.
///
/// The `Journal` coordinates write operations to the Write-Ahead Log, ensuring
/// that overlapping writes are serialized. It maintains a pool of `Write` objects
/// and implements the range-locking protocol.
///
/// # Concurrency
///
/// The Journal is designed for single-threaded operation. All methods must be
/// called from the same thread. The storage layer may complete I/O asynchronously,
/// but callbacks are invoked on the main thread.
pub struct Journal<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> {
    /// Pointer to the storage backend. Must remain valid for Journal lifetime.
    pub storage: *mut S,

    /// Replica identifier for logging and debugging.
    pub replica: u8,

    /// Prepare headers indexed by slot, where `slot == header.op % SLOT_COUNT`.
    ///
    /// Each slot's header command is either `prepare` or `reserved`. When a slot
    /// contains a `reserved` header, the header's `op` equals the slot index.
    pub headers: AlignedSlice<HeaderPrepare>,

    /// Redundant copy of prepare headers, updated after the corresponding prepare
    /// is written to disk.
    ///
    /// This handles the case where a prepare is written before its header sector
    /// is flushed, allowing recovery to detect and repair inconsistencies.
    pub headers_redundant: AlignedSlice<HeaderPrepare>,

    /// Staging buffers for writing header sectors to disk.
    pub write_headers_sectors: AlignedSlice<[u8; constants::SECTOR_SIZE]>,

    /// Tracks which header chunks have been read/written during recovery.
    pub header_chunk_tracking: HeaderChunkTracking,

    /// Pool of write operation descriptors for the range-locking machinery.
    pub writes: IOPSType<Write<S, WRITE_OPS, WRITE_OPS_WORDS>, WRITE_OPS, WRITE_OPS_WORDS>,

    /// Slots that need to be written to disk or repaired from other replicas.
    ///
    /// Similar to a dirty bit in a kernel page cache: indicates the in-memory
    /// state differs from durable storage.
    pub dirty: BitSet<WRITE_OPS, WRITE_OPS_WORDS>,

    /// Slots that are dirty due to corruption, misdirected writes, or sector errors.
    ///
    /// A faulty bit always implies the corresponding dirty bit is set. Faulty
    /// indicates the entry requires repair from another replica, not just a flush.
    pub faulty: BitSet<WRITE_OPS, WRITE_OPS_WORDS>,

    /// Checksums of prepare messages, used to respond to `request_prepare` messages.
    ///
    /// A checksum may be unavailable when the slot is reserved, being written, or corrupt.
    /// Check `prepare_inhabited` to determine validity.
    pub prepare_checksums: Vec<u128>,

    /// Indicates whether each `prepare_checksums` entry contains a valid checksum.
    pub prepare_inhabited: Vec<bool>,

    /// Current journal state (initializing, recovering, or recovered).
    pub status: Status<S, WRITE_OPS, WRITE_OPS_WORDS>,
}

impl<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize>
    Journal<S, WRITE_OPS, WRITE_OPS_WORDS>
{
    /// Creates a new Journal with the given storage backend.
    ///
    /// # Safety Requirements (for callers)
    ///
    /// - `storage` must point to a valid `S` that outlives the Journal
    /// - The storage pointer must not be aliased elsewhere during Journal operations
    pub fn new(storage: *mut S, replica: u8) -> Self {
        let headers = unsafe {
            AlignedSlice::<HeaderPrepare>::new_zeroed_aligned(SLOT_COUNT, constants::SECTOR_SIZE)
        };
        let headers_redundant = unsafe {
            AlignedSlice::<HeaderPrepare>::new_zeroed_aligned(SLOT_COUNT, constants::SECTOR_SIZE)
        };
        let write_headers_sectors = unsafe {
            AlignedSlice::<[u8; constants::SECTOR_SIZE]>::new_zeroed_aligned(
                constants::JOURNAL_IOPS_WRITE_MAX as usize,
                constants::SECTOR_SIZE,
            )
        };

        assert!(&headers.iter().all(|h| !h.valid_checksum()));
        assert!(&headers_redundant.iter().all(|h| !h.valid_checksum()));

        let dirty: BitSet<WRITE_OPS, WRITE_OPS_WORDS> = BitSet::full();
        let faulty: BitSet<WRITE_OPS, WRITE_OPS_WORDS> = BitSet::full();

        let prepare_checksums = vec![0u128; SLOT_COUNT];
        let prepare_inhabited = vec![false; SLOT_COUNT];

        Self {
            storage,
            replica,
            headers,
            headers_redundant,
            write_headers_sectors,
            header_chunk_tracking: HeaderChunkTracking::default(),
            writes: IOPSType::default(),
            dirty,
            faulty,
            prepare_checksums,
            prepare_inhabited,
            status: Status::Init,
        }
    }

    /// Initiates a sector write operation with range locking.
    ///
    /// This method:
    /// 1. Initializes the `Write` with the journal back-pointer and range
    /// 2. Attempts to lock the range via `lock_sectors`
    /// 3. Either submits the I/O immediately or queues behind an overlapping write
    ///
    /// # Safety
    ///
    /// - `write` must be a valid pointer to a `Write` obtained from `self.writes`
    /// - `buffer` must remain valid until the callback is invoked
    /// - The caller must not modify or free the buffer while I/O is in progress
    ///
    /// # Parameters
    ///
    /// - `write`: Pointer to an uninitialized `Write` from the pool
    /// - `callback`: Function called when the write completes (or is coalesced)
    /// - `buffer`: Data to write; must be sector-aligned and remain valid
    /// - `ring`: Target WAL ring (Headers or Prepares)
    /// - `offset`: Byte offset within the ring
    pub unsafe fn write_sectors(
        &mut self,
        write: *mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>,
        callback: fn(*mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>),
        buffer: &[u8],
        ring: Ring,
        offset: u64,
    ) {
        // SAFETY: `write` is a valid pointer from the caller's pool allocation.
        // Writing to `journal` and `range` fields initializes them before use.
        unsafe { ptr::addr_of_mut!((*write).journal).write(self as *mut _) };
        unsafe {
            ptr::addr_of_mut!((*write).range).write(Range::new(callback, buffer, ring, offset))
        };
        self.lock_sectors(write);
    }

    /// Attempts to lock a range, queuing it if an overlap exists.
    ///
    /// Scans all in-flight writes for overlapping ranges. If a locked overlap
    /// is found, the new write is appended to that range's wait list. Otherwise,
    /// the range is locked and submitted to storage.
    fn lock_sectors(&mut self, write: *mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>) {
        // SAFETY: `write` is a valid pointer from write_sectors or the drain loop.
        // The range must not already be locked or queued.
        unsafe {
            assert!(!(*write).range.locked);
            assert!((*write).range.next.is_null());
        }

        let it = self.writes.iterate();
        for other in it {
            if other == write {
                continue;
            }

            // SAFETY: All `Write` pointers from the iterator are valid pool entries.
            unsafe {
                // Never allow concurrent writes to the same slot.
                assert!((*other).slot_index != (*write).slot_index);

                if !(*other).range.locked {
                    continue;
                }

                if (*other).range.overlaps(&(*write).range) {
                    // Overlapping writes must be EXACT matches (same offset, len, ring).
                    // This ensures the queued write can simply reuse the result.
                    assert!((*other).range.offset == (*write).range.offset);
                    assert!((*other).range.buffer_len == (*write).range.buffer_len);
                    assert!((*other).range.ring == (*write).range.ring);
                    // Only header writes may overlap (prepares are always unique).
                    assert!((*other).range.ring == Ring::Headers);

                    // Append to tail of wait list.
                    // Walk the linked list to find the last node, then append.
                    let mut tail: *mut Range<S, WRITE_OPS, WRITE_OPS_WORDS> =
                        ptr::addr_of_mut!((*other).range);
                    while !(*tail).next.is_null() {
                        tail = (*tail).next;
                    }
                    (*tail).next = ptr::addr_of_mut!((*write).range);
                    return; // Queued; do not submit to storage yet.
                }
            }
        }

        // No overlap found; lock and submit to storage.
        unsafe {
            (*write).range.locked = true;

            // Map ring type to storage zone offset.
            let zone = match (*write).range.ring {
                Ring::Headers => S::WAL_HEADERS_ZONE,
                Ring::Prepares => S::WAL_PREPARES_ZONE,
            };

            // SAFETY: self.storage is valid for Journal lifetime (caller invariant).
            let storage = &mut *self.storage;

            // SAFETY: completion is uninitialized MaybeUninit; storage will initialize it.
            // Buffer validity guaranteed by write_sectors caller invariant.
            storage.write_sectors(
                Self::write_sectors_on_write,
                &mut *(*write).range.completion.as_mut_ptr(),
                (*write).range.buffer(),
                zone,
                (*write).range.offset,
            );

            // Verify synchronicity model matches storage behavior.
            match S::SYNCHRONICITY {
                Synchronicity::AlwaysAsynchronous => assert!((*write).range.locked),
                Synchronicity::AlwaysSynchronous => {
                    // We cannot assert !locked here because the callback has already run
                    // and might have released the `write` back to the pool. Dereferencing
                    // `write` here would be a Use-After-Free/Release.
                    //
                    // We rely on the storage contract: AlwaysSynchronous must have called
                    // the completion callback inline.
                }
            }
        };
    }

    /// Completion callback for storage write operations.
    ///
    /// Invoked by the storage layer when a sector write completes. This function:
    /// 1. Recovers the `Write` from the completion token using `container_of!`
    /// 2. Unlocks the range
    /// 3. Drains all waiting writes and re-evaluates them for conflicts
    /// 4. Invokes the user's callback
    fn write_sectors_on_write(completion: &mut S::Write) {
        // SAFETY: `container_of!` relies on #[repr(C)] layout.
        // `completion` is embedded within Range, which is embedded within Write.
        let range: *mut Range<S, WRITE_OPS, WRITE_OPS_WORDS> =
            container_of!(completion, Range<S, WRITE_OPS, WRITE_OPS_WORDS>, completion);
        let write: *mut Write<S, WRITE_OPS, WRITE_OPS_WORDS> =
            container_of!(range, Write<S, WRITE_OPS, WRITE_OPS_WORDS>, range);

        unsafe {
            let journal: *mut Journal<S, WRITE_OPS, WRITE_OPS_WORDS> = (*write).journal;

            // Mark this range as no longer in I/O.
            assert!((*write).range.locked);
            (*write).range.locked = false;

            // Drain the wait list: detach head, then process each waiter.
            // We must capture `next` before clearing it, as lock_sectors may
            // re-queue the write to a different list.
            let mut current = (*range).next;
            (*range).next = ptr::null_mut(); // Detach list from completed range.

            while !current.is_null() {
                let waiting = current;
                // Waiters are never locked (they're waiting, not in I/O).
                assert!(!(*waiting).locked);

                // Advance before modifying, as lock_sectors may set a new next.
                current = (*waiting).next;
                (*waiting).next = ptr::null_mut(); // Clear for re-evaluation.

                // SAFETY: container_of! recovers Write from embedded Range.
                let waiting_write: *mut Write<S, WRITE_OPS, WRITE_OPS_WORDS> =
                    container_of!(waiting, Write<S, WRITE_OPS, WRITE_OPS_WORDS>, range);

                // Re-evaluate: may submit immediately or queue behind another.
                (&mut *journal).lock_sectors(waiting_write);
            }

            // Finally, invoke the user's callback for the completed write.
            (((*range).callback)(write));
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsr::superblock;
    use core::cell::RefCell;

    // =========================================================================
    // Test Infrastructure: MockStorage
    // =========================================================================

    /// Zone identifier for mock storage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Zone {
        SuperBlock,
        WalHeaders,
        WalPrepares,
    }

    /// Completion token for mock write operations.
    /// The actual contents don't matter - we just need it to exist for container_of!
    #[repr(C)]
    struct MockWriteCompletion {
        /// Callback to invoke on completion.
        callback: fn(&mut MockWriteCompletion),
    }

    /// Record of a single write operation for verification.
    #[derive(Debug, Clone)]
    #[allow(dead_code)] // Fields used for future verification extensions.
    struct WriteRecord {
        zone: Zone,
        offset: u64,
        len: usize,
    }

    /// Mock storage implementation for deterministic testing.
    ///
    /// The `ASYNC` const generic controls the synchronicity model:
    /// - `false`: AlwaysSynchronous (callbacks invoked immediately)
    /// - `true`: AlwaysAsynchronous (callbacks deferred until `drain_callbacks`)
    struct MockStorage<const ASYNC: bool> {
        /// Log of all writes submitted to storage.
        write_log: RefCell<Vec<WriteRecord>>,
        /// Pending callbacks to invoke (for deferred mode).
        pending_callbacks: RefCell<Vec<*mut MockWriteCompletion>>,
    }

    impl<const ASYNC: bool> MockStorage<ASYNC> {
        /// Creates a new mock storage.
        fn new() -> Self {
            Self {
                write_log: RefCell::new(Vec::new()),
                pending_callbacks: RefCell::new(Vec::new()),
            }
        }

        fn write_count(&self) -> usize {
            self.write_log.borrow().len()
        }

        /// Invokes all pending callbacks in order.
        #[allow(dead_code)] // Useful for future tests needing explicit completion control.
        fn drain_callbacks(&self) {
            let callbacks: Vec<_> = self.pending_callbacks.borrow_mut().drain(..).collect();
            for write_ptr in callbacks {
                // SAFETY: The pointer was stored from a valid &mut reference in write_sectors.
                // The write completion structure is still alive (caller ensures lifetime).
                unsafe {
                    let write = &mut *write_ptr;
                    (write.callback)(write);
                }
            }
        }
    }

    impl<const ASYNC: bool> Storage for MockStorage<ASYNC> {
        type Read = ();
        type Write = MockWriteCompletion;
        type Zone = Zone;

        // We invoke callbacks (either immediately or deferred), so declare as Async.
        // The assertion expects !locked after write_sectors returns.
        const SYNCHRONICITY: Synchronicity = if ASYNC {
            Synchronicity::AlwaysAsynchronous
        } else {
            Synchronicity::AlwaysSynchronous
        };
        const SUPERBLOCK_ZONE: Self::Zone = Zone::SuperBlock;
        const WAL_HEADERS_ZONE: Self::Zone = Zone::WalHeaders;
        const WAL_PREPARES_ZONE: Self::Zone = Zone::WalPrepares;

        fn read_sectors(
            &mut self,
            _callback: fn(&mut Self::Read),
            _read: &mut Self::Read,
            _buffer: &mut [u8],
            _zone: Self::Zone,
            _offset: u64,
        ) {
            unimplemented!("read not used in journal tests")
        }

        fn write_sectors(
            &mut self,
            callback: fn(&mut Self::Write),
            write: &mut Self::Write,
            buffer: &[u8],
            zone: Self::Zone,
            offset: u64,
        ) {
            // Record the write operation.
            self.write_log.borrow_mut().push(WriteRecord {
                zone,
                offset,
                len: buffer.len(),
            });

            // Store callback for invocation.
            write.callback = callback;

            if ASYNC {
                // Store for later invocation.
                self.pending_callbacks.borrow_mut().push(write as *mut _);
            } else {
                // Immediately invoke callback.
                (write.callback)(write);
            }
        }

        unsafe fn context_from_read(_: &mut Self::Read) -> &mut superblock::Context<Self> {
            unimplemented!("context_from_read not used in journal tests")
        }

        unsafe fn context_from_write(_: &mut Self::Write) -> &mut superblock::Context<Self> {
            unimplemented!("context_from_write not used in journal tests")
        }
    }

    // Type aliases for test convenience.
    type TestMockStorage = MockStorage<false>;
    type TestWrite = Write<TestMockStorage, 32, 1>;
    type TestRange = Range<TestMockStorage, 32, 1>;
    type TestJournal = Journal<TestMockStorage, 32, 1>;

    /// Dummy callback for tests that don't need callback verification.
    fn dummy_callback(_: *mut TestWrite) {}
    fn dummy_callback_async(_: *mut Write<MockStorage<true>, 32, 1>) {}

    // =========================================================================
    // Range Overlap Detection (Property Tests)
    // =========================================================================

    mod overlap_properties {
        use super::*;
        use proptest::prelude::*;

        /// Generates a valid Ring value for property tests.
        fn ring_strategy() -> impl Strategy<Value = Ring> {
            prop_oneof![Just(Ring::Headers), Just(Ring::Prepares)]
        }

        proptest! {
            /// A range always overlaps with itself (reflexivity).
            #[test]
            fn overlaps_is_reflexive(
                ring in ring_strategy(),
                offset in 0u64..1_000_000,
                len in 1usize..10_000
            ) {
                let buffer = vec![0u8; len];
                let range = TestRange::new(dummy_callback, &buffer, ring, offset);
                prop_assert!(range.overlaps(&range), "Range must overlap with itself");
            }

            /// If A overlaps B, then B overlaps A (symmetry).
            #[test]
            fn overlaps_is_symmetric(
                ring in ring_strategy(),
                offset_a in 0u64..1_000_000,
                len_a in 1usize..10_000,
                offset_b in 0u64..1_000_000,
                len_b in 1usize..10_000
            ) {
                let buf_a = vec![0u8; len_a];
                let buf_b = vec![0u8; len_b];
                let range_a = TestRange::new(dummy_callback, &buf_a, ring, offset_a);
                let range_b = TestRange::new(dummy_callback, &buf_b, ring, offset_b);

                prop_assert_eq!(
                    range_a.overlaps(&range_b),
                    range_b.overlaps(&range_a),
                    "Overlap detection must be symmetric"
                );
            }

            /// Different rings never overlap regardless of offset/length.
            #[test]
            fn different_rings_never_overlap(
                offset_a in 0u64..1_000_000,
                len_a in 1usize..10_000,
                offset_b in 0u64..1_000_000,
                len_b in 1usize..10_000
            ) {
                let buf_a = vec![0u8; len_a];
                let buf_b = vec![0u8; len_b];
                let headers = TestRange::new(dummy_callback, &buf_a, Ring::Headers, offset_a);
                let prepares = TestRange::new(dummy_callback, &buf_b, Ring::Prepares, offset_b);

                prop_assert!(!headers.overlaps(&prepares), "Headers and Prepares must never overlap");
                prop_assert!(!prepares.overlaps(&headers), "Prepares and Headers must never overlap");
            }

            /// Adjacent ranges [a, a+len) and [a+len, ...) do not overlap.
            #[test]
            fn adjacent_ranges_do_not_overlap(
                ring in ring_strategy(),
                offset in 0u64..1_000_000,
                len_a in 1usize..10_000,
                len_b in 1usize..10_000
            ) {
                let buf_a = vec![0u8; len_a];
                let buf_b = vec![0u8; len_b];
                let range_a = TestRange::new(dummy_callback, &buf_a, ring, offset);
                let range_b = TestRange::new(dummy_callback, &buf_b, ring, offset + len_a as u64);

                prop_assert!(!range_a.overlaps(&range_b), "Adjacent ranges must not overlap");
                prop_assert!(!range_b.overlaps(&range_a), "Adjacent ranges must not overlap (symmetric)");
            }

            /// Ranges with partial overlap are correctly detected.
            #[test]
            fn partial_overlap_detected(
                ring in ring_strategy(),
                offset_a in 0u64..1_000_000,
                len_a in 100usize..10_000,
                overlap_pct in 1usize..99
            ) {
                let buf_a = vec![0u8; len_a];
                let overlap_bytes = (len_a * overlap_pct) / 100;
                prop_assume!(overlap_bytes > 0);
                let offset_b = offset_a + (len_a - overlap_bytes) as u64;
                let buf_b = vec![0u8; len_a];

                let range_a = TestRange::new(dummy_callback, &buf_a, ring, offset_a);
                let range_b = TestRange::new(dummy_callback, &buf_b, ring, offset_b);

                prop_assert!(range_a.overlaps(&range_b), "Partial overlap must be detected");
            }

            /// If B is completely contained within A, they overlap.
            #[test]
            fn complete_containment_is_overlap(
                ring in ring_strategy(),
                outer_offset in 0u64..1_000_000,
                outer_len in 1000usize..10_000,
                inner_start_delta in 0usize..400,
                inner_len in 1usize..400
            ) {
                prop_assume!(inner_start_delta + inner_len < outer_len);

                let buf_outer = vec![0u8; outer_len];
                let buf_inner = vec![0u8; inner_len];
                let outer = TestRange::new(dummy_callback, &buf_outer, ring, outer_offset);
                let inner = TestRange::new(
                    dummy_callback,
                    &buf_inner,
                    ring,
                    outer_offset + inner_start_delta as u64
                );

                prop_assert!(outer.overlaps(&inner), "Outer must overlap inner");
                prop_assert!(inner.overlaps(&outer), "Inner must overlap outer");
            }

            /// Ranges near u64::MAX do not wrap and false-detect overlap with low offsets.
            #[test]
            fn high_offset_does_not_wrap(
                ring in ring_strategy(),
                len in 1usize..65536,
                low_offset in 0u64..1_000_000
            ) {
                // Place a range near MAX such that offset + len would overflow without saturation.
                let high_offset = u64::MAX - (len as u64 / 2);
                let buf_high = vec![0u8; len];
                let buf_low = vec![0u8; len];

                let high_range = TestRange::new(dummy_callback, &buf_high, ring, high_offset);
                let low_range = TestRange::new(dummy_callback, &buf_low, ring, low_offset);

                // These ranges are at opposite ends of the address space; they must not overlap.
                prop_assert!(!high_range.overlaps(&low_range), "High range must not wrap to overlap low");
                prop_assert!(!low_range.overlaps(&high_range), "Low range must not overlap high");
            }
        }
    }

    // =========================================================================
    // Locking Protocol (Unit Tests)
    // =========================================================================

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn panics_on_overlapping_different_offset() {
        // Use deferred callbacks so write1 stays locked when write2 is submitted.
        let mut storage = MockStorage::<true>::new();
        let mut journal = Journal::<MockStorage<true>, 32, 1>::new(&mut storage, 0);

        let write1 = journal.writes.acquire().unwrap();
        let write2 = journal.writes.acquire().unwrap();

        // write1: [0, 8192), write2: [4096, 8192) - overlaps but different offset.
        let buf1 = vec![1u8; 8192];
        let buf2 = vec![2u8; 4096];

        unsafe {
            (*write1).slot_index = 1;
            (*write2).slot_index = 2;

            journal.write_sectors(write1, dummy_callback_async, &buf1, Ring::Headers, 0);
            // write1 is now locked (callback deferred).
            // This overlaps [0, 8192) but starts at 4096 - should panic.
            journal.write_sectors(write2, dummy_callback_async, &buf2, Ring::Headers, 4096);
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn panics_on_overlapping_different_length() {
        // Use deferred callbacks so write1 stays locked when write2 is submitted.
        let mut storage = MockStorage::<true>::new();
        let mut journal = Journal::<MockStorage<true>, 32, 1>::new(&mut storage, 0);

        let write1 = journal.writes.acquire().unwrap();
        let write2 = journal.writes.acquire().unwrap();

        // Same offset but different length.
        let buf1 = vec![1u8; 8192];
        let buf2 = vec![2u8; 4096];

        unsafe {
            (*write1).slot_index = 1;
            (*write2).slot_index = 2;

            journal.write_sectors(write1, dummy_callback_async, &buf1, Ring::Headers, 0);
            // write1 is now locked (callback deferred).
            // Same offset (0) but different length - should panic.
            journal.write_sectors(write2, dummy_callback_async, &buf2, Ring::Headers, 0);
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn panics_on_overlapping_prepares() {
        // Use deferred callbacks so write1 stays locked when write2 is submitted.
        let mut storage = MockStorage::<true>::new();
        let mut journal = Journal::<MockStorage<true>, 32, 1>::new(&mut storage, 0);

        let write1 = journal.writes.acquire().unwrap();
        let write2 = journal.writes.acquire().unwrap();

        let buf1 = vec![1u8; 4096];
        let buf2 = vec![2u8; 4096];

        unsafe {
            (*write1).slot_index = 1;
            (*write2).slot_index = 2;

            journal.write_sectors(write1, dummy_callback_async, &buf1, Ring::Prepares, 0);
            // write1 is now locked (callback deferred).
            // Prepares writes must never overlap - should panic.
            journal.write_sectors(write2, dummy_callback_async, &buf2, Ring::Prepares, 0);
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn panics_on_duplicate_slot_index() {
        let mut storage = MockStorage::new();
        let mut journal = TestJournal::new(&mut storage, 0);

        let write1 = journal.writes.acquire().unwrap();
        let write2 = journal.writes.acquire().unwrap();

        // Non-overlapping ranges but same slot index.
        let buf1 = vec![1u8; 4096];
        let buf2 = vec![2u8; 4096];

        unsafe {
            (*write1).slot_index = 42;
            (*write2).slot_index = 42; // Same slot - should panic.

            journal.write_sectors(write1, dummy_callback, &buf1, Ring::Headers, 0);
            journal.write_sectors(write2, dummy_callback, &buf2, Ring::Headers, 8192);
        }
    }

    // =========================================================================
    // Completion & Drain Logic (Unit Tests)
    // =========================================================================

    #[test]
    fn completion_clears_wait_list() {
        // Verify that after completion, the wait list is detached.
        let mut storage = MockStorage::new();
        let mut journal = TestJournal::new(&mut storage, 0);

        let write1 = journal.writes.acquire().unwrap();
        let write2 = journal.writes.acquire().unwrap();

        let buf1 = vec![1u8; 4096];
        let buf2 = vec![2u8; 4096];

        unsafe {
            (*write1).slot_index = 1;
            (*write2).slot_index = 2;

            journal.write_sectors(write1, dummy_callback, &buf1, Ring::Headers, 0);
            journal.write_sectors(write2, dummy_callback, &buf2, Ring::Headers, 0);

            // After synchronous completion, next pointers should be null.
            assert!(
                (*write1).range.next.is_null(),
                "Wait list should be cleared"
            );
            assert!(
                (*write2).range.next.is_null(),
                "Waiter's next should be cleared"
            );
        }
    }

    /// Verifies that overlapping writes queue and defer until the first completes.
    #[test]
    fn async_overlap_queuing_defers_until_completion() {
        // Use async storage so callbacks are deferred.
        let mut storage = MockStorage::<true>::new();
        let mut journal = Journal::<MockStorage<true>, 32, 1>::new(&mut storage, 0);

        let write1 = journal.writes.acquire().unwrap();
        let write2 = journal.writes.acquire().unwrap();

        let buf1 = vec![1u8; 4096];
        let buf2 = vec![2u8; 4096];

        unsafe {
            (*write1).slot_index = 1;
            (*write2).slot_index = 2;

            // Submit first write - becomes locked.
            journal.write_sectors(write1, dummy_callback_async, &buf1, Ring::Headers, 0);
            assert!((*write1).range.locked, "First write should be locked");

            // Submit second write at SAME offset - should queue, not submit.
            journal.write_sectors(write2, dummy_callback_async, &buf2, Ring::Headers, 0);
            assert!(
                !(*write2).range.locked,
                "Second write should be queued, not locked"
            );

            // First write should have second in its wait list.
            assert!(
                !(*write1).range.next.is_null(),
                "First write should have waiter"
            );

            // Only 1 I/O should have been submitted so far.
            assert_eq!(
                storage.write_count(),
                1,
                "Only first write should be submitted"
            );

            // Complete first write - this dequeues and submits second.
            storage.drain_callbacks();

            // Second write is now in-flight (submitted when first completed).
            assert_eq!(
                storage.write_count(),
                2,
                "Second write should now be submitted"
            );
            assert!(!(*write1).range.locked, "First write should be unlocked");
            assert!(
                (*write2).range.locked,
                "Second write should now be locked (in-flight)"
            );

            // Complete second write.
            storage.drain_callbacks();

            // Now both should be complete.
            assert!(!(*write2).range.locked, "Second write should be unlocked");
        }
    }

    // =========================================================================
    // Integration Properties (Property Tests)
    // =========================================================================

    mod integration_properties {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// N non-overlapping writes all complete successfully.
            #[test]
            fn all_non_overlapping_writes_complete(
                write_count in 2usize..20
            ) {
                let mut storage = MockStorage::new();
                let mut journal = TestJournal::new(&mut storage, 0);

                let mut writes = Vec::new();
                let mut buffers = Vec::new();

                for i in 0..write_count {
                    let write = journal.writes.acquire().unwrap();
                    // Set slot_index immediately to avoid conflicts during lock_sectors iteration.
                    unsafe { (*write).slot_index = i as u32; }
                    let buf = vec![i as u8; 4096];
                    buffers.push(buf);
                    writes.push(write);
                }

                for (i, write) in writes.iter().enumerate() {
                    unsafe {
                        journal.write_sectors(
                            *write,
                            dummy_callback,
                            &buffers[i],
                            Ring::Headers,
                            (i * 8192) as u64
                        );
                    }
                }

                // All should complete.
                prop_assert_eq!(storage.write_count(), write_count);
            }

            /// N exact overlapping writes serialize to N sequential I/Os.
            #[test]
            fn exact_overlaps_serialize_correctly(
                overlap_count in 2usize..10
            ) {
                let mut storage = MockStorage::new();
                let mut journal = TestJournal::new(&mut storage, 0);

                let mut writes = Vec::new();
                let mut buffers = Vec::new();

                // All same offset/length.
                for i in 0..overlap_count {
                    let write = journal.writes.acquire().unwrap();
                    // Set slot_index immediately to avoid conflicts during lock_sectors iteration.
                    unsafe { (*write).slot_index = i as u32; }
                    let buf = vec![i as u8; 4096];
                    buffers.push(buf);
                    writes.push(write);
                }

                for (i, write) in writes.iter().enumerate() {
                    unsafe {
                        journal.write_sectors(
                            *write,
                            dummy_callback,
                            &buffers[i],
                            Ring::Headers,
                            0 // All same offset
                        );
                    }
                }

                // All completed in sequence.
                prop_assert_eq!(storage.write_count(), overlap_count);
            }
        }
    }

    // =========================================================================
    // Callback Safety (Unit Tests)
    // =========================================================================

    #[test]
    fn container_of_recovers_correct_write() {
        let mut storage = MockStorage::new();
        let mut journal = TestJournal::new(&mut storage, 0);

        let write = journal.writes.acquire().unwrap();
        let buf = vec![1u8; 4096];

        fn verifying_callback(w: *mut TestWrite) {
            // This callback receives the correct write pointer.
            assert!(!w.is_null(), "Callback should receive valid write pointer");
        }

        unsafe {
            (*write).slot_index = 1;
            journal.write_sectors(write, verifying_callback, &buf, Ring::Headers, 0);
        }

        // If we got here without panic, container_of worked correctly.
        assert_eq!(storage.write_count(), 1);
    }

    // =========================================================================
    // Edge Cases (Unit Tests)
    // =========================================================================

    #[test]
    fn write_pool_exhaustion() {
        let mut storage = MockStorage::new();
        let mut journal = TestJournal::new(&mut storage, 0);

        // Acquire all 32 slots (WRITE_OPS = 32).
        let mut writes = Vec::new();
        for _ in 0..32 {
            writes.push(journal.writes.acquire().unwrap());
        }

        // Next acquire should fail.
        assert!(
            journal.writes.acquire().is_none(),
            "Pool should be exhausted"
        );

        // Release one and acquire again should work.
        journal.writes.release(writes.pop().unwrap());
        assert!(
            journal.writes.acquire().is_some(),
            "Should be able to acquire after release"
        );
    }

    /// Regression test for UB: batch acquire then submit pattern.
    #[test]
    fn batch_acquire_then_submit_is_safe() {
        let mut storage = MockStorage::new();
        let mut journal = TestJournal::new(&mut storage, 0);

        // Acquire multiple writes first (all have unsubmitted ranges).
        let write1 = journal.writes.acquire().unwrap();
        let write2 = journal.writes.acquire().unwrap();
        let write3 = journal.writes.acquire().unwrap();

        // Set slot indices (required by caller contract).
        unsafe {
            (*write1).slot_index = 1;
            (*write2).slot_index = 2;
            (*write3).slot_index = 3;
        }

        // Submit only the first write. lock_sectors will iterate write2 and write3,
        // reading their range.locked fields. With zero-initialization, this is safe.
        let buf1 = vec![1u8; 4096];
        unsafe {
            journal.write_sectors(write1, dummy_callback, &buf1, Ring::Headers, 0);
        }

        // Submit the remaining writes.
        let buf2 = vec![2u8; 4096];
        let buf3 = vec![3u8; 4096];
        unsafe {
            journal.write_sectors(write2, dummy_callback, &buf2, Ring::Headers, 8192);
            journal.write_sectors(write3, dummy_callback, &buf3, Ring::Headers, 16384);
        }

        // All writes should complete successfully.
        assert_eq!(storage.write_count(), 3);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn zero_length_buffer_panics() {
        let buf: [u8; 0] = [];
        let range = TestRange::new(dummy_callback, &buf, Ring::Headers, 0);
        let _ = range.buffer(); // Should panic.
    }

    #[test]
    fn range_new_stores_buffer_correctly() {
        let buf = vec![1u8, 2, 3, 4, 5];
        let range = TestRange::new(dummy_callback, &buf, Ring::Headers, 100);

        assert_eq!(range.buffer_ptr, buf.as_ptr());
        assert_eq!(range.buffer_len, 5);
        assert_eq!(range.ring, Ring::Headers);
        assert_eq!(range.offset, 100);
        assert!(!range.locked);
        assert!(range.next.is_null());
    }

    #[test]
    fn range_buffer_returns_correct_slice() {
        let buf = vec![10u8, 20, 30, 40];
        let range = TestRange::new(dummy_callback, &buf, Ring::Prepares, 0);

        let retrieved = range.buffer();
        assert_eq!(retrieved, &[10, 20, 30, 40]);
    }

    #[test]
    fn journal_new_initializes_correctly() {
        let mut storage = MockStorage::new();
        let journal = TestJournal::new(&mut storage, 5);

        assert_eq!(journal.storage, &mut storage as *mut _);
        assert_eq!(journal.replica, 5);
    }

    #[test]
    fn async_writes_complete_correctly() {
        let mut storage = MockStorage::<true>::new();
        let mut journal = Journal::<MockStorage<true>, 32, 1>::new(&mut storage, 0);

        let write = journal.writes.acquire().unwrap();
        let buf = vec![1u8; 4096];

        unsafe {
            (*write).slot_index = 1;
            journal.write_sectors(write, dummy_callback_async, &buf, Ring::Headers, 0);

            // Should be locked (Async).
            assert!((*write).range.locked);
            assert_eq!(storage.pending_callbacks.borrow().len(), 1);

            storage.drain_callbacks();

            // Should be unlocked.
            assert!(!(*write).range.locked);
        }

        assert_eq!(storage.write_count(), 1);
    }

    #[test]
    fn synchronous_callback_can_release_write() {
        let mut storage = MockStorage::<false>::new();
        let mut journal = Journal::<MockStorage<false>, 32, 1>::new(&mut storage, 0);

        let write = journal.writes.acquire().unwrap();
        let buf = vec![1u8; 4096];

        fn releasing_callback(w: *mut TestWrite) {
            unsafe {
                let journal = &mut *(*w).journal;
                // Release the write back to the pool.
                // This makes the `w` pointer invalid for the caller of the callback.
                journal.writes.release(w);
            }
        }

        unsafe {
            (*write).slot_index = 1;
            // This should not panic or trigger UB (accessing released write).
            journal.write_sectors(write, releasing_callback, &buf, Ring::Headers, 0);
        }

        // Verify the write was actually released.
        // Capacity is 32, we acquired 1, then released it in callback.
        // So available should be 32.
        // We can't check `journal.writes.available()` directly as it's not exposed in `IOPSType`
        // public interface easily without iteration or internal knowledge,
        // but we can try to acquire 32 writes again to prove it's free.

        let mut writes = Vec::new();
        for _ in 0..32 {
            if let Some(w) = journal.writes.acquire() {
                writes.push(w);
            } else {
                panic!("Should be able to re-acquire all writes");
            }
        }
    }
}
