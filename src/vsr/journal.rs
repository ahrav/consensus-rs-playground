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
use core::cmp::min;
use core::mem::MaybeUninit;
use core::ptr;

#[allow(unused_imports)]
use crate::constants;
use crate::container_of;
use crate::message_pool::Message;
use crate::stdx::BitSet;
use crate::util::utils::AlignedSlice;
use crate::vsr::command::PrepareCmd;
use crate::vsr::journal_primitives::Slot;
use crate::vsr::{Checksum128, Command, Operation};
use crate::vsr::{
    Header,
    HeaderPrepare,
    iops::IOPSType,
    journal_primitives::{
        HEADER_CHUNK_COUNT, HEADER_CHUNK_WORDS, HEADERS_SIZE, Ring, SLOT_COUNT, SLOT_COUNT_WORDS,
        WAL_HEADER_SIZE,
    },
    storage::{Storage, Synchronicity},
    // Command, Operation,
};

// Compile-time assertions to verify header chunk constant relationships.
const _: () = {
    // HEADERS_SIZE must be sector-aligned for I/O.
    assert!(HEADERS_SIZE.is_multiple_of(constants::SECTOR_SIZE as u64));
    // HEADERS_SIZE must be a multiple of header size for correct slicing.
    assert!(HEADERS_SIZE.is_multiple_of(WAL_HEADER_SIZE as u64));
    // MESSAGE_SIZE_MAX must be a multiple of header size.
    assert!(constants::MESSAGE_SIZE_MAX.is_multiple_of(WAL_HEADER_SIZE as u32));
    // MESSAGE_SIZE_MAX must be sector-aligned.
    assert!((constants::MESSAGE_SIZE_MAX as usize).is_multiple_of(constants::SECTOR_SIZE));
};

/// Bitset tracking header chunk state. `HEADER_CHUNK_WORDS` is the number of `u64`s
/// needed to store `HEADER_CHUNK_COUNT` bits.
pub type HeaderChunks = BitSet<HEADER_CHUNK_COUNT, HEADER_CHUNK_WORDS>;

/// Descriptor for a header chunk read request.
///
/// Returned by [`Journal::header_chunk_next_to_request`] to specify what to read.
/// The caller issues the I/O and calls [`Journal::header_chunk_mark_recovered`]
/// with the `index` on completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderChunk {
    /// Chunk index in `[0, HEADER_CHUNK_COUNT)`.
    pub index: usize,
    /// Byte offset within the header region.
    pub offset: u64,
    /// Bytes to read (at most `MESSAGE_SIZE_MAX`).
    pub len: usize,
}

impl HeaderChunk {
    #[inline]
    pub fn end(self) -> u64 {
        self.offset + self.len as u64
    }

    #[inline]
    pub fn assert_invariants(&self) {
        assert!(self.index < HEADER_CHUNK_COUNT);
        assert!(self.len > 0);
        assert!(self.len <= constants::MESSAGE_SIZE_MAX as usize);

        assert!(self.offset < HEADERS_SIZE);
        assert!(self.end() <= HEADERS_SIZE);

        assert!(self.offset.is_multiple_of(constants::SECTOR_SIZE as u64));
        assert!(self.len.is_multiple_of(constants::SECTOR_SIZE));

        assert!(self.len.is_multiple_of(WAL_HEADER_SIZE));
    }
}

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

pub type MessagePrepare = Message<PrepareCmd>;

/// Callback invoked when `write_prepare` completes.
pub type WritePrepareCallback<S, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> =
    fn(*mut Journal<S, WRITE_OPS, WRITE_OPS_WORDS>, Option<*mut MessagePrepare>);

/// A write operation descriptor for the range-locking machinery.
///
/// `Write` represents a single in-flight write operation. It is allocated from
/// the `Journal`'s `IOPSType` pool and contains:
/// - A back-pointer to the owning `Journal`
/// - The op number (from the prepare header) used for concurrent-slot detection assertions
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

    /// User callback for `write_prepare`.
    ///
    /// `None` for plain `write_sectors` tests; always `Some` for the `write_prepare` flow.
    pub prepare_callback: Option<WritePrepareCallback<S, WRITE_OPS, WRITE_OPS_WORDS>>,

    /// Prepare message being written.
    pub message: *mut MessagePrepare,

    /// Operation number of the prepare whose slot is being written.
    ///
    /// Used to derive the slot (`op % SLOT_COUNT`) and assert that we never have
    /// two concurrent writes to the same slot.
    pub op: u64,

    /// Cached checksum of `message.header.checksum` (used by `writing()`).
    pub checksum: Checksum128,

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Writing {
    None,
    Slot,
    Exact,
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

    /// Chunks with outstanding or completed read requests.
    pub header_chunks_requested: HeaderChunks,

    /// Chunks whose reads have completed and data copied.
    pub header_chunks_recovered: HeaderChunks,

    /// Pool of write operation descriptors for the range-locking machinery.
    pub writes: IOPSType<Write<S, WRITE_OPS, WRITE_OPS_WORDS>, WRITE_OPS, WRITE_OPS_WORDS>,

    /// Slots that need to be written to disk or repaired from other replicas.
    ///
    /// Similar to a dirty bit in a kernel page cache: indicates the in-memory
    /// state differs from durable storage.
    pub dirty: BitSet<SLOT_COUNT, SLOT_COUNT_WORDS>,

    /// Slots that are dirty due to corruption, misdirected writes, or sector errors.
    ///
    /// A faulty bit always implies the corresponding dirty bit is set. Faulty
    /// indicates the entry requires repair from another replica, not just a flush.
    pub faulty: BitSet<SLOT_COUNT, SLOT_COUNT_WORDS>,

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

        let dirty: BitSet<SLOT_COUNT, SLOT_COUNT_WORDS> = BitSet::full();
        let faulty: BitSet<SLOT_COUNT, SLOT_COUNT_WORDS> = BitSet::full();

        let prepare_checksums = vec![0u128; SLOT_COUNT];
        let prepare_inhabited = vec![false; SLOT_COUNT];

        let mut journal = Self {
            storage,
            replica,
            headers,
            headers_redundant,
            write_headers_sectors,
            header_chunks_requested: HeaderChunks::empty(),
            header_chunks_recovered: HeaderChunks::empty(),
            writes: IOPSType::default(),
            dirty,
            faulty,
            prepare_checksums,
            prepare_inhabited,
            status: Status::Init,
        };

        unsafe {
            ptr::write_bytes(
                journal.headers.as_mut_ptr() as *mut u8,
                0,
                SLOT_COUNT * size_of::<HeaderPrepare>(),
            );
            ptr::write_bytes(
                journal.headers_redundant.as_mut_ptr() as *mut u8,
                0,
                SLOT_COUNT * size_of::<HeaderPrepare>(),
            );
        }

        assert!(journal.headers.iter().all(|h| !h.valid_checksum()));
        assert!(
            journal
                .headers_redundant
                .iter()
                .all(|h| !h.valid_checksum())
        );

        journal.prepare_checksums.fill(0);
        journal.prepare_inhabited.fill(false);

        journal
    }

    // =========================================================================
    // Slot/Header Lookup: Pure Calculation (no verification)
    // =========================================================================

    /// Returns the slot where `op` would be stored. Does NOT verify occupancy.
    #[inline]
    fn slot_for_op(&self, op: u64) -> Slot {
        Slot::from_op(op)
    }

    /// Returns whatever header occupies the slot for `op`.
    /// Returns `None` only if the slot is reserved.
    /// The returned header may have a DIFFERENT op (circular buffer wrap).
    #[inline]
    fn header_for_op(&self, op: u64) -> Option<&HeaderPrepare> {
        let slot = self.slot_for_op(op);
        let existing = &self.headers[slot.index()];
        assert!(existing.command == Command::Prepare);

        if existing.operation == Operation::RESERVED {
            assert!(existing.op == slot.index() as u64);
            None
        } else {
            assert_eq!(self.slot_for_op(existing.op).index(), slot.index());
            Some(existing)
        }
    }

    // =========================================================================
    // Slot/Header Lookup: Op Verification (checks op matches)
    // =========================================================================

    /// Returns the slot for `op` only if a header with that exact op exists.
    #[inline]
    #[allow(dead_code)]
    fn slot_with_op(&self, op: u64) -> Option<Slot> {
        self.header_with_op(op).map(|_| self.slot_for_op(op))
    }

    /// Returns the header only if it exists AND has the requested op.
    #[inline]
    pub fn header_with_op(&self, op: u64) -> Option<&HeaderPrepare> {
        if let Some(existing) = self.header_for_op(op)
            && existing.op == op
        {
            Some(existing)
        } else {
            None
        }
    }

    // =========================================================================
    // Slot/Header Lookup: Identity Verification (checks op AND checksum)
    // =========================================================================

    /// Returns the slot only if this EXACT header (op + checksum) exists.
    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    fn slot_with_op_and_checksum(&self, op: u64, checksum: Checksum128) -> Option<Slot> {
        self.header_with_op_and_checksum(op, checksum)
            .map(|_| self.slot_for_op(op))
    }

    /// Returns the header only if it matches EXACTLY (op + checksum).
    #[inline]
    pub fn header_with_op_and_checksum(
        &self,
        op: u64,
        checksum: Checksum128,
    ) -> Option<&HeaderPrepare> {
        if let Some(existing) = self.header_with_op(op) {
            assert!(existing.op == op);
            if existing.checksum == checksum {
                return Some(existing);
            }
        }
        None
    }

    // =========================================================================
    // Slot/Header Lookup: From Header Reference
    // =========================================================================

    /// Returns the slot for a header known to exist. Asserts presence.
    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    fn slot_for_header(&self, header: &HeaderPrepare) -> Slot {
        assert!(self.slot_with_op(header.op).is_some());
        self.slot_for_op(header.op)
    }

    /// Returns the slot only if this exact header exists in the journal.
    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    fn slot_with_header(&self, header: &HeaderPrepare) -> Option<Slot> {
        self.slot_with_op_and_checksum(header.op, header.checksum)
    }

    // =========================================================================
    // Header Predicates
    // =========================================================================

    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    fn has_header(&self, header: &HeaderPrepare) -> bool {
        self.header_with_op_and_checksum(header.op, header.checksum)
            .is_some()
    }

    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    fn has_dirty(&self, header: &HeaderPrepare) -> bool {
        assert!(self.has_header(header));
        let slot = self.slot_for_header(header);
        self.dirty.is_set(slot.index())
    }

    // =========================================================================
    // Header Traversal
    // =========================================================================

    #[inline]
    pub fn previous_entry(&self, header: &HeaderPrepare) -> Option<&HeaderPrepare> {
        if header.op == 0 {
            None
        } else {
            self.header_with_op(header.op - 1)
        }
    }

    #[inline]
    pub fn next_entry(&self, header: &HeaderPrepare) -> Option<&HeaderPrepare> {
        self.header_with_op(header.op + 1)
    }

    pub fn op_maximum(&self) -> u64 {
        assert!(matches!(self.status, Status::Recovered));

        self.headers
            .iter()
            .filter_map(|h| (h.operation != Operation::RESERVED).then_some(h.op))
            .max()
            .unwrap_or(0)
    }

    // ---------------------------------------------------------------------
    // Header chunk tracking (recovery)
    // ---------------------------------------------------------------------

    #[inline]
    pub fn header_chunks_reset(&mut self) {
        self.header_chunks_recovered = HeaderChunks::empty();
        self.header_chunks_requested = HeaderChunks::empty();
        self.header_chunks_assert_invariants();
    }

    #[inline]
    pub fn header_chunks_done(&self) -> bool {
        self.header_chunks_recovered.is_full()
    }

    /// Returns the byte offset for a given chunk index.
    ///
    /// Chunks are `MESSAGE_SIZE_MAX`-sized, so offset = index * MESSAGE_SIZE_MAX.
    #[inline]
    pub fn header_chunk_offset(chunk_index: usize) -> u64 {
        assert!(chunk_index < HEADER_CHUNK_COUNT);
        (chunk_index as u64)
            .checked_mul(constants::MESSAGE_SIZE_MAX as u64)
            .expect("chunk_offset overflow")
    }

    /// Returns the chunk length for a given offset.
    ///
    /// Returns `MESSAGE_SIZE_MAX` for all chunks except possibly the last,
    /// which may be smaller if `HEADERS_SIZE` isn't evenly divisible.
    #[inline]
    pub fn header_chunk_len_for_offset(offset: u64) -> usize {
        assert!(offset < HEADERS_SIZE);

        let remaining = HEADERS_SIZE
            .checked_sub(offset)
            .expect("HEADER_SIZE - offset underflow");
        let max_u64 = min(constants::MESSAGE_SIZE_MAX as u64, remaining);

        assert!(max_u64 <= usize::MAX as u64);
        let max = max_u64 as usize;

        assert!(max > 0);
        assert!(max.is_multiple_of(constants::SECTOR_SIZE));
        assert!(max.is_multiple_of(WAL_HEADER_SIZE));

        max
    }

    /// Returns the next chunk to read, marking it as requested.
    ///
    /// Returns `None` in two cases:
    /// - All chunks recovered (done)
    /// - All chunks requested but some not yet recovered (waiting for I/O)
    #[inline]
    pub fn header_chunk_next_to_request(&mut self) -> Option<HeaderChunk> {
        self.header_chunks_assert_invariants();

        if self.header_chunks_recovered.is_full() {
            assert!(self.header_chunks_requested.is_full());
            return None;
        }

        let Some(chunk_index) = self.header_chunks_requested.first_unset() else {
            // All reads have been issued; waiting for completions.
            return None;
        };

        assert!(!self.header_chunks_requested.is_set(chunk_index));
        self.header_chunks_requested.set(chunk_index);

        let offset = Self::header_chunk_offset(chunk_index);
        assert!(offset < HEADERS_SIZE);

        let len = Self::header_chunk_len_for_offset(offset);
        let chunk = HeaderChunk {
            index: chunk_index,
            offset,
            len,
        };
        chunk.assert_invariants();

        self.header_chunks_assert_invariants();
        Some(chunk)
    }

    /// Marks a chunk as recovered after its read completes.
    ///
    /// # Panics
    ///
    /// Panics if the chunk was not previously requested or was already recovered.
    #[inline]
    pub fn header_chunk_mark_recovered(&mut self, chunk_index: usize) {
        self.header_chunks_assert_invariants();

        assert!(chunk_index < HEADER_CHUNK_COUNT);
        assert!(self.header_chunks_requested.is_set(chunk_index));
        assert!(!self.header_chunks_recovered.is_set(chunk_index));

        self.header_chunks_recovered.set(chunk_index);
        self.header_chunks_assert_invariants();
    }

    #[inline]
    fn header_chunks_assert_invariants(&self) {
        assert!(self.header_chunks_recovered.count() <= self.header_chunks_requested.count());
        assert!(
            self.header_chunks_recovered
                .is_subset(&self.header_chunks_requested)
        );
        if self.header_chunks_recovered.is_full() {
            assert!(self.header_chunks_requested.is_full());
        }
    }

    /// Reinterprets a byte slice as a slice of headers.
    ///
    /// # Safety
    ///
    /// This is safe because:
    /// - `HeaderPrepare` is `#[repr(C)]` with fixed `WAL_HEADER_SIZE` layout
    /// - Alignment is verified at runtime before the cast
    /// - Returned slice lifetime is tied to input slice
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `chunk_bytes.len()` is not a positive multiple of `WAL_HEADER_SIZE`
    /// - `chunk_bytes` is not properly aligned for `HeaderPrepare`
    #[inline]
    pub fn header_chunk_bytes_as_headers(chunk_bytes: &[u8]) -> &[HeaderPrepare] {
        assert!(chunk_bytes.len() >= WAL_HEADER_SIZE);
        assert!(chunk_bytes.len().is_multiple_of(WAL_HEADER_SIZE));

        // HeaderPrepare requires specific alignment (typically 16-byte).
        const REQUIRED_ALIGN: usize = core::mem::align_of::<HeaderPrepare>();
        assert!(
            (chunk_bytes.as_ptr() as usize).is_multiple_of(REQUIRED_ALIGN),
            "chunk_bytes must be {}-byte aligned for HeaderPrepare",
            REQUIRED_ALIGN
        );

        let count = chunk_bytes.len() / WAL_HEADER_SIZE;
        let headers = chunk_bytes.as_ptr() as *const HeaderPrepare;
        // SAFETY: Length and alignment are verified by assertions above.
        unsafe { std::slice::from_raw_parts(headers, count) }
    }

    /// Installs `header` into its slot and marks that slot dirty.
    ///
    /// If the exact header already exists, this is a no-op (the dirty bit must
    /// already be set). When overwriting a non-reserved slot, this clears the
    /// `faulty` flag and resets the redundant header to a reserved placeholder
    /// so recovery can detect out-of-order or partial writes.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The journal is not in `Status::Recovered`
    /// - `header` is not a valid `Prepare` header (reserved operation or too small)
    /// - The slot would move backwards (`existing.op > header.op`)
    /// - The slot is reserved but the redundant header is not a reserved placeholder
    #[cfg_attr(not(test), allow(dead_code))]
    fn set_header_as_dirty(&mut self, header: &HeaderPrepare) {
        assert!(matches!(self.status, Status::Recovered));
        assert!(header.command == Command::Prepare);
        assert!(header.operation != Operation::RESERVED);
        assert!((header.size as usize) >= size_of::<HeaderPrepare>());

        let slot = self.slot_for_op(header.op);

        if self.has_header(header) {
            assert!(self.dirty.is_set(slot.index()));
            return;
        }

        assert!(self.headers[slot.index()].op <= header.op);

        if self.headers[slot.index()].operation == Operation::RESERVED {
            assert!(self.headers_redundant[slot.index()].operation == Operation::RESERVED);
            assert!(self.headers_redundant[slot.index()].checksum == 0);
        } else {
            self.faulty.unset(slot.index());
            self.headers_redundant[slot.index()] =
                HeaderPrepare::reserve(header.cluster, slot.index() as u64);
        }

        self.headers[slot.index()] = *header;
        self.dirty.set(slot.index());
    }

    /// Returns whether a write is currently in-flight for `header`'s slot.
    ///
    /// Scans the write pool for a busy entry with the same slot (`op % SLOT_COUNT`).
    /// If a write is found:
    /// - `Writing::Exact` when the checksum matches `header`
    /// - `Writing::Slot` when the slot/op matches but the checksum differs
    ///
    /// Otherwise returns `Writing::None`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The journal is not in `Status::Recovered`
    /// - More than one write targets the same slot
    /// - A write targets the same slot but a different `op`
    #[cfg_attr(not(test), allow(dead_code))]
    fn writing(&self, header: &HeaderPrepare) -> Writing {
        assert!(matches!(self.status, Status::Recovered));

        let slot = self.slot_for_op(header.op);
        let mut found = Writing::None;

        for write in self.writes.iterate_const() {
            let write_slot = self.slot_for_op(unsafe { (*write).op });
            if write_slot.index() != slot.index() {
                continue;
            }

            assert!(unsafe { (*write).op } == header.op);
            assert!(found == Writing::None);

            if unsafe { (*write).checksum } == header.checksum {
                found = Writing::Exact
            } else {
                found = Writing::Slot
            }
        }

        found
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
                let other_slot = Slot::from_op((*other).op);
                let write_slot = Slot::from_op((*write).op);
                assert!(other_slot != write_slot);

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
#[path = "journal_tests.rs"]
mod tests;
