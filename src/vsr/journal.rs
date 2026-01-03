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

use core::{cmp::min, mem::MaybeUninit, ptr};

#[allow(unused_imports)]
use crate::{
    constants, container_of,
    message_pool::Message,
    stdx::BitSet,
    util::{utils::AlignedSlice, zero},
    vsr::{
        Checksum128, Command, Header, HeaderPrepare, Operation,
        command::PrepareCmd,
        iops::IOPSType,
        journal_primitives::{
            HEADER_CHUNK_COUNT, HEADER_CHUNK_WORDS, HEADERS_PER_SECTOR, HEADERS_SIZE, Ring,
            SLOT_COUNT, SLOT_COUNT_WORDS, Slot, WAL_HEADER_SIZE,
        },
        storage::{Storage, Synchronicity},
    },
};

use super::replica::Replica;

// ============================================================================
// Read IOP accounting
// ============================================================================

// The read path shares a single IOP pool across commit and repair reads. We reserve
// a small number of slots for commit-path reads so repair traffic cannot starve
// client commits. Accounting is enforced in `read_prepare_with_op_and_checksum`.

/// Total read IOPs available to the journal.
///
/// We intentionally reserve a small number of reads for the commit path so that an
/// asymmetrically partitioned replica cannot starve the cluster with repair reads.
const READ_OPS: usize = constants::JOURNAL_IOPS_READ_MAX as usize;

/// `IOPSType` uses a bitset internally; `READ_OPS_WORDS` is the number of `u64` words
/// required to represent `READ_OPS` bits.
const READ_OPS_WORDS: usize = READ_OPS.div_ceil(64);

/// Reads reserved for commit path (`destination_replica == None`).
const READS_COMMIT_COUNT_MAX: usize = 2;

/// Reads available for repair path (`destination_replica != None`).
const READS_REPAIR_COUNT_MAX: usize = READ_OPS - READS_COMMIT_COUNT_MAX;

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

    assert!(READ_OPS > 0);
    assert!(READS_REPAIR_COUNT_MAX > 0);
    assert!(READS_REPAIR_COUNT_MAX + READS_COMMIT_COUNT_MAX == READ_OPS);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Parameters describing a prepare read request.
///
/// These options are copied into the read descriptor and passed back to the
/// completion callback so callers can associate results with their request.
pub struct ReadOptions {
    /// Operation number of the prepare to read.
    pub op: u64,
    /// Expected checksum for identity verification.
    pub checksum: Checksum128,
    /// `None` for commit-path reads, `Some(replica)` for repair/request_prepare reads.
    pub destination_replica: Option<u8>,
}

impl ReadOptions {
    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    fn commit(op: u64, checksum: Checksum128) -> Self {
        Self {
            op,
            checksum,
            destination_replica: None,
        }
    }

    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    fn repair(op: u64, checksum: Checksum128, destination_replica: u8) -> Self {
        Self {
            op,
            checksum,
            destination_replica: Some(destination_replica),
        }
    }
}

#[repr(C)]
/// Descriptor for an in-flight prepare read.
///
/// Allocated from `Journal.reads` and populated before issuing storage I/O. The
/// completion token is first to support `container_of!` in the read callback.
pub struct Read<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> {
    /// Storage completion token (must be first for `container_of!`).
    pub completion: MaybeUninit<S::Read>,
    /// Back-pointer to the owning journal.
    pub journal: *mut Journal<S, WRITE_OPS, WRITE_OPS_WORDS>,
    /// Owned message buffer used for the read; moved out on completion.
    pub message: MaybeUninit<MessagePrepare>,
    /// Read options passed through to the callback.
    pub options: ReadOptions,
    /// User callback invoked when the read completes (or is bypassed).
    pub callback: ReadPrepareCallback<S, WRITE_OPS, WRITE_OPS_WORDS>,
}

/// Callback invoked when `read_prepare` completes.
///
/// `prepare` is `Some(message)` only when the prepare was read successfully and passed all
/// validation. `None` indicates the prepare is unavailable, was rewritten while reading,
/// or was detected as corrupt/misdirected (and the slot may have been marked faulty).
/// The callback may run without issuing I/O (e.g. header-only fast path or no-entry cases).
/// If the caller needs to retain the message beyond the callback, it must clone it.
pub type ReadPrepareCallback<S, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> =
    fn(*mut Replica<S, WRITE_OPS, WRITE_OPS_WORDS>, Option<*mut MessagePrepare>, ReadOptions);

/// Callback invoked when `write_prepare` completes.
///
/// The second parameter is `Some(message)` when the prepare was successfully written to disk
/// and the in-memory header still matches. `None` indicates:
/// - The slot was already durable (no I/O needed)
/// - No write IOP was available
/// - The in-memory header changed during the write
///
/// The callback is always invoked exactly once per `write_prepare` call.
pub type WritePrepareCallback<S, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> =
    fn(*mut Replica<S, WRITE_OPS, WRITE_OPS_WORDS>, Option<*mut MessagePrepare>);

/// Callback invoked when journal recovery completes.
///
/// Called once after all header chunks have been read and validated, and the
/// journal transitions from `Status::Recovering` to `Status::Recovered`.
/// At this point the journal is ready for normal read/write operations.
pub type RecoverCallback<S, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> =
    fn(*mut Journal<S, WRITE_OPS, WRITE_OPS_WORDS>);

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Phases of the recovery pipeline.
enum RecoveryStage {
    Headers,
    Prepares,
    Slots,
    Fix,
    Done,
}

#[allow(dead_code)]
/// In-progress recovery bookkeeping.
struct Recovery<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> {
    callback: RecoverCallback<S, WRITE_OPS, WRITE_OPS_WORDS>,
    cluster: u128,
    op_prepare_max: u64,
    op_checkpoint: u64,
    solo: bool,

    stage: RecoveryStage,
    prepare_next_slot: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Identifies what kind of recovery data a read buffer contains.
enum RecoveryReadKind {
    Headers,
    Prepares,
}

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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Status {
    /// Initial state before recovery begins.
    Init,
    /// Journal is recovering; header chunks being read and validated.
    Recovering {
        /// Number of header sectors successfully read and validated.
        headers_recovered: u64,
        /// Total number of header sectors to recover.
        headers_total: u64,
    },
    /// Recovery complete; journal is ready for normal operation.
    Recovered,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Writing {
    /// No write in progress for this slot.
    None,
    /// A write is in progress for this exact op+checksum.
    Exact,
    /// A write is in progress for this slot but different op+checksum.
    Slot,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Recovery action to take for a header/prepare pair.
enum RecoveryDecision {
    /// Redundant headers and prepares match.
    Eql,
    /// Entry is empty (reserved).
    Nil,
    /// Redundant headers must be repaired locally from the prepare.
    Fix,
    /// Entry is faulty and must be repaired over VSR.
    Vsr,
    /// Truncate prepares beyond `op_prepare_max`.
    Cut,
    /// Truncate torn prepares beyond the recovery head.
    CutTorn,
    /// Should never happen (solo-only).
    Unreachable,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Matcher for a single boolean parameter in a recovery-case pattern.
enum Matcher {
    Any,
    IsFalse,
    IsTrue,
    AssertFalse,
    AssertTrue,
}

/// One row in the recovery decision table.
///
/// `pattern` matches 11 boolean parameters derived from a header/prepare pair.
/// The `decision_*` fields capture the action to take depending on solo/multi mode.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct RecoveryCase {
    label: &'static str,
    decision_multiple: RecoveryDecision,
    decision_single: RecoveryDecision,
    pattern: [Matcher; 11],
}

#[allow(dead_code)]
impl RecoveryCase {
    #[inline]
    const fn decision(&self, solo: bool) -> RecoveryDecision {
        if solo {
            self.decision_single
        } else {
            self.decision_multiple
        }
    }

    #[inline]
    fn check(&self, parameters: [bool; 11]) -> Result<bool, ()> {
        for (m, p) in self.pattern.iter().copied().zip(parameters.iter().copied()) {
            match m {
                Matcher::Any => {}
                Matcher::IsFalse => {
                    if p {
                        return Ok(false);
                    }
                }
                Matcher::IsTrue => {
                    if !p {
                        return Ok(false);
                    }
                }
                Matcher::AssertFalse => {
                    if p {
                        return Err(());
                    }
                }
                Matcher::AssertTrue => {
                    if !p {
                        return Err(());
                    }
                }
            }
        }
        Ok(true)
    }
}

const __: Matcher = Matcher::Any;
const _0: Matcher = Matcher::IsFalse;
const _1: Matcher = Matcher::IsTrue;
#[allow(dead_code)]
const A0: Matcher = Matcher::AssertFalse;
#[allow(dead_code)]
const A1: Matcher = Matcher::AssertTrue;

/// Pseudo-case used when torn prepares are detected during recovery.
#[allow(dead_code)]
const CASE_CUT_TORN: RecoveryCase = RecoveryCase {
    label: "@CUT_TORN",
    decision_multiple: RecoveryDecision::CutTorn,
    decision_single: RecoveryDecision::CutTorn,
    pattern: [__; 11],
};

/// Ordered recovery decision table (first and only match wins).
#[allow(dead_code)]
const RECOVERY_CASES: [RecoveryCase; 16] = [
    // @A - both invalid.
    RecoveryCase {
        label: "@A",
        decision_multiple: RecoveryDecision::Vsr,
        decision_single: RecoveryDecision::Vsr,
        pattern: [_0, __, _0, __, __, A0, A0, __, __, __, __],
    },
    // @B - header ok reserved, prepare invalid.
    RecoveryCase {
        label: "@B",
        decision_multiple: RecoveryDecision::Vsr,
        decision_single: RecoveryDecision::Vsr,
        pattern: [_1, _1, _0, __, __, A0, __, __, __, __, __],
    },
    // @C - header ok not reserved, prepare invalid.
    RecoveryCase {
        label: "@C",
        decision_multiple: RecoveryDecision::Vsr,
        decision_single: RecoveryDecision::Vsr,
        pattern: [_1, _0, _0, __, __, A0, __, __, __, __, __],
    },
    // @D - prepare ok reserved, header invalid.
    RecoveryCase {
        label: "@D",
        decision_multiple: RecoveryDecision::Vsr,
        decision_single: RecoveryDecision::Fix,
        pattern: [_0, __, _1, _1, __, __, A0, __, __, __, __],
    },
    // @E - prepare ok, not reserved, not maximum.
    RecoveryCase {
        label: "@E",
        decision_multiple: RecoveryDecision::Vsr,
        decision_single: RecoveryDecision::Fix,
        pattern: [_0, __, _1, _0, _0, _0, A0, __, __, __, __],
    },
    // @F - prepare ok, not reserved, maximum.
    RecoveryCase {
        label: "@F",
        decision_multiple: RecoveryDecision::Fix,
        decision_single: RecoveryDecision::Fix,
        pattern: [_0, __, _1, _0, _1, _0, A0, __, __, __, __],
    },
    // @G - header ok reserved, prepare ok not reserved.
    RecoveryCase {
        label: "@G",
        decision_multiple: RecoveryDecision::Fix,
        decision_single: RecoveryDecision::Fix,
        pattern: [_1, _1, _1, _0, __, _0, __, __, __, __, __],
    },
    // @H - prepare beyond op_prepare_max.
    RecoveryCase {
        label: "@H",
        decision_multiple: RecoveryDecision::Cut,
        decision_single: RecoveryDecision::Unreachable,
        pattern: [__, __, _1, _0, __, _1, __, __, __, __, __],
    },
    // @I - header.op > op_prepare_max, prepare not reserved.
    RecoveryCase {
        label: "@I",
        decision_multiple: RecoveryDecision::Cut,
        decision_single: RecoveryDecision::Unreachable,
        pattern: [_1, _0, _1, _0, __, _0, _1, __, __, A0, __],
    },
    // @J - header beyond op_prepare_max, prepare reserved.
    RecoveryCase {
        label: "@J",
        decision_multiple: RecoveryDecision::Cut,
        decision_single: RecoveryDecision::Unreachable,
        pattern: [_1, _0, _1, _1, __, __, _1, __, __, A0, __],
    },
    // @K - prepare reserved, header not reserved.
    RecoveryCase {
        label: "@K",
        decision_multiple: RecoveryDecision::Vsr,
        decision_single: RecoveryDecision::Vsr,
        pattern: [_1, _0, _1, _1, __, __, _0, __, __, __, __],
    },
    // @L normal path: reserved.
    RecoveryCase {
        label: "@L",
        decision_multiple: RecoveryDecision::Nil,
        decision_single: RecoveryDecision::Nil,
        pattern: [_1, _1, _1, _1, __, __, __, A1, A1, A0, A1],
    },
    // @M - header.op < prepare.op.
    RecoveryCase {
        label: "@M",
        decision_multiple: RecoveryDecision::Fix,
        decision_single: RecoveryDecision::Fix,
        pattern: [_1, _0, _1, _0, __, _0, _0, _0, _0, _1, __],
    },
    // @N - header.op > prepare.op.
    RecoveryCase {
        label: "@N",
        decision_multiple: RecoveryDecision::Vsr,
        decision_single: RecoveryDecision::Vsr,
        pattern: [_1, _0, _1, _0, __, _0, _0, _0, _0, _0, __],
    },
    // @O - header.view != prepare.view (op matches, checksum differs).
    RecoveryCase {
        label: "@O",
        decision_multiple: RecoveryDecision::Vsr,
        decision_single: RecoveryDecision::Vsr,
        pattern: [_1, _0, _1, _0, __, _0, _0, _0, _1, A0, A0],
    },
    // @P - normal path: prepare.
    RecoveryCase {
        label: "@P",
        decision_multiple: RecoveryDecision::Fix,
        decision_single: RecoveryDecision::Fix,
        pattern: [_1, _0, _1, _0, __, _0, _0, _1, A1, A0, A1],
    },
];

/// Selects the recovery case for the given header/prepare pair.
///
/// `op_max` and `op_prepare_max` describe the current recovery head and are used
/// to decide when to truncate or repair entries.
#[allow(dead_code)]
#[inline]
fn recovery_case(
    header: Option<&HeaderPrepare>,
    prepare: Option<&HeaderPrepare>,
    op_max: u64,
    op_prepare_max: u64,
    solo: bool,
) -> &'static RecoveryCase {
    let is_header = header.is_some();
    let is_prepare = prepare.is_some();
    let parameters = [
        is_header,
        is_header && header.unwrap().operation == Operation::RESERVED,
        is_prepare,
        is_prepare && prepare.unwrap().operation == Operation::RESERVED,
        is_prepare && prepare.unwrap().op == op_max,
        is_prepare && prepare.unwrap().op > op_prepare_max,
        is_header && header.unwrap().op > op_prepare_max,
        is_header && is_prepare && header.unwrap().checksum == prepare.unwrap().checksum,
        is_header && is_prepare && header.unwrap().op == prepare.unwrap().op,
        is_header && is_prepare && header.unwrap().op < prepare.unwrap().op,
        is_header && is_prepare && header.unwrap().view == prepare.unwrap().view,
    ];

    let mut result = None;
    for c in RECOVERY_CASES.iter() {
        match c.check(parameters) {
            Ok(true) => {
                assert!(result.is_none());
                result = Some(c);
            }
            Ok(false) => {}
            Err(()) => {
                panic!(
                    "recovery_case: impossible state: case={} decision={:?} parameters={:?}",
                    c.label,
                    c.decision(solo),
                    parameters
                );
            }
        }
    }
    result.unwrap_or_else(|| panic!("recovery_case: no match for parameters={:?}", parameters))
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

    /// Recovery-time: header chunks with outstanding or completed read requests.
    pub header_chunks_requested: HeaderChunks,

    /// Recovery-time: header chunks whose reads have completed and data copied.
    pub header_chunks_recovered: HeaderChunks,

    /// Pool of read operation descriptors for prepare reads (commit + repair).
    pub reads: IOPSType<Read<S, WRITE_OPS, WRITE_OPS_WORDS>, READ_OPS, READ_OPS_WORDS>,

    /// In-flight reads attributed to the commit path (`destination_replica == None`).
    pub reads_commit_count: usize,

    /// In-flight reads attributed to the repair path (`destination_replica != None`).
    pub reads_repair_count: usize,

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
    pub status: Status,
}

impl<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize>
    Journal<S, WRITE_OPS, WRITE_OPS_WORDS>
{
    #[inline]
    fn replica_from_journal(
        journal: *mut Journal<S, WRITE_OPS, WRITE_OPS_WORDS>,
    ) -> *mut Replica<S, WRITE_OPS, WRITE_OPS_WORDS> {
        container_of!(journal, Replica<S, WRITE_OPS, WRITE_OPS_WORDS>, journal)
    }

    #[inline]
    fn replica_ptr(&mut self) -> *mut Replica<S, WRITE_OPS, WRITE_OPS_WORDS> {
        Self::replica_from_journal(self as *mut _)
    }

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
            reads: IOPSType::default(),
            reads_commit_count: 0,
            reads_repair_count: 0,
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
    fn slot_for_header(&self, header: &HeaderPrepare) -> Slot {
        assert!(self.slot_with_op(header.op).is_some());
        self.slot_for_op(header.op)
    }

    /// Returns the slot only if this exact header exists in the journal.
    #[inline]
    fn slot_with_header(&self, header: &HeaderPrepare) -> Option<Slot> {
        self.slot_with_op_and_checksum(header.op, header.checksum)
    }

    // =========================================================================
    // Header Predicates
    // =========================================================================

    #[inline]
    fn has_header(&self, header: &HeaderPrepare) -> bool {
        self.header_with_op_and_checksum(header.op, header.checksum)
            .is_some()
    }

    #[inline]
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

    /// Read a prepare from disk. There must be a matching in-memory header.
    ///
    /// This is the safe entry point used by the normal commit path and by request/repair
    /// handlers that already have an exact (op, checksum) identity to read.
    ///
    /// On failure, the callback is invoked with `None` and **no read I/O is issued**.
    /// This includes cases where the in-memory slot is not inhabited or no exact match exists.
    #[cfg_attr(not(test), allow(dead_code))]
    fn read_prepare(
        &mut self,
        callback: ReadPrepareCallback<S, WRITE_OPS, WRITE_OPS_WORDS>,
        options: ReadOptions,
    ) {
        assert!(matches!(self.status, Status::Recovered));
        assert!(options.checksum != 0);
        assert!(self.reads.available() > 0);

        let replica_ptr = self.replica_ptr();
        let replica = unsafe { &mut *replica_ptr };

        if options.op > replica.op {
            self.read_prepare_log(options.op, Some(options.checksum), "beyond replica.op");
            callback(replica_ptr, None, options);
            return;
        }

        let Some(slot) = self.slot_with_op_and_checksum(options.op, options.checksum) else {
            self.read_prepare_log(options.op, Some(options.checksum), "no entry exactly");
            callback(replica_ptr, None, options);
            return;
        };

        if self.prepare_inhabited[slot.index()]
            && self.prepare_checksums[slot.index()] == options.checksum
        {
            self.read_prepare_with_op_and_checksum(callback, options);
        } else {
            self.read_prepare_log(options.op, Some(options.checksum), "no matching prepare");
            callback(replica_ptr, None, options);
        }
    }

    /// Read a prepare from disk. There may or may not be a matching in-memory header.
    ///
    /// This is the shared primitive used by:
    /// - `read_prepare` (normal path)
    /// - request/repair paths once the checksum is known
    /// - recovery checks (step 4) that need to validate/repair prepares
    ///
    /// The caller must ensure:
    /// - `prepare_inhabited[slot] == true`
    /// - `prepare_checksums[slot] == options.checksum`
    ///
    /// This method enforces read IOP accounting (commit vs repair) to prevent repair reads from
    /// starving commit reads. It may complete inline (no I/O) for header-only prepares.
    /// Full validation happens in `read_prepare_with_op_and_checksum_on_read`.
    #[cfg_attr(not(test), allow(dead_code))]
    fn read_prepare_with_op_and_checksum(
        &mut self,
        callback: ReadPrepareCallback<S, WRITE_OPS, WRITE_OPS_WORDS>,
        options: ReadOptions,
    ) {
        assert!(matches!(self.status, Status::Recovered));
        assert!(options.checksum != 0);

        let replica_ptr = self.replica_ptr();
        let replica = unsafe { &mut *replica_ptr };

        let slot = self.slot_for_op(options.op);
        assert!(self.prepare_inhabited[slot.index()]);
        assert!(self.prepare_checksums[slot.index()] == options.checksum);

        if options.destination_replica.is_none() {
            assert!(self.reads.available() > 0);
        }

        let mut message = replica.message_pool.get::<PrepareCmd>();

        // Default to full slot reads; exact headers can tighten this.
        let mut message_size: usize = constants::MESSAGE_SIZE_MAX_USIZE;

        // Optimization: if the exact header is in-memory, we can read fewer bytes (or skip entirely).
        if let Some(exact) = self.header_with_op_and_checksum(options.op, options.checksum) {
            if exact.size as usize == size_of::<HeaderPrepare>() {
                // Header-only prepare; no disk required. Populate the message buffer
                // with the header and zero the remainder of the first sector.
                let buf_ptr = message.buffer_ptr();
                let buf_len = constants::MESSAGE_SIZE_MAX_USIZE;
                assert!(buf_len >= constants::SECTOR_SIZE);

                unsafe {
                    ptr::copy_nonoverlapping(
                        exact as *const HeaderPrepare as *const u8,
                        buf_ptr,
                        size_of::<HeaderPrepare>(),
                    );

                    let sector = core::slice::from_raw_parts_mut(buf_ptr, constants::SECTOR_SIZE);
                    sector[size_of::<HeaderPrepare>()..].fill(0);
                }

                callback(replica_ptr, Some(&mut message as *mut _), options);
                return;
            } else {
                // As an optimization, read only the exact message size (sector-aligned).
                message_size = constants::sector_ceil(exact.size as usize);
                assert!(message_size <= constants::MESSAGE_SIZE_MAX_USIZE);
            }
        }

        // ---------------------------------------------------------------------
        // IOP accounting (commit vs repair)
        // ---------------------------------------------------------------------

        if options.destination_replica.is_none() {
            self.reads_commit_count += 1;
        } else {
            // Repair reads are capped so they cannot starve commit reads.
            if self.reads_repair_count == READS_REPAIR_COUNT_MAX {
                self.read_prepare_log(options.op, Some(options.checksum), "waiting for IOP");
                callback(replica_ptr, None, options);
                return;
            }
            self.reads_repair_count += 1;
        }

        assert!(self.reads_commit_count <= READS_COMMIT_COUNT_MAX);
        assert!(self.reads_repair_count <= READS_REPAIR_COUNT_MAX);

        let read = self.reads.acquire().expect("read pool exhausted");

        unsafe {
            ptr::addr_of_mut!((*read).journal).write(self as *mut _);
            ptr::addr_of_mut!((*read).options).write(options);
            ptr::addr_of_mut!((*read).callback).write(callback);
            ptr::addr_of_mut!((*read).message).write(MaybeUninit::new(message));
        }

        let message_ptr = unsafe { (*read).message.as_mut_ptr() };
        let buf_ptr = unsafe { (*message_ptr).buffer_ptr() };
        let buf_len_total = constants::MESSAGE_SIZE_MAX_USIZE;
        assert!(message_size <= buf_len_total);

        let buffer: &mut [u8] = unsafe { core::slice::from_raw_parts_mut(buf_ptr, message_size) };
        let offset = (constants::MESSAGE_SIZE_MAX as u64) * (slot.index() as u64);
        assert!(offset.is_multiple_of(constants::SECTOR_SIZE as u64));

        let storage = unsafe { &mut *self.storage };

        storage.read_sectors(
            Self::read_prepare_with_op_and_checksum_on_read,
            unsafe { &mut *(*read).completion.as_mut_ptr() },
            buffer,
            S::WAL_PREPARES_ZONE,
            offset,
        );
    }

    /// Completion callback for prepare reads.
    ///
    /// Recovers the read descriptor, updates IOP accounting, and validates the
    /// on-disk prepare (header, body, padding) against the expected identity.
    /// On validation failure, marks the slot faulty/dirty if it still refers to
    /// the same prepare, then invokes the callback with `None`.
    fn read_prepare_with_op_and_checksum_on_read(completion: &mut S::Read) {
        let read: *mut Read<S, WRITE_OPS, WRITE_OPS_WORDS> =
            container_of!(completion, Read<S, WRITE_OPS, WRITE_OPS_WORDS>, completion);

        let journal = unsafe { &mut *(*read).journal };
        assert!(matches!(journal.status, Status::Recovered));

        let callback = unsafe { (*read).callback };
        let options = unsafe { (*read).options };
        let mut message = unsafe { ptr::read((*read).message.as_ptr()) };
        let message_ptr: *mut MessagePrepare = &mut message;

        // Release IOP accounting before invoking user callbacks.
        if options.destination_replica.is_none() {
            assert!(journal.reads_commit_count > 0);
            journal.reads_commit_count -= 1;
        } else {
            assert!(journal.reads_repair_count > 0);
            journal.reads_repair_count -= 1;
        }
        journal.reads.release(read);

        let replica_ptr = Self::replica_from_journal(journal as *mut _);
        let replica = unsafe { &mut *replica_ptr };

        if options.op > replica.op {
            journal.read_prepare_log(options.op, Some(options.checksum), "beyond replica.op");
            callback(replica_ptr, None, options);
            return;
        }

        // The prepare may have been re-written since the read began.
        let slot = journal.slot_for_op(options.op);
        let checksum_inhabited = journal.prepare_inhabited[slot.index()];
        let checksum_match = journal.prepare_checksums[slot.index()] == options.checksum;
        if !checksum_inhabited || !checksum_match {
            journal.read_prepare_log(
                options.op,
                Some(options.checksum),
                "prepare changed during read",
            );
            callback(replica_ptr, None, options);
            return;
        }

        // Validate header identity and message integrity.
        let header = message.header();
        let expected_cluster = replica.cluster;

        let error_reason: Option<&'static str> = (|| {
            if !header.valid_checksum() {
                return Some("corrupt header after read");
            }

            assert!(header.invalid().is_none());

            if header.cluster != expected_cluster {
                return Some("wrong cluster");
            }

            if header.op != options.op {
                return Some("op changed during read");
            }

            if header.checksum != options.checksum {
                return Some("checksum changed during read");
            }

            let size = header.size as usize;
            if size < size_of::<HeaderPrepare>() || size > constants::MESSAGE_SIZE_MAX_USIZE {
                return Some("invalid message size");
            }

            let bytes = unsafe { core::slice::from_raw_parts(message.buffer_ptr(), size) };
            let body_used = &bytes[size_of::<HeaderPrepare>()..size];
            if !header.valid_checksum_body(body_used) {
                return Some("corrupt body after read");
            }

            let padded = constants::sector_ceil(size);
            assert!(padded.is_multiple_of(constants::SECTOR_SIZE));
            if padded > constants::MESSAGE_SIZE_MAX_USIZE {
                return Some("invalid message size");
            }

            // Padding must be zero to detect torn or partial writes.
            let padding = unsafe {
                core::slice::from_raw_parts(message.buffer_ptr().add(size), padded - size)
            };
            if !zero::is_all_zeros(padding) {
                return Some("corrupt sector padding");
            }

            None
        })();

        if let Some(reason) = error_reason {
            // Mark the slot faulty if the exact header still matches what we were trying to read.
            if let Some(slot) = journal.slot_with_op_and_checksum(options.op, options.checksum) {
                journal.faulty.set(slot.index());
                journal.dirty.set(slot.index());
            }

            journal.read_prepare_log(options.op, Some(options.checksum), reason);
            callback(replica_ptr, None, options);
        } else {
            callback(replica_ptr, Some(message_ptr), options);
        }
    }

    /// Debug hook for read-path notices; intentionally a no-op in production builds.
    #[inline]
    fn read_prepare_log(&self, op: u64, checksum: Option<Checksum128>, notice: &str) {
        let _ = (op, checksum, notice);
    }

    /// Writes a prepare message and its header to the WAL.
    ///
    /// This is a two-stage write:
    /// 1. Write the prepare payload to the Prepares ring
    /// 2. Update the header sector in the Headers ring
    ///
    /// If the slot is already durable or no IOP is available, the callback is
    /// invoked with `None` and no I/O is issued. The callback receives `Some(message)`
    /// only when both writes complete and the in-memory header still matches.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The journal is not in `Status::Recovered`
    /// - `message` is null or not a non-reserved `Prepare` header
    /// - The header does not exist in memory or a write is already in flight
    #[cfg_attr(not(test), allow(dead_code))]
    fn write_prepare(
        &mut self,
        callback: WritePrepareCallback<S, WRITE_OPS, WRITE_OPS_WORDS>,
        message: *mut MessagePrepare,
    ) {
        assert!(matches!(self.status, Status::Recovered));
        assert!(!message.is_null());

        let replica_ptr = self.replica_ptr();

        let header = unsafe { (*message).header() };
        assert!(header.command == Command::Prepare);
        assert!(header.operation != Operation::RESERVED);

        let header_size = header.size as usize;
        assert!(header_size <= constants::MESSAGE_SIZE_MAX as usize);

        assert!(self.has_header(header));
        assert!(self.writing(header) == Writing::None);

        let slot = self
            .slot_with_header(header)
            .expect("write_prepare expects the header to exist in memory");

        if !self.dirty.is_set(slot.index()) {
            assert!(!self.faulty.is_set(slot.index()));
            assert!(self.prepare_inhabited[slot.index()]);
            assert!(self.prepare_checksums[slot.index()] == header.checksum);
            assert!(self.headers_redundant[slot.index()].checksum == header.checksum);

            // Already durable; no I/O required.
            callback(replica_ptr, None);
            return;
        }

        assert!(self.has_dirty(header));

        let Some(write) = self.writes.acquire() else {
            // No IOP available; caller must retry later.
            callback(replica_ptr, None);
            return;
        };

        // Don't read the data within `Write`, as it may be uninitialized.
        // Doing so will drop the old value which could lead to undefined behavior.
        // Write directly to the fields of the `Write` struct.
        unsafe {
            ptr::addr_of_mut!((*write).journal).write(self as *mut _);
            ptr::addr_of_mut!((*write).prepare_callback).write(Some(callback));
            ptr::addr_of_mut!((*write).message).write(message);
            ptr::addr_of_mut!((*write).op).write(header.op);
            ptr::addr_of_mut!((*write).checksum).write(header.checksum);
        }

        let buffer_len = constants::sector_ceil(header_size);
        assert!(buffer_len <= constants::MESSAGE_SIZE_MAX as usize);

        let buffer_ptr = unsafe { (*message).buffer_ptr() };
        // SAFETY: Message buffers are MESSAGE_SIZE_MAX bytes and sector-aligned.
        let buffer = unsafe { core::slice::from_raw_parts_mut(buffer_ptr, buffer_len) };
        buffer[header_size..].fill(0);
        assert!(zero::is_all_zeros(&buffer[header_size..]));
        self.prepare_inhabited[slot.index()] = false;
        self.prepare_checksums[slot.index()] = 0;

        let offset = (constants::MESSAGE_SIZE_MAX as u64) * (slot.index() as u64);
        unsafe {
            self.write_sectors(
                write,
                Self::write_prepare_header,
                buffer,
                Ring::Prepares,
                offset,
            )
        };
    }

    /// Completion callback after the prepare payload write.
    ///
    /// Restores per-slot prepare bookkeeping, updates the redundant header, and
    /// submits a header sector write. If the in-memory header changed while the
    /// payload write was in flight, the slot is left dirty and the write is
    /// released without completing the prepare.
    fn write_prepare_header(write: *mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>) {
        let journal = unsafe { &mut *(*write).journal };
        assert!(matches!(journal.status, Status::Recovered));

        let message = unsafe { (*write).message };
        assert!(!message.is_null());
        let header = unsafe { (*message).header() };

        assert!(journal.writing(header) == Writing::Exact);

        let slot = journal.slot_for_header(header);

        journal.prepare_inhabited[slot.index()] = true;
        journal.prepare_checksums[slot.index()] = header.checksum;

        if !journal.has_header(header) {
            // The in-memory entry changed while the prepare was being written.
            Self::write_prepare_release(journal as *mut _, write, None);
            journal.dirty.set(slot.index());
            return;
        }

        if journal.headers_redundant[slot.index()].operation == Operation::RESERVED
            && journal.headers_redundant[slot.index()].checksum == 0
        {
            assert!(journal.faulty.is_set(slot.index()));
        }

        journal.headers_redundant[slot.index()] = *header;

        let sector_index = slot.index() / HEADERS_PER_SECTOR;
        let buffer = journal.header_sector(sector_index, write);
        // Extract pointer and length to end the mutable borrow of journal,
        // allowing write_sectors to borrow journal again.
        let buffer_ptr = buffer.as_ptr();
        let buffer_len = buffer.len();

        let offset = Ring::Headers.offset(slot);
        assert!(offset.is_multiple_of(constants::SECTOR_SIZE as u64));

        // SAFETY: buffer_ptr/buffer_len come from header_sector which returns a slice
        // into write_headers_sectors. The buffer remains valid for the duration of the
        // write operation due to the range locking protocol.
        unsafe {
            let buffer = core::slice::from_raw_parts(buffer_ptr, buffer_len);
            journal.write_sectors(
                write,
                Self::write_prepare_on_write_header,
                buffer,
                Ring::Headers,
                offset,
            );
        }
    }

    /// Builds the header sector buffer for a write.
    ///
    /// Uses a per-write staging buffer keyed by the write's pool index to avoid
    /// concurrent mutation. The buffer is populated from `headers_redundant`.
    ///
    /// # Panics
    ///
    /// Panics if the sector index is out of range or `write` does not come from
    /// the journal's write pool.
    fn header_sector(
        &mut self,
        sector_index: usize,
        write: *const Write<S, WRITE_OPS, WRITE_OPS_WORDS>,
    ) -> &mut [u8] {
        assert!(matches!(self.status, Status::Recovered));
        assert!(sector_index < SLOT_COUNT / HEADERS_PER_SECTOR);

        let base = self.writes.items.as_ptr() as *const Write<S, WRITE_OPS, WRITE_OPS_WORDS>;
        let diff = (write as usize).wrapping_sub(base as usize);
        assert!(diff.is_multiple_of(size_of::<Write<S, WRITE_OPS, WRITE_OPS_WORDS>>()));
        let write_index = diff / size_of::<Write<S, WRITE_OPS, WRITE_OPS_WORDS>>();
        assert!(write_index < self.write_headers_sectors.len());
        assert!(self.writes.items.len() == self.write_headers_sectors.len());

        let sector_slot = Slot::new(sector_index * HEADERS_PER_SECTOR);

        let sector_bytes = &mut self.write_headers_sectors[write_index];
        let sector_headers: &mut [HeaderPrepare] = unsafe {
            core::slice::from_raw_parts_mut(
                sector_bytes.as_mut_ptr() as *mut HeaderPrepare,
                HEADERS_PER_SECTOR,
            )
        };

        sector_headers.copy_from_slice(
            &self.headers_redundant[sector_slot.index()..sector_slot.index() + HEADERS_PER_SECTOR],
        );

        for (i, sh) in sector_headers.iter().enumerate() {
            let slot = Slot::new(sector_slot.index() + i);

            if sh.operation == Operation::RESERVED && sh.checksum == 0 {
                assert!(self.faulty.is_set(slot.index()));
            } else if self.faulty.is_set(slot.index()) {
                // An entry may be faulty even with a non-zero checksum (eg. torn write).
            }
        }

        &mut sector_bytes[..]
    }

    /// Completion callback after the header sector write.
    ///
    /// Verifies the header still matches the in-memory slot and redundant copy,
    /// clears dirty/faulty flags, and releases the write. If any check fails, the
    /// write is released without marking the slot durable.
    fn write_prepare_on_write_header(write: *mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>) {
        let journal = unsafe { &mut *(*write).journal };
        assert!(matches!(journal.status, Status::Recovered));
        let message = unsafe { (*write).message };
        assert!(!message.is_null());

        let header = unsafe { (*message).header() };
        assert!(journal.writing(header) == Writing::Exact);

        if !journal.has_header(header) {
            Self::write_prepare_release(journal as *mut _, write, None);
            return;
        }

        let slot = journal
            .slot_with_header(header)
            .expect("header must exist after has_header()");
        if journal.headers_redundant[slot.index()].checksum != header.checksum {
            assert!(journal.dirty.is_set(slot.index()));
            Self::write_prepare_release(journal as *mut _, write, None);
            return;
        }

        if !journal.prepare_inhabited[slot.index()]
            || journal.prepare_checksums[slot.index()] != header.checksum
        {
            Self::write_prepare_release(journal as *mut _, write, None);
            return;
        }

        journal.dirty.unset(slot.index());
        journal.faulty.unset(slot.index());

        Self::write_prepare_release(journal as *mut _, write, Some(message));
    }

    /// Releases a write and invokes the user callback.
    ///
    /// The IOP is returned to the pool before calling into user code to avoid
    /// reentrancy issues and to allow immediate reuse.
    fn write_prepare_release(
        journal: *mut Journal<S, WRITE_OPS, WRITE_OPS_WORDS>,
        write: *mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>,
        wrote: Option<*mut MessagePrepare>,
    ) {
        assert!(!journal.is_null());
        assert!(!write.is_null());

        let callback =
            unsafe { (*write).prepare_callback }.expect("write_prepare callback missing");
        let write_message = unsafe { (*write).message };

        // Release the IOP back to the pool before calling into user code.
        {
            let journal_mut = unsafe { &mut *journal };
            journal_mut.writes.release(write);

            if !write_message.is_null() {
                let header = unsafe { (*write_message).header() };
                assert!(journal_mut.writing(header) == Writing::None);
            }
        }

        let replica_ptr = Self::replica_from_journal(journal);
        callback(replica_ptr, wrote);
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
