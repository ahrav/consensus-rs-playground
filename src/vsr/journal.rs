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

use core::mem::MaybeUninit;
use core::ptr;

use crate::container_of;
use crate::vsr::iops::IOPSType;
use crate::vsr::journal_primitives::Ring;
use crate::vsr::storage::{Storage, Synchronicity};

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
/// - A range cannot be both `locked` and have a non-null `next` (mutually exclusive)
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
    #[inline]
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.ring != other.ring {
            return false;
        }

        let len_a = self.buffer_len as u64;
        let len_b = other.buffer_len as u64;

        if self.offset < other.offset {
            self.offset + len_a > other.offset
        } else {
            other.offset + len_b > self.offset
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
///
/// # Current Limitations
///
/// This is a minimal implementation for the range-locking machinery. The full
/// Journal port will add: message pointer, high-level callback, and additional
/// metadata.
#[repr(C)]
pub struct Write<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> {
    /// Back-pointer to the owning Journal. Set by `write_sectors`.
    pub journal: *mut Journal<S, WRITE_OPS, WRITE_OPS_WORDS>,

    /// Slot index within the log. Used to assert no concurrent writes to same slot.
    // TODO: This will be replaced with full slot tracking in the complete Journal.
    pub slot_index: u32,

    /// The range descriptor containing buffer, offset, and locking state.
    pub range: Range<S, WRITE_OPS, WRITE_OPS_WORDS>,
}

/// WAL journal skeleton implementing range-locked sector writes.
///
/// The `Journal` coordinates write operations to the Write-Ahead Log, ensuring
/// that overlapping writes are serialized. It maintains a pool of `Write` objects
/// and implements the range-locking protocol.
///
/// # Current Limitations
///
/// This is a partial implementation containing only the range-locking machinery.
/// The full Journal will include: headers/prepares index arrays, dirty/faulty
/// bitsets, view change state, and recovery logic.
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

    /// Pool of write operation descriptors.
    pub writes: IOPSType<Write<S, WRITE_OPS, WRITE_OPS_WORDS>, WRITE_OPS, WRITE_OPS_WORDS>,
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
        Self {
            storage,
            replica,
            writes: IOPSType::default(),
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
                Synchronicity::AlwaysAsynchronous => assert!(!(*write).range.locked),
                Synchronicity::AlwaysSynchronous => assert!((*write).range.locked),
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
