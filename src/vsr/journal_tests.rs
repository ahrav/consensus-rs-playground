use super::*;
use crate::message_pool::MessagePool;
use crate::storage::AlignedBuf;
use crate::vsr::Release;
use crate::vsr::superblock;
use core::cell::RefCell;
use std::collections::HashMap;

/// Alignment required for `HeaderPrepareRaw` (16 bytes).
const HEADER_ALIGN: usize = core::mem::align_of::<HeaderPrepareRaw>();

// =========================================================================
// Test Infrastructure: MockStorage
// =========================================================================

/// Zone identifier for mock storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Completion token for mock read operations.
/// The actual contents don't matter - we just need it to exist for container_of!
#[repr(C)]
struct MockReadCompletion {
    /// Callback to invoke on completion.
    callback: fn(&mut MockReadCompletion),
}

/// Record of a single write operation for verification.
#[derive(Debug, Clone)]
struct WriteRecord {
    zone: Zone,
    offset: u64,
    len: usize,
}

/// Record of a single read operation for verification.
#[derive(Debug, Clone)]
#[allow(dead_code)] // zone/offset fields reserved for future verification extensions.
struct ReadRecord {
    zone: Zone,
    offset: u64,
    len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ReadKey {
    zone: Zone,
    offset: u64,
}

/// Mock storage implementation for deterministic testing.
///
/// The `ASYNC` const generic controls the synchronicity model:
/// - `false`: AlwaysSynchronous (callbacks invoked immediately)
/// - `true`: AlwaysAsynchronous (callbacks deferred until `drain_callbacks`)
struct MockStorage<const ASYNC: bool> {
    /// Log of all writes submitted to storage.
    write_log: RefCell<Vec<WriteRecord>>,
    /// Log of all reads submitted to storage.
    read_log: RefCell<Vec<ReadRecord>>,
    /// Data to return for reads keyed by (zone, offset).
    read_data: RefCell<HashMap<ReadKey, Vec<u8>>>,
    /// Pending callbacks to invoke (for deferred mode).
    pending_callbacks: RefCell<Vec<*mut MockWriteCompletion>>,
    /// Pending read callbacks to invoke (for deferred mode).
    pending_read_callbacks: RefCell<Vec<*mut MockReadCompletion>>,
}

impl<const ASYNC: bool> MockStorage<ASYNC> {
    /// Creates a new mock storage.
    fn new() -> Self {
        Self {
            write_log: RefCell::new(Vec::new()),
            read_log: RefCell::new(Vec::new()),
            read_data: RefCell::new(HashMap::new()),
            pending_callbacks: RefCell::new(Vec::new()),
            pending_read_callbacks: RefCell::new(Vec::new()),
        }
    }

    fn write_count(&self) -> usize {
        self.write_log.borrow().len()
    }

    fn read_count(&self) -> usize {
        self.read_log.borrow().len()
    }

    fn set_read_data(&self, zone: Zone, offset: u64, data: Vec<u8>) {
        self.read_data
            .borrow_mut()
            .insert(ReadKey { zone, offset }, data);
    }

    fn store_write_data(&self, zone: Zone, offset: u64, data: &[u8]) {
        let mut map = self.read_data.borrow_mut();
        let entry = map.entry(ReadKey { zone, offset }).or_default();
        if entry.len() < data.len() {
            entry.resize(data.len(), 0);
        }
        entry[..data.len()].copy_from_slice(data);
    }

    /// Invokes all pending callbacks in order.
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

    /// Invokes all pending read callbacks in order.
    fn drain_read_callbacks(&self) {
        let callbacks: Vec<_> = self.pending_read_callbacks.borrow_mut().drain(..).collect();
        for read_ptr in callbacks {
            // SAFETY: The pointer was stored from a valid &mut reference in read_sectors.
            // The read completion structure is still alive (caller ensures lifetime).
            unsafe {
                let read = &mut *read_ptr;
                (read.callback)(read);
            }
        }
    }
}

impl<const ASYNC: bool> Storage for MockStorage<ASYNC> {
    type Read = MockReadCompletion;
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
        callback: fn(&mut Self::Read),
        read: &mut Self::Read,
        buffer: &mut [u8],
        zone: Self::Zone,
        offset: u64,
    ) {
        self.read_log.borrow_mut().push(ReadRecord {
            zone,
            offset,
            len: buffer.len(),
        });

        let data = self
            .read_data
            .borrow()
            .get(&ReadKey { zone, offset })
            .cloned()
            .expect("missing read data for zone/offset");
        assert!(buffer.len() <= data.len());
        buffer.copy_from_slice(&data[..buffer.len()]);

        read.callback = callback;

        if ASYNC {
            self.pending_read_callbacks
                .borrow_mut()
                .push(read as *mut _);
        } else {
            (read.callback)(read);
        }
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
        self.store_write_data(zone, offset, buffer);

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
type TestReplica = Replica<TestMockStorage, 32, 1>;

/// Dummy callback for tests that don't need callback verification.
fn dummy_callback(_: *mut TestWrite) {}
fn dummy_callback_async(_: *mut Write<MockStorage<true>, 32, 1>) {}

thread_local! {
    static PREPARE_CALLBACKS: RefCell<Vec<Option<*mut MessagePrepare>>> =
        const { RefCell::new(Vec::new()) };
}

thread_local! {
    static READ_PREPARE_CALLBACKS: RefCell<Vec<(Option<MessagePrepare>, ReadOptions)>> =
        const { RefCell::new(Vec::new()) };
}

fn record_prepare_callback(_replica: *mut TestReplica, message: Option<*mut MessagePrepare>) {
    PREPARE_CALLBACKS.with(|callbacks| callbacks.borrow_mut().push(message));
}

fn record_prepare_callback_async(
    _replica: *mut Replica<MockStorage<true>, 32, 1>,
    message: Option<*mut MessagePrepare>,
) {
    PREPARE_CALLBACKS.with(|callbacks| callbacks.borrow_mut().push(message));
}

fn record_read_prepare_callback(
    _replica: *mut TestReplica,
    message: Option<*mut MessagePrepare>,
    options: ReadOptions,
) {
    let cloned = message.map(|ptr| unsafe { (*ptr).clone() });
    READ_PREPARE_CALLBACKS.with(|callbacks| callbacks.borrow_mut().push((cloned, options)));
}

fn record_read_prepare_callback_async(
    _replica: *mut Replica<MockStorage<true>, 32, 1>,
    message: Option<*mut MessagePrepare>,
    options: ReadOptions,
) {
    let cloned = message.map(|ptr| unsafe { (*ptr).clone() });
    READ_PREPARE_CALLBACKS.with(|callbacks| callbacks.borrow_mut().push((cloned, options)));
}

fn reset_prepare_callbacks() {
    PREPARE_CALLBACKS.with(|callbacks| callbacks.borrow_mut().clear());
}

fn take_prepare_callbacks() -> Vec<Option<*mut MessagePrepare>> {
    PREPARE_CALLBACKS.with(|callbacks| callbacks.borrow_mut().drain(..).collect())
}

fn reset_read_prepare_callbacks() {
    READ_PREPARE_CALLBACKS.with(|callbacks| callbacks.borrow_mut().clear());
}

fn take_read_prepare_callbacks() -> Vec<(Option<MessagePrepare>, ReadOptions)> {
    READ_PREPARE_CALLBACKS.with(|callbacks| callbacks.borrow_mut().drain(..).collect())
}

// =========================================================================
// HeaderChunk Unit Tests
// =========================================================================

#[test]
fn header_chunk_fields_accessible() {
    let chunk = HeaderChunk {
        index: 0,
        offset: 0,
        len: WAL_HEADER_SIZE,
    };
    assert_eq!(chunk.index, 0);
    assert_eq!(chunk.offset, 0);
    assert_eq!(chunk.len, WAL_HEADER_SIZE);
}

#[test]
fn header_chunk_end_calculation() {
    let chunk = HeaderChunk {
        index: 0,
        offset: 4096,
        len: 8192,
    };
    assert_eq!(chunk.end(), 4096 + 8192);
}

#[test]
fn header_chunk_assert_invariants_valid_first_chunk() {
    let chunk = HeaderChunk {
        index: 0,
        offset: 0,
        len: TestJournal::header_chunk_len_for_offset(0),
    };
    chunk.assert_invariants(); // Should not panic
}

#[test]
#[should_panic]
fn header_chunk_assert_invariants_index_out_of_bounds() {
    let chunk = HeaderChunk {
        index: HEADER_CHUNK_COUNT,
        offset: 0,
        len: 4096,
    };
    chunk.assert_invariants();
}

#[test]
#[should_panic]
fn header_chunk_assert_invariants_zero_length() {
    let chunk = HeaderChunk {
        index: 0,
        offset: 0,
        len: 0,
    };
    chunk.assert_invariants();
}

#[test]
#[should_panic]
fn header_chunk_assert_invariants_exceeds_message_size_max() {
    let chunk = HeaderChunk {
        index: 0,
        offset: 0,
        len: constants::MESSAGE_SIZE_MAX as usize + 1,
    };
    chunk.assert_invariants();
}

#[test]
#[should_panic]
fn header_chunk_assert_invariants_offset_exceeds_headers_size() {
    let chunk = HeaderChunk {
        index: 0,
        offset: HEADERS_SIZE,
        len: 4096,
    };
    chunk.assert_invariants();
}

#[test]
#[should_panic]
fn header_chunk_assert_invariants_end_exceeds_headers_size() {
    let chunk = HeaderChunk {
        index: 0,
        offset: HEADERS_SIZE - 4096,
        len: 8192,
    };
    chunk.assert_invariants();
}

#[test]
#[should_panic]
fn header_chunk_assert_invariants_offset_not_sector_aligned() {
    let chunk = HeaderChunk {
        index: 0,
        offset: constants::SECTOR_SIZE as u64 - 1,
        len: 4096,
    };
    chunk.assert_invariants();
}

#[test]
#[should_panic]
fn header_chunk_assert_invariants_len_not_sector_aligned() {
    let chunk = HeaderChunk {
        index: 0,
        offset: 0,
        len: constants::SECTOR_SIZE - 1,
    };
    chunk.assert_invariants();
}

// ==========================================================================
// HeaderChunk Tracking Default/Reset Tests
// ==========================================================================

#[test]
fn tracking_default_is_empty() {
    let mut storage = MockStorage::new();
    let journal = TestJournal::new(&mut storage, 0);
    assert!(journal.header_chunks_requested.is_empty());
    assert!(journal.header_chunks_recovered.is_empty());
    assert!(!journal.header_chunks_done());
}

#[test]
fn tracking_reset_clears_state() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);

    // Request and recover some chunks
    let chunk = journal.header_chunk_next_to_request().unwrap();
    journal.header_chunk_mark_recovered(chunk.index);

    journal.header_chunks_reset();

    assert!(journal.header_chunks_requested.is_empty());
    assert!(journal.header_chunks_recovered.is_empty());
    assert!(!journal.header_chunks_done());
}

// ==========================================================================
// Chunk Offset/Length Calculation Tests
// ==========================================================================

#[test]
fn chunk_offset_first_chunk_is_zero() {
    assert_eq!(TestJournal::header_chunk_offset(0), 0);
}

#[test]
fn chunk_offset_second_chunk_is_message_size_max() {
    if HEADER_CHUNK_COUNT > 1 {
        assert_eq!(
            TestJournal::header_chunk_offset(1),
            constants::MESSAGE_SIZE_MAX as u64
        );
    }
}

#[test]
#[allow(clippy::reversed_empty_ranges)] // Range may be empty when HEADER_CHUNK_COUNT == 1
fn chunk_offset_monotonically_increasing() {
    for i in 1..HEADER_CHUNK_COUNT {
        assert!(TestJournal::header_chunk_offset(i) > TestJournal::header_chunk_offset(i - 1));
    }
}

#[test]
#[should_panic]
fn chunk_offset_panics_at_boundary() {
    TestJournal::header_chunk_offset(HEADER_CHUNK_COUNT);
}

#[test]
fn chunk_len_for_offset_zero_returns_valid_length() {
    let len = TestJournal::header_chunk_len_for_offset(0);
    assert!(len > 0);
    assert!(len <= constants::MESSAGE_SIZE_MAX as usize);
    assert!(len.is_multiple_of(constants::SECTOR_SIZE));
    assert!(len.is_multiple_of(WAL_HEADER_SIZE));
}

#[test]
fn chunk_len_covers_entire_headers_region() {
    let mut total_covered = 0u64;
    for i in 0..HEADER_CHUNK_COUNT {
        let offset = TestJournal::header_chunk_offset(i);
        let len = TestJournal::header_chunk_len_for_offset(offset);
        total_covered += len as u64;
    }
    assert_eq!(total_covered, HEADERS_SIZE);
}

#[test]
#[should_panic]
fn chunk_len_for_offset_panics_at_headers_size() {
    TestJournal::header_chunk_len_for_offset(HEADERS_SIZE);
}

// ==========================================================================
// State Machine Transition Tests
// ==========================================================================

#[test]
fn next_to_request_returns_first_chunk_initially() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    let chunk = journal.header_chunk_next_to_request().unwrap();
    assert_eq!(chunk.index, 0);
    assert_eq!(chunk.offset, 0);
}

#[test]
fn next_to_request_marks_chunk_as_requested() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    let chunk = journal.header_chunk_next_to_request().unwrap();
    assert!(journal.header_chunks_requested.is_set(chunk.index));
    assert!(!journal.header_chunks_recovered.is_set(chunk.index));
}

#[test]
fn next_to_request_returns_chunks_in_order() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    for expected_index in 0..HEADER_CHUNK_COUNT {
        let chunk = journal.header_chunk_next_to_request().unwrap();
        assert_eq!(chunk.index, expected_index);
    }
}

#[test]
fn next_to_request_returns_none_when_all_requested() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);

    // Request all chunks
    for _ in 0..HEADER_CHUNK_COUNT {
        journal.header_chunk_next_to_request();
    }

    // Should return None since all are requested but not recovered
    assert!(journal.header_chunk_next_to_request().is_none());
}

#[test]
fn next_to_request_returns_none_when_done() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);

    // Request and recover all chunks
    for _ in 0..HEADER_CHUNK_COUNT {
        let chunk = journal.header_chunk_next_to_request().unwrap();
        journal.header_chunk_mark_recovered(chunk.index);
    }

    assert!(journal.header_chunks_done());
    assert!(journal.header_chunk_next_to_request().is_none());
}

#[test]
fn mark_recovered_updates_state() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    let chunk = journal.header_chunk_next_to_request().unwrap();

    assert!(!journal.header_chunks_recovered.is_set(chunk.index));
    journal.header_chunk_mark_recovered(chunk.index);
    assert!(journal.header_chunks_recovered.is_set(chunk.index));
}

#[test]
#[should_panic]
fn mark_recovered_panics_if_not_requested() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    journal.header_chunk_mark_recovered(0); // Never requested
}

#[test]
#[should_panic]
fn mark_recovered_panics_if_already_recovered() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    let chunk = journal.header_chunk_next_to_request().unwrap();
    journal.header_chunk_mark_recovered(chunk.index);
    journal.header_chunk_mark_recovered(chunk.index); // Double recovery
}

#[test]
#[should_panic]
fn mark_recovered_panics_for_out_of_bounds_index() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    journal.header_chunk_mark_recovered(HEADER_CHUNK_COUNT);
}

// ==========================================================================
// done() Tests
// ==========================================================================

#[test]
fn done_false_when_empty() {
    let mut storage = MockStorage::new();
    let journal = TestJournal::new(&mut storage, 0);
    assert!(!journal.header_chunks_done());
}

#[test]
fn done_false_when_partially_recovered() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    let chunk = journal.header_chunk_next_to_request().unwrap();
    journal.header_chunk_mark_recovered(chunk.index);

    if HEADER_CHUNK_COUNT > 1 {
        assert!(!journal.header_chunks_done());
    }
}

#[test]
fn done_true_when_all_recovered() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);

    for _ in 0..HEADER_CHUNK_COUNT {
        let chunk = journal.header_chunk_next_to_request().unwrap();
        journal.header_chunk_mark_recovered(chunk.index);
    }

    assert!(journal.header_chunks_done());
}

// ==========================================================================
// bytes_as_headers Tests
// ==========================================================================

#[test]
fn bytes_as_headers_single_header() {
    let buffer = AlignedBuf::new_zeroed(WAL_HEADER_SIZE, HEADER_ALIGN);
    let headers = TestJournal::header_chunk_bytes_as_headers(buffer.as_slice());
    assert_eq!(headers.len(), 1);
}

#[test]
fn bytes_as_headers_multiple_headers() {
    let count = 4;
    let size = WAL_HEADER_SIZE * count;
    let buffer = AlignedBuf::new_zeroed(size, HEADER_ALIGN);
    let headers = TestJournal::header_chunk_bytes_as_headers(buffer.as_slice());
    assert_eq!(headers.len(), count);
}

#[test]
#[should_panic]
fn bytes_as_headers_panics_on_empty_slice() {
    TestJournal::header_chunk_bytes_as_headers(&[]);
}

#[test]
#[should_panic]
fn bytes_as_headers_panics_on_too_small_slice() {
    let buffer = vec![0u8; WAL_HEADER_SIZE - 1];
    TestJournal::header_chunk_bytes_as_headers(&buffer);
}

#[test]
#[should_panic]
fn bytes_as_headers_panics_on_non_multiple_length() {
    let size = WAL_HEADER_SIZE + 1;
    let buffer = AlignedBuf::new_zeroed(size, HEADER_ALIGN);
    TestJournal::header_chunk_bytes_as_headers(buffer.as_slice());
}

#[test]
#[should_panic]
fn bytes_as_headers_panics_on_misaligned_input() {
    // Create buffer with extra byte to force misalignment
    let size = WAL_HEADER_SIZE + 16;
    let buffer = vec![0u8; size];

    // Find a misaligned slice within the buffer
    let align = core::mem::align_of::<HeaderPrepareRaw>();
    let base_addr = buffer.as_ptr() as usize;
    let offset = if base_addr.is_multiple_of(align) {
        1
    } else {
        0
    };

    let slice = &buffer[offset..offset + WAL_HEADER_SIZE];
    assert!(
        !(slice.as_ptr() as usize).is_multiple_of(align),
        "Slice should be misaligned"
    );

    TestJournal::header_chunk_bytes_as_headers(slice);
}

// ==========================================================================
// Integration/Workflow Tests
// ==========================================================================

#[test]
fn full_recovery_workflow() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    let mut recovered_chunks = Vec::new();

    // Phase 1: Request all chunks
    while let Some(chunk) = journal.header_chunk_next_to_request() {
        chunk.assert_invariants();
        recovered_chunks.push(chunk);
    }

    assert_eq!(recovered_chunks.len(), HEADER_CHUNK_COUNT);
    assert!(journal.header_chunks_requested.is_full());
    assert!(!journal.header_chunks_done());

    // Phase 2: Recover all chunks (in reverse order for variety)
    for chunk in recovered_chunks.iter().rev() {
        journal.header_chunk_mark_recovered(chunk.index);
    }

    assert!(journal.header_chunks_done());
    assert!(journal.header_chunk_next_to_request().is_none());
}

#[test]
fn partial_recovery_with_reset() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);

    // Request and recover half the chunks
    let half = HEADER_CHUNK_COUNT / 2;
    for _ in 0..half {
        let chunk = journal.header_chunk_next_to_request().unwrap();
        journal.header_chunk_mark_recovered(chunk.index);
    }

    if HEADER_CHUNK_COUNT > 1 {
        assert!(!journal.header_chunks_done());
    }

    // Reset and verify clean state
    journal.header_chunks_reset();

    assert!(!journal.header_chunks_done());
    assert!(journal.header_chunks_requested.is_empty());

    // Should be able to request from beginning
    let first_chunk = journal.header_chunk_next_to_request().unwrap();
    assert_eq!(first_chunk.index, 0);
}

#[test]
fn out_of_order_recovery() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    let mut chunks = Vec::new();

    // Request all chunks
    while let Some(chunk) = journal.header_chunk_next_to_request() {
        chunks.push(chunk);
    }

    // Recover in reverse order
    for chunk in chunks.iter().rev() {
        journal.header_chunk_mark_recovered(chunk.index);
    }

    assert!(journal.header_chunks_done());
}

#[test]
fn interleaved_request_recovery() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);

    // Request 2, recover 1, repeat
    let mut pending = Vec::new();

    while !journal.header_chunks_done() {
        // Request up to 2 chunks
        for _ in 0..2 {
            if let Some(chunk) = journal.header_chunk_next_to_request() {
                pending.push(chunk.index);
            }
        }

        // Recover 1 chunk
        if let Some(idx) = pending.pop() {
            journal.header_chunk_mark_recovered(idx);
        }
    }

    assert!(journal.header_chunks_done());
}

// ==========================================================================
// Edge Case Tests
// ==========================================================================

#[test]
fn single_chunk_scenario() {
    // This test is relevant when HEADER_CHUNK_COUNT == 1
    if HEADER_CHUNK_COUNT == 1 {
        let mut storage = MockStorage::new();
        let mut journal = TestJournal::new(&mut storage, 0);

        let chunk = journal.header_chunk_next_to_request().unwrap();
        assert_eq!(chunk.index, 0);
        assert_eq!(chunk.offset, 0);
        assert_eq!(chunk.len as u64, HEADERS_SIZE);

        assert!(journal.header_chunk_next_to_request().is_none());

        journal.header_chunk_mark_recovered(0);
        assert!(journal.header_chunks_done());
    }
}

#[test]
fn boundary_chunk_sizes() {
    // Verify last chunk has correct size
    let last_chunk_index = HEADER_CHUNK_COUNT - 1;
    let last_offset = TestJournal::header_chunk_offset(last_chunk_index);
    let last_len = TestJournal::header_chunk_len_for_offset(last_offset);

    // Last chunk should end exactly at HEADERS_SIZE
    assert_eq!(last_offset + last_len as u64, HEADERS_SIZE);
}

#[test]
fn zero_remaining_after_full_recovery() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);

    while let Some(chunk) = journal.header_chunk_next_to_request() {
        journal.header_chunk_mark_recovered(chunk.index);
    }

    // Verify no chunks remain unrequested or unrecovered
    assert!(journal.header_chunks_requested.is_full());
    assert!(journal.header_chunks_recovered.is_full());
    assert_eq!(
        journal.header_chunks_requested.count(),
        journal.header_chunks_recovered.count()
    );
}

// ==========================================================================
// Invariant Verification Tests (Bug #1 fix validation)
// ==========================================================================

#[test]
fn invariant_recovered_subset_of_requested() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);

    // Request some but don't recover - this verifies the fixed invariant
    // works correctly (recovered is empty, which is a subset of requested)
    journal.header_chunk_next_to_request();
    if HEADER_CHUNK_COUNT > 1 {
        journal.header_chunk_next_to_request();
    }

    // Invariant check happens inside header_chunk_next_to_request via assert_invariants
    // If the bug #1 fix is correct, this test passes.
    // With the buggy code (checking requested.is_subset(recovered)),
    // this would have panicked.
}

#[test]
fn invariant_holds_during_partial_recovery() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);

    // Request all chunks
    let mut indices = Vec::new();
    while let Some(chunk) = journal.header_chunk_next_to_request() {
        indices.push(chunk.index);
    }

    // Recover half of them
    for &idx in indices.iter().take(HEADER_CHUNK_COUNT / 2) {
        journal.header_chunk_mark_recovered(idx);
    }

    // Invariant should hold: recovered (half) is subset of requested (all)
    assert!(
        journal.header_chunks_recovered.count() < journal.header_chunks_requested.count()
            || HEADER_CHUNK_COUNT <= 1
    );
}

// ==========================================================================
// HeaderChunk Property Tests
// ==========================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Invariant: requested count >= recovered count always
        #[test]
        fn prop_requested_count_gte_recovered_count(
            num_requests in 0usize..=HEADER_CHUNK_COUNT,
            num_recoveries in 0usize..=HEADER_CHUNK_COUNT
        ) {
            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);
            let mut requested_indices = Vec::new();

            // Request some chunks
            for _ in 0..num_requests {
                if let Some(chunk) = journal.header_chunk_next_to_request() {
                    requested_indices.push(chunk.index);
                }
            }

            // Recover some of the requested chunks
            let recoveries = num_recoveries.min(requested_indices.len());
            for &idx in requested_indices.iter().take(recoveries) {
                journal.header_chunk_mark_recovered(idx);
            }

            prop_assert!(
                journal.header_chunks_requested.count() >=
                journal.header_chunks_recovered.count()
            );
        }

        /// Invariant: recovered is always a subset of requested
        #[test]
        fn prop_recovered_subset_of_requested(
            operations in prop::collection::vec(0u8..2, 0..50)
        ) {
            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);
            let mut pending_recoveries: Vec<usize> = Vec::new();

            for op in operations {
                match op {
                    0 => {
                        // Request
                        if let Some(chunk) = journal.header_chunk_next_to_request() {
                            pending_recoveries.push(chunk.index);
                        }
                    }
                    1 => {
                        // Recover
                        if let Some(idx) = pending_recoveries.pop() {
                            journal.header_chunk_mark_recovered(idx);
                        }
                    }
                    _ => unreachable!()
                }
            }

            // Check invariant: every recovered bit must be requested
            for idx in 0..HEADER_CHUNK_COUNT {
                if journal.header_chunks_recovered.is_set(idx) {
                    prop_assert!(
                        journal.header_chunks_requested.is_set(idx),
                        "Recovered chunk {} was not requested", idx
                    );
                }
            }
        }

        /// All chunks are eventually returned by header_chunk_next_to_request
        #[test]
        fn prop_all_chunks_requestable(_unused in Just(())) {
            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);
            let mut seen = [false; HEADER_CHUNK_COUNT];

            for _ in 0..HEADER_CHUNK_COUNT {
                let chunk = journal.header_chunk_next_to_request();
                prop_assert!(chunk.is_some());
                let chunk = chunk.unwrap();
                prop_assert!(!seen[chunk.index], "Chunk {} returned twice", chunk.index);
                seen[chunk.index] = true;
            }

            prop_assert!(seen.iter().all(|&x| x));
        }

        /// Chunks are contiguous (no gaps)
        #[test]
        fn prop_chunks_are_contiguous(_unused in Just(())) {
            let mut expected_offset = 0u64;

            for i in 0..HEADER_CHUNK_COUNT {
                let offset = TestJournal::header_chunk_offset(i);
                let len = TestJournal::header_chunk_len_for_offset(offset);

                prop_assert_eq!(offset, expected_offset);
                expected_offset = offset + len as u64;
            }

            prop_assert_eq!(expected_offset, HEADERS_SIZE);
        }

        /// Interleaved request/recover operations maintain invariants
        #[test]
        fn prop_interleaved_operations_maintain_invariants(seed: u64) {
            use std::collections::VecDeque;

            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);
            let mut pending: VecDeque<usize> = VecDeque::new();
            let mut rng = seed;

            // Simple PRNG for determinism
            let next_bool = |r: &mut u64| -> bool {
                *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
                (*r >> 63) == 0
            };

            for _ in 0..100 {
                if pending.is_empty() || (next_bool(&mut rng) && pending.len() < HEADER_CHUNK_COUNT) {
                    // Try to request
                    if let Some(chunk) = journal.header_chunk_next_to_request() {
                        pending.push_back(chunk.index);
                    }
                } else {
                    // Recover oldest pending
                    if let Some(idx) = pending.pop_front() {
                        journal.header_chunk_mark_recovered(idx);
                    }
                }

                // Verify invariant after each operation
                prop_assert!(
                    journal.header_chunks_recovered.count() <=
                    journal.header_chunks_requested.count()
                );
            }
        }

        /// Chunk invariants are always valid for returned chunks
        #[test]
        fn prop_chunk_invariants_always_valid(_unused in Just(())) {
            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);

            while let Some(chunk) = journal.header_chunk_next_to_request() {
                // This will panic if invariants don't hold
                chunk.assert_invariants();
                journal.header_chunk_mark_recovered(chunk.index);
            }
        }
    }
}

// =========================================================================
// Recovery Case Table (Unit Tests)
// =========================================================================

#[test]
fn recovery_cases_match_expected_decisions() {
    fn assert_case(
        label: &str,
        header: Option<HeaderPrepare>,
        prepare: Option<HeaderPrepare>,
        op_max: u64,
        op_prepare_max: u64,
        decision_multiple: RecoveryDecision,
        decision_single: RecoveryDecision,
    ) {
        let case = recovery_case(
            header.as_ref(),
            prepare.as_ref(),
            op_max,
            op_prepare_max,
            false,
        );
        assert_eq!(case.label, label);
        assert_eq!(case.decision(false), decision_multiple);
        assert_eq!(case.decision(true), decision_single);
    }

    assert_case(
        "@A",
        None,
        None,
        0,
        0,
        RecoveryDecision::Vsr,
        RecoveryDecision::Vsr,
    );

    assert_case(
        "@B",
        Some(make_reserved_header(1)),
        None,
        0,
        0,
        RecoveryDecision::Vsr,
        RecoveryDecision::Vsr,
    );

    assert_case(
        "@C",
        Some(make_recovery_header(5, 1, 0x11, Operation::REGISTER)),
        None,
        10,
        10,
        RecoveryDecision::Vsr,
        RecoveryDecision::Vsr,
    );

    assert_case(
        "@D",
        None,
        Some(make_reserved_header(2)),
        10,
        10,
        RecoveryDecision::Vsr,
        RecoveryDecision::Fix,
    );

    assert_case(
        "@E",
        None,
        Some(make_recovery_header(9, 1, 0x22, Operation::REGISTER)),
        10,
        10,
        RecoveryDecision::Vsr,
        RecoveryDecision::Fix,
    );

    assert_case(
        "@F",
        None,
        Some(make_recovery_header(10, 1, 0x33, Operation::REGISTER)),
        10,
        10,
        RecoveryDecision::Fix,
        RecoveryDecision::Fix,
    );

    assert_case(
        "@G",
        Some(make_reserved_header(3)),
        Some(make_recovery_header(7, 1, 0x44, Operation::REGISTER)),
        20,
        10,
        RecoveryDecision::Fix,
        RecoveryDecision::Fix,
    );

    assert_case(
        "@H",
        None,
        Some(make_recovery_header(12, 1, 0x55, Operation::REGISTER)),
        20,
        10,
        RecoveryDecision::Cut,
        RecoveryDecision::Unreachable,
    );

    assert_case(
        "@I",
        Some(make_recovery_header(12, 1, 0x66, Operation::REGISTER)),
        Some(make_recovery_header(9, 1, 0x77, Operation::REGISTER)),
        20,
        10,
        RecoveryDecision::Cut,
        RecoveryDecision::Unreachable,
    );

    assert_case(
        "@J",
        Some(make_recovery_header(12, 1, 0x88, Operation::REGISTER)),
        Some(make_reserved_header(2)),
        20,
        10,
        RecoveryDecision::Cut,
        RecoveryDecision::Unreachable,
    );

    assert_case(
        "@K",
        Some(make_recovery_header(8, 1, 0x99, Operation::REGISTER)),
        Some(make_reserved_header(2)),
        20,
        10,
        RecoveryDecision::Vsr,
        RecoveryDecision::Vsr,
    );

    let reserved = make_reserved_header(4);
    assert_case(
        "@L",
        Some(reserved),
        Some(reserved),
        10,
        10,
        RecoveryDecision::Nil,
        RecoveryDecision::Nil,
    );

    assert_case(
        "@M",
        Some(make_recovery_header(5, 1, 0x100, Operation::REGISTER)),
        Some(make_recovery_header(7, 1, 0x200, Operation::REGISTER)),
        20,
        10,
        RecoveryDecision::Fix,
        RecoveryDecision::Fix,
    );

    assert_case(
        "@N",
        Some(make_recovery_header(7, 1, 0x300, Operation::REGISTER)),
        Some(make_recovery_header(5, 1, 0x400, Operation::REGISTER)),
        20,
        10,
        RecoveryDecision::Vsr,
        RecoveryDecision::Vsr,
    );

    assert_case(
        "@O",
        Some(make_recovery_header(6, 1, 0x500, Operation::REGISTER)),
        Some(make_recovery_header(6, 2, 0x600, Operation::REGISTER)),
        20,
        10,
        RecoveryDecision::Vsr,
        RecoveryDecision::Vsr,
    );

    let header = make_recovery_header(6, 1, 0x700, Operation::REGISTER);
    assert_case(
        "@P",
        Some(header),
        Some(header),
        20,
        10,
        RecoveryDecision::Fix,
        RecoveryDecision::Fix,
    );
}

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
// Header Helper Method Properties (Property Tests)
// =========================================================================

mod header_helper_properties {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for op values that span multiple wraparounds.
    fn op_strategy() -> impl Strategy<Value = u64> {
        prop_oneof![
            0u64..SLOT_COUNT as u64,                          // First epoch
            SLOT_COUNT as u64..(SLOT_COUNT * 2) as u64,       // Second epoch (wraparound)
            (SLOT_COUNT * 5) as u64..(SLOT_COUNT * 6) as u64, // Later epoch
        ]
    }

    /// Strategy for slot indices.
    fn slot_strategy() -> impl Strategy<Value = usize> {
        0..SLOT_COUNT
    }

    proptest! {
        /// Property: slot_for_op maps to op % SLOT_COUNT.
        #[test]
        fn prop_slot_for_op_maps_to_modulo(op in any::<u64>()) {
            let mut storage = MockStorage::new();
            let journal = TestJournal::new(&mut storage, 0);

            let slot = journal.slot_for_op(op);
            prop_assert_eq!(slot.index(), (op % SLOT_COUNT as u64) as usize);
        }

        /// Property: header_with_op returns Some iff stored header.op == query op.
        /// Only tests within the same slot to avoid accessing uninitialized headers.
        #[test]
        fn prop_header_with_op_returns_iff_op_matches(
            slot in slot_strategy(),
            epoch in 0u64..10
        ) {
            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);

            // Store a header at a specific op for this slot
            let stored_op = slot as u64 + epoch * SLOT_COUNT as u64;
            journal.headers[slot] = make_header(stored_op, Operation::REGISTER);

            let slot_from_op = journal.slot_for_op(stored_op);
            prop_assert_eq!(slot_from_op.index(), slot);

            // Query with the exact op - should return the header
            let result = journal.header_with_op(stored_op);
            prop_assert!(result.is_some());
            prop_assert_eq!(result.unwrap().op, stored_op);

            let slot_with_op = journal.slot_with_op(stored_op);
            prop_assert_eq!(slot_with_op.map(|slot| slot.index()), Some(slot));

            let header_for_op = journal.header_for_op(stored_op).unwrap();
            prop_assert_eq!(header_for_op.op, stored_op);

            let slot_for_header = journal.slot_for_header(header_for_op);
            prop_assert_eq!(slot_for_header.index(), slot);

            // Query with a different epoch for the same slot - should return None
            let different_epoch = if epoch == 0 { 1 } else { epoch - 1 };
            let different_op = slot as u64 + different_epoch * SLOT_COUNT as u64;
            let result_different = journal.header_with_op(different_op);
            prop_assert!(result_different.is_none());

            let different_slot = journal.slot_for_op(different_op);
            prop_assert_eq!(different_slot.index(), slot);

            let header_for_different_op = journal.header_for_op(different_op).unwrap();
            prop_assert_eq!(header_for_different_op.op, stored_op);

            let slot_with_different_op = journal.slot_with_op(different_op);
            prop_assert!(slot_with_different_op.is_none());
        }

        /// Property: reserved slots always return None regardless of query op.
        #[test]
        fn prop_reserved_slots_always_return_none(
            slot in slot_strategy(),
            epoch in 0u64..10
        ) {
            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);

            // Initialize the slot as reserved (default behavior but explicit here)
            let reserved = make_reserved_header(slot);
            journal.headers[slot] = reserved;

            // Query for any op that maps to this slot - should return None
            let query_op = slot as u64 + epoch * SLOT_COUNT as u64;
            let result = journal.header_with_op(query_op);
            prop_assert!(result.is_none());

            let header_for_op = journal.header_for_op(query_op);
            prop_assert!(header_for_op.is_none());

            let slot_with_op = journal.slot_with_op(query_op);
            prop_assert!(slot_with_op.is_none());

            let slot_with_op_and_checksum =
                journal.slot_with_op_and_checksum(query_op, reserved.checksum);
            prop_assert!(slot_with_op_and_checksum.is_none());

            let slot_with_header = journal.slot_with_header(&reserved);
            prop_assert!(slot_with_header.is_none());

            let has_header = journal.has_header(&reserved);
            prop_assert!(!has_header);
        }

        /// Property: header_with_op_and_checksum requires BOTH op AND checksum match.
        #[test]
        fn prop_checksum_match_required(
            op in op_strategy(),
            use_correct_checksum in proptest::bool::ANY
        ) {
            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);

            let header = make_header(op, Operation::REGISTER);
            let correct_checksum = header.checksum;
            let slot = (op % SLOT_COUNT as u64) as usize;
            journal.headers[slot] = header;

            let query_checksum = if use_correct_checksum {
                correct_checksum
            } else {
                correct_checksum.wrapping_add(1)
            };

            let result = journal.header_with_op_and_checksum(op, query_checksum);
            let slot_result = journal.slot_with_op_and_checksum(op, query_checksum);

            let query_header = if use_correct_checksum {
                header
            } else {
                let mut mutated = header;
                mutated.checksum = query_checksum;
                mutated
            };

            let slot_with_header = journal.slot_with_header(&query_header);
            let has_header = journal.has_header(&query_header);

            if use_correct_checksum {
                prop_assert!(result.is_some());
                prop_assert_eq!(slot_result.map(|slot| slot.index()), Some(slot));
                prop_assert_eq!(slot_with_header.map(|slot| slot.index()), Some(slot));
                prop_assert!(has_header);
            } else {
                prop_assert!(result.is_none());
                prop_assert!(slot_result.is_none());
                prop_assert!(slot_with_header.is_none());
                prop_assert!(!has_header);
            }
        }

        /// Property: has_dirty reflects the dirty bit for an existing header.
        #[test]
        fn prop_has_dirty_tracks_dirty_bit(
            op in op_strategy(),
            is_dirty in proptest::bool::ANY
        ) {
            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);

            let header = make_header(op, Operation::REGISTER);
            let slot = (op % SLOT_COUNT as u64) as usize;
            journal.headers[slot] = header;
            journal.dirty.set_value(slot, is_dirty);

            prop_assert_eq!(journal.has_dirty(&header), is_dirty);
        }

        /// Property: previous_entry returns op-1 when it exists.
        #[test]
        fn prop_previous_entry_navigation(
            base_op in 1u64..50 // Keep small for simplicity
        ) {
            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);

            // Setup: headers at base_op-1 and base_op
            journal.headers[(base_op - 1) as usize] = make_header(base_op - 1, Operation::REGISTER);
            journal.headers[base_op as usize] = make_header(base_op, Operation::REGISTER);

            let current = &journal.headers[base_op as usize];
            let result = journal.previous_entry(current);

            prop_assert!(result.is_some());
            prop_assert_eq!(result.unwrap().op, base_op - 1);
        }

        /// Property: next_entry returns op+1 when it exists.
        #[test]
        fn prop_next_entry_navigation(
            base_op in 0u64..49
        ) {
            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);

            journal.headers[base_op as usize] = make_header(base_op, Operation::REGISTER);
            journal.headers[(base_op + 1) as usize] = make_header(base_op + 1, Operation::REGISTER);

            let current = &journal.headers[base_op as usize];
            let result = journal.next_entry(current);

            prop_assert!(result.is_some());
            prop_assert_eq!(result.unwrap().op, base_op + 1);
        }

        /// Property: op_maximum returns actual maximum of non-reserved ops.
        #[test]
        fn prop_op_maximum_is_actual_max(
            num_headers in 1usize..20,
            base_op in 0u64..100
        ) {
            let mut storage = MockStorage::new();
            let mut journal = TestJournal::new(&mut storage, 0);
            journal.status = Status::Recovered;

            // Insert headers at consecutive slots starting from base_op
            let mut expected_max = 0u64;
            for i in 0..num_headers {
                let op = base_op + i as u64;
                if (op as usize) < SLOT_COUNT {
                    journal.headers[op as usize] = make_header(op, Operation::REGISTER);
                    expected_max = expected_max.max(op);
                }
            }

            let actual_max = journal.op_maximum();
            prop_assert_eq!(actual_max, expected_max);
        }
    }
}

// =========================================================================
// Journal Internal State (Unit Tests)
// =========================================================================

#[test]
fn set_header_as_dirty_updates_reserved_slot() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    journal.status = Status::Recovered;
    journal.dirty = BitSet::empty();
    journal.faulty = BitSet::empty();

    let op = 5u64;
    let slot = (op % SLOT_COUNT as u64) as usize;
    journal.headers[slot] = make_reserved_header(slot);
    let redundant_before = journal.headers_redundant[slot];

    let header = make_header(op, Operation::REGISTER);
    journal.set_header_as_dirty(&header);

    assert_eq!(journal.headers[slot], header);
    assert!(journal.dirty.is_set(slot));
    assert_eq!(journal.headers_redundant[slot], redundant_before);
    assert_eq!(journal.headers_redundant[slot].checksum, 0);
    assert!(!journal.faulty.is_set(slot));
}

#[test]
fn set_header_as_dirty_overwrite_clears_faulty_and_resets_redundant() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    journal.status = Status::Recovered;
    journal.dirty = BitSet::empty();
    journal.faulty = BitSet::empty();

    let op = 7u64;
    let slot = (op % SLOT_COUNT as u64) as usize;
    let existing = make_header(op, Operation::REGISTER);
    journal.headers[slot] = existing;
    journal.faulty.set(slot);
    journal.headers_redundant[slot] = make_header(op + 1, Operation::ROOT);

    let mut header = make_header(op, Operation::RECONFIGURE);
    header.cluster = 7u128;
    header.set_checksum();

    journal.set_header_as_dirty(&header);

    assert_eq!(journal.headers[slot], header);
    assert!(journal.dirty.is_set(slot));
    assert!(!journal.faulty.is_set(slot));
    assert_eq!(
        journal.headers_redundant[slot],
        HeaderPrepare::reserve(header.cluster, slot as u64)
    );
}

#[test]
fn set_header_as_dirty_noop_when_header_exists() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    journal.status = Status::Recovered;
    journal.dirty = BitSet::empty();
    journal.faulty = BitSet::empty();

    let op = 9u64;
    let slot = (op % SLOT_COUNT as u64) as usize;
    let header = make_header(op, Operation::REGISTER);
    journal.headers[slot] = header;
    journal.dirty.set(slot);
    journal.faulty.set(slot);
    let redundant = make_header(op + 1, Operation::NOOP);
    journal.headers_redundant[slot] = redundant;

    journal.set_header_as_dirty(&header);

    assert_eq!(journal.headers[slot], header);
    assert!(journal.dirty.is_set(slot));
    assert!(journal.faulty.is_set(slot));
    assert_eq!(journal.headers_redundant[slot], redundant);
}

#[test]
fn writing_returns_none_when_no_matching_slot() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    journal.status = Status::Recovered;

    let header = make_header(3, Operation::REGISTER);
    let write = journal.writes.acquire().unwrap();
    unsafe {
        (*write).op = header.op + 1;
        (*write).checksum = header.checksum;
    }

    assert_eq!(journal.writing(&header), Writing::None);
}

#[test]
fn writing_returns_exact_when_checksum_matches() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    journal.status = Status::Recovered;

    let header = make_header(11, Operation::REGISTER);
    let write = journal.writes.acquire().unwrap();
    unsafe {
        (*write).op = header.op;
        (*write).checksum = header.checksum;
    }

    assert_eq!(journal.writing(&header), Writing::Exact);
}

#[test]
fn writing_returns_slot_when_checksum_differs() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    journal.status = Status::Recovered;

    let header = make_header(12, Operation::REGISTER);
    let write = journal.writes.acquire().unwrap();
    unsafe {
        (*write).op = header.op;
        (*write).checksum = header.checksum.wrapping_add(1);
    }

    assert_eq!(journal.writing(&header), Writing::Slot);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn writing_panics_on_multiple_writes_same_slot() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    journal.status = Status::Recovered;

    let header = make_header(13, Operation::REGISTER);
    let write1 = journal.writes.acquire().unwrap();
    let write2 = journal.writes.acquire().unwrap();
    unsafe {
        (*write1).op = header.op;
        (*write1).checksum = header.checksum;
        (*write2).op = header.op;
        (*write2).checksum = header.checksum;
    }

    let _ = journal.writing(&header);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn writing_panics_on_same_slot_different_op() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);
    journal.status = Status::Recovered;

    let header = make_header(1, Operation::REGISTER);
    let write = journal.writes.acquire().unwrap();
    unsafe {
        (*write).op = header.op + SLOT_COUNT as u64;
        (*write).checksum = header.checksum;
    }

    let _ = journal.writing(&header);
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
        (*write1).op = 1;
        (*write2).op = 2;

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
        (*write1).op = 1;
        (*write2).op = 2;

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
        (*write1).op = 1;
        (*write2).op = 2;

        journal.write_sectors(write1, dummy_callback_async, &buf1, Ring::Prepares, 0);
        // write1 is now locked (callback deferred).
        // Prepares writes must never overlap - should panic.
        journal.write_sectors(write2, dummy_callback_async, &buf2, Ring::Prepares, 0);
    }
}

#[test]
#[should_panic(expected = "assertion failed")]
fn panics_on_duplicate_slot() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);

    let write1 = journal.writes.acquire().unwrap();
    let write2 = journal.writes.acquire().unwrap();

    // Non-overlapping ranges but ops map to the same slot.
    // op 42 and op 42 + SLOT_COUNT both map to slot 42.
    let buf1 = vec![1u8; 4096];
    let buf2 = vec![2u8; 4096];

    unsafe {
        (*write1).op = 42;
        (*write2).op = 42 + SLOT_COUNT as u64; // Same slot - should panic.

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
        (*write1).op = 1;
        (*write2).op = 2;

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
        (*write1).op = 1;
        (*write2).op = 2;

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

#[test]
fn async_overlap_queuing_handles_multiple_waiters() {
    let mut storage = MockStorage::<true>::new();
    let mut journal = Journal::<MockStorage<true>, 32, 1>::new(&mut storage, 0);

    let write1 = journal.writes.acquire().unwrap();
    let write2 = journal.writes.acquire().unwrap();
    let write3 = journal.writes.acquire().unwrap();

    let buf1 = vec![1u8; 4096];
    let buf2 = vec![2u8; 4096];
    let buf3 = vec![3u8; 4096];

    unsafe {
        (*write1).op = 1;
        (*write2).op = 2;
        (*write3).op = 3;

        journal.write_sectors(write1, dummy_callback_async, &buf1, Ring::Headers, 0);
        journal.write_sectors(write2, dummy_callback_async, &buf2, Ring::Headers, 0);
        journal.write_sectors(write3, dummy_callback_async, &buf3, Ring::Headers, 0);

        assert_eq!(storage.write_count(), 1);
        assert!((*write1).range.locked);
        assert!(!(*write2).range.locked);
        assert!(!(*write3).range.locked);

        storage.drain_callbacks();
        assert_eq!(storage.write_count(), 2);
        assert!(!(*write1).range.locked);
        assert!((*write2).range.locked);
        assert!(!(*write3).range.locked);

        storage.drain_callbacks();
        assert_eq!(storage.write_count(), 3);
        assert!(!(*write2).range.locked);
        assert!((*write3).range.locked);

        storage.drain_callbacks();
        assert!(!(*write3).range.locked);
    }
}

// =========================================================================
// Read Prepare Path (Unit Tests)
// =========================================================================

#[test]
fn read_prepare_returns_none_when_header_missing() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::new();
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    let header = make_prepare_header_with_size(1, Operation::REGISTER, constants::SECTOR_SIZE);
    replica.op = header.op;
    let journal = &mut replica.journal;
    let slot = Slot::from_op(header.op);
    journal.headers[slot.index()] = make_reserved_header(slot.index());
    let options = ReadOptions::commit(header.op, header.checksum);

    journal.read_prepare(record_read_prepare_callback, options);

    assert_eq!(storage.read_count(), 0);
    assert_eq!(journal.reads_commit_count, 0);
    assert_eq!(journal.reads_repair_count, 0);
    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert!(callbacks[0].0.is_none());
    assert_eq!(callbacks[0].1, options);
}

#[test]
fn read_prepare_returns_none_when_prepare_not_inhabited() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::new();
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    let header = make_prepare_header_with_size(2, Operation::REGISTER, constants::SECTOR_SIZE);
    replica.op = header.op;
    let journal = &mut replica.journal;
    let slot = Slot::from_op(header.op);
    journal.headers[slot.index()] = header;
    journal.prepare_inhabited[slot.index()] = false;
    journal.prepare_checksums[slot.index()] = header.checksum;

    let options = ReadOptions::commit(header.op, header.checksum);
    journal.read_prepare(record_read_prepare_callback, options);

    assert_eq!(storage.read_count(), 0);
    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert!(callbacks[0].0.is_none());
    assert_eq!(callbacks[0].1, options);
}

#[test]
fn read_prepare_works_with_fresh_message_buffer() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::new();
    let op = 3u64;
    let cluster = 42u128;
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    replica.op = op;
    replica.cluster = cluster;
    let journal = &mut replica.journal;
    let body = vec![0x11, 0x22, 0x33];
    let header =
        install_prepare_for_read(journal, &storage, op, Operation::REGISTER, cluster, &body);

    let options = ReadOptions::commit(op, header.checksum);

    journal.read_prepare(record_read_prepare_callback, options);

    assert_eq!(storage.read_count(), 1);
    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert_eq!(callbacks[0].1, options);
    let message = callbacks[0].0.as_ref().expect("message missing");
    assert_eq!(message.header().size, header.size);
    assert_eq!(message.body_used(), body.as_slice());
}

#[test]
fn read_prepare_header_only_fast_path_skips_io() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::new();
    let op = 3u64;
    let cluster = 7u128;
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    replica.op = op;
    replica.cluster = cluster;
    let journal = &mut replica.journal;
    let header = install_prepare_for_read(journal, &storage, op, Operation::REGISTER, cluster, &[]);

    let options = ReadOptions::commit(op, header.checksum);

    journal.read_prepare(record_read_prepare_callback, options);

    assert_eq!(storage.read_count(), 0);
    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert_eq!(callbacks[0].1, options);
    let message = callbacks[0].0.as_ref().expect("message missing");
    let bytes =
        unsafe { core::slice::from_raw_parts(message.buffer_ptr(), constants::SECTOR_SIZE) };
    assert_eq!(&bytes[..HeaderPrepare::SIZE], header.as_bytes());
    assert!(bytes[HeaderPrepare::SIZE..].iter().all(|&b| b == 0));
}

#[test]
fn read_prepare_reads_exact_size_and_validates_padding() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::new();
    let op = 4u64;
    let cluster = 9u128;
    let body = vec![0x5a];
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    replica.op = op;
    replica.cluster = cluster;
    let journal = &mut replica.journal;
    let header =
        install_prepare_for_read(journal, &storage, op, Operation::REGISTER, cluster, &body);

    let options = ReadOptions::commit(op, header.checksum);

    journal.read_prepare(record_read_prepare_callback, options);

    let log = storage.read_log.borrow();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].len, constants::sector_ceil(header.size as usize));
    drop(log);

    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert_eq!(callbacks[0].1, options);
    let message = callbacks[0].0.as_ref().expect("message missing");
    assert_eq!(message.header().checksum, header.checksum);
    assert_eq!(message.header().size, header.size);
    assert_eq!(message.body_used(), body.as_slice());

    let slot = Slot::from_op(op);
    assert!(!journal.faulty.is_set(slot.index()));
}

#[test]
fn read_prepare_reads_full_slot_when_header_not_exact() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::new();
    let op = 5u64;
    let cluster = 3u128;
    let body = vec![0x11, 0x22];
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    replica.op = op;
    replica.cluster = cluster;
    let journal = &mut replica.journal;
    let header =
        install_prepare_for_read(journal, &storage, op, Operation::REGISTER, cluster, &body);

    let slot = Slot::from_op(op);
    journal.headers[slot.index()].checksum = header.checksum.wrapping_add(1);

    let options = ReadOptions::commit(op, header.checksum);

    journal.read_prepare_with_op_and_checksum(record_read_prepare_callback, options);

    let log = storage.read_log.borrow();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].len, constants::MESSAGE_SIZE_MAX_USIZE);
    drop(log);

    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert_eq!(callbacks[0].1, options);
    assert!(callbacks[0].0.is_some());
}

#[test]
fn read_prepare_marks_faulty_on_body_corruption() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::new();
    let op = 6u64;
    let cluster = 1u128;
    let body = vec![0x10, 0x20, 0x30];
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    replica.op = op;
    replica.cluster = cluster;
    let journal = &mut replica.journal;

    let (header, mut data) = make_prepare_disk_image(op, Operation::REGISTER, cluster, &body);
    data[HeaderPrepare::SIZE] ^= 0xFF;

    let slot = Slot::from_op(op);
    journal.headers[slot.index()] = header;
    journal.prepare_inhabited[slot.index()] = true;
    journal.prepare_checksums[slot.index()] = header.checksum;
    let offset = (constants::MESSAGE_SIZE_MAX as u64) * slot.index() as u64;
    storage.set_read_data(Zone::WalPrepares, offset, data);

    let options = ReadOptions::commit(op, header.checksum);

    journal.read_prepare(record_read_prepare_callback, options);

    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert!(callbacks[0].0.is_none());
    assert_eq!(callbacks[0].1, options);
    assert!(journal.faulty.is_set(slot.index()));
    assert!(journal.dirty.is_set(slot.index()));
}

#[test]
fn read_prepare_rejects_nonzero_padding() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::new();
    let op = 7u64;
    let cluster = 1u128;
    let body = vec![0x42];
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    replica.op = op;
    replica.cluster = cluster;
    let journal = &mut replica.journal;

    let (header, mut data) = make_prepare_disk_image(op, Operation::REGISTER, cluster, &body);
    let size = header.size as usize;
    let padded = constants::sector_ceil(size);
    assert!(size < padded);
    data[size] = 0x99;

    let slot = Slot::from_op(op);
    journal.headers[slot.index()] = header;
    journal.prepare_inhabited[slot.index()] = true;
    journal.prepare_checksums[slot.index()] = header.checksum;
    let offset = (constants::MESSAGE_SIZE_MAX as u64) * slot.index() as u64;
    storage.set_read_data(Zone::WalPrepares, offset, data);

    let options = ReadOptions::commit(op, header.checksum);

    journal.read_prepare(record_read_prepare_callback, options);

    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert!(callbacks[0].0.is_none());
    assert_eq!(callbacks[0].1, options);
    assert!(journal.faulty.is_set(slot.index()));
    assert!(journal.dirty.is_set(slot.index()));
}

#[test]
fn read_prepare_rejects_structurally_invalid_header() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::new();
    let op = 8u64;
    let cluster = 1u128;
    let body = vec![0x55];
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    replica.op = op;
    replica.cluster = cluster;
    let journal = &mut replica.journal;

    let (mut header, mut data) = make_prepare_disk_image(op, Operation::REGISTER, cluster, &body);
    header.protocol = constants::VSR_VERSION + 1;
    header.set_checksum();
    data[..HeaderPrepare::SIZE].copy_from_slice(header.as_bytes());

    let slot = Slot::from_op(op);
    journal.headers[slot.index()] = header;
    journal.prepare_inhabited[slot.index()] = true;
    journal.prepare_checksums[slot.index()] = header.checksum;
    let offset = (constants::MESSAGE_SIZE_MAX as u64) * slot.index() as u64;
    storage.set_read_data(Zone::WalPrepares, offset, data);

    let options = ReadOptions::commit(op, header.checksum);

    journal.read_prepare(record_read_prepare_callback, options);

    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert!(callbacks[0].0.is_none());
    assert_eq!(callbacks[0].1, options);
    assert!(journal.faulty.is_set(slot.index()));
    assert!(journal.dirty.is_set(slot.index()));
}

#[test]
fn read_prepare_detects_prepare_changed_during_read() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::<true>::new();
    let op = 8u64;
    let cluster = 1u128;
    let body = vec![0x1];
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    replica.op = op;
    replica.cluster = cluster;
    let journal = &mut replica.journal;
    let header =
        install_prepare_for_read(journal, &storage, op, Operation::REGISTER, cluster, &body);

    let options = ReadOptions::commit(op, header.checksum);

    journal.read_prepare(record_read_prepare_callback_async, options);

    assert_eq!(journal.reads_commit_count, 1);
    assert_eq!(storage.pending_read_callbacks.borrow().len(), 1);

    let slot = Slot::from_op(op);
    journal.prepare_inhabited[slot.index()] = false;

    storage.drain_read_callbacks();

    assert_eq!(journal.reads_commit_count, 0);
    assert!(!journal.faulty.is_set(slot.index()));
    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert!(callbacks[0].0.is_none());
    assert_eq!(callbacks[0].1, options);
}

#[test]
fn read_prepare_returns_none_when_read_pool_exhausted() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::<true>::new();
    let pool = MessagePool::new(READ_OPS + 1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    replica.op = 1000;
    replica.cluster = 1;
    let journal = &mut replica.journal;

    let mut op = 1u64;

    for _ in 0..READS_REPAIR_COUNT_MAX {
        let header = install_prepare_for_read(journal, &storage, op, Operation::REGISTER, 1, &[1]);
        let options = ReadOptions::repair(op, header.checksum, 1);
        journal.read_prepare(record_read_prepare_callback_async, options);
        op += 1;
    }

    for _ in 0..READS_COMMIT_COUNT_MAX {
        let header = install_prepare_for_read(journal, &storage, op, Operation::REGISTER, 1, &[2]);
        let options = ReadOptions::commit(op, header.checksum);
        journal.read_prepare(record_read_prepare_callback_async, options);
        op += 1;
    }

    assert_eq!(journal.reads_repair_count, READS_REPAIR_COUNT_MAX);
    assert_eq!(journal.reads_commit_count, READS_COMMIT_COUNT_MAX);
    assert_eq!(storage.read_count(), READ_OPS);

    reset_read_prepare_callbacks();

    let header = install_prepare_for_read(journal, &storage, op, Operation::REGISTER, 1, &[3]);
    let options = ReadOptions::commit(op, header.checksum);
    journal.read_prepare(record_read_prepare_callback_async, options);

    assert_eq!(storage.read_count(), READ_OPS);
    assert_eq!(journal.reads_commit_count, READS_COMMIT_COUNT_MAX);
    assert_eq!(journal.reads_repair_count, READS_REPAIR_COUNT_MAX);
    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert!(callbacks[0].0.is_none());
    assert_eq!(callbacks[0].1, options);
}

#[test]
fn read_prepare_reserves_commit_reads_over_repairs() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::<true>::new();
    let pool = MessagePool::new(READ_OPS + 4);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    replica.op = 1000;
    replica.cluster = 1;
    let journal = &mut replica.journal;

    for i in 0..READS_REPAIR_COUNT_MAX {
        let op = 100 + i as u64;
        let header =
            install_prepare_for_read(journal, &storage, op, Operation::REGISTER, 1, &[0x7f]);
        let options = ReadOptions::repair(op, header.checksum, 1);

        journal.read_prepare(record_read_prepare_callback_async, options);
    }

    assert_eq!(journal.reads_repair_count, READS_REPAIR_COUNT_MAX);
    assert_eq!(storage.read_count(), READS_REPAIR_COUNT_MAX);

    let op = 200u64;
    let header = install_prepare_for_read(journal, &storage, op, Operation::REGISTER, 1, &[0x1]);
    let options = ReadOptions::repair(op, header.checksum, 1);
    journal.read_prepare(record_read_prepare_callback_async, options);

    assert_eq!(journal.reads_repair_count, READS_REPAIR_COUNT_MAX);
    assert_eq!(storage.read_count(), READS_REPAIR_COUNT_MAX);
    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert!(callbacks[0].0.is_none());
    assert_eq!(callbacks[0].1, options);

    let op = 300u64;
    let header = install_prepare_for_read(journal, &storage, op, Operation::REGISTER, 1, &[0x2]);
    let options = ReadOptions::commit(op, header.checksum);
    journal.read_prepare(record_read_prepare_callback_async, options);

    assert_eq!(journal.reads_commit_count, 1);
    assert_eq!(storage.read_count(), READS_REPAIR_COUNT_MAX + 1);

    storage.drain_read_callbacks();

    assert_eq!(journal.reads_commit_count, 0);
    assert_eq!(journal.reads_repair_count, 0);
    assert_eq!(
        take_read_prepare_callbacks().len(),
        READS_REPAIR_COUNT_MAX + 1
    );
}

#[test]
fn read_prepare_accepts_max_size_prepare() {
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::new();
    let op = 400u64;
    let cluster = 1u128;
    let body = vec![0x5a; constants::MESSAGE_BODY_SIZE_MAX_USIZE];
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica(&mut storage, 0, pool);
    replica.op = op;
    replica.cluster = cluster;
    let journal = &mut replica.journal;
    let header =
        install_prepare_for_read(journal, &storage, op, Operation::REGISTER, cluster, &body);

    let options = ReadOptions::commit(op, header.checksum);

    journal.read_prepare(record_read_prepare_callback, options);

    let log = storage.read_log.borrow();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].len, constants::MESSAGE_SIZE_MAX_USIZE);
    drop(log);

    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert_eq!(callbacks[0].1, options);
    let message = callbacks[0].0.as_ref().expect("message missing");
    assert_eq!(message.body_used().len(), body.len());
    assert_eq!(message.body_used(), body.as_slice());
}

// =========================================================================
// Write Prepare Path (Unit Tests)
// =========================================================================

#[test]
fn write_prepare_skips_clean_slot() {
    reset_prepare_callbacks();

    let mut storage = MockStorage::new();
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica_dirty(&mut storage, 0, pool.clone());
    let journal = &mut replica.journal;
    let op = 5u64;
    let slot = (op % SLOT_COUNT as u64) as usize;
    let header = make_prepare_header_with_size(op, Operation::REGISTER, constants::SECTOR_SIZE);

    journal.headers[slot] = header;
    journal.headers_redundant[slot] = header;
    journal.prepare_inhabited[slot] = true;
    journal.prepare_checksums[slot] = header.checksum;
    journal.dirty.unset(slot);
    journal.faulty.unset(slot);

    let mut message = make_prepare_message(&pool, header);
    let message_ptr = message.as_mut() as *mut MessagePrepare;

    journal.write_prepare(record_prepare_callback, message_ptr);

    assert_eq!(storage.write_count(), 0);
    assert_eq!(take_prepare_callbacks(), vec![None]);
    assert!(!journal.dirty.is_set(slot));
    assert!(!journal.faulty.is_set(slot));
    assert!(journal.prepare_inhabited[slot]);
    assert_eq!(journal.prepare_checksums[slot], header.checksum);
    assert_eq!(journal.headers_redundant[slot], header);
}

#[test]
fn write_prepare_returns_none_when_pool_exhausted() {
    reset_prepare_callbacks();

    let mut storage = MockStorage::new();
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica_dirty(&mut storage, 0, pool.clone());
    let journal = &mut replica.journal;
    let op = 1u64;
    let slot = (op % SLOT_COUNT as u64) as usize;
    let header = make_prepare_header_with_size(op, Operation::REGISTER, constants::SECTOR_SIZE);
    journal.headers[slot] = header;

    let mut busy = Vec::new();
    for _ in 0..32 {
        busy.push(journal.writes.acquire().unwrap());
    }
    assert!(journal.writes.acquire().is_none());

    let mut message = make_prepare_message(&pool, header);
    let message_ptr = message.as_mut() as *mut MessagePrepare;

    journal.write_prepare(record_prepare_callback, message_ptr);

    assert_eq!(storage.write_count(), 0);
    assert_eq!(take_prepare_callbacks(), vec![None]);

    for write in busy {
        journal.writes.release(write);
    }
}

#[test]
fn write_prepare_happy_path_updates_state_and_writes() {
    reset_prepare_callbacks();

    let mut storage = MockStorage::new();
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica_dirty(&mut storage, 0, pool.clone());
    let journal = &mut replica.journal;
    let op = 7u64;
    let slot = (op % SLOT_COUNT as u64) as usize;
    let header = make_prepare_header_with_size(op, Operation::REGISTER, constants::SECTOR_SIZE);
    journal.headers[slot] = header;

    let mut message = make_prepare_message(&pool, header);
    let message_ptr = message.as_mut() as *mut MessagePrepare;

    journal.write_prepare(record_prepare_callback, message_ptr);

    let log = storage.write_log.borrow();
    assert_eq!(log.len(), 2);
    assert_eq!(log[0].zone, Zone::WalPrepares);
    assert_eq!(
        log[0].offset,
        (constants::MESSAGE_SIZE_MAX as u64) * slot as u64
    );
    assert_eq!(log[0].len, constants::SECTOR_SIZE);

    let expected_header_offset = (slot / HEADERS_PER_SECTOR * constants::SECTOR_SIZE) as u64;
    assert_eq!(log[1].zone, Zone::WalHeaders);
    assert_eq!(log[1].offset, expected_header_offset);
    assert_eq!(log[1].len, constants::SECTOR_SIZE);
    drop(log);

    assert!(!journal.dirty.is_set(slot));
    assert!(!journal.faulty.is_set(slot));
    assert_eq!(journal.headers_redundant[slot], header);
    assert!(journal.prepare_inhabited[slot]);
    assert_eq!(journal.prepare_checksums[slot], header.checksum);
    assert_eq!(take_prepare_callbacks(), vec![Some(message_ptr)]);
}

#[test]
fn write_prepare_non_sector_aligned_size_zero_pads() {
    reset_prepare_callbacks();

    let mut storage = MockStorage::new();
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica_dirty(&mut storage, 0, pool.clone());
    let journal = &mut replica.journal;
    let op = 11u64;
    let slot = (op % SLOT_COUNT as u64) as usize;

    let header_size = HeaderPrepare::SIZE + 1;
    let buffer_len = constants::sector_ceil(header_size);
    assert!(buffer_len > header_size);

    let mut message = Box::new(pool.get::<PrepareCmd>());
    message.set_used_len(header_size);
    message.body_used_mut().fill(0x5a);

    let buffer_ptr = message.buffer_ptr();
    let body_len = header_size - HeaderPrepare::SIZE;

    {
        let header = message.header_mut();
        header.operation = Operation::REGISTER;
        header.op = op;
        let body = unsafe {
            core::slice::from_raw_parts(buffer_ptr.add(constants::HEADER_SIZE_USIZE), body_len)
        };
        header.set_checksum_body(body);
        header.set_checksum();
    }

    unsafe {
        let padding =
            core::slice::from_raw_parts_mut(buffer_ptr.add(header_size), buffer_len - header_size);
        padding.fill(0xaa);
    }

    let header = *message.header();
    journal.headers[slot] = header;

    let message_ptr = message.as_mut() as *mut MessagePrepare;
    journal.write_prepare(record_prepare_callback, message_ptr);

    let log = storage.write_log.borrow();
    assert_eq!(log.len(), 2);
    assert_eq!(log[0].zone, Zone::WalPrepares);
    assert_eq!(
        log[0].offset,
        (constants::MESSAGE_SIZE_MAX as u64) * slot as u64
    );
    assert_eq!(log[0].len, buffer_len);
    drop(log);

    let padding = unsafe {
        core::slice::from_raw_parts(buffer_ptr.add(header_size), buffer_len - header_size)
    };
    assert!(padding.iter().all(|&b| b == 0));
    assert_eq!(take_prepare_callbacks(), vec![Some(message_ptr)]);
}

#[test]
fn write_prepare_aborts_when_header_changes_during_payload_write() {
    reset_prepare_callbacks();

    let mut storage = MockStorage::<true>::new();
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica_dirty(&mut storage, 0, pool.clone());
    let journal = &mut replica.journal;
    let op = 9u64;
    let slot = (op % SLOT_COUNT as u64) as usize;
    let header = make_prepare_header_with_size(op, Operation::REGISTER, constants::SECTOR_SIZE);
    journal.headers[slot] = header;

    let mut message = make_prepare_message(&pool, header);
    let message_ptr = message.as_mut() as *mut MessagePrepare;

    journal.write_prepare(record_prepare_callback_async, message_ptr);
    assert_eq!(storage.write_count(), 1);
    assert_eq!(storage.pending_callbacks.borrow().len(), 1);

    let new_header = make_prepare_header_with_size(op, Operation::ROOT, constants::SECTOR_SIZE);
    journal.headers[slot] = new_header;

    storage.drain_callbacks();

    assert_eq!(storage.write_count(), 1);
    assert_eq!(storage.pending_callbacks.borrow().len(), 0);
    assert!(journal.dirty.is_set(slot));
    assert_eq!(take_prepare_callbacks(), vec![None]);
}

#[test]
fn write_prepare_aborts_on_header_write_mismatch() {
    reset_prepare_callbacks();

    let mut storage = MockStorage::<true>::new();
    let pool = MessagePool::new(1);
    let mut replica = make_recovered_replica_dirty(&mut storage, 0, pool.clone());
    let journal = &mut replica.journal;
    let op = 11u64;
    let slot = (op % SLOT_COUNT as u64) as usize;
    let header = make_prepare_header_with_size(op, Operation::REGISTER, constants::SECTOR_SIZE);
    journal.headers[slot] = header;

    let mut message = make_prepare_message(&pool, header);
    let message_ptr = message.as_mut() as *mut MessagePrepare;

    journal.write_prepare(record_prepare_callback_async, message_ptr);
    assert_eq!(storage.write_count(), 1);

    storage.drain_callbacks();
    assert_eq!(storage.write_count(), 2);
    assert_eq!(storage.pending_callbacks.borrow().len(), 1);

    journal.headers_redundant[slot].checksum = header.checksum.wrapping_add(1);

    storage.drain_callbacks();

    assert_eq!(storage.write_count(), 2);
    assert_eq!(storage.pending_callbacks.borrow().len(), 0);
    assert!(journal.dirty.is_set(slot));
    assert!(journal.faulty.is_set(slot));
    assert_eq!(take_prepare_callbacks(), vec![None]);
}

// =========================================================================
// Write + Read Integration (Unit Tests)
// =========================================================================

#[test]
fn write_then_read_prepare_round_trip() {
    reset_prepare_callbacks();
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::new();
    let op = 24u64;
    let cluster = 9u128;
    let pool = MessagePool::new(2);
    let mut replica = make_recovered_replica_dirty(&mut storage, 0, pool.clone());
    replica.op = op;
    replica.cluster = cluster;
    let journal = &mut replica.journal;
    let body = vec![0x9a, 0xbc, 0xde, 0xf0, 0x11];

    let mut header = HeaderPrepare::new();
    header.command = Command::Prepare;
    header.operation = Operation::REGISTER;
    header.op = op;
    header.cluster = cluster;
    header.size = (HeaderPrepare::SIZE + body.len()) as u32;
    header.release = Release(1);
    header.parent = 1;
    header.commit = op - 1;
    header.timestamp = 1;
    header.client = 1;
    header.request = 0;
    header.set_checksum_body(&body);
    header.set_checksum();

    let slot = Slot::from_op(op);
    journal.headers[slot.index()] = header;

    let mut write_message = Box::new(pool.get::<PrepareCmd>());
    write_message.set_used_len(header.size as usize);
    write_message.body_used_mut().copy_from_slice(&body);
    *write_message.header_mut() = header;
    let write_message_ptr = write_message.as_mut() as *mut MessagePrepare;

    journal.write_prepare(record_prepare_callback, write_message_ptr);

    assert_eq!(storage.write_count(), 2);
    assert_eq!(take_prepare_callbacks(), vec![Some(write_message_ptr)]);
    assert!(journal.prepare_inhabited[slot.index()]);
    assert_eq!(journal.prepare_checksums[slot.index()], header.checksum);

    let options = ReadOptions::commit(op, header.checksum);

    journal.read_prepare(record_read_prepare_callback, options);

    assert_eq!(storage.read_count(), 1);
    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert_eq!(callbacks[0].1, options);
    let message = callbacks[0].0.as_ref().expect("message missing");
    assert_eq!(message.header().op, header.op);
    assert_eq!(message.header().checksum, header.checksum);
    assert_eq!(message.header().size, header.size);
    assert_eq!(message.body_used(), body.as_slice());
}

#[test]
fn async_write_then_read_prepare_round_trip() {
    reset_prepare_callbacks();
    reset_read_prepare_callbacks();

    let mut storage = MockStorage::<true>::new();
    let op = 41u64;
    let cluster = 5u128;
    let body = vec![0xa1, 0xb2, 0xc3, 0xd4];
    let pool = MessagePool::new(2);
    let mut replica = make_recovered_replica_dirty(&mut storage, 0, pool.clone());
    replica.op = op;
    replica.cluster = cluster;
    let journal = &mut replica.journal;

    let mut header = HeaderPrepare::new();
    header.command = Command::Prepare;
    header.operation = Operation::REGISTER;
    header.op = op;
    header.cluster = cluster;
    header.size = (HeaderPrepare::SIZE + body.len()) as u32;
    header.release = Release(1);
    header.parent = 1;
    header.commit = op - 1;
    header.timestamp = 1;
    header.client = 1;
    header.request = 0;
    header.set_checksum_body(&body);
    header.set_checksum();

    let slot = Slot::from_op(op);
    journal.headers[slot.index()] = header;

    let mut write_message = Box::new(pool.get::<PrepareCmd>());
    write_message.set_used_len(header.size as usize);
    write_message.body_used_mut().copy_from_slice(&body);
    *write_message.header_mut() = header;
    let write_message_ptr = write_message.as_mut() as *mut MessagePrepare;

    journal.write_prepare(record_prepare_callback_async, write_message_ptr);

    assert_eq!(storage.pending_callbacks.borrow().len(), 1);
    storage.drain_callbacks();
    assert_eq!(storage.pending_callbacks.borrow().len(), 1);
    storage.drain_callbacks();

    assert_eq!(storage.write_count(), 2);
    assert_eq!(take_prepare_callbacks(), vec![Some(write_message_ptr)]);
    assert!(journal.prepare_inhabited[slot.index()]);
    assert_eq!(journal.prepare_checksums[slot.index()], header.checksum);

    let options = ReadOptions::commit(op, header.checksum);

    journal.read_prepare(record_read_prepare_callback_async, options);

    assert_eq!(storage.pending_read_callbacks.borrow().len(), 1);
    storage.drain_read_callbacks();

    assert_eq!(storage.read_count(), 1);
    let callbacks = take_read_prepare_callbacks();
    assert_eq!(callbacks.len(), 1);
    assert_eq!(callbacks[0].1, options);
    let message = callbacks[0].0.as_ref().expect("message missing");
    assert_eq!(message.header().op, header.op);
    assert_eq!(message.header().checksum, header.checksum);
    assert_eq!(message.header().size, header.size);
    assert_eq!(message.body_used(), body.as_slice());
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
                // Set op immediately to avoid conflicts during lock_sectors iteration.
                unsafe { (*write).op = i as u64; }
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
                // Set op immediately to avoid conflicts during lock_sectors iteration.
                unsafe { (*write).op = i as u64; }
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
        (*write).op = 1;
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

    // Set op values (required by caller contract).
    unsafe {
        (*write1).op = 1;
        (*write2).op = 2;
        (*write3).op = 3;
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

// =========================================================================
// Journal::new Initialization Tests
// =========================================================================

#[test]
fn journal_new_initializes_all_fields_correctly() {
    let mut storage = MockStorage::new();
    let journal = TestJournal::new(&mut storage, 7);

    // Basic pointer and replica fields
    assert_eq!(journal.storage, &mut storage as *mut _);
    assert_eq!(journal.replica, 7);

    // Status should be Init
    assert!(matches!(journal.status, Status::Init));

    // BitSets should be full (all slots marked dirty/faulty initially)
    assert!(
        journal.dirty.is_full(),
        "dirty BitSet should be full initially"
    );
    assert!(
        journal.faulty.is_full(),
        "faulty BitSet should be full initially"
    );

    // Verify dirty/faulty bitsets have correct capacity for all slots
    assert_eq!(
        BitSet::<SLOT_COUNT, SLOT_COUNT_WORDS>::capacity(),
        SLOT_COUNT,
        "dirty/faulty bitsets should track all journal slots"
    );

    // Vector lengths and contents - prepare_checksums
    assert_eq!(
        journal.prepare_checksums.len(),
        SLOT_COUNT,
        "prepare_checksums should have SLOT_COUNT entries"
    );
    assert!(
        journal.prepare_checksums.iter().all(|&c| c == 0),
        "all prepare_checksums should be zero"
    );

    // Vector lengths and contents - prepare_inhabited
    assert_eq!(
        journal.prepare_inhabited.len(),
        SLOT_COUNT,
        "prepare_inhabited should have SLOT_COUNT entries"
    );
    assert!(
        journal.prepare_inhabited.iter().all(|&i| !i),
        "all prepare_inhabited should be false"
    );

    // AlignedSlice lengths
    assert_eq!(
        journal.headers.len(),
        SLOT_COUNT,
        "headers should have SLOT_COUNT entries"
    );
    assert_eq!(
        journal.headers_redundant.len(),
        SLOT_COUNT,
        "headers_redundant should have SLOT_COUNT entries"
    );
    assert_eq!(
        journal.headers_redundant_raw.len(),
        SLOT_COUNT,
        "headers_redundant_raw should have SLOT_COUNT entries"
    );
    assert_eq!(
        journal.write_headers_sectors.len(),
        constants::JOURNAL_IOPS_WRITE_MAX as usize,
        "write_headers_sectors should have JOURNAL_IOPS_WRITE_MAX entries"
    );

    // IOPSType write pool should be empty (all 32 slots available)
    assert_eq!(
        journal.writes.available(),
        32,
        "writes pool should have all slots available"
    );

    // Header chunk tracking should be in default (not done) state
    assert!(journal.header_chunks_requested.is_empty());
    assert!(journal.header_chunks_recovered.is_empty());
    assert!(
        !journal.header_chunks_done(),
        "header chunks should not be done initially"
    );
}

#[test]
fn journal_new_headers_have_invalid_checksums() {
    let mut storage = MockStorage::new();
    let journal = TestJournal::new(&mut storage, 0);

    // Critical safety invariant: headers must have invalid checksums initially.
    // This prevents interpreting uninitialized/zeroed data as valid prepares,
    // which could lead to data corruption during recovery.
    assert!(
        journal.headers.iter().all(|h| !h.valid_checksum()),
        "all headers should have invalid checksums after initialization"
    );
    assert!(
        journal
            .headers_redundant
            .iter()
            .all(|h| !h.valid_checksum()),
        "all redundant headers should have invalid checksums after initialization"
    );
    assert!(
        journal
            .headers_redundant_raw
            .iter()
            .all(|h| !h.valid_checksum()),
        "all raw redundant headers should have invalid checksums after initialization"
    );
}

#[test]
fn journal_new_works_with_edge_case_replica_values() {
    // Test minimum replica value (0)
    let mut storage_min = MockStorage::new();
    let journal_min = TestJournal::new(&mut storage_min, 0);
    assert_eq!(journal_min.replica, 0);
    assert!(matches!(journal_min.status, Status::Init));
    assert!(journal_min.dirty.is_full());

    // Test maximum replica value (255)
    let mut storage_max = MockStorage::new();
    let journal_max = TestJournal::new(&mut storage_max, 255);
    assert_eq!(journal_max.replica, 255);
    assert!(matches!(journal_max.status, Status::Init));
    assert!(journal_max.faulty.is_full());
}

#[test]
fn async_writes_complete_correctly() {
    let mut storage = MockStorage::<true>::new();
    let mut journal = Journal::<MockStorage<true>, 32, 1>::new(&mut storage, 0);

    let write = journal.writes.acquire().unwrap();
    let buf = vec![1u8; 4096];

    unsafe {
        (*write).op = 1;
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
        (*write).op = 1;
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

// =========================================================================
// Header Helper Method Tests
// =========================================================================

/// Creates a valid HeaderPrepare with the given op and operation.
fn make_header(op: u64, operation: Operation) -> HeaderPrepare {
    let mut header = HeaderPrepare::zeroed();
    header.command = Command::Prepare;
    header.operation = operation;
    header.op = op;
    header.size = HeaderPrepare::SIZE as u32;
    header.set_checksum();
    header
}

/// Creates a reserved header for the given slot index.
/// Reserved headers have operation = RESERVED and op = slot_index.
fn make_reserved_header(slot_index: usize) -> HeaderPrepare {
    let mut header = HeaderPrepare::zeroed();
    header.command = Command::Prepare;
    header.operation = Operation::RESERVED;
    header.op = slot_index as u64;
    header.size = HeaderPrepare::SIZE as u32;
    header.set_checksum();
    header
}

fn make_recovery_header(op: u64, view: u32, checksum: u128, operation: Operation) -> HeaderPrepare {
    let mut header = HeaderPrepare::new();
    header.command = Command::Prepare;
    header.operation = operation;
    header.op = op;
    header.view = view;
    header.checksum = checksum;
    header
}

/// Creates a valid Prepare header with a sector-aligned size.
fn make_prepare_header_with_size(op: u64, operation: Operation, size: usize) -> HeaderPrepare {
    assert!(size.is_multiple_of(constants::SECTOR_SIZE));
    assert!(size >= HeaderPrepare::SIZE);

    let mut header = HeaderPrepare::new();
    header.command = Command::Prepare;
    header.operation = operation;
    header.op = op;
    header.size = size as u32;
    if operation != Operation::ROOT && operation != Operation::RESERVED {
        header.release = Release(1);
        header.parent = 1;
        header.commit = op.saturating_sub(1);
        header.timestamp = 1;

        if operation == Operation::PULSE || operation == Operation::UPGRADE {
            header.client = 0;
            header.request = 0;
        } else if operation == Operation::REGISTER {
            header.client = 1;
            header.request = 0;
        } else {
            header.client = 1;
            header.request = 1;
        }
    }

    let body_len = size - HeaderPrepare::SIZE;
    let body = vec![0u8; body_len];
    header.set_checksum_body(&body);
    header.set_checksum();

    header
}

fn make_prepare_message(pool: &MessagePool, header: HeaderPrepare) -> Box<MessagePrepare> {
    let mut message = Box::new(pool.get::<PrepareCmd>());
    *message.header_mut() = header;
    message
}

fn make_recovered_replica<const ASYNC: bool>(
    storage: &mut MockStorage<ASYNC>,
    replica: u8,
    message_pool: MessagePool,
) -> Replica<MockStorage<ASYNC>, 32, 1> {
    let mut replica = Replica::new(storage, replica, 0, message_pool);
    replica.journal.status = Status::Recovered;
    replica.journal.dirty = BitSet::empty();
    replica.journal.faulty = BitSet::empty();
    replica
}

fn make_recovered_replica_dirty<const ASYNC: bool>(
    storage: &mut MockStorage<ASYNC>,
    replica: u8,
    message_pool: MessagePool,
) -> Replica<MockStorage<ASYNC>, 32, 1> {
    let mut replica = make_recovered_replica(storage, replica, message_pool);
    replica.journal.dirty = BitSet::full();
    replica.journal.faulty = BitSet::full();
    replica
}

fn make_prepare_disk_image(
    op: u64,
    operation: Operation,
    cluster: u128,
    body: &[u8],
) -> (HeaderPrepare, Vec<u8>) {
    assert!(body.len() <= constants::MESSAGE_BODY_SIZE_MAX_USIZE);

    let mut header = HeaderPrepare::new();
    header.command = Command::Prepare;
    header.operation = operation;
    header.op = op;
    header.cluster = cluster;
    header.size = (HeaderPrepare::SIZE + body.len()) as u32;
    if operation != Operation::ROOT && operation != Operation::RESERVED {
        header.release = Release(1);
        header.parent = 1;
        header.commit = op.saturating_sub(1);
        header.timestamp = 1;

        if operation == Operation::PULSE || operation == Operation::UPGRADE {
            header.client = 0;
            header.request = 0;
        } else if operation == Operation::REGISTER {
            header.client = 1;
            header.request = 0;
        } else {
            header.client = 1;
            header.request = 1;
        }
    }
    header.set_checksum_body(body);
    header.set_checksum();

    let mut data = vec![0u8; constants::MESSAGE_SIZE_MAX_USIZE];
    data[..HeaderPrepare::SIZE].copy_from_slice(header.as_bytes());
    data[HeaderPrepare::SIZE..HeaderPrepare::SIZE + body.len()].copy_from_slice(body);

    (header, data)
}

fn install_prepare_for_read<const ASYNC: bool>(
    journal: &mut Journal<MockStorage<ASYNC>, 32, 1>,
    storage: &MockStorage<ASYNC>,
    op: u64,
    operation: Operation,
    cluster: u128,
    body: &[u8],
) -> HeaderPrepare {
    let (header, data) = make_prepare_disk_image(op, operation, cluster, body);
    let slot = Slot::from_op(op);
    journal.headers[slot.index()] = header;
    journal.prepare_inhabited[slot.index()] = true;
    journal.prepare_checksums[slot.index()] = header.checksum;

    let offset = (constants::MESSAGE_SIZE_MAX as u64) * slot.index() as u64;
    storage.set_read_data(Zone::WalPrepares, offset, data);
    header
}

// -------------------------------------------------------------------------
// previous_entry / next_entry Boundary Tests
// -------------------------------------------------------------------------

#[test]
fn previous_entry_returns_none_for_op_zero() {
    let mut storage = MockStorage::new();
    let mut journal = TestJournal::new(&mut storage, 0);

    // Insert header at op 0.
    let header = make_header(0, Operation::ROOT);
    journal.headers[0] = header;

    // previous_entry for op 0 should return None.
    let result = journal.previous_entry(&journal.headers[0]);
    assert!(result.is_none());
}

// -------------------------------------------------------------------------
// op_maximum Panic Tests
// -------------------------------------------------------------------------

#[test]
#[should_panic]
fn op_maximum_panics_when_not_recovered() {
    let mut storage = MockStorage::new();
    let journal = TestJournal::new(&mut storage, 0);

    // Status is Init, should panic.
    let _ = journal.op_maximum();
}
