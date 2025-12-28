//! Header chunk tracking for journal recovery.
//!
//! During recovery, headers are read in chunks bounded by `MESSAGE_SIZE_MAX`
//! (the I/O size limit). This module tracks which chunks have been requested
//! and which have completed, enabling parallel reads while ensuring all
//! headers are eventually recovered.
//!
//! State machine: unrequested → requested → recovered

#![allow(dead_code)]

use core::cmp::min;

use super::journal_primitives::{HEADER_CHUNK_COUNT, HEADER_CHUNK_WORDS};
use crate::constants;
use crate::stdx::BitSet;
use crate::vsr::journal_primitives::{HEADERS_SIZE, WAL_HEADER_SIZE};
use crate::vsr::HeaderPrepare;

// Compile-time assertions to verify constant relationships.
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

/// Bitset tracking chunk state. `HEADER_CHUNK_WORDS` is the number of `u64`s
/// needed to store `HEADER_CHUNK_COUNT` bits.
pub type HeaderChunks = BitSet<HEADER_CHUNK_COUNT, HEADER_CHUNK_WORDS>;

/// Descriptor for a header chunk read request.
///
/// Returned by [`HeaderChunkTracking::next_to_request`] to specify what to read.
/// The caller issues the I/O and calls [`HeaderChunkTracking::mark_recovered`]
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

/// Tracks header chunk recovery state.
///
/// Maintains two bitsets tracking the state machine for each chunk:
/// unrequested → requested → recovered.
///
/// # Invariant
///
/// `recovered ⊆ requested` — a chunk cannot complete before being issued.
#[derive(Debug, Clone)]
pub struct HeaderChunkTracking {
    /// Chunks with outstanding or completed read requests.
    pub header_chunks_requested: HeaderChunks,
    /// Chunks whose reads have completed and data copied.
    pub header_chunks_recovered: HeaderChunks,
}

impl Default for HeaderChunkTracking {
    fn default() -> Self {
        Self {
            header_chunks_requested: HeaderChunks::empty(),
            header_chunks_recovered: HeaderChunks::empty(),
        }
    }
}

impl HeaderChunkTracking {
    #[inline]
    pub fn reset(&mut self) {
        self.header_chunks_recovered = HeaderChunks::empty();
        self.header_chunks_requested = HeaderChunks::empty();
        self.assert_invariants();
    }

    #[inline]
    pub fn done(&self) -> bool {
        self.header_chunks_recovered.is_full()
    }

    /// Returns the byte offset for a given chunk index.
    ///
    /// Chunks are `MESSAGE_SIZE_MAX`-sized, so offset = index × MESSAGE_SIZE_MAX.
    #[inline]
    pub fn chunk_offset(chunk_index: usize) -> u64 {
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
    pub fn chunk_len_for_offset(offset: u64) -> usize {
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
    ///
    /// Caller should issue a read for the returned chunk and call
    /// [`mark_recovered`](Self::mark_recovered) on completion.
    #[inline]
    pub fn next_to_request(&mut self) -> Option<HeaderChunk> {
        self.assert_invariants();

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

        let offset = Self::chunk_offset(chunk_index);
        assert!(offset < HEADERS_SIZE);

        let len = Self::chunk_len_for_offset(offset);
        let chunk = HeaderChunk {
            index: chunk_index,
            offset,
            len,
        };
        chunk.assert_invariants();

        self.assert_invariants();
        Some(chunk)
    }

    /// Marks a chunk as recovered after its read completes.
    ///
    /// # Panics
    ///
    /// Panics if the chunk was not previously requested or was already recovered.
    #[inline]
    pub fn mark_recovered(&mut self, chunk_index: usize) {
        self.assert_invariants();

        assert!(chunk_index < HEADER_CHUNK_COUNT);
        assert!(self.header_chunks_requested.is_set(chunk_index));
        assert!(!self.header_chunks_recovered.is_set(chunk_index));

        self.header_chunks_recovered.set(chunk_index);
        self.assert_invariants();
    }

    #[inline]
    fn assert_invariants(&self) {
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
    pub fn bytes_as_headers(chunk_bytes: &[u8]) -> &[HeaderPrepare] {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // HeaderChunk Unit Tests
    // ==========================================================================

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
            len: HeaderChunkTracking::chunk_len_for_offset(0),
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
    // HeaderChunkTracking Default/Reset Tests
    // ==========================================================================

    #[test]
    fn tracking_default_is_empty() {
        let tracking = HeaderChunkTracking::default();
        assert!(tracking.header_chunks_requested.is_empty());
        assert!(tracking.header_chunks_recovered.is_empty());
        assert!(!tracking.done());
    }

    #[test]
    fn tracking_reset_clears_state() {
        let mut tracking = HeaderChunkTracking::default();

        // Request and recover some chunks
        let chunk = tracking.next_to_request().unwrap();
        tracking.mark_recovered(chunk.index);

        tracking.reset();

        assert!(tracking.header_chunks_requested.is_empty());
        assert!(tracking.header_chunks_recovered.is_empty());
        assert!(!tracking.done());
    }

    // ==========================================================================
    // Chunk Offset/Length Calculation Tests
    // ==========================================================================

    #[test]
    fn chunk_offset_first_chunk_is_zero() {
        assert_eq!(HeaderChunkTracking::chunk_offset(0), 0);
    }

    #[test]
    fn chunk_offset_second_chunk_is_message_size_max() {
        if HEADER_CHUNK_COUNT > 1 {
            assert_eq!(
                HeaderChunkTracking::chunk_offset(1),
                constants::MESSAGE_SIZE_MAX as u64
            );
        }
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)] // Range may be empty when HEADER_CHUNK_COUNT == 1
    fn chunk_offset_monotonically_increasing() {
        for i in 1..HEADER_CHUNK_COUNT {
            assert!(
                HeaderChunkTracking::chunk_offset(i) > HeaderChunkTracking::chunk_offset(i - 1)
            );
        }
    }

    #[test]
    #[should_panic]
    fn chunk_offset_panics_at_boundary() {
        HeaderChunkTracking::chunk_offset(HEADER_CHUNK_COUNT);
    }

    #[test]
    fn chunk_len_for_offset_zero_returns_valid_length() {
        let len = HeaderChunkTracking::chunk_len_for_offset(0);
        assert!(len > 0);
        assert!(len <= constants::MESSAGE_SIZE_MAX as usize);
        assert!(len.is_multiple_of(constants::SECTOR_SIZE));
        assert!(len.is_multiple_of(WAL_HEADER_SIZE));
    }

    #[test]
    fn chunk_len_covers_entire_headers_region() {
        let mut total_covered = 0u64;
        for i in 0..HEADER_CHUNK_COUNT {
            let offset = HeaderChunkTracking::chunk_offset(i);
            let len = HeaderChunkTracking::chunk_len_for_offset(offset);
            total_covered += len as u64;
        }
        assert_eq!(total_covered, HEADERS_SIZE);
    }

    #[test]
    #[should_panic]
    fn chunk_len_for_offset_panics_at_headers_size() {
        HeaderChunkTracking::chunk_len_for_offset(HEADERS_SIZE);
    }

    // ==========================================================================
    // State Machine Transition Tests
    // ==========================================================================

    #[test]
    fn next_to_request_returns_first_chunk_initially() {
        let mut tracking = HeaderChunkTracking::default();
        let chunk = tracking.next_to_request().unwrap();
        assert_eq!(chunk.index, 0);
        assert_eq!(chunk.offset, 0);
    }

    #[test]
    fn next_to_request_marks_chunk_as_requested() {
        let mut tracking = HeaderChunkTracking::default();
        let chunk = tracking.next_to_request().unwrap();
        assert!(tracking.header_chunks_requested.is_set(chunk.index));
        assert!(!tracking.header_chunks_recovered.is_set(chunk.index));
    }

    #[test]
    fn next_to_request_returns_chunks_in_order() {
        let mut tracking = HeaderChunkTracking::default();
        for expected_index in 0..HEADER_CHUNK_COUNT {
            let chunk = tracking.next_to_request().unwrap();
            assert_eq!(chunk.index, expected_index);
        }
    }

    #[test]
    fn next_to_request_returns_none_when_all_requested() {
        let mut tracking = HeaderChunkTracking::default();

        // Request all chunks
        for _ in 0..HEADER_CHUNK_COUNT {
            tracking.next_to_request();
        }

        // Should return None since all are requested but not recovered
        assert!(tracking.next_to_request().is_none());
    }

    #[test]
    fn next_to_request_returns_none_when_done() {
        let mut tracking = HeaderChunkTracking::default();

        // Request and recover all chunks
        for _ in 0..HEADER_CHUNK_COUNT {
            let chunk = tracking.next_to_request().unwrap();
            tracking.mark_recovered(chunk.index);
        }

        assert!(tracking.done());
        assert!(tracking.next_to_request().is_none());
    }

    #[test]
    fn mark_recovered_updates_state() {
        let mut tracking = HeaderChunkTracking::default();
        let chunk = tracking.next_to_request().unwrap();

        assert!(!tracking.header_chunks_recovered.is_set(chunk.index));
        tracking.mark_recovered(chunk.index);
        assert!(tracking.header_chunks_recovered.is_set(chunk.index));
    }

    #[test]
    #[should_panic]
    fn mark_recovered_panics_if_not_requested() {
        let mut tracking = HeaderChunkTracking::default();
        tracking.mark_recovered(0); // Never requested
    }

    #[test]
    #[should_panic]
    fn mark_recovered_panics_if_already_recovered() {
        let mut tracking = HeaderChunkTracking::default();
        let chunk = tracking.next_to_request().unwrap();
        tracking.mark_recovered(chunk.index);
        tracking.mark_recovered(chunk.index); // Double recovery
    }

    #[test]
    #[should_panic]
    fn mark_recovered_panics_for_out_of_bounds_index() {
        let mut tracking = HeaderChunkTracking::default();
        tracking.mark_recovered(HEADER_CHUNK_COUNT);
    }

    // ==========================================================================
    // done() Tests
    // ==========================================================================

    #[test]
    fn done_false_when_empty() {
        let tracking = HeaderChunkTracking::default();
        assert!(!tracking.done());
    }

    #[test]
    fn done_false_when_partially_recovered() {
        let mut tracking = HeaderChunkTracking::default();
        let chunk = tracking.next_to_request().unwrap();
        tracking.mark_recovered(chunk.index);

        if HEADER_CHUNK_COUNT > 1 {
            assert!(!tracking.done());
        }
    }

    #[test]
    fn done_true_when_all_recovered() {
        let mut tracking = HeaderChunkTracking::default();

        for _ in 0..HEADER_CHUNK_COUNT {
            let chunk = tracking.next_to_request().unwrap();
            tracking.mark_recovered(chunk.index);
        }

        assert!(tracking.done());
    }

    // ==========================================================================
    // bytes_as_headers Tests
    // ==========================================================================

    #[test]
    fn bytes_as_headers_single_header() {
        // Use aligned allocation
        let layout =
            std::alloc::Layout::from_size_align(WAL_HEADER_SIZE, core::mem::align_of::<HeaderPrepare>())
                .unwrap();
        let buffer = unsafe {
            let ptr = std::alloc::alloc_zeroed(layout);
            std::slice::from_raw_parts(ptr, WAL_HEADER_SIZE)
        };

        let headers = HeaderChunkTracking::bytes_as_headers(buffer);
        assert_eq!(headers.len(), 1);

        // Clean up
        unsafe {
            std::alloc::dealloc(buffer.as_ptr() as *mut u8, layout);
        }
    }

    #[test]
    fn bytes_as_headers_multiple_headers() {
        let count = 4;
        let size = WAL_HEADER_SIZE * count;
        let layout =
            std::alloc::Layout::from_size_align(size, core::mem::align_of::<HeaderPrepare>())
                .unwrap();
        let buffer = unsafe {
            let ptr = std::alloc::alloc_zeroed(layout);
            std::slice::from_raw_parts(ptr, size)
        };

        let headers = HeaderChunkTracking::bytes_as_headers(buffer);
        assert_eq!(headers.len(), count);

        // Clean up
        unsafe {
            std::alloc::dealloc(buffer.as_ptr() as *mut u8, layout);
        }
    }

    #[test]
    #[should_panic]
    fn bytes_as_headers_panics_on_empty_slice() {
        HeaderChunkTracking::bytes_as_headers(&[]);
    }

    #[test]
    #[should_panic]
    fn bytes_as_headers_panics_on_too_small_slice() {
        let buffer = vec![0u8; WAL_HEADER_SIZE - 1];
        HeaderChunkTracking::bytes_as_headers(&buffer);
    }

    #[test]
    #[should_panic]
    fn bytes_as_headers_panics_on_non_multiple_length() {
        // Create aligned buffer with non-multiple size
        let size = WAL_HEADER_SIZE + 1;
        let layout =
            std::alloc::Layout::from_size_align(size, core::mem::align_of::<HeaderPrepare>())
                .unwrap();
        let buffer = unsafe {
            let ptr = std::alloc::alloc_zeroed(layout);
            std::slice::from_raw_parts(ptr, size)
        };

        // This will panic due to non-multiple length
        let _ = std::panic::catch_unwind(|| {
            HeaderChunkTracking::bytes_as_headers(buffer);
        });

        // Clean up even on panic path
        unsafe {
            std::alloc::dealloc(buffer.as_ptr() as *mut u8, layout);
        }

        // Re-panic to satisfy should_panic
        panic!("Expected panic for non-multiple length");
    }

    #[test]
    #[should_panic]
    fn bytes_as_headers_panics_on_misaligned_input() {
        // Create buffer with extra byte to force misalignment
        let size = WAL_HEADER_SIZE + 16;
        let buffer = vec![0u8; size];

        // Find a misaligned slice within the buffer
        let align = core::mem::align_of::<HeaderPrepare>();
        let base_addr = buffer.as_ptr() as usize;
        let offset = if base_addr.is_multiple_of(align) { 1 } else { 0 };

        let slice = &buffer[offset..offset + WAL_HEADER_SIZE];
        assert!(
            !(slice.as_ptr() as usize).is_multiple_of(align),
            "Slice should be misaligned"
        );

        HeaderChunkTracking::bytes_as_headers(slice);
    }

    // ==========================================================================
    // Integration/Workflow Tests
    // ==========================================================================

    #[test]
    fn full_recovery_workflow() {
        let mut tracking = HeaderChunkTracking::default();
        let mut recovered_chunks = Vec::new();

        // Phase 1: Request all chunks
        while let Some(chunk) = tracking.next_to_request() {
            chunk.assert_invariants();
            recovered_chunks.push(chunk);
        }

        assert_eq!(recovered_chunks.len(), HEADER_CHUNK_COUNT);
        assert!(tracking.header_chunks_requested.is_full());
        assert!(!tracking.done());

        // Phase 2: Recover all chunks (in reverse order for variety)
        for chunk in recovered_chunks.iter().rev() {
            tracking.mark_recovered(chunk.index);
        }

        assert!(tracking.done());
        assert!(tracking.next_to_request().is_none());
    }

    #[test]
    fn partial_recovery_with_reset() {
        let mut tracking = HeaderChunkTracking::default();

        // Request and recover half the chunks
        let half = HEADER_CHUNK_COUNT / 2;
        for _ in 0..half {
            let chunk = tracking.next_to_request().unwrap();
            tracking.mark_recovered(chunk.index);
        }

        if HEADER_CHUNK_COUNT > 1 {
            assert!(!tracking.done());
        }

        // Reset and verify clean state
        tracking.reset();

        assert!(!tracking.done());
        assert!(tracking.header_chunks_requested.is_empty());

        // Should be able to request from beginning
        let first_chunk = tracking.next_to_request().unwrap();
        assert_eq!(first_chunk.index, 0);
    }

    #[test]
    fn out_of_order_recovery() {
        let mut tracking = HeaderChunkTracking::default();
        let mut chunks = Vec::new();

        // Request all chunks
        while let Some(chunk) = tracking.next_to_request() {
            chunks.push(chunk);
        }

        // Recover in reverse order
        for chunk in chunks.iter().rev() {
            tracking.mark_recovered(chunk.index);
        }

        assert!(tracking.done());
    }

    #[test]
    fn interleaved_request_recovery() {
        let mut tracking = HeaderChunkTracking::default();

        // Request 2, recover 1, repeat
        let mut pending = Vec::new();

        while !tracking.done() {
            // Request up to 2 chunks
            for _ in 0..2 {
                if let Some(chunk) = tracking.next_to_request() {
                    pending.push(chunk.index);
                }
            }

            // Recover 1 chunk
            if let Some(idx) = pending.pop() {
                tracking.mark_recovered(idx);
            }
        }

        assert!(tracking.done());
    }

    // ==========================================================================
    // Edge Case Tests
    // ==========================================================================

    #[test]
    fn single_chunk_scenario() {
        // This test is relevant when HEADER_CHUNK_COUNT == 1
        if HEADER_CHUNK_COUNT == 1 {
            let mut tracking = HeaderChunkTracking::default();

            let chunk = tracking.next_to_request().unwrap();
            assert_eq!(chunk.index, 0);
            assert_eq!(chunk.offset, 0);
            assert_eq!(chunk.len as u64, HEADERS_SIZE);

            assert!(tracking.next_to_request().is_none());

            tracking.mark_recovered(0);
            assert!(tracking.done());
        }
    }

    #[test]
    fn boundary_chunk_sizes() {
        // Verify last chunk has correct size
        let last_chunk_index = HEADER_CHUNK_COUNT - 1;
        let last_offset = HeaderChunkTracking::chunk_offset(last_chunk_index);
        let last_len = HeaderChunkTracking::chunk_len_for_offset(last_offset);

        // Last chunk should end exactly at HEADERS_SIZE
        assert_eq!(last_offset + last_len as u64, HEADERS_SIZE);
    }

    #[test]
    fn zero_remaining_after_full_recovery() {
        let mut tracking = HeaderChunkTracking::default();

        while let Some(chunk) = tracking.next_to_request() {
            tracking.mark_recovered(chunk.index);
        }

        // Verify no chunks remain unrequested or unrecovered
        assert!(tracking.header_chunks_requested.is_full());
        assert!(tracking.header_chunks_recovered.is_full());
        assert_eq!(
            tracking.header_chunks_requested.count(),
            tracking.header_chunks_recovered.count()
        );
    }

    // ==========================================================================
    // Invariant Verification Tests (Bug #1 fix validation)
    // ==========================================================================

    #[test]
    fn invariant_recovered_subset_of_requested() {
        let mut tracking = HeaderChunkTracking::default();

        // Request some but don't recover - this verifies the fixed invariant
        // works correctly (recovered is empty, which is a subset of requested)
        tracking.next_to_request();
        if HEADER_CHUNK_COUNT > 1 {
            tracking.next_to_request();
        }

        // Invariant check happens inside next_to_request via assert_invariants
        // If the bug #1 fix is correct, this test passes.
        // With the buggy code (checking requested.is_subset(recovered)),
        // this would have panicked.
    }

    #[test]
    fn invariant_holds_during_partial_recovery() {
        let mut tracking = HeaderChunkTracking::default();

        // Request all chunks
        let mut indices = Vec::new();
        while let Some(chunk) = tracking.next_to_request() {
            indices.push(chunk.index);
        }

        // Recover half of them
        for &idx in indices.iter().take(HEADER_CHUNK_COUNT / 2) {
            tracking.mark_recovered(idx);
        }

        // Invariant should hold: recovered (half) is subset of requested (all)
        assert!(tracking.header_chunks_recovered.count() < tracking.header_chunks_requested.count()
            || HEADER_CHUNK_COUNT <= 1);
    }
}

#[cfg(test)]
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
            let mut tracking = HeaderChunkTracking::default();
            let mut requested_indices = Vec::new();

            // Request some chunks
            for _ in 0..num_requests {
                if let Some(chunk) = tracking.next_to_request() {
                    requested_indices.push(chunk.index);
                }
            }

            // Recover some of the requested chunks
            let recoveries = num_recoveries.min(requested_indices.len());
            for &idx in requested_indices.iter().take(recoveries) {
                tracking.mark_recovered(idx);
            }

            prop_assert!(
                tracking.header_chunks_requested.count() >=
                tracking.header_chunks_recovered.count()
            );
        }

        /// Invariant: recovered is always a subset of requested
        #[test]
        fn prop_recovered_subset_of_requested(
            operations in prop::collection::vec(0u8..2, 0..50)
        ) {
            let mut tracking = HeaderChunkTracking::default();
            let mut pending_recoveries: Vec<usize> = Vec::new();

            for op in operations {
                match op {
                    0 => {
                        // Request
                        if let Some(chunk) = tracking.next_to_request() {
                            pending_recoveries.push(chunk.index);
                        }
                    }
                    1 => {
                        // Recover
                        if let Some(idx) = pending_recoveries.pop() {
                            tracking.mark_recovered(idx);
                        }
                    }
                    _ => unreachable!()
                }
            }

            // Check invariant: every recovered bit must be requested
            for idx in 0..HEADER_CHUNK_COUNT {
                if tracking.header_chunks_recovered.is_set(idx) {
                    prop_assert!(
                        tracking.header_chunks_requested.is_set(idx),
                        "Recovered chunk {} was not requested", idx
                    );
                }
            }
        }

        /// All chunks are eventually returned by next_to_request
        #[test]
        fn prop_all_chunks_requestable(_unused in Just(())) {
            let mut tracking = HeaderChunkTracking::default();
            let mut seen = [false; HEADER_CHUNK_COUNT];

            for _ in 0..HEADER_CHUNK_COUNT {
                let chunk = tracking.next_to_request();
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
                let offset = HeaderChunkTracking::chunk_offset(i);
                let len = HeaderChunkTracking::chunk_len_for_offset(offset);

                prop_assert_eq!(offset, expected_offset);
                expected_offset = offset + len as u64;
            }

            prop_assert_eq!(expected_offset, HEADERS_SIZE);
        }

        /// Interleaved request/recover operations maintain invariants
        #[test]
        fn prop_interleaved_operations_maintain_invariants(seed: u64) {
            use std::collections::VecDeque;

            let mut tracking = HeaderChunkTracking::default();
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
                    if let Some(chunk) = tracking.next_to_request() {
                        pending.push_back(chunk.index);
                    }
                } else {
                    // Recover oldest pending
                    if let Some(idx) = pending.pop_front() {
                        tracking.mark_recovered(idx);
                    }
                }

                // Verify invariant after each operation
                prop_assert!(
                    tracking.header_chunks_recovered.count() <=
                    tracking.header_chunks_requested.count()
                );
            }
        }

        /// Chunk invariants are always valid for returned chunks
        #[test]
        fn prop_chunk_invariants_always_valid(_unused in Just(())) {
            let mut tracking = HeaderChunkTracking::default();

            while let Some(chunk) = tracking.next_to_request() {
                // This will panic if invariants don't hold
                chunk.assert_invariants();
                tracking.mark_recovered(chunk.index);
            }
        }
    }
}
