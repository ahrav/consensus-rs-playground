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
use core::mem::size_of;

use super::journal_primitives::{HEADER_CHUNK_COUNT, HEADER_CHUNK_WORDS};
use crate::constants;
use crate::stdx::BitSet;
use crate::vsr::journal_primitives::{HEADERS_SIZE, WAL_HEADER_SIZE};
use crate::vsr::{Header, HeaderPrepare};

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
            self.header_chunks_requested
                .is_subset(&self.header_chunks_recovered)
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
    /// - Chunk buffers are sector-aligned, satisfying alignment requirements
    /// - Returned slice lifetime is tied to input slice
    ///
    /// # Panics
    ///
    /// Panics if `chunk_bytes.len()` is not a positive multiple of `WAL_HEADER_SIZE`.
    #[inline]
    pub fn bytes_as_headers(chunk_bytes: &[u8]) -> &[HeaderPrepare] {
        assert!(chunk_bytes.len() >= WAL_HEADER_SIZE);
        assert!(chunk_bytes.len().is_multiple_of(WAL_HEADER_SIZE));

        let count = chunk_bytes.len() / WAL_HEADER_SIZE;
        let headers = chunk_bytes.as_ptr() as *const HeaderPrepare;
        // SAFETY: See doc comment above. The assertions guarantee length/alignment.
        unsafe { std::slice::from_raw_parts(headers, count) }
    }
}
