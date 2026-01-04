//! Free block tracking for the VSR storage layer.
//!
//! The free set tracks which storage blocks are available for allocation. It uses a
//! sharded bitset structure where each shard spans multiple cache lines to optimize
//! for sequential scanning and minimize cache contention during concurrent access.
//!
//! # Shard Structure
//!
//! The free set is divided into fixed-size shards of [`SHARD_BITS`] bits (4096 bits
//! per shard). Each shard occupies [`SHARD_CACHE_LINES`] cache lines (8 x 64 bytes),
//! ensuring that sequential bit searches remain cache-friendly.
//!
//! # Checkpoint Durability and Block Safety
//!
//! Released blocks cannot be immediately reused. Consider this crash scenario:
//!
//! 1. Block B is released and immediately reallocated for new data
//! 2. The system crashes before the checkpoint completes
//! 3. On recovery, the old checkpoint references block B, but it now contains
//!    unrelated new data, causing data corruption
//!
//! To prevent this, released blocks are buffered in
//! [`FreeSet::blocks_released_prior_checkpoint_durability`] until the current
//! checkpoint becomes durable on a quorum of replicas. Only then are they moved
//! to [`FreeSet::blocks_released`] and made available for reuse.
//!
//! # Reservation Workflow
//!
//! Block acquisition follows a three-phase workflow to maintain determinism:
//!
//! 1. **Reserve**: Jobs call `reserve()` in deterministic order to claim upper
//!    bounds on the blocks they need
//! 2. **Acquire**: Jobs run concurrently, acquiring blocks only from their
//!    respective reservations
//! 3. **Forfeit**: Jobs call `forfeit()` when finished, releasing reserved but
//!    unacquired blocks back to the free pool
//!
//! This separation ensures deterministic allocation order despite concurrent
//! execution - critical for state machine replication across replicas.
//!
//! # Persistence
//!
//! The free set's bitsets are EWAH-compressed and stored in grid blocks as
//! "trailers" during checkpoints. See [`block_count_for_trailer_size`] for
//! the block capacity calculation.

// Allow dead code during development as methods are implemented incrementally.
#![allow(dead_code)]

use crate::{
    constants,
    ewah::Ewah,
    stdx::{DynamicBitSet, ReleasedSet},
};

/// Storage block size in bytes.
///
/// Each bit in the free set represents one block of this size. The total tracked
/// capacity is `bit_length * BLOCK_SIZE` bytes.
pub const BLOCK_SIZE: u64 = constants::BLOCK_SIZE;

/// CPU cache line size in bytes.
///
/// Used to size shards for optimal cache utilization during sequential scanning.
/// Each shard spans [`SHARD_CACHE_LINES`] cache lines.
pub const CACHE_LINE_SIZE: u64 = constants::CACHE_LINE_SIZE;

/// The word type used for bitset storage and manipulation.
///
/// Using `u64` allows efficient use of hardware bit-manipulation instructions
/// (`trailing_zeros`, `leading_zeros`, `count_ones`) on 64-bit platforms.
pub type Word = u64;

/// Number of cache lines per shard.
///
/// Set to 8 cache lines (512 bytes per shard) to balance between:
/// - Enough bits per shard (4096) to amortize search overhead
/// - Small enough to fit in L1 cache during sequential scanning
pub const SHARD_CACHE_LINES: usize = 8;

/// Number of bits per shard.
///
/// Derived from [`SHARD_CACHE_LINES`] * [`CACHE_LINE_SIZE`] * 8 bits/byte.
/// This value (4096) is chosen to be a power of 2 for efficient division
/// and to align with common storage allocation granularities.
pub const SHARD_BITS: usize = SHARD_CACHE_LINES * (CACHE_LINE_SIZE as usize) * 8;

const _: () = {
    assert!(SHARD_BITS == 4096);
    assert!(SHARD_BITS.is_multiple_of(64));
};

/// Specifies the type of bit to search for in [`find_bit`].
///
/// This enum controls whether the search looks for allocated blocks (set bits)
/// or free blocks (unset bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitKind {
    /// Search for bits that are set (1).
    ///
    /// Used when scanning for allocated blocks, e.g., during defragmentation
    /// or when verifying block ownership.
    Set,

    /// Search for bits that are unset (0).
    ///
    /// Used when searching for free blocks to allocate.
    Unset,
}

/// State machine for the block reservation workflow.
///
/// See the module-level documentation for the full reservation lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationState {
    /// Blocks are being reserved and acquired for active operations.
    Reserving,

    /// Reserved but unacquired blocks are being returned to the free pool.
    Forfeiting,
}

/// Configuration options for [`FreeSet::init`].
#[derive(Debug, Clone, Copy)]
pub struct InitOptions {
    /// Maximum addressable storage size in bytes.
    ///
    /// Determines the number of blocks tracked by the free set. The actual
    /// bit capacity is rounded up to the nearest shard boundary.
    pub grid_size_limit: usize,

    /// Maximum blocks that may be released before checkpoint durability.
    ///
    /// This bounds the capacity of [`FreeSet::blocks_released_prior_checkpoint_durability`].
    /// Set this to the maximum number of blocks any single checkpoint interval
    /// might release. The actual capacity also includes headroom for trailer
    /// blocks used to store the compressed bitsets themselves.
    pub blocks_released_prior_checkpoint_durability_max: usize,
}

/// Tracks free and allocated blocks in the VSR storage grid.
///
/// See the module-level documentation for checkpoint durability semantics
/// and the reservation workflow.
#[derive(Debug)]
pub struct FreeSet {
    /// Whether the free set has been opened and is ready for use.
    ///
    /// Must be `true` before any acquire/release operations.
    pub opened: bool,

    /// Whether the current checkpoint is durable on a quorum of replicas.
    ///
    /// When `false`, released blocks are buffered in
    /// [`blocks_released_prior_checkpoint_durability`]. When this transitions
    /// to `true`, those blocks are moved to [`blocks_released`] and become
    /// available for reuse.
    pub checkpoint_durable: bool,

    /// Shard index for fast free-block lookup.
    ///
    /// A set bit indicates the corresponding shard has NO free blocks,
    /// allowing the allocator to skip entire shards during sequential scans.
    pub index: DynamicBitSet,

    /// Maximum addressable blocks based on grid size limit (not rounded to shard).
    ///
    /// Blocks at indices >= this value are outside the configured storage limit
    /// and must never be allocated, even though the bitsets have capacity for them.
    pub blocks_count_limit: u64,

    /// Bitset tracking acquired (in-use) blocks. Set bit = block is acquired.
    pub blocks_acquired: DynamicBitSet,

    /// Bitset of blocks released during the previous (now durable) checkpoint.
    ///
    /// These blocks are safe to reallocate. Updated when checkpoint durability
    /// is confirmed.
    pub blocks_released: DynamicBitSet,

    /// Blocks released during the current (not yet durable) checkpoint interval.
    ///
    /// These blocks CANNOT be reused yet. If the system crashes before the
    /// checkpoint becomes durable, recovery will restore from the previous
    /// checkpoint which may still reference these blocks. Once checkpoint
    /// durability is confirmed, these move to [`blocks_released`].
    ///
    /// Stores 0-based block indices (not 1-based addresses).
    pub blocks_released_prior_checkpoint_durability: ReleasedSet,

    /// Number of blocks reserved by the current reservation session.
    reservation_blocks: usize,
    /// Number of active reservations in the current session.
    reservation_count: usize,
    /// Current phase of the reservation workflow.
    reservation_state: ReservationState,
    /// Session counter to detect stale reservation handles.
    reservation_session: usize,
}

impl FreeSet {
    /// Creates a new `FreeSet` configured for the specified storage capacity.
    ///
    /// Allocates all internal bitsets and indices sized to track blocks up to the
    /// grid size limit. The block capacity is rounded up to the nearest shard
    /// boundary ([`SHARD_BITS`]) for efficient index operations.
    ///
    /// The returned `FreeSet` is in an uninitialized state (`opened = false`) and
    /// must be opened before use. See module-level documentation for the full
    /// lifecycle.
    ///
    /// # Parameters
    ///
    /// * `options.grid_size_limit` - Maximum addressable storage in bytes. Determines
    ///   the number of blocks tracked (`grid_size_limit / BLOCK_SIZE`).
    /// * `options.blocks_released_prior_checkpoint_durability_max` - Upper bound on
    ///   blocks that may be released during a single checkpoint interval. Additional
    ///   capacity is added for EWAH-compressed trailer blocks.
    ///
    /// # Panics
    ///
    /// Panics if the capacity calculation for `blocks_released_prior_checkpoint_durability`
    /// overflows.
    pub fn init(options: InitOptions) -> Self {
        let blocks_count = Self::block_count_max(options.grid_size_limit);
        assert!(blocks_count.is_multiple_of(SHARD_BITS));
        assert!(blocks_count.is_multiple_of(64));

        let shard_count = blocks_count / SHARD_BITS;
        let blocks_count_limit = (options.grid_size_limit as u64) / BLOCK_SIZE;
        assert!(blocks_count_limit as usize <= blocks_count);

        // Two bitsets (blocks_acquired and blocks_released) are EWAH-compressed
        // and stored in grid blocks during checkpoints. Reserve capacity for both.
        // TODO: Consider using word_count instead of bit count to reduce over-allocation
        // (Zig uses bits here too, but this may be overly conservative for large grids).
        let extra_trailer_blocks =
            2 * block_count_for_trailer_size(Ewah::<Word>::encode_size_max(blocks_count) as u64);

        let desired_capacity = options
            .blocks_released_prior_checkpoint_durability_max
            .checked_add(extra_trailer_blocks)
            .expect("released set capacity overflow");

        let index = DynamicBitSet::empty(shard_count);
        let blocks_acquired = DynamicBitSet::empty(blocks_count);
        let blocks_released = DynamicBitSet::empty(blocks_count);

        Self {
            opened: false,
            checkpoint_durable: false,

            index,
            blocks_count_limit,

            blocks_acquired,
            blocks_released,

            blocks_released_prior_checkpoint_durability: ReleasedSet::with_capacity(
                desired_capacity,
            ),

            reservation_blocks: 0,
            reservation_count: 0,
            reservation_state: ReservationState::Reserving,
            reservation_session: 0,
        }
    }

    /// Marks a block as released, scheduling it for deallocation.
    ///
    /// Released blocks remain in [`blocks_acquired`](Self::blocks_acquired) until the
    /// current checkpoint becomes durable. This prevents premature reuse that could
    /// cause data corruption on crash recovery. See module-level documentation for
    /// the checkpoint durability workflow.
    ///
    /// # Parameters
    ///
    /// * `address` - The 1-based block address to release. Block addresses are 1-indexed
    ///   so that address 0 can serve as a sentinel/null value.
    ///
    /// # Checkpoint Behavior
    ///
    /// * If `checkpoint_durable` is `true`: The block is added directly to
    ///   [`blocks_released`](Self::blocks_released) for immediate deallocation on the
    ///   next durable checkpoint.
    /// * If `checkpoint_durable` is `false`: The block is buffered in
    ///   [`blocks_released_prior_checkpoint_durability`](Self::blocks_released_prior_checkpoint_durability)
    ///   until the checkpoint becomes durable.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The free set is not opened
    /// * `address` is 0
    /// * The block is not currently acquired
    /// * The block is already marked as released
    pub fn release(&mut self, address: u64) {
        assert!(self.opened);
        assert!(address > 0);

        let block = (address - 1) as usize;
        assert!(self.blocks_acquired.is_set(block));
        assert!(!self.blocks_released.is_set(block));
        assert!(
            !self
                .blocks_released_prior_checkpoint_durability
                .contains(block as u64)
        );

        if self.checkpoint_durable {
            self.blocks_released.set(block);
        } else {
            self.blocks_released_prior_checkpoint_durability
                .insert(block as u64);
        }
    }

    /// Deallocates a block, making it available for future acquisition.
    ///
    /// This is an internal method called by [`mark_checkpoint_durable`](Self::mark_checkpoint_durable)
    /// to complete the deallocation of blocks that were previously released via
    /// [`release`](Self::release). Unlike `release`, which only *schedules* deallocation,
    /// `free` actually clears the block from both [`blocks_acquired`](Self::blocks_acquired)
    /// and [`blocks_released`](Self::blocks_released), and updates the shard index to
    /// indicate that the shard may now have free blocks.
    ///
    /// # Parameters
    ///
    /// * `address` - The 1-based block address to free.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The free set is not opened
    /// * The checkpoint is not durable (blocks can only be freed after durability is confirmed)
    /// * `address` is 0
    /// * The block is not currently acquired
    /// * The block is not marked as released (must call `release` before `free`)
    /// * The block is in the pre-durability buffer (indicates state machine error)
    /// * There are active reservations (freeing during reservation could cause races)
    fn free(&mut self, address: u64) {
        assert!(self.opened);
        assert!(self.checkpoint_durable);
        assert!(address > 0);

        let block = (address - 1) as usize;
        assert!(self.blocks_acquired.is_set(block));
        assert!(self.blocks_released.is_set(block));
        assert!(
            !self
                .blocks_released_prior_checkpoint_durability
                .contains(block as u64)
        );
        assert_eq!(self.reservation_count, 0);
        assert_eq!(self.reservation_blocks, 0);

        let shard = block / SHARD_BITS;
        self.index.unset(shard);

        self.blocks_acquired.unset(block);
        self.blocks_released.unset(block);
    }

    /// Transitions the checkpoint state from durable to not-durable.
    ///
    /// Called when starting a new checkpoint. After this call, any blocks released
    /// via [`release`](Self::release) will be buffered in
    /// [`blocks_released_prior_checkpoint_durability`](Self::blocks_released_prior_checkpoint_durability)
    /// rather than added directly to [`blocks_released`](Self::blocks_released).
    ///
    /// This ensures crash safety: if the system fails before the new checkpoint
    /// becomes durable, recovery will use the previous checkpoint, which may still
    /// reference blocks released during the failed checkpoint interval.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The free set is not opened
    /// * The checkpoint is already not-durable (double transition)
    /// * The pre-durability buffer is not empty (previous checkpoint's releases not yet processed)
    pub fn mark_checkpoint_not_durable(&mut self) {
        assert!(self.opened);
        assert!(self.checkpoint_durable);
        assert!(self.blocks_released_prior_checkpoint_durability.is_empty());

        self.checkpoint_durable = false;
    }

    /// Transitions the checkpoint state from not-durable to durable.
    ///
    /// Called when a checkpoint has been confirmed durable on a quorum of replicas.
    /// This method performs two critical operations:
    ///
    /// 1. **Free previously released blocks**: Iterates through all blocks in
    ///    [`blocks_released`](Self::blocks_released) and calls [`free`](Self::free)
    ///    on each, making them available for reallocation. Uses word-level iteration
    ///    with `trailing_zeros` for efficient scanning.
    ///
    /// 2. **Promote buffered releases**: Moves all blocks from
    ///    [`blocks_released_prior_checkpoint_durability`](Self::blocks_released_prior_checkpoint_durability)
    ///    to [`blocks_released`](Self::blocks_released). These blocks were released
    ///    during the now-durable checkpoint interval and will be freed on the *next*
    ///    durability confirmation.
    ///
    /// # Crash Safety
    ///
    /// This two-phase approach ensures blocks are never reused until their release
    /// is persisted across a quorum. If the system crashes before durability is
    /// confirmed, recovery restores from the previous checkpoint, which may still
    /// reference the "released" blocks.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * The free set is not opened
    /// * The checkpoint is already durable (double transition)
    pub fn mark_checkpoint_durable(&mut self) {
        assert!(self.opened);
        assert!(!self.checkpoint_durable);

        self.checkpoint_durable = true;

        let released_word_len = self.blocks_released.word_len();
        assert!(self.blocks_released.bit_length().is_multiple_of(64));

        for wi in 0..released_word_len {
            let mut word = self.blocks_released.words()[wi];
            while word != 0 {
                let tz = word.trailing_zeros() as usize;
                let block = wi * 64 + tz;
                debug_assert!(block < self.blocks_count());
                self.free((block as u64) + 1);
                word &= word - 1;
            }
        }

        assert_eq!(self.blocks_released.count(), 0);

        while let Some(block) = self.blocks_released_prior_checkpoint_durability.pop() {
            self.blocks_released.set(block as usize);
        }
        assert!(self.blocks_released_prior_checkpoint_durability.is_empty());

        self.verify_index();
    }

    /// Asserts that the shard index is consistent with the block-level bitset.
    ///
    /// For each shard, verifies that `index.is_set(shard)` is `true` if and only if
    /// all blocks in that shard are acquired (no free blocks). This is a debug
    /// assertion to catch index corruption or synchronization bugs.
    ///
    /// # Panics
    ///
    /// Panics if any shard's index bit doesn't match its actual free-block state.
    fn verify_index(&self) {
        for shard in 0..self.shards_count() {
            let shard_full = self.find_free_block_in_shard(shard).is_none();
            assert_eq!(shard_full, self.index.is_set(shard));
        }
    }

    /// Searches for a free (unacquired) block within a specific shard.
    ///
    /// Scans the block range `[shard * SHARD_BITS, (shard + 1) * SHARD_BITS)` in
    /// `blocks_acquired` looking for an unset bit, which indicates a free block.
    ///
    /// # Parameters
    ///
    /// * `shard` - The shard index to search within (0-based).
    ///
    /// # Returns
    ///
    /// Returns `Some(block_index)` if a free block is found, or `None` if all
    /// blocks in the shard are acquired.
    ///
    /// # Panics
    ///
    /// Panics if `shard >= self.shards_count()`.
    fn find_free_block_in_shard(&self, shard: usize) -> Option<usize> {
        let shard_start = shard * SHARD_BITS;
        let shard_end = shard_start + SHARD_BITS;

        assert!(shard < self.shards_count());
        assert!(shard_start < self.blocks_count());
        assert!(shard_end <= self.blocks_count());

        find_bit(
            &self.blocks_acquired,
            shard_start,
            shard_end,
            BitKind::Unset,
        )
    }

    /// Returns the total number of blocks tracked by the free set.
    ///
    /// This is the capacity of the block-level bitsets, rounded up to the nearest
    /// shard boundary. Note that blocks at indices >= `blocks_count_limit` are
    /// outside the configured storage limit and must not be allocated.
    #[inline]
    fn blocks_count(&self) -> usize {
        self.blocks_acquired.bit_length()
    }

    /// Returns the number of shards in the free set index.
    ///
    /// Each shard covers [`SHARD_BITS`] blocks. The shard index enables fast
    /// skipping of fully-allocated regions during block allocation searches.
    #[inline]
    fn shards_count(&self) -> usize {
        self.index.bit_length()
    }

    /// Calculates the block capacity for a given storage size limit.
    ///
    /// Computes the number of blocks needed to cover `grid_size_limit` bytes,
    /// then rounds up to the nearest shard boundary ([`SHARD_BITS`]). This
    /// ensures the internal bitsets align with shard boundaries for efficient
    /// index operations.
    ///
    /// # Parameters
    ///
    /// * `grid_size_limit` - Maximum addressable storage in bytes.
    ///
    /// # Returns
    ///
    /// The number of blocks (bits) to allocate, guaranteed to be a multiple of
    /// [`SHARD_BITS`].
    ///
    /// # Example
    ///
    /// With `BLOCK_SIZE = 512 KiB` and `SHARD_BITS = 4096`:
    /// - 1 GiB storage = 2048 blocks -> rounds to 4096 blocks (1 shard)
    /// - 10 GiB storage = 20480 blocks -> rounds to 20480 blocks (5 shards)
    pub fn block_count_max(grid_size_limit: usize) -> usize {
        let block_count_limit = (grid_size_limit as u64 / BLOCK_SIZE) as usize;
        block_count_limit.div_ceil(SHARD_BITS) * SHARD_BITS
    }
}

/// Searches for the first set or unset bit within a range of a [`DynamicBitSet`].
///
/// This function performs a linear scan through the specified bit range, examining
/// one 64-bit word at a time. It uses hardware `trailing_zeros` to efficiently
/// find the lowest matching bit within each word.
///
/// # Parameters
///
/// * `bit_set` - The bitset to search within.
/// * `bit_min` - The starting bit index (inclusive). The search begins at this position.
/// * `bit_max` - The ending bit index (exclusive). The search stops before this position.
/// * `bit_kind` - Whether to search for set bits ([`BitKind::Set`]) or unset bits
///   ([`BitKind::Unset`]).
///
/// # Returns
///
/// Returns `Some(index)` containing the index of the first matching bit in the range
/// `[bit_min, bit_max)`, or `None` if no matching bit exists in that range.
///
/// # Panics
///
/// Panics if:
/// * `bit_max < bit_min` (invalid range)
/// * `bit_max > bit_set.bit_length()` (range exceeds bitset capacity)
///
/// # Algorithm
///
/// The search proceeds word-by-word (64 bits at a time):
///
/// 1. For [`BitKind::Unset`], each word is inverted so that unset bits become set bits.
/// 2. The first word is masked to ignore bits below `bit_min`.
/// 3. The last word is masked to ignore bits at or above `bit_max`.
/// 4. Each masked word is checked for any set bits using `trailing_zeros`.
/// 5. The first matching bit index is returned, or `None` if the range is exhausted.
///
/// # Performance
///
/// - Time complexity: O(n/64) where n = `bit_max - bit_min`
/// - The function processes 64 bits per iteration using efficient bitwise operations
/// - Best case (match in first word): O(1)
fn find_bit(
    bit_set: &DynamicBitSet,
    bit_min: usize,
    bit_max: usize,
    bit_kind: BitKind,
) -> Option<usize> {
    assert!(bit_max >= bit_min);
    assert!(bit_max <= bit_set.bit_length());

    if bit_min == bit_max {
        return None;
    }

    let word_start = bit_min / 64; // inclusive
    let word_offset = bit_min % 64;
    let word_end = bit_max.div_ceil(64); // exclusive
    if word_start == word_end {
        return None;
    }

    for wi in word_start..word_end {
        let mut w = bit_set.words()[wi];
        if bit_kind == BitKind::Unset {
            w = !w;
        }

        // The first word may contain bits before our search range. Clear them
        // to prevent false matches. The mask `(!0u64) << word_offset` keeps only
        // bits at positions >= word_offset.
        if wi == word_start && word_offset != 0 {
            w &= (!0u64) << word_offset;
        }

        // The last word may contain bits beyond our search range. Clear them
        // to prevent false matches. The mask `(1u64 << end_off) - 1` keeps only
        // bits at positions < end_off.
        if wi == word_end - 1 {
            let end_off = bit_max % 64;
            if end_off != 0 {
                w &= (1u64 << end_off) - 1;
            }
        }

        if w != 0 {
            let tz = w.trailing_zeros() as usize;
            let bit = wi * 64 + tz;
            if bit < bit_max {
                return Some(bit);
            } else {
                return None;
            }
        }
    }

    None
}

/// Calculates the number of grid blocks needed to store trailer data.
///
/// During checkpoints, the free set's bitsets are EWAH-compressed and persisted
/// as "trailer" data in grid blocks. Each grid block has a fixed header
/// ([`constants::HEADER_SIZE`] bytes), leaving `BLOCK_SIZE - HEADER_SIZE` bytes
/// of usable payload per block.
///
/// This function computes the minimum number of blocks required to store
/// `trailer_size` bytes of compressed bitset data.
///
/// # Example
///
/// With `BLOCK_SIZE = 512 KiB` and `HEADER_SIZE = 256 bytes`:
/// - Usable space per block: 524,032 bytes
/// - A 1 MiB trailer requires: ceil(1,048,576 / 524,032) = 2 blocks
#[inline]
fn block_count_for_trailer_size(trailer_size: u64) -> usize {
    let chunk_size_max = BLOCK_SIZE - constants::HEADER_SIZE as u64;
    let q = trailer_size / chunk_size_max;
    let r = trailer_size % chunk_size_max;
    (q + u64::from(r != 0)) as usize
}

#[cfg(test)]
mod tests {
    use super::{BitKind, find_bit};
    use crate::stdx::DynamicBitSet;
    use proptest::prelude::*;

    fn expected_find_bit(
        bit_set: &DynamicBitSet,
        bit_min: usize,
        bit_max: usize,
        bit_kind: BitKind,
    ) -> Option<usize> {
        (bit_min..bit_max).find(|&idx| bit_set.is_set(idx) == (bit_kind == BitKind::Set))
    }

    #[test]
    fn find_bit_handles_empty_range() {
        let mut bits = DynamicBitSet::empty(8);
        bits.set(3);
        assert_eq!(find_bit(&bits, 4, 4, BitKind::Set), None);
        assert_eq!(find_bit(&bits, 4, 4, BitKind::Unset), None);
    }

    #[test]
    fn find_bit_respects_word_boundaries() {
        let mut bits = DynamicBitSet::empty(130);
        bits.set(1);
        bits.set(63);
        bits.set(64);
        bits.set(129);

        assert_eq!(find_bit(&bits, 0, 64, BitKind::Set), Some(1));
        assert_eq!(find_bit(&bits, 2, 64, BitKind::Set), Some(63));
        assert_eq!(find_bit(&bits, 64, 128, BitKind::Set), Some(64));
        assert_eq!(find_bit(&bits, 128, 130, BitKind::Set), Some(129));

        assert_eq!(find_bit(&bits, 0, 2, BitKind::Unset), Some(0));
        assert_eq!(find_bit(&bits, 63, 65, BitKind::Unset), None);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]
        #[test]
        fn prop_find_bit_matches_linear_scan(
            (bit_length, set_bits, range, bit_kind) in (1usize..=256).prop_flat_map(|len| {
                let set_bits = prop::collection::hash_set(0..len, 0..=len);
                let range = (0..=len, 0..=len).prop_map(|(a, b)| {
                    if a <= b { (a, b) } else { (b, a) }
                });
                let bit_kind = prop_oneof![Just(BitKind::Set), Just(BitKind::Unset)];
                (Just(len), set_bits, range, bit_kind)
            })
        ) {
            let mut bits = DynamicBitSet::empty(bit_length);
            for idx in &set_bits {
                bits.set(*idx);
            }

            let (bit_min, bit_max) = range;
            let expected = expected_find_bit(&bits, bit_min, bit_max, bit_kind);
            let actual = find_bit(&bits, bit_min, bit_max, bit_kind);

            prop_assert_eq!(actual, expected);

            if let Some(idx) = actual {
                prop_assert!(idx >= bit_min);
                prop_assert!(idx < bit_max);
                prop_assert_eq!(bits.is_set(idx), bit_kind == BitKind::Set);
            }
        }
    }
}
