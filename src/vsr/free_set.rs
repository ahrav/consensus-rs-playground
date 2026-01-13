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
//! checkpoint becomes durable on a quorum of replicas. At durability, those
//! blocks move to [`FreeSet::blocks_released`]. They are actually freed (and
//! become reusable) when the *next* checkpoint is confirmed durable.
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

use std::cmp::max;

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

/// Identifies which free-set bitset is being encoded or decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitsetKind {
    /// Bitset tracking acquired (in-use) blocks.
    BlocksAcquired,
    /// Bitset tracking released blocks awaiting final deallocation.
    BlocksReleased,
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

/// A reservation over a contiguous block-index range in the free set.
///
/// The fields are block indices (0-based) for ease of calculation. A reservation
/// may span both free and acquired blocks; at creation time it is guaranteed to
/// include exactly the number of free blocks requested by `reserve()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reservation {
    /// Start of the reserved range in block indices (0-based, inclusive).
    pub block_base: usize,
    /// Length of the reserved range in blocks.
    pub block_count: usize,
    /// Reservation session token used to detect stale handles.
    pub session: usize,
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

/// Sizes returned by [`FreeSet::encode_chunks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodedSizes {
    /// Bytes of EWAH-encoded `blocks_acquired` data to persist.
    pub encoded_size_blocks_acquired: u64,
    /// Bytes of EWAH-encoded `blocks_released` data to persist.
    pub encoded_size_blocks_released: u64,
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
    /// to `true`, those blocks move to [`blocks_released`] and will be freed
    /// on the next durability transition.
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

    /// Bitset of blocks released during the previous checkpoint interval.
    ///
    /// These blocks are freed (and become reusable) when the next checkpoint
    /// becomes durable.
    pub blocks_released: DynamicBitSet,

    /// Blocks released during the current (not yet durable) checkpoint interval.
    ///
    /// These blocks CANNOT be reused yet. If the system crashes before the
    /// checkpoint becomes durable, recovery will restore from the previous
    /// checkpoint which may still reference these blocks. Once checkpoint
    /// durability is confirmed, these move to [`blocks_released`] and remain
    /// acquired until the next durability transition.
    ///
    /// Stores 0-based block indices (not 1-based addresses).
    pub blocks_released_prior_checkpoint_durability: ReleasedSet,

    /// Number of blocks reserved by the current reservation session.
    /// High-water mark (cursor position)
    reservation_blocks: usize,
    /// Number of active reservations in the current session.
    reservation_count: usize,
    /// Current phase of the reservation workflow. (Reserving or Forfeiting)
    reservation_state: ReservationState,
    /// Session counter to detect stale reservation handles.
    /// Used to invalidate old reservations.
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

    /// Opens a free set from persisted state.
    ///
    /// Inputs:
    /// - EWAH-encoded bitset chunks for `blocks_acquired` and `blocks_released`.
    /// - The block addresses used to store those bitsets in the grid.
    ///
    /// The free-set block addresses are not included in the encoded acquired bitset
    /// (see `CheckpointTrailer`), so they are patched in via [`mark_released`](Self::mark_released).
    pub fn open(
        &mut self,
        encoded_blocks_acquired: &[&[u8]],
        encoded_blocks_released: &[&[u8]],
        free_set_block_addresses_blocks_acquired: &[u64],
        free_set_block_addresses_blocks_released: &[u64],
    ) {
        assert!(!self.opened);
        assert!(!self.checkpoint_durable);

        let encoded_empty =
            encoded_blocks_acquired.is_empty() && encoded_blocks_released.is_empty();
        let addrs_empty = free_set_block_addresses_blocks_acquired.is_empty()
            && free_set_block_addresses_blocks_released.is_empty();
        assert_eq!(encoded_empty, addrs_empty);

        self.decode_chunks(encoded_blocks_acquired, encoded_blocks_released);

        self.mark_released(free_set_block_addresses_blocks_acquired);
        self.mark_released(free_set_block_addresses_blocks_released);

        self.opened = true;
    }

    /// Decodes EWAH-compressed bitset chunks into the free set.
    ///
    /// The free set must be unopened and empty. After decoding, the shard index
    /// is rebuilt by scanning for fully-allocated shards. Panics if the encoding
    /// is invalid or chunk alignment is incorrect.
    pub fn decode_chunks(
        &mut self,
        encoded_blocks_acquired: &[&[u8]],
        encoded_blocks_released: &[&[u8]],
    ) {
        assert!(!self.opened);
        assert!(!self.checkpoint_durable);

        assert_eq!(self.index.count(), 0);
        assert_eq!(self.blocks_acquired.count(), 0);
        assert_eq!(self.blocks_released.count(), 0);
        assert!(self.blocks_released_prior_checkpoint_durability.is_empty());

        assert_eq!(self.reservation_count, 0);
        assert_eq!(self.reservation_blocks, 0);

        self.decode(BitsetKind::BlocksAcquired, encoded_blocks_acquired);
        self.decode(BitsetKind::BlocksReleased, encoded_blocks_released);

        for shard in 0..self.shards_count() {
            if self.find_free_block_in_shard(shard).is_none() {
                self.index.set(shard);
            }
        }

        self.verify_index();
    }

    /// Encodes both bitsets into the provided trailer chunks.
    ///
    /// Chunks must be `Word`-aligned and large enough for the EWAH output.
    /// Returns the exact number of bytes to persist for each bitset (trailing zero runs omitted).
    pub fn encode_chunks(
        &self,
        target_chunks_blocks_acquired: &mut [&mut [u8]],
        target_chunks_blocks_released: &mut [&mut [u8]],
    ) -> EncodedSizes {
        assert!(self.opened);
        assert!(self.checkpoint_durable);
        assert_eq!(self.reservation_count, 0);
        assert_eq!(self.reservation_blocks, 0);

        EncodedSizes {
            encoded_size_blocks_acquired: self
                .encode(BitsetKind::BlocksAcquired, target_chunks_blocks_acquired)
                as u64,
            encoded_size_blocks_released: self
                .encode(BitsetKind::BlocksReleased, target_chunks_blocks_released)
                as u64,
        }
    }

    /// Reserve `reserve_count` free blocks. The blocks are not acquired yet.
    ///
    /// Invariants:
    ///
    /// - If a reservation is returned, it covers exactly `reserve_count` free blocks, along with
    ///   any interleaved already-acquired blocks.
    /// - Active reservations are exclusive (i.e. disjoint).
    ///   (A reservation is active until `forfeit()` is called.)
    ///
    /// Returns `None` if there are not enough blocks free and vacant.
    /// Returns a reservation which can be used with `acquire()`:
    /// - The caller should consider the returned `Reservation` as opaque and immutable.
    /// - Each `reserve()` call which returns `Some` must correspond to exactly one `forfeit()`
    ///   call.
    pub fn reserve(&mut self, reserve_count: usize) -> Option<Reservation> {
        assert!(self.opened);
        assert!(reserve_count > 0);
        assert_eq!(self.reservation_state, ReservationState::Reserving);

        let shard_min = self.reservation_blocks / SHARD_BITS;
        let shard_max = self.shards_count();

        let shard_start = find_bit(&self.index, shard_min, shard_max, BitKind::Unset)?;

        let mut block = max(shard_start * SHARD_BITS, self.reservation_blocks);

        for _ in 0..reserve_count {
            let free_bit = find_bit(
                &self.blocks_acquired,
                block,
                self.blocks_count(),
                BitKind::Unset,
            )?;
            block = free_bit + 1;

            if (block as u64) > self.blocks_count_limit {
                return None;
            }
        }

        let block_base = self.reservation_blocks;
        let block_count = block - block_base;

        self.reservation_blocks += block_count;
        self.reservation_count += 1;

        Some(Reservation {
            block_base,
            block_count,
            session: self.reservation_session,
        })
    }

    /// After invoking `forfeit()`, the reservation must never be used again.
    pub fn forfeit(&mut self, reservation: Reservation) {
        assert!(self.opened);
        assert_eq!(reservation.session, self.reservation_session);
        assert!(reservation.block_base + reservation.block_count <= self.reservation_blocks);
        assert!(self.reservation_count > 0);

        self.reservation_count -= 1;
        if self.reservation_count == 0 {
            self.reservation_blocks = 0;
            self.reservation_session = self.reservation_session.wrapping_add(1);
            self.reservation_state = ReservationState::Reserving;
        } else {
            self.reservation_state = ReservationState::Forfeiting;
        }
    }

    /// Marks a free block from the reservation as acquired, and returns the address.
    /// The reservation must not have been forfeited yet.
    /// The reservation must belong to the current cycle of reservations.
    ///
    /// Invariants:
    ///
    /// - An acquired block cannot be acquired again until it has been released and the release
    ///   has been checkpointed.
    ///
    /// Returns `None` if no free block is available in the reservation.
    pub fn acquire(&mut self, reservation: Reservation) -> Option<u64> {
        assert!(self.opened);
        assert!(reservation.block_count > 0);

        assert!(self.reservation_count > 0);
        assert!(reservation.block_base < self.reservation_blocks);
        assert!(reservation.block_base + reservation.block_count <= self.reservation_blocks);
        assert_eq!(reservation.session, self.reservation_session);

        let shard_min = reservation.block_base / SHARD_BITS;
        let shard_max = (reservation.block_base + reservation.block_count).div_ceil(SHARD_BITS);
        let shard_start = find_bit(&self.index, shard_min, shard_max, BitKind::Unset)?;
        assert!(!self.index.is_set(shard_start));

        let reservation_start = max(shard_start * SHARD_BITS, reservation.block_base);
        let reservation_end = reservation.block_base + reservation.block_count;

        let block = find_bit(
            &self.blocks_acquired,
            reservation_start,
            reservation_end,
            BitKind::Unset,
        )?;
        assert!(block >= reservation.block_base);
        assert!(block <= reservation.block_base + reservation.block_count);

        assert!(!self.blocks_acquired.is_set(block));
        assert!(!self.blocks_released.is_set(block));
        assert!(
            !self
                .blocks_released_prior_checkpoint_durability
                .contains(block as u64)
        );

        let shard = block / SHARD_BITS;
        assert!(shard >= shard_start);

        self.blocks_acquired.set(block);

        if self.find_free_block_in_shard(shard).is_none() {
            self.index.set(shard);
        }

        Some((block as u64) + 1)
    }

    /// Returns `true` if the address is free (not acquired).
    ///
    /// If the free set is not open yet, returns `false` conservatively.
    pub fn is_free(&self, address: u64) -> bool {
        if !self.opened {
            return false;
        }

        assert!(address > 0);
        !self.blocks_acquired.is_set((address - 1) as usize)
    }

    /// Returns `true` if the address is marked released in either buffer.
    pub fn is_released(&self, address: u64) -> bool {
        if !self.opened {
            return false;
        }
        assert!(address > 0);

        let block = (address - 1) as usize;
        self.blocks_released_prior_checkpoint_durability
            .contains(block as u64)
            || self.blocks_released.is_set(block)
    }

    /// Returns `true` if the block would be freed when the current checkpoint
    /// becomes durable.
    ///
    /// Only valid while `checkpoint_durable` is `false`. During this time,
    /// `blocks_released` holds releases from the previous interval.
    pub fn to_be_freed_at_checkpoint_durability(&self, address: u64) -> bool {
        assert!(self.opened);
        assert!(!self.checkpoint_durable);
        assert!(address > 0);

        let block = (address - 1) as usize;
        assert!(self.blocks_acquired.is_set(block));
        assert!(
            !self.blocks_released.is_set(block)
                || !self
                    .blocks_released_prior_checkpoint_durability
                    .contains(block as u64)
        );

        self.blocks_released.is_set(block)
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

    /// Marks the blocks that store the free set itself as acquired and released.
    ///
    /// The on-disk free-set bitsets do not include the blocks that store them, so
    /// they must be patched in after decoding. Because the next checkpoint will
    /// write a different set of free-set blocks, these can be released in the
    /// current interval. Addresses must be strictly increasing and nonzero.
    fn mark_released(&mut self, addresses: &[u64]) {
        assert!(!self.opened);
        assert!(!self.checkpoint_durable);

        let mut prev: u64 = 0;
        for &address in addresses {
            assert!(address > 0);
            assert!(address > prev);
            prev = address;

            let block = (address - 1) as usize;

            assert!(!self.blocks_acquired.is_set(block));
            assert!(!self.blocks_released.is_set(block));
            assert!(
                !self
                    .blocks_released_prior_checkpoint_durability
                    .contains(block as u64)
            );

            self.blocks_acquired.set(block);
            let shard = block / SHARD_BITS;
            if self.find_free_block_in_shard(shard).is_none() {
                self.index.set(shard);
            }

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

    /// Returns the highest acquired address, or `None` if no blocks are acquired.
    pub fn highest_address_acquired(&self) -> Option<u64> {
        assert!(self.opened);
        self.blocks_acquired
            .highest_set_bit()
            .map(|bit| bit as u64 + 1)
    }

    /// Returns the highest released address in `blocks_released`, or `None` if none.
    pub fn highest_address_released(&self) -> Option<u64> {
        assert!(self.opened);
        self.blocks_released
            .highest_set_bit()
            .map(|bit| bit as u64 + 1)
    }

    /// Returns the number of active reservations in the current session.
    pub fn count_reservations(&self) -> usize {
        assert!(self.opened);
        self.reservation_count
    }

    /// Returns the number of acquired blocks.
    pub fn count_acquired(&self) -> usize {
        assert!(self.opened);
        self.blocks_acquired.count()
    }

    /// Returns the number of free blocks (total minus acquired).
    pub fn count_free(&self) -> usize {
        assert!(self.opened);
        self.blocks_count() - self.blocks_acquired.count()
    }

    /// Returns the number of blocks released across both release buffers.
    pub fn count_released(&self) -> usize {
        assert!(self.opened);
        self.blocks_released.count() + self.blocks_released_prior_checkpoint_durability.count()
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

    /// Decodes compressed bitset chunks into the selected bitset.
    ///
    /// Panics if the encoding is invalid. The decoder does not emit trailing
    /// zero runs, so any words beyond the decoded range must remain zero.
    fn decode(&mut self, bitset_kind: BitsetKind, source_chunks: &[&[u8]]) {
        assert!(!self.opened);
        assert!(!self.checkpoint_durable);
        assert_words_aligned_chunks(source_chunks);

        let word_count = self.blocks_word_count();
        let blocks_count = self.blocks_count();

        let target_words: &mut [Word] = match bitset_kind {
            BitsetKind::BlocksAcquired => unsafe { self.blocks_acquired.words_mut() },
            BitsetKind::BlocksReleased => unsafe { self.blocks_released.words_mut() },
        };
        assert_eq!(target_words.len(), word_count);

        let source_size = source_chunks.iter().map(|c| c.len()).sum();

        let mut decoder = Ewah::<Word>::decode_chunks(target_words, source_size);
        let mut words_decoded = 0;
        for &chunk in source_chunks {
            words_decoded += decoder.decode_chunk(chunk);
        }
        assert!(decoder.done());

        assert!(words_decoded * 64 <= blocks_count);

        for &w in &target_words[words_decoded..] {
            assert_eq!(w, 0);
        }
    }

    /// EWAH-encodes the selected bitset into `target_chunks`.
    ///
    /// Streams output across the chunk slices until completion. The returned size excludes
    /// trailing zero runs so the encoding is stable regardless of the configured storage limit.
    fn encode(&self, bitset_kind: BitsetKind, target_chunks: &mut [&mut [u8]]) -> usize {
        assert!(self.opened);
        assert!(self.checkpoint_durable);
        assert_words_aligned_chunks_mut(target_chunks);

        let word_count = self.blocks_word_count();

        let source_words: &[Word] = match bitset_kind {
            BitsetKind::BlocksAcquired => self.blocks_acquired.words(),
            BitsetKind::BlocksReleased => self.blocks_released.words(),
        };
        assert_eq!(source_words.len(), word_count);

        let mut encoder = Ewah::<Word>::encode_chunk(source_words);

        let mut bytes_encode_total: usize = 0;
        let mut done = false;

        for chunk in target_chunks.iter_mut() {
            // Stream the EWAH output across chunk buffers until the encoder finishes.
            let bytes_encoded = encoder.encode_chunk(chunk);
            // The encoder must always make forward progress while data remains.
            assert!(bytes_encoded > 0);
            bytes_encode_total += bytes_encoded;

            if encoder.done() {
                done = true;
                break;
            }
        }

        // Caller must provide enough chunk space to hold the full encoding.
        assert!(done);

        // Drop trailing zero runs so output is independent of the grid size limit.
        let bytes_trailing_zero_runs =
            encoder.trailing_zero_runs_count * core::mem::size_of::<Word>();
        bytes_encode_total - bytes_trailing_zero_runs
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

    /// Returns the number of 64-bit words backing the block bitsets.
    #[inline]
    fn blocks_word_count(&self) -> usize {
        assert!(self.blocks_count().is_multiple_of(64));
        self.blocks_count() / 64
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

/// Ensures each chunk is `Word`-aligned and sized to whole words for EWAH I/O.
fn assert_words_aligned_chunks(chunks: &[&[u8]]) {
    for chunk in chunks {
        // SAFETY: We only inspect the alignment split, no elements are reinterpreted.
        let (prefix, _, suffix) = unsafe { chunk.align_to::<Word>() };
        assert!(prefix.is_empty(), "source chunk not word-aligned");
        assert!(suffix.is_empty(), "source chunk length not word-aligned");
    }
}

/// Ensures each mutable chunk is `Word`-aligned and sized to whole words for EWAH I/O.
fn assert_words_aligned_chunks_mut(chunks: &mut [&mut [u8]]) {
    for chunk in chunks.iter_mut() {
        // SAFETY: We only inspect the alignment split, no elements are reinterpreted.
        let (prefix, _, suffix) = unsafe { (*chunk).align_to_mut::<Word>() };
        assert!(prefix.is_empty(), "target chunk not word-aligned");
        assert!(suffix.is_empty(), "target chunk length not word-aligned");
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
    use super::{BitKind, BitsetKind, FreeSet, InitOptions, SHARD_BITS, Word, find_bit};
    use crate::ewah::Ewah;
    use crate::stdx::DynamicBitSet;
    use proptest::prelude::*;
    use std::mem::size_of;

    /// Reference implementation of [`find_bit`] using linear scan.
    /// Used to verify the optimized word-based implementation.
    fn expected_find_bit(
        bit_set: &DynamicBitSet,
        bit_min: usize,
        bit_max: usize,
        bit_kind: BitKind,
    ) -> Option<usize> {
        (bit_min..bit_max).find(|&idx| bit_set.is_set(idx) == (bit_kind == BitKind::Set))
    }

    /// Empty ranges should return `None` regardless of bitset contents.
    #[test]
    fn find_bit_handles_empty_range() {
        let mut bits = DynamicBitSet::empty(8);
        bits.set(3);
        assert_eq!(find_bit(&bits, 4, 4, BitKind::Set), None);
        assert_eq!(find_bit(&bits, 4, 4, BitKind::Unset), None);
    }

    /// Tests masking logic at word boundaries (bits 63/64) and partial final words.
    /// The implementation must correctly mask bits outside the search range when
    /// `bit_min` or `bit_max` don't align to word boundaries.
    #[test]
    fn find_bit_respects_word_boundaries() {
        // 130 bits spans 3 words: [0-63], [64-127], [128-129]
        let mut bits = DynamicBitSet::empty(130);
        bits.set(1); // First word
        bits.set(63); // Last bit of first word
        bits.set(64); // First bit of second word
        bits.set(129); // Partial third word

        assert_eq!(find_bit(&bits, 0, 64, BitKind::Set), Some(1));
        assert_eq!(find_bit(&bits, 2, 64, BitKind::Set), Some(63));
        assert_eq!(find_bit(&bits, 64, 128, BitKind::Set), Some(64));
        assert_eq!(find_bit(&bits, 128, 130, BitKind::Set), Some(129));

        assert_eq!(find_bit(&bits, 0, 2, BitKind::Unset), Some(0));
        // Range [63, 65) contains bits 63 and 64, both set - no unset bits
        assert_eq!(find_bit(&bits, 63, 65, BitKind::Unset), None);
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: crate::test_utils::proptest_cases(16),
            ..ProptestConfig::default()
        })]

        /// Verifies [`find_bit`] matches the reference linear scan for arbitrary
        /// bitset patterns and search ranges. Catches off-by-one errors and
        /// incorrect word masking that unit tests might miss.
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

    // -------------------------------------------------------------------------
    // Test Helpers
    // -------------------------------------------------------------------------

    /// Creates an uninitialized [`FreeSet`] with the given block capacity.
    /// The set is not opened and cannot be used for acquire/release operations.
    fn init_free_set(blocks_count: usize) -> FreeSet {
        assert!(blocks_count.is_multiple_of(SHARD_BITS));
        let grid_size_limit = (blocks_count as u64) * super::BLOCK_SIZE;
        assert!(grid_size_limit <= usize::MAX as u64);

        FreeSet::init(InitOptions {
            grid_size_limit: grid_size_limit as usize,
            blocks_released_prior_checkpoint_durability_max: 0,
        })
    }

    /// Creates an opened [`FreeSet`] with all blocks free and checkpoint marked durable.
    /// Ready for immediate reserve/acquire/release operations.
    fn open_empty_free_set(blocks_count: usize) -> FreeSet {
        let mut set = init_free_set(blocks_count);
        set.open(&[], &[], &[], &[]);
        set.checkpoint_durable = true;
        set
    }

    /// Returns the maximum EWAH-encoded size in bytes for a bitset with `blocks_count` bits.
    fn encode_size_max_bytes(blocks_count: usize) -> usize {
        let word_count = blocks_count / 64;
        Ewah::<Word>::encode_size_max(word_count)
    }

    /// Reinterprets a word slice as bytes for EWAH encoding/decoding.
    /// Safe because `Word` has no padding and any bit pattern is valid.
    fn words_as_bytes(words: &[Word]) -> &[u8] {
        let byte_len = std::mem::size_of_val(words);
        unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, byte_len) }
    }

    /// Mutable variant of [`words_as_bytes`].
    fn words_as_bytes_mut(words: &mut [Word]) -> &mut [u8] {
        let byte_len = std::mem::size_of_val(words);
        unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, byte_len) }
    }

    fn assert_bitset_eq(a: &DynamicBitSet, b: &DynamicBitSet) {
        assert_eq!(a.bit_length(), b.bit_length());
        assert_eq!(a.word_len(), b.word_len());
        assert_eq!(a.words(), b.words());
    }

    /// Compares two free sets for logical equality of their persistent state.
    fn assert_free_set_eq(a: &FreeSet, b: &FreeSet) {
        assert_bitset_eq(&a.blocks_acquired, &b.blocks_acquired);
        assert_bitset_eq(&a.blocks_released, &b.blocks_released);
        assert_bitset_eq(&a.index, &b.index);
        assert_eq!(
            a.blocks_released_prior_checkpoint_durability.count(),
            b.blocks_released_prior_checkpoint_durability.count()
        );
    }

    // -------------------------------------------------------------------------
    // FreeSet Tests
    // -------------------------------------------------------------------------

    /// Verifies `highest_address_acquired` correctly tracks the maximum as blocks
    /// are acquired and freed in various orders. Critical for determining storage
    /// bounds during checkpointing.
    #[test]
    fn highest_address_acquired_tracks_maximum_after_acquire_and_free() {
        let blocks_count = SHARD_BITS;
        let mut set = open_empty_free_set(blocks_count);

        {
            let reservation = set.reserve(6).unwrap();
            assert_eq!(None, set.highest_address_acquired());
            assert_eq!(Some(1), set.acquire(reservation));
            assert_eq!(Some(2), set.acquire(reservation));
            assert_eq!(Some(3), set.acquire(reservation));
            set.forfeit(reservation);
        }

        assert_eq!(Some(3), set.highest_address_acquired());

        set.release(2);
        set.free(2);
        assert_eq!(Some(3), set.highest_address_acquired());

        set.release(3);
        set.free(3);
        assert_eq!(Some(1), set.highest_address_acquired());

        set.release(1);
        set.free(1);
        assert_eq!(None, set.highest_address_acquired());

        {
            let reservation = set.reserve(6).unwrap();
            assert_eq!(Some(1), set.acquire(reservation));
            assert_eq!(Some(2), set.acquire(reservation));
            assert_eq!(Some(3), set.acquire(reservation));
            set.forfeit(reservation);
        }

        {
            set.release(3);
            assert_eq!(Some(3), set.highest_address_acquired());

            set.free(3);
            assert_eq!(Some(2), set.highest_address_acquired());
        }
    }

    /// Verifies that acquiring all blocks then releasing them returns the free set
    /// to its initial empty state. Tests at shard boundary sizes (1, 2, 63, 64, 65)
    /// to catch off-by-one errors in shard index maintenance.
    #[test]
    fn acquire_all_blocks_then_release_restores_initial_state() {
        // Test at shard boundaries: 63/64/65 shards exercise word-boundary edge cases
        // in the shard index (which is itself a bitset).
        for blocks_count in [
            SHARD_BITS,
            2 * SHARD_BITS,
            63 * SHARD_BITS,
            64 * SHARD_BITS,
            65 * SHARD_BITS,
        ] {
            let mut set = open_empty_free_set(blocks_count);
            let empty = open_empty_free_set(blocks_count);

            {
                let reservation = set.reserve(blocks_count).unwrap();
                for i in 0..blocks_count {
                    assert_eq!(Some(i as u64 + 1), set.acquire(reservation));
                }
                assert_eq!(None, set.acquire(reservation));
                set.forfeit(reservation);
            }

            assert_eq!(blocks_count, set.count_acquired());
            assert_eq!(0, set.count_free());

            for i in 0..blocks_count {
                let address = i as u64 + 1;
                set.release(address);
                set.free(address);
            }

            assert_free_set_eq(&empty, &set);

            assert_eq!(0, set.count_acquired());
            assert_eq!(blocks_count, set.count_free());

            {
                let reservation = set.reserve(blocks_count).unwrap();
                for i in 0..blocks_count {
                    assert_eq!(Some(i as u64 + 1), set.acquire(reservation));
                }
                assert_eq!(None, set.acquire(reservation));
                set.forfeit(reservation);
            }
        }
    }

    /// Verifies that `acquire` only allocates blocks within its own reservation's
    /// range, even when earlier reservations contain free blocks.
    ///
    /// Reservations form contiguous, non-overlapping ranges. When reservation A
    /// claims range [0, 1) and reservation B claims [1, 8193), acquiring from B
    /// must not "steal" block 0 from A's range - determinism requires each
    /// reservation to allocate only within its bounds.
    #[test]
    fn acquire_respects_existing_reservations_across_shards() {
        let mut set = open_empty_free_set(SHARD_BITS * 3);

        // Reservation A: 1 block -> range [0, 1)
        // Reservation B: 8192 blocks -> range [1, 8193), spans shards 0-2
        let reservation_a = set.reserve(1).unwrap();
        let reservation_b = set.reserve(2 * SHARD_BITS).unwrap();

        // Each acquire from B should return blocks starting at index 1 (address 2),
        // skipping block 0 which belongs to A's range.
        // Formula: address - 1 = block_index = reservation_a.block_count + i = 1 + i
        for i in 0..reservation_b.block_count {
            let address = set.acquire(reservation_b).unwrap();
            assert_eq!(address - 1, reservation_a.block_count as u64 + i as u64);
            set.verify_index();
        }
        assert_eq!(None, set.acquire(reservation_b));

        set.forfeit(reservation_b);
        set.forfeit(reservation_a);
    }

    // -------------------------------------------------------------------------
    // EWAH Encode/Decode Tests
    // -------------------------------------------------------------------------

    /// Tests EWAH roundtrip with a hand-crafted pattern that exercises:
    /// - Zero runs (compressed as run-length)
    /// - Literal words (stored verbatim)
    /// - One runs (compressed as run-length)
    ///
    /// The encoded format uses marker words where low bit indicates literal (1)
    /// or run (0), and remaining bits encode counts.
    #[test]
    fn encode_decode_roundtrip_preserves_bitset_patterns() {
        // EWAH marker format: [zero_run_count:31 | literal_count:31 | is_ones_run:1]
        // Word 0: 2 zero-words, then 3 literal words follow
        // Words 1-3: literal data (alternating bit patterns)
        // Word 4: 1 one-run marker, followed by (64-5) = 59 all-ones words
        let encoded_expect_words: [Word; 5] = [
            (2u64 << 1) | (3u64 << 32), // marker: 2 zeros, 3 literals
            0b10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010u64,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101u64,
            0b10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010u64,
            1u64 | ((64u64 - 5) << 1), // marker: ones-run of 59 words
        ];

        // Decoded: 2 zero words + 3 literal words + 59 all-ones words = 64 words
        let mut decoded_expect = Vec::with_capacity(64);
        decoded_expect.extend_from_slice(&[
            0u64,
            0u64,
            0b10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010u64,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101u64,
            0b10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010u64,
        ]);
        decoded_expect.extend(std::iter::repeat_n(u64::MAX, 64 - 5));

        let blocks_count = decoded_expect.len() * 64;
        let mut decoded_actual = init_free_set(blocks_count);

        let encoded_expect_bytes = words_as_bytes(&encoded_expect_words);
        decoded_actual.decode(BitsetKind::BlocksAcquired, &[encoded_expect_bytes]);

        assert_eq!(
            decoded_expect.len(),
            decoded_actual.blocks_acquired.word_len()
        );
        assert_eq!(&decoded_expect[..], decoded_actual.blocks_acquired.words());

        decoded_actual.opened = true;
        decoded_actual.checkpoint_durable = true;

        let encoded_size = encode_size_max_bytes(blocks_count);
        assert_eq!(encoded_size % size_of::<Word>(), 0);
        let mut encoded_actual_words = vec![0u64; encoded_size / size_of::<Word>()];
        let encoded_actual_len = {
            let mut encoded_chunk = words_as_bytes_mut(&mut encoded_actual_words);
            decoded_actual.encode(BitsetKind::BlocksAcquired, &mut [&mut encoded_chunk])
        };

        assert_eq!(encoded_actual_len, encoded_expect_bytes.len());
        assert_eq!(
            &words_as_bytes(&encoded_actual_words)[..encoded_actual_len],
            encoded_expect_bytes
        );
    }

    // -------------------------------------------------------------------------
    // Pattern-Based EWAH Test Infrastructure
    // -------------------------------------------------------------------------

    /// Fill type for generating test bitset patterns. EWAH compresses uniform
    /// runs efficiently, so we test all three cases to exercise different
    /// encoding paths.
    #[derive(Clone, Copy, Debug)]
    enum TestPatternFill {
        /// All bits set (0xFFFF...) - compresses to a "ones run" marker.
        UniformOnes,
        /// All bits clear (0x0000...) - compresses to a "zeros run" marker.
        UniformZeros,
        /// Mixed bits - stored as literal words (no compression).
        Literal,
    }

    /// Describes a contiguous region of words with uniform fill type.
    #[derive(Clone, Copy, Debug)]
    struct TestPattern {
        fill: TestPatternFill,
        words: usize,
    }

    /// Deterministic xorshift64 PRNG for reproducible test data.
    ///
    /// Using a custom PRNG instead of `rand` ensures:
    /// - Identical sequences across Rust versions and platforms
    /// - No external crate dependency for tests
    /// - Failures can be reproduced exactly from the seed
    #[derive(Clone)]
    struct TestRng {
        state: u64,
    }

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        /// Xorshift64 with constants (13, 7, 17) for full 2^64-1 period.
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        fn range_inclusive(&mut self, min: u64, max: u64) -> u64 {
            assert!(min <= max);
            let span = max - min + 1;
            min + (self.next_u64() % span)
        }

        fn index(&mut self, len: usize) -> usize {
            assert!(len > 0);
            (self.next_u64() % len as u64) as usize
        }
    }

    /// Builds a bitset from pattern descriptors, encodes it, decodes it, and
    /// verifies the roundtrip preserves all data. The seed controls literal
    /// word generation for reproducibility.
    fn test_encode_patterns(patterns: &[TestPattern], seed: u64) {
        let blocks_count: usize = patterns.iter().map(|p| p.words * 64).sum();
        let mut decoded_expect = init_free_set(blocks_count);

        // Start with all shards marked full (index bit set), then clear bits
        // for shards containing any non-full words.
        for shard in 0..decoded_expect.index.bit_length() {
            decoded_expect.index.set(shard);
        }

        let mut rng = TestRng::new(seed);
        let blocks_words = unsafe { decoded_expect.blocks_acquired.words_mut() };
        let mut blocks_offset: usize = 0;
        for pattern in patterns {
            for _ in 0..pattern.words {
                let word = match pattern.fill {
                    TestPatternFill::UniformOnes => u64::MAX,
                    TestPatternFill::UniformZeros => 0,
                    // Exclude 0 and MAX to ensure literal words aren't accidentally
                    // treated as uniform runs.
                    TestPatternFill::Literal => rng.range_inclusive(1, u64::MAX - 1),
                };
                blocks_words[blocks_offset] = word;
                // Update shard index: if any word in shard isn't all-ones,
                // the shard has free blocks.
                let index_bit = (blocks_offset * 64) / SHARD_BITS;
                if word != u64::MAX {
                    decoded_expect.index.unset(index_bit);
                }
                blocks_offset += 1;
            }
        }
        assert_eq!(blocks_offset, blocks_words.len());

        decoded_expect.verify_index();
        decoded_expect.opened = true;
        decoded_expect.checkpoint_durable = true;

        // Encode -> decode roundtrip
        let encoded_size = encode_size_max_bytes(blocks_count);
        assert_eq!(encoded_size % size_of::<Word>(), 0);
        let mut encoded_words = vec![0u64; encoded_size / size_of::<Word>()];
        let encoded_len = {
            let mut encoded_chunk = words_as_bytes_mut(&mut encoded_words);
            decoded_expect.encode(BitsetKind::BlocksAcquired, &mut [&mut encoded_chunk])
        };

        let mut decoded_actual = init_free_set(blocks_count);
        let encoded_bytes = words_as_bytes(&encoded_words);
        decoded_actual.decode_chunks(&[&encoded_bytes[..encoded_len]], &[]);

        assert_free_set_eq(&decoded_expect, &decoded_actual);
    }

    /// Exercises EWAH encoding/decoding across various fill patterns to ensure
    /// the codec handles all edge cases:
    /// - Single fill type (pure runs)
    /// - Mixed patterns (run/literal transitions)
    /// - Long runs exceeding marker field limits
    /// - Randomly interleaved patterns
    #[test]
    fn encode_decode_handles_various_fill_patterns() {
        let words_per_shard = SHARD_BITS / 64;

        // Single fill type tests - verify basic run compression
        test_encode_patterns(
            &[TestPattern {
                fill: TestPatternFill::UniformOnes,
                words: words_per_shard,
            }],
            0xA1B2_C3D4,
        );
        test_encode_patterns(
            &[TestPattern {
                fill: TestPatternFill::UniformZeros,
                words: words_per_shard,
            }],
            0xB2C3_D4E5,
        );
        test_encode_patterns(
            &[TestPattern {
                fill: TestPatternFill::Literal,
                words: words_per_shard,
            }],
            0xC3D4_E5F6,
        );

        // Long run exceeding u16::MAX - tests run count overflow handling
        test_encode_patterns(
            &[TestPattern {
                fill: TestPatternFill::UniformOnes,
                words: u16::MAX as usize + 1,
            }],
            0xD4E5_F607,
        );

        // Mixed pattern - exercises run/literal/run transitions
        test_encode_patterns(
            &[
                TestPattern {
                    fill: TestPatternFill::UniformOnes,
                    words: words_per_shard / 4,
                },
                TestPattern {
                    fill: TestPatternFill::UniformZeros,
                    words: words_per_shard / 4,
                },
                TestPattern {
                    fill: TestPatternFill::Literal,
                    words: words_per_shard / 4,
                },
                TestPattern {
                    fill: TestPatternFill::UniformOnes,
                    words: words_per_shard / 4,
                },
            ],
            0xE5F6_0718,
        );

        // Random interleaving - stress test with worst-case fragmentation
        // (single-word patterns force frequent marker emissions)
        let fills = [
            TestPatternFill::UniformOnes,
            TestPatternFill::UniformZeros,
            TestPatternFill::Literal,
        ];
        let mut rng = TestRng::new(0xF607_1829);
        for _ in 0..10 {
            let mut patterns = Vec::with_capacity(words_per_shard);
            for _ in 0..words_per_shard {
                let fill = fills[rng.index(fills.len())];
                patterns.push(TestPattern { fill, words: 1 });
            }
            test_encode_patterns(&patterns, rng.next_u64());
        }
    }
}
