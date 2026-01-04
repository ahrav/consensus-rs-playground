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

use crate::{constants, stdx::DynamicBitSet};

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

#[cfg(test)]
mod tests {
    use super::{find_bit, BitKind};
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
