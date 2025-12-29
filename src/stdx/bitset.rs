//! Fixed-size bitset backed by an array of `u64` words; capacity is compile-time known.

/// Computes the number of `u64` words needed to store `N` bits.
pub const fn words_for_bits(n: usize) -> usize {
    n.div_ceil(64)
}

/// Fixed-size bitset backed by an array of `u64` words.
///
/// The const parameter `N` is the bit capacity, and `WORDS` is the number of `u64` words.
/// Use `WORDS = words_for_bits(N)` to get the correct value.
///
/// All indexing operations panic when `idx >= N`. Use `iter` to traverse set bits
/// in ascending order without allocation.
///
/// # Examples
/// ```
/// use consensus::stdx::bitset::{BitSet, words_for_bits};
///
/// let mut bits: BitSet<8, { words_for_bits(8) }> = BitSet::empty();
/// bits.set(1);
/// bits.set(3);
/// assert_eq!(bits.iter().collect::<Vec<_>>(), vec![1, 3]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitSet<const N: usize, const WORDS: usize> {
    words: [u64; WORDS],
}

impl<const N: usize, const WORDS: usize> BitSet<N, WORDS> {
    const fn validate() {
        assert!(N > 0, "BitSet capacity must be > 0");
        assert!(WORDS == N.div_ceil(64), "WORDS must equal N.div_ceil(64)");
    }

    /// Returns the number of addressable bits (`N`).
    ///
    /// Panics at compile time if `N` is zero or `WORDS` is incorrect.
    #[inline(always)]
    pub const fn capacity() -> usize {
        Self::validate();
        N
    }

    /// Creates an empty bitset.
    #[inline]
    pub const fn empty() -> Self {
        Self::validate();
        Self {
            words: [0u64; WORDS],
        }
    }

    /// Counts set bits; never exceeds `N`.
    #[inline]
    pub const fn count(&self) -> usize {
        let mut total: usize = 0;
        let mut i = 0;
        while i < WORDS {
            total += self.words[i].count_ones() as usize;
            i += 1;
        }
        debug_assert!(total <= N);
        total
    }

    /// Returns `true` when no bits are set.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        let mut i = 0;
        while i < WORDS {
            if self.words[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Returns `true` when every bit in `[0, N)` is set.
    #[inline]
    pub const fn is_full(&self) -> bool {
        let mut i = 0;
        // Check all complete words.
        while i + 1 < WORDS {
            if self.words[i] != u64::MAX {
                return false;
            }
            i += 1;
        }

        // Check last word with proper mask for remaining bits.
        if WORDS > 0 {
            let remaining_bits = N % 64;
            let mask = if remaining_bits == 0 {
                u64::MAX
            } else {
                (1u64 << remaining_bits) - 1
            };
            if self.words[WORDS - 1] != mask {
                return false;
            }
        }
        true
    }

    /// Creates a bitset with all bits set.
    #[inline]
    pub const fn full() -> Self {
        Self::validate();

        let mut words = [0u64; WORDS];
        let mut i = 0;

        // Fill all complete words with all ones.
        while i + 1 < WORDS {
            words[i] = u64::MAX;
            i += 1;
        }

        // Handle the last word specially - only set the valid bits.
        if WORDS > 0 {
            let remaining_bits = N % 64;
            if remaining_bits == 0 {
                // N is a multiple of 64, so last word is also full.
                words[WORDS - 1] = u64::MAX;
            } else {
                // Only set the lower `remaining_bits` bits.
                words[WORDS - 1] = (1u64 << remaining_bits) - 1;
            }
        }

        Self { words }
    }

    /// Lowest set bit, if any.
    #[inline]
    pub const fn first_set(&self) -> Option<usize> {
        let mut word_idx = 0;
        while word_idx < WORDS {
            let word = self.words[word_idx];
            if word != 0 {
                let bit_idx = word.trailing_zeros() as usize;
                let idx = word_idx * 64 + bit_idx;
                debug_assert!(idx < N);
                return Some(idx);
            }
            word_idx += 1;
        }
        None
    }

    /// Lowest unset bit; `None` when full.
    #[inline]
    pub const fn first_unset(&self) -> Option<usize> {
        let mut word_idx = 0;
        while word_idx < WORDS {
            let word = self.words[word_idx];
            let inverted = !word;
            if inverted != 0 {
                let bit_idx = inverted.trailing_zeros() as usize;
                let idx = word_idx * 64 + bit_idx;
                if idx < N {
                    return Some(idx);
                }
                // The unset bit is beyond our capacity.
                return None;
            }
            word_idx += 1;
        }
        None
    }

    /// Returns whether `idx` is set.
    ///
    /// Panics if `idx >= N`.
    #[inline]
    pub const fn is_set(&self, idx: usize) -> bool {
        assert!(idx < N, "bit index out of bounds");
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        (self.words[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Sets the bit at `idx`.
    ///
    /// Panics if `idx >= N`.
    #[inline]
    pub const fn set(&mut self, idx: usize) {
        assert!(idx < N, "bit index out of bounds");
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        self.words[word_idx] |= 1u64 << bit_idx;
        debug_assert!(self.is_set(idx));
    }

    /// Clears the bit at `idx`.
    ///
    /// Panics if `idx >= N`.
    #[inline]
    pub const fn unset(&mut self, idx: usize) {
        assert!(idx < N, "bit index out of bounds");
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        self.words[word_idx] &= !(1u64 << bit_idx);
        debug_assert!(!self.is_set(idx));
    }

    /// Sets or clears the bit at `idx` based on `value`.
    ///
    /// Panics if `idx >= N`.
    #[inline]
    pub const fn set_value(&mut self, idx: usize, value: bool) {
        assert!(idx < N, "bit index out of bounds");
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        let bit_mask = 1u64 << bit_idx;

        // Branchless: clear the bit, then OR in the new value.
        let val_mask = (value as u64) << bit_idx;
        self.words[word_idx] = (self.words[word_idx] & !bit_mask) | val_mask;

        debug_assert!(self.is_set(idx) == value);
    }

    /// Clears all bits.
    #[inline]
    pub const fn clear(&mut self) {
        let mut i = 0;
        while i < WORDS {
            self.words[i] = 0;
            i += 1;
        }

        debug_assert!(self.is_empty());
    }

    /// Returns `true` if every set bit in `self` is also set in `other`.
    ///
    /// An empty bitset is a subset of any bitset; any bitset is a subset of a full bitset.
    #[inline]
        for i in 0..WORDS {
            if self.words[i] & !other.words[i] != 0 {
                return false;
            }
        }
        true
            }
            i += 1;
        }
        true
    }

    /// Iterates over set bits in ascending order using a snapshot of the current state.
    #[inline]
    pub const fn iter(&self) -> BitSetIterator<N, WORDS> {
        BitSetIterator {
            words: self.words,
            word_idx: 0,
            current_word: if WORDS > 0 { self.words[0] } else { 0 },
        }
    }
}

/// Iterator over set bit indices in ascending order, produced by `BitSet::iter`.
#[derive(Clone, Copy)]
pub struct BitSetIterator<const N: usize, const WORDS: usize> {
    words: [u64; WORDS],
    word_idx: usize,
    current_word: u64,
}

impl<const N: usize, const WORDS: usize> Iterator for BitSetIterator<N, WORDS> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        // Find the next set bit.
        loop {
            if self.current_word != 0 {
                let bit_idx = self.current_word.trailing_zeros() as usize;
                let idx = self.word_idx * 64 + bit_idx;

                // Clear the lowest set bit.
                self.current_word &= self.current_word.wrapping_sub(1);

                if idx < N {
                    return Some(idx);
                }
                // Beyond capacity - this word has no more valid bits.
                self.current_word = 0;
            }

            // Move to the next word.
            self.word_idx += 1;
            if self.word_idx >= WORDS {
                return None;
            }
            self.current_word = self.words[self.word_idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BitSet, words_for_bits};
    use std::collections::HashSet;

    use proptest::prelude::*;

    // Type aliases for convenience in tests.
    type BitSet64 = BitSet<64, { words_for_bits(64) }>;
    type BitSet128 = BitSet<128, { words_for_bits(128) }>;
    type BitSet256 = BitSet<256, { words_for_bits(256) }>;

    // ============================================
    // Property-Based Tests
    // ============================================

    proptest! {
        #[test]
        fn set_is_idempotent(idx in 0usize..64) {
            let mut b1: BitSet64 = BitSet::empty();
            let mut b2: BitSet64 = BitSet::empty();

            b1.set(idx);
            b2.set(idx);
            b2.set(idx); // set again

            prop_assert_eq!(b1, b2);
            prop_assert_eq!(b1.count(), 1);
        }

        #[test]
        fn count_equals_iter_len(bits in prop::collection::vec(0usize..64, 0..64)) {
            let mut b: BitSet64 = BitSet::empty();
            for &idx in &bits {
                b.set(idx);
            }

            let count = b.count();
            let iter_len = b.iter().count();
            prop_assert_eq!(count, iter_len);
        }

        #[test]
        fn first_set_matches_iter_first(bits in prop::collection::vec(0usize..64, 1..64)) {
            let mut b: BitSet64 = BitSet::empty();
            for &idx in &bits {
                b.set(idx);
            }

            prop_assert_eq!(b.first_set(), b.iter().next());
        }

        #[test]
        fn set_unset_is_identity(idx in 0usize..64) {
            let mut b: BitSet64 = BitSet::empty();
            let original = b;

            b.set(idx);
            b.unset(idx);

            prop_assert_eq!(b, original);
            prop_assert!(b.is_empty());
        }

        #[test]
        fn is_empty_iff_count_zero(bits in prop::collection::vec(0usize..64, 0..64)) {
            let mut b: BitSet64 = BitSet::empty();
            for &idx in &bits {
                b.set(idx);
            }

            prop_assert_eq!(b.is_empty(), b.count() == 0);
        }

        #[test]
        fn is_full_iff_count_n(bits in prop::collection::vec(0usize..64, 0..64)) {
            let mut b: BitSet64 = BitSet::empty();
            for &idx in &bits {
                b.set(idx);
            }

            prop_assert_eq!(b.is_full(), b.count() == 64);
        }

        #[test]
        fn iter_roundtrip(bits in prop::collection::hash_set(0usize..64, 0..64)) {
            let mut b1: BitSet64 = BitSet::empty();
            for &idx in &bits {
                b1.set(idx);
            }

            let mut b2: BitSet64 = BitSet::empty();
            for idx in b1.iter() {
                b2.set(idx);
            }

            prop_assert_eq!(b1, b2);

            // Also verify the collected indices match the input set
            let collected: HashSet<usize> = b1.iter().collect();
            prop_assert_eq!(collected, bits);
        }

        #[test]
        fn subset_is_reflexive(bits in prop::collection::vec(0usize..64, 0..64)) {
            let mut b: BitSet64 = BitSet::empty();
            for &idx in &bits {
                b.set(idx);
            }

            prop_assert!(b.is_subset(&b));
        }

        #[test]
        fn subset_with_additional_bits(
            base_bits in prop::collection::hash_set(0usize..64, 0..32),
            extra_bits in prop::collection::hash_set(0usize..64, 1..32),
        ) {
            let mut subset: BitSet64 = BitSet::empty();
            let mut superset: BitSet64 = BitSet::empty();

            for &idx in &base_bits {
                subset.set(idx);
                superset.set(idx);
            }

            for &idx in &extra_bits {
                superset.set(idx);
            }

            // subset should always be a subset of superset
            prop_assert!(subset.is_subset(&superset));

            // superset is only a subset of subset if extra_bits were already in base_bits
            let extra_outside_base: HashSet<_> = extra_bits.difference(&base_bits).collect();
            prop_assert_eq!(superset.is_subset(&subset), extra_outside_base.is_empty());
        }
    }

    // ============================================
    // Unit Tests
    // ============================================

    #[test]
    fn empty_bitset() {
        let b: BitSet64 = BitSet::empty();

        assert!(b.is_empty());
        assert!(!b.is_full());
        assert_eq!(b.count(), 0);
        assert_eq!(b.first_set(), None);
        assert_eq!(b.first_unset(), Some(0));
    }

    #[test]
    fn full_bitset() {
        let b: BitSet<8, { words_for_bits(8) }> = BitSet::full();

        assert!(b.is_full());
        assert!(!b.is_empty());
        assert_eq!(b.count(), 8);
        assert_eq!(b.first_set(), Some(0));
        assert_eq!(b.first_unset(), None);
    }

    #[test]
    fn set_unset() {
        let mut b: BitSet128 = BitSet::empty();

        b.set(0);
        b.set(64);
        b.set(127);

        assert!(b.is_set(0));
        assert!(b.is_set(64));
        assert!(b.is_set(127));
        assert!(!b.is_set(1));
        assert_eq!(b.count(), 3);

        b.unset(64);

        assert!(!b.is_set(64));
        assert_eq!(b.count(), 2);
    }

    #[test]
    fn first_set_unset() {
        let mut b: BitSet<8, { words_for_bits(8) }> = BitSet::empty();

        assert_eq!(b.first_set(), None);
        assert_eq!(b.first_unset(), Some(0));

        b.set(3);

        assert_eq!(b.first_set(), Some(3));
        assert_eq!(b.first_unset(), Some(0));

        b.set(0);

        assert_eq!(b.first_set(), Some(0));
        assert_eq!(b.first_unset(), Some(1));
    }

    #[test]
    fn set_value() {
        let mut b: BitSet<8, { words_for_bits(8) }> = BitSet::empty();

        b.set_value(3, true);
        assert!(b.is_set(3));

        b.set_value(3, false);
        assert!(!b.is_set(3));
    }

    #[test]
    fn clear() {
        let mut b: BitSet64 = BitSet::empty();
        b.set(0);
        b.set(32);
        b.set(63);

        b.clear();

        assert!(b.is_empty());
        assert_eq!(b.count(), 0);
    }

    #[test]
    fn iterate() {
        let mut b: BitSet64 = BitSet::empty();
        b.set(3);
        b.set(7);
        b.set(15);
        b.set(63);

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![3, 7, 15, 63]);
    }

    #[test]
    fn iterate_empty() {
        let b: BitSet64 = BitSet::empty();
        let indices: Vec<usize> = b.iter().collect();
        assert!(indices.is_empty());
    }

    #[test]
    fn small_capacity() {
        let mut b: BitSet<1, { words_for_bits(1) }> = BitSet::empty();

        assert!(b.is_empty());
        assert_eq!(b.first_unset(), Some(0));

        b.set(0);

        assert!(b.is_full());
        assert_eq!(b.first_unset(), None);
        assert_eq!(b.first_set(), Some(0));
    }

    #[test]
    fn max_capacity_128() {
        let mut b: BitSet128 = BitSet::empty();

        b.set(0);
        b.set(127);

        assert_eq!(b.count(), 2);
        assert_eq!(b.first_set(), Some(0));

        let full: BitSet128 = BitSet::full();
        assert_eq!(full.count(), 128);
        assert_eq!(full.first_unset(), None);
    }

    #[test]
    fn large_capacity_256() {
        let mut b: BitSet256 = BitSet::empty();

        b.set(0);
        b.set(127);
        b.set(128);
        b.set(255);

        assert_eq!(b.count(), 4);
        assert_eq!(b.first_set(), Some(0));

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![0, 127, 128, 255]);

        let full: BitSet256 = BitSet::full();
        assert_eq!(full.count(), 256);
        assert_eq!(full.first_unset(), None);
    }

    #[test]
    fn large_capacity_1024() {
        type BitSet1024 = BitSet<1024, { words_for_bits(1024) }>;
        let mut b: BitSet1024 = BitSet::empty();

        b.set(0);
        b.set(511);
        b.set(512);
        b.set(1023);

        assert_eq!(b.count(), 4);
        assert_eq!(b.first_set(), Some(0));

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![0, 511, 512, 1023]);

        let full: BitSet1024 = BitSet::full();
        assert_eq!(full.count(), 1024);
        assert_eq!(full.first_unset(), None);
    }

    // ============================================
    // full() Edge Cases
    // ============================================

    #[test]
    fn full_bitset_n1() {
        let b: BitSet<1, { words_for_bits(1) }> = BitSet::full();

        assert!(b.is_full());
        assert_eq!(b.count(), 1);
        assert!(b.is_set(0));
        assert_eq!(b.first_unset(), None);

        // Verify iterator yields exactly one element
        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn full_bitset_n63() {
        let b: BitSet<63, { words_for_bits(63) }> = BitSet::full();

        assert!(b.is_full());
        assert_eq!(b.count(), 63);
        assert!(b.is_set(0));
        assert!(b.is_set(62));
        assert_eq!(b.first_unset(), None);

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices.len(), 63);
    }

    #[test]
    fn full_bitset_n64() {
        let b: BitSet64 = BitSet::full();

        assert!(b.is_full());
        assert_eq!(b.count(), 64);
        assert!(b.is_set(0));
        assert!(b.is_set(63));
        assert_eq!(b.first_unset(), None);

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices.len(), 64);
    }

    #[test]
    fn full_bitset_n65() {
        let b: BitSet<65, { words_for_bits(65) }> = BitSet::full();

        assert!(b.is_full());
        assert_eq!(b.count(), 65);
        assert!(b.is_set(0));
        assert!(b.is_set(64));
        assert_eq!(b.first_unset(), None);

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices.len(), 65);
    }

    #[test]
    fn full_bitset_n127() {
        let b: BitSet<127, { words_for_bits(127) }> = BitSet::full();

        assert!(b.is_full());
        assert_eq!(b.count(), 127);
        assert!(b.is_set(0));
        assert!(b.is_set(126));
        assert_eq!(b.first_unset(), None);

        // Verify all 127 bits are set
        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices.len(), 127);
    }

    #[test]
    fn full_bitset_n128() {
        let b: BitSet128 = BitSet::full();

        assert!(b.is_full());
        assert_eq!(b.count(), 128);
        assert!(b.is_set(0));
        assert!(b.is_set(127));
        assert_eq!(b.first_unset(), None);

        // Verify all 128 bits are set
        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices.len(), 128);
    }

    // ============================================
    // Iterator Boundary Tests
    // ============================================

    #[test]
    fn iterate_full_bitset() {
        let b: BitSet<8, { words_for_bits(8) }> = BitSet::full();
        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn iterate_full_bitset_n128() {
        let b: BitSet128 = BitSet::full();
        let indices: Vec<usize> = b.iter().collect();

        // Should yield exactly 0..128
        assert_eq!(indices.len(), 128);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[127], 127);

        // Verify sequence is correct
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(idx, i);
        }
    }

    #[test]
    fn iterate_consecutive_bits() {
        let mut b: BitSet<16, { words_for_bits(16) }> = BitSet::empty();
        for i in 5..10 {
            b.set(i);
        }

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn iterate_first_bit_only() {
        let mut b: BitSet64 = BitSet::empty();
        b.set(0);

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn iterate_last_bit_only_n64() {
        let mut b: BitSet64 = BitSet::empty();
        b.set(63);

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![63]);
    }

    #[test]
    fn iterate_last_bit_only_n128() {
        let mut b: BitSet128 = BitSet::empty();
        b.set(127);

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![127]);
    }

    #[test]
    fn iterate_across_word_boundary() {
        let mut b: BitSet128 = BitSet::empty();
        b.set(63);
        b.set(64);

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![63, 64]);
    }

    #[test]
    fn iterate_sparse_across_words() {
        let mut b: BitSet256 = BitSet::empty();
        b.set(0);
        b.set(63);
        b.set(64);
        b.set(127);
        b.set(128);
        b.set(191);
        b.set(192);
        b.set(255);

        let indices: Vec<usize> = b.iter().collect();
        assert_eq!(indices, vec![0, 63, 64, 127, 128, 191, 192, 255]);
    }

    // ============================================
    // State Transition Tests
    // ============================================

    #[test]
    fn transition_empty_to_full() {
        let mut b: BitSet<4, { words_for_bits(4) }> = BitSet::empty();

        assert!(b.is_empty());
        assert!(!b.is_full());

        b.set(0);
        assert!(!b.is_empty());
        assert!(!b.is_full());
        assert_eq!(b.count(), 1);

        b.set(1);
        b.set(2);
        assert!(!b.is_full());
        assert_eq!(b.count(), 3);

        b.set(3);
        assert!(b.is_full());
        assert_eq!(b.count(), 4);
        assert_eq!(b.first_unset(), None);
    }

    #[test]
    fn transition_full_to_empty() {
        let mut b: BitSet<4, { words_for_bits(4) }> = BitSet::full();

        assert!(b.is_full());

        for i in 0..4 {
            b.unset(i);
            assert_eq!(b.count(), 3 - i);
        }

        assert!(b.is_empty());
        assert_eq!(b.first_set(), None);
    }

    // ============================================
    // Clone/PartialEq Tests
    // ============================================

    #[test]
    fn clone_produces_identical_bitset() {
        let mut b1: BitSet64 = BitSet::empty();
        b1.set(5);
        b1.set(10);
        b1.set(42);

        let b2 = b1;

        assert_eq!(b1, b2);
        assert_eq!(b1.count(), b2.count());
        assert_eq!(b1.first_set(), b2.first_set());

        // Verify independence - mutating copy doesn't affect original
        let mut b3 = b1;
        b3.set(20);
        assert_ne!(b1, b3);
    }

    #[test]
    fn partial_eq_empty_bitsets() {
        let b1: BitSet64 = BitSet::empty();
        let b2: BitSet64 = BitSet::empty();
        assert_eq!(b1, b2);
    }

    #[test]
    fn partial_eq_full_bitsets() {
        let b1: BitSet64 = BitSet::full();
        let b2: BitSet64 = BitSet::full();
        assert_eq!(b1, b2);
    }

    #[test]
    fn partial_eq_different_bits() {
        let mut b1: BitSet64 = BitSet::empty();
        let mut b2: BitSet64 = BitSet::empty();

        b1.set(5);
        b2.set(6);

        assert_ne!(b1, b2);
    }

    // ============================================
    // first_unset Edge Cases
    // ============================================

    #[test]
    fn first_unset_on_full_various_sizes() {
        let b1: BitSet<1, { words_for_bits(1) }> = BitSet::full();
        assert_eq!(b1.first_unset(), None);

        let b8: BitSet<8, { words_for_bits(8) }> = BitSet::full();
        assert_eq!(b8.first_unset(), None);

        let b64: BitSet64 = BitSet::full();
        assert_eq!(b64.first_unset(), None);

        let b127: BitSet<127, { words_for_bits(127) }> = BitSet::full();
        assert_eq!(b127.first_unset(), None);

        let b128: BitSet128 = BitSet::full();
        assert_eq!(b128.first_unset(), None);

        let b256: BitSet256 = BitSet::full();
        assert_eq!(b256.first_unset(), None);
    }

    #[test]
    fn first_unset_with_holes() {
        let mut b: BitSet<16, { words_for_bits(16) }> = BitSet::empty();

        // Set all except index 5
        for i in 0..16 {
            if i != 5 {
                b.set(i);
            }
        }

        assert_eq!(b.first_unset(), Some(5));
    }

    #[test]
    fn first_unset_last_position() {
        let mut b: BitSet<8, { words_for_bits(8) }> = BitSet::empty();

        // Set all except last
        for i in 0..7 {
            b.set(i);
        }

        assert_eq!(b.first_unset(), Some(7));
    }

    #[test]
    fn first_unset_across_word_boundary() {
        let mut b: BitSet128 = BitSet::empty();

        // Set all bits in first word (0-63)
        for i in 0..64 {
            b.set(i);
        }

        assert_eq!(b.first_unset(), Some(64));
    }

    #[test]
    fn first_unset_in_second_word() {
        let mut b: BitSet128 = BitSet::empty();

        // Set all bits except bit 70
        for i in 0..128 {
            if i != 70 {
                b.set(i);
            }
        }

        assert_eq!(b.first_unset(), Some(70));
    }

    // ============================================
    // is_subset tests
    // ============================================

    #[test]
    fn empty_is_subset_of_any() {
        let empty: BitSet64 = BitSet::empty();
        let full: BitSet64 = BitSet::full();
        let mut partial: BitSet64 = BitSet::empty();
        partial.set(10);
        partial.set(30);

        assert!(empty.is_subset(&empty));
        assert!(empty.is_subset(&partial));
        assert!(empty.is_subset(&full));
    }

    #[test]
    fn any_is_subset_of_full() {
        let empty: BitSet64 = BitSet::empty();
        let full: BitSet64 = BitSet::full();
        let mut partial: BitSet64 = BitSet::empty();
        partial.set(10);
        partial.set(30);

        assert!(empty.is_subset(&full));
        assert!(partial.is_subset(&full));
        assert!(full.is_subset(&full));
    }

    #[test]
    fn subset_not_superset() {
        let mut a: BitSet64 = BitSet::empty();
        let mut b: BitSet64 = BitSet::empty();

        a.set(5);
        b.set(5);
        b.set(10);

        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));
    }

    #[test]
    fn disjoint_sets_not_subsets() {
        let mut a: BitSet64 = BitSet::empty();
        let mut b: BitSet64 = BitSet::empty();

        a.set(5);
        b.set(10);

        assert!(!a.is_subset(&b));
        assert!(!b.is_subset(&a));
    }

    #[test]
    fn subset_across_word_boundary() {
        let mut subset: BitSet128 = BitSet::empty();
        let mut superset: BitSet128 = BitSet::empty();

        // Set bits in both words for subset
        subset.set(10); // first word
        subset.set(70); // second word

        // Superset has all of subset's bits plus more
        superset.set(10);
        superset.set(50);
        superset.set(70);
        superset.set(100);

        assert!(subset.is_subset(&superset));
        assert!(!superset.is_subset(&subset));
    }
}
