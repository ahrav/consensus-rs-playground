//! Fixed-size bitset backed by a single `u128`; optimized for up to 128 compile-time-known flags.

// Compile-time proof that u32 -> usize is safe on this platform.
// This fails to compile on 16-bit platforms.
const _: () = assert!(
    std::mem::size_of::<usize>() >= std::mem::size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

/// Fixed-size bitset backed by a single `u128`; capacity `N` must be in 1..=128.
///
/// All indexing operations panic when `idx >= N`. Use `iter` to traverse set bits
/// in ascending order without allocation.
///
/// # Examples
/// ```
/// use crate::stdx::bitset::BitSet;
///
/// let mut bits: BitSet<8> = BitSet::empty();
/// bits.set(1);
/// bits.set(3);
/// assert_eq!(bits.iter().collect::<Vec<_>>(), vec![1, 3]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitSet<const N: u32> {
    bits: u128,
}

impl<const N: u32> BitSet<N> {
    const fn validate() {
        assert!(N > 0, "BitSet capacity must be > 0");
        assert!(N <= 128, "BitSet capacity must be <= 128");
    }

    /// Returns the number of addressable bits (`N`).
    ///
    /// Panics if `N` is zero or greater than 128.
    #[inline(always)]
    pub const fn capacity() -> u32 {
        Self::validate();
        N
    }

    #[inline(always)]
    const fn bit(idx: u32) -> u128 {
        assert!(idx < N, "bit index of out bound");
        1u128 << idx
    }

    /// Creates an empty bitset.
    #[inline]
    pub const fn empty() -> Self {
        Self::validate();
        Self { bits: 0 }
    }

    /// Counts set bits; never exceeds `N`.
    #[inline]
    pub const fn count(&self) -> u32 {
        let count = self.bits.count_ones();
        assert!(count <= N);
        count
    }

    /// Returns `true` when no bits are set.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// Returns `true` when every bit in `[0, N)` is set.
    #[inline]
    pub const fn is_full(&self) -> bool {
        self.count() == N
    }

    /// Creates a bitset with all bits set, special-casing `N == 128` to avoid shifting by 128.
    #[inline]
    pub fn full() -> Self {
        Self::validate();

        let bits = if N == 128 {
            u128::MAX
        } else {
            (1u128 << N) - 1
        };

        let bitset = Self { bits };
        assert!(bitset.is_full());
        assert!(bitset.count() == N);

        bitset
    }

    /// Lowest set bit, if any.
    #[inline]
    pub const fn first_set(&self) -> Option<u32> {
        if self.bits == 0 {
            return None;
        }

        let idx = self.bits.trailing_zeros();
        assert!(idx < N);
        Some(idx)
    }

    /// Lowest unset bit; `None` when full.
    #[inline]
    pub const fn first_unset(&self) -> Option<u32> {
        let idx = (!self.bits).trailing_zeros();
        if idx < N { Some(idx) } else { None }
    }

    /// Returns whether `idx` is set.
    ///
    /// Panics if `idx >= N`.
    #[inline]
    pub const fn is_set(&self, idx: u32) -> bool {
        assert!(idx < N);
        (self.bits & Self::bit(idx)) != 0
    }

    /// Sets the bit at `idx`.
    ///
    /// Panics if `idx >= N`.
    #[inline]
    pub const fn set(&mut self, idx: u32) {
        assert!(idx < N);
        self.bits |= Self::bit(idx);
        assert!(self.is_set(idx));
    }

    /// Clears the bit at `idx`.
    ///
    /// Panics if `idx >= N`.
    #[inline]
    pub const fn unset(&mut self, idx: u32) {
        assert!(idx < N);
        self.bits &= !Self::bit(idx);
        assert!(!self.is_set(idx));
    }

    /// Sets or clears the bit at `idx` based on `value`.
    ///
    /// Panics if `idx >= N`.
    #[inline]
    pub const fn set_value(&mut self, idx: u32, value: bool) {
        assert!(idx < N);
        if value {
            self.set(idx)
        } else {
            self.unset(idx);
        }
        assert!(self.is_set(idx) == value);
    }

    /// Clears all bits.
    #[inline]
    pub const fn clear(&mut self) {
        self.bits = 0;

        assert!(self.is_empty());
        assert!(self.count() == 0);
    }

    /// Iterates over set bits in ascending order using a snapshot of the current state.
    #[inline]
    pub const fn iter(&self) -> BitSetIterator<N> {
        BitSetIterator {
            bits_remain: self.bits,
        }
    }
}

/// Iterator over set bit indices in ascending order, produced by `BitSet::iter`.
#[derive(Clone, Copy)]
pub struct BitSetIterator<const N: u32> {
    bits_remain: u128,
}

impl<const N: u32> Iterator for BitSetIterator<N> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.bits_remain == 0 {
            return None;
        }

        let idx = self.bits_remain.trailing_zeros();
        if idx >= N {
            return None;
        }

        // Clear lowest set bit: bits & (bits - 1)
        self.bits_remain &= self.bits_remain.wrapping_sub(1);

        Some(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::BitSet;
    use std::collections::HashSet;

    use proptest::prelude::*;

    // ============================================
    // Property-Based Tests
    // ============================================

    proptest! {
        #[test]
        fn set_is_idempotent(idx in 0u32..64) {
            let mut b1: BitSet<64> = BitSet::empty();
            let mut b2: BitSet<64> = BitSet::empty();

            b1.set(idx);
            b2.set(idx);
            b2.set(idx); // set again

            prop_assert_eq!(b1, b2);
            prop_assert_eq!(b1.count(), 1);
        }

        #[test]
        fn count_equals_iter_len(bits in prop::collection::vec(0u32..64, 0..64)) {
            let mut b: BitSet<64> = BitSet::empty();
            for &idx in &bits {
                b.set(idx);
            }

            let count = b.count() as usize;
            let iter_len = b.iter().count();
            prop_assert_eq!(count, iter_len);
        }

        #[test]
        fn first_set_matches_iter_first(bits in prop::collection::vec(0u32..64, 1..64)) {
            let mut b: BitSet<64> = BitSet::empty();
            for &idx in &bits {
                b.set(idx);
            }

            prop_assert_eq!(b.first_set(), b.iter().next());
        }

        #[test]
        fn set_unset_is_identity(idx in 0u32..64) {
            let mut b: BitSet<64> = BitSet::empty();
            let original = b;

            b.set(idx);
            b.unset(idx);

            prop_assert_eq!(b, original);
            prop_assert!(b.is_empty());
        }

        #[test]
        fn is_empty_iff_count_zero(bits in prop::collection::vec(0u32..64, 0..64)) {
            let mut b: BitSet<64> = BitSet::empty();
            for &idx in &bits {
                b.set(idx);
            }

            prop_assert_eq!(b.is_empty(), b.count() == 0);
        }

        #[test]
        fn is_full_iff_count_n(bits in prop::collection::vec(0u32..64, 0..64)) {
            let mut b: BitSet<64> = BitSet::empty();
            for &idx in &bits {
                b.set(idx);
            }

            prop_assert_eq!(b.is_full(), b.count() == 64);
        }

        #[test]
        fn iter_roundtrip(bits in prop::collection::hash_set(0u32..64, 0..64)) {
            let mut b1: BitSet<64> = BitSet::empty();
            for &idx in &bits {
                b1.set(idx);
            }

            let mut b2: BitSet<64> = BitSet::empty();
            for idx in b1.iter() {
                b2.set(idx);
            }

            prop_assert_eq!(b1, b2);

            // Also verify the collected indices match the input set
            let collected: HashSet<u32> = b1.iter().collect();
            prop_assert_eq!(collected, bits);
        }
    }

    // ============================================
    // Unit Tests
    // ============================================

    #[test]
    fn empty_bitset() {
        let b: BitSet<64> = BitSet::empty();

        assert!(b.is_empty());
        assert!(!b.is_full());
        assert_eq!(b.count(), 0);
        assert_eq!(b.first_set(), None);
        assert_eq!(b.first_unset(), Some(0));
    }

    #[test]
    fn full_bitset() {
        let b: BitSet<8> = BitSet::full();

        assert!(b.is_full());
        assert!(!b.is_empty());
        assert_eq!(b.count(), 8);
        assert_eq!(b.first_set(), Some(0));
        assert_eq!(b.first_unset(), None);
    }

    #[test]
    fn set_unset() {
        let mut b: BitSet<128> = BitSet::empty();

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
        let mut b: BitSet<8> = BitSet::empty();

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
        let mut b: BitSet<8> = BitSet::empty();

        b.set_value(3, true);
        assert!(b.is_set(3));

        b.set_value(3, false);
        assert!(!b.is_set(3));
    }

    #[test]
    fn clear() {
        let mut b: BitSet<64> = BitSet::empty();
        b.set(0);
        b.set(32);
        b.set(63);

        b.clear();

        assert!(b.is_empty());
        assert_eq!(b.count(), 0);
    }

    #[test]
    fn iterate() {
        let mut b: BitSet<64> = BitSet::empty();
        b.set(3);
        b.set(7);
        b.set(15);
        b.set(63);

        let indices: Vec<u32> = b.iter().collect();
        assert_eq!(indices, vec![3, 7, 15, 63]);
    }

    #[test]
    fn iterate_empty() {
        let b: BitSet<64> = BitSet::empty();
        let indices: Vec<u32> = b.iter().collect();
        assert!(indices.is_empty());
    }

    #[test]
    fn small_capacity() {
        let mut b: BitSet<1> = BitSet::empty();

        assert!(b.is_empty());
        assert_eq!(b.first_unset(), Some(0));

        b.set(0);

        assert!(b.is_full());
        assert_eq!(b.first_unset(), None);
        assert_eq!(b.first_set(), Some(0));
    }

    #[test]
    fn max_capacity() {
        let mut b: BitSet<128> = BitSet::empty();

        b.set(0);
        b.set(127);

        assert_eq!(b.count(), 2);
        assert_eq!(b.first_set(), Some(0));

        let full: BitSet<128> = BitSet::full();
        assert_eq!(full.count(), 128);
        assert_eq!(full.first_unset(), None);
    }

    // ============================================
    // full() Edge Cases
    // ============================================

    #[test]
    fn full_bitset_n1() {
        let b: BitSet<1> = BitSet::full();

        assert!(b.is_full());
        assert_eq!(b.count(), 1);
        assert!(b.is_set(0));
        assert_eq!(b.first_unset(), None);

        // Verify iterator yields exactly one element
        let indices: Vec<u32> = b.iter().collect();
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn full_bitset_n127() {
        let b: BitSet<127> = BitSet::full();

        assert!(b.is_full());
        assert_eq!(b.count(), 127);
        assert!(b.is_set(0));
        assert!(b.is_set(126));
        assert_eq!(b.first_unset(), None);

        // Verify all 127 bits are set
        let indices: Vec<u32> = b.iter().collect();
        assert_eq!(indices.len(), 127);
    }

    #[test]
    fn full_bitset_n128_special_case() {
        // This exercises the N==128 branch in full() (u128::MAX)
        let b: BitSet<128> = BitSet::full();

        assert!(b.is_full());
        assert_eq!(b.count(), 128);
        assert!(b.is_set(0));
        assert!(b.is_set(127));
        assert_eq!(b.first_unset(), None);

        // Verify all 128 bits are set
        let indices: Vec<u32> = b.iter().collect();
        assert_eq!(indices.len(), 128);
    }

    // ============================================
    // Iterator Boundary Tests
    // ============================================

    #[test]
    fn iterate_full_bitset() {
        let b: BitSet<8> = BitSet::full();
        let indices: Vec<u32> = b.iter().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn iterate_full_bitset_n128() {
        let b: BitSet<128> = BitSet::full();
        let indices: Vec<u32> = b.iter().collect();

        // Should yield exactly 0..128
        assert_eq!(indices.len(), 128);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[127], 127);

        // Verify sequence is correct
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(idx, i as u32);
        }
    }

    #[test]
    fn iterate_consecutive_bits() {
        let mut b: BitSet<16> = BitSet::empty();
        for i in 5..10 {
            b.set(i);
        }

        let indices: Vec<u32> = b.iter().collect();
        assert_eq!(indices, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn iterate_first_bit_only() {
        let mut b: BitSet<64> = BitSet::empty();
        b.set(0);

        let indices: Vec<u32> = b.iter().collect();
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn iterate_last_bit_only_n64() {
        let mut b: BitSet<64> = BitSet::empty();
        b.set(63);

        let indices: Vec<u32> = b.iter().collect();
        assert_eq!(indices, vec![63]);
    }

    #[test]
    fn iterate_last_bit_only_n128() {
        let mut b: BitSet<128> = BitSet::empty();
        b.set(127);

        let indices: Vec<u32> = b.iter().collect();
        assert_eq!(indices, vec![127]);
    }

    // ============================================
    // State Transition Tests
    // ============================================

    #[test]
    fn transition_empty_to_full() {
        let mut b: BitSet<4> = BitSet::empty();

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
        let mut b: BitSet<4> = BitSet::full();

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
        let mut b1: BitSet<64> = BitSet::empty();
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
        let b1: BitSet<64> = BitSet::empty();
        let b2: BitSet<64> = BitSet::empty();
        assert_eq!(b1, b2);
    }

    #[test]
    fn partial_eq_full_bitsets() {
        let b1: BitSet<64> = BitSet::full();
        let b2: BitSet<64> = BitSet::full();
        assert_eq!(b1, b2);
    }

    #[test]
    fn partial_eq_different_bits() {
        let mut b1: BitSet<64> = BitSet::empty();
        let mut b2: BitSet<64> = BitSet::empty();

        b1.set(5);
        b2.set(6);

        assert_ne!(b1, b2);
    }

    // ============================================
    // first_unset Edge Cases
    // ============================================

    #[test]
    fn first_unset_on_full_various_sizes() {
        let b1: BitSet<1> = BitSet::full();
        assert_eq!(b1.first_unset(), None);

        let b8: BitSet<8> = BitSet::full();
        assert_eq!(b8.first_unset(), None);

        let b127: BitSet<127> = BitSet::full();
        assert_eq!(b127.first_unset(), None);

        let b128: BitSet<128> = BitSet::full();
        assert_eq!(b128.first_unset(), None);
    }

    #[test]
    fn first_unset_with_holes() {
        let mut b: BitSet<16> = BitSet::empty();

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
        let mut b: BitSet<8> = BitSet::empty();

        // Set all except last
        for i in 0..7 {
            b.set(i);
        }

        assert_eq!(b.first_unset(), Some(7));
    }
}
