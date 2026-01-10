#![allow(dead_code)]

use std::cell::Cell;

/// Indicates whether an upsert operation updated an existing entry or inserted a new one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateOrInsert {
    /// An existing entry with the same key was found and its value was replaced.
    Update,
    /// No existing entry matched the key, so a new entry was created.
    Insert,
}

/// A short, partial hash of a key, stored alongside cached values.
///
/// Because the tag is small, collisions are possible: `tag(k1) == tag(k2)` does not
/// imply `k1 == k2`. However, most of the time, where the tag differs, a full key
/// comparison can be avoided. Since tags are 16-32x smaller than keys, they can also
/// be kept hot in cache.
pub trait Tag: Copy + Eq + PartialEq + Default {
    /// The number of bits in this tag type.
    const BITS: usize;

    /// Extracts a tag from hash entropy by truncating to the tag width.
    fn truncate(entropy: u64) -> Self;
}

/// 8-bit tag implementation.
impl Tag for u8 {
    const BITS: usize = 8;

    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u8
    }
}

/// 16-bit tag implementation.
impl Tag for u16 {
    const BITS: usize = 16;

    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u16
    }
}

/// Defines the key/value types and operations required by a set-associative cache.
///
/// This mirrors Zig's comptime `key_from_value` and `hash` parameters.
pub trait SetAssociativeCacheContext {
    /// The key type used for lookups.
    type Key: Copy + Eq;

    /// The value type stored in the cache.
    type Value: Copy;

    /// Extracts the key from a cached value.
    fn key_from_value(value: &Self::Value) -> Self::Key;

    /// Computes a hash of the given key.
    fn hash(key: Self::Key) -> u64;
}

/// Tracks cache performance statistics using interior mutability.
#[derive(Debug)]
pub struct Metrics {
    /// Count of cache lookups that found the requested key.
    hits: Cell<u64>,
    /// Count of cache lookups that did not find the requested key.
    misses: Cell<u64>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            hits: Cell::new(0),
            misses: Cell::new(0),
        }
    }
}

impl Metrics {
    /// Resets all counters to zero.
    #[inline]
    pub fn reset(&self) {
        self.hits.set(0);
        self.misses.set(0);
    }

    /// Returns the number of cache hits since the last reset.
    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits.get()
    }

    /// Returns the number of cache misses since the last reset.
    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses.get()
    }
}

/// A little simpler than `PackedIntArray` in the standard library, restricted to
/// little-endian 64-bit words and using words exactly without padding.
#[derive(Debug)]
pub struct PackedUnsignedIntegerArray<const BITS: usize> {
    words: Box<[u64]>,
}

impl<const BITS: usize> PackedUnsignedIntegerArray<BITS> {
    const WORD_BITS: usize = 64;

    #[inline]
    const fn uints_per_word() -> usize {
        Self::WORD_BITS / BITS
    }

    #[inline]
    const fn mask_value() -> u64 {
        (1u64 << BITS) - 1
    }

    #[inline]
    pub const fn words_for_len(len: usize) -> usize {
        let bits = len * BITS;
        bits.div_ceil(Self::WORD_BITS)
    }

    pub fn new_zeroed(words_len: usize) -> Self {
        const { assert!(cfg!(target_endian = "little")) };
        assert!(BITS < 8);
        assert!(BITS.is_power_of_two());
        assert!(Self::WORD_BITS.is_multiple_of(BITS));
        Self {
            words: vec![0u64; words_len].into_boxed_slice(),
        }
    }

    pub fn from_words(words: Vec<u64>) -> Self {
        const { assert!(cfg!(target_endian = "little")) };
        assert!(BITS < 8);
        assert!(BITS.is_power_of_two());
        assert!(Self::WORD_BITS.is_multiple_of(BITS));
        Self {
            words: words.into_boxed_slice(),
        }
    }

    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    #[inline]
    pub fn get(&self, index: u64) -> u8 {
        let uints_per_word = Self::uints_per_word() as u64;
        let word_index = index / uints_per_word;
        let within = index % uints_per_word;
        let shift = (within as usize) * BITS;
        debug_assert!(word_index < self.words.len() as u64);
        let word = self.words[word_index as usize];
        ((word >> shift) & Self::mask_value()) as u8
    }

    #[inline]
    pub fn set(&mut self, index: u64, value: u8) {
        debug_assert!((value as u64) <= Self::mask_value());
        let uints_per_word = Self::uints_per_word() as u64;
        let word_index = index / uints_per_word;
        let within = index % uints_per_word;
        let shift = (within as usize) * BITS;
        let mask = Self::mask_value() << shift;
        assert!(word_index < self.words.len() as u64);
        let w = &mut self.words[word_index as usize];
        *w = (*w & !mask) | ((value as u64) << shift);
    }
}

#[cfg(test)]
mod tests {
    use super::PackedUnsignedIntegerArray;
    use proptest::prelude::*;

    #[test]
    fn packed_unsigned_integer_array_unit() {
        let mut array =
            PackedUnsignedIntegerArray::<2>::from_words(vec![0, 0b10110010, 0, 0, 0, 0, 0, 0]);

        assert_eq!(0b10, array.get(32));
        assert_eq!(0b00, array.get(32 + 1));
        assert_eq!(0b11, array.get(32 + 2));
        assert_eq!(0b10, array.get(32 + 3));

        array.set(0, 0b01);
        assert_eq!(0b00000001u64, array.words()[0]);
        assert_eq!(0b01, array.get(0));
        array.set(1, 0b10);
        assert_eq!(0b00001001u64, array.words()[0]);
        assert_eq!(0b10, array.get(1));
        array.set(2, 0b11);
        assert_eq!(0b00111001u64, array.words()[0]);
        assert_eq!(0b11, array.get(2));
        array.set(3, 0b11);
        assert_eq!(0b11111001u64, array.words()[0]);
        assert_eq!(0b11, array.get(3));
        array.set(3, 0b01);
        assert_eq!(0b01111001u64, array.words()[0]);
        assert_eq!(0b01, array.get(3));
        array.set(3, 0b00);
        assert_eq!(0b00111001u64, array.words()[0]);
        assert_eq!(0b00, array.get(3));

        array.set(4, 0b11);
        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000001100111001u64,
            array.words()[0],
        );
        array.set(31, 0b11);
        assert_eq!(
            0b1100000000000000000000000000000000000000000000000000001100111001u64,
            array.words()[0],
        );
    }

    const LEN: usize = 1024;

    fn packed_unsigned_integer_array_case<const BITS: usize>(ops: &[(usize, u8)]) {
        let words_len = PackedUnsignedIntegerArray::<BITS>::words_for_len(LEN);
        let mut array = PackedUnsignedIntegerArray::<BITS>::new_zeroed(words_len);
        let mut reference = vec![0u8; LEN];

        for &(index, value) in ops {
            array.set(index as u64, value);
            reference[index] = value;

            for (i, &expected) in reference.iter().enumerate() {
                assert_eq!(expected, array.get(i as u64));
            }
        }
    }

    fn packed_unsigned_integer_array_ops<const BITS: usize>()
    -> impl Strategy<Value = Vec<(usize, u8)>> {
        let mask = ((1u16 << BITS) - 1) as u8;
        prop::collection::vec((0usize..LEN, 0u8..=mask), 0..512)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn packed_unsigned_integer_array_prop_u1(ops in packed_unsigned_integer_array_ops::<1>()) {
            packed_unsigned_integer_array_case::<1>(&ops);
        }

        #[test]
        fn packed_unsigned_integer_array_prop_u2(ops in packed_unsigned_integer_array_ops::<2>()) {
            packed_unsigned_integer_array_case::<2>(&ops);
        }

        #[test]
        fn packed_unsigned_integer_array_prop_u4(ops in packed_unsigned_integer_array_ops::<4>()) {
            packed_unsigned_integer_array_case::<4>(&ops);
        }
    }
}
