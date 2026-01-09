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
}
