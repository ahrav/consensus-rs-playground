#![allow(dead_code)]

use std::cell::Cell;

/// Indicates whether an upsert operation updated an existing entry or inserted a new one.
///
/// When upserting a value into the cache, the cache must determine whether a matching
/// key already exists. This enum communicates the outcome to the caller, allowing them
/// to take different actions based on whether the value was newly inserted or an
/// existing entry was modified.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateOrInsert {
    /// An existing entry with the same key was found and its value was replaced.
    Update,
    /// No existing entry matched the key, so a new entry was created.
    Insert,
}

/// A partial hash stored alongside cached values to enable fast mismatch detection.
///
/// Tags are short hashes (typically 8 or 16 bits) that allow the cache to quickly
/// determine that a key does *not* match without performing an expensive full-key
/// comparison. When looking up a key, the cache first compares tags; only if the
/// tags match does it proceed to compare full keys.
///
/// # Collision Semantics
///
/// Tag collisions are possible: `tag(k1) == tag(k2)` does **not** imply `k1 == k2`.
/// However, this is acceptable because:
/// - Most lookups for absent keys will have non-matching tags, avoiding full comparisons
/// - Tags are 16-32x smaller than keys and stay hot in CPU cache
/// - The rare false positive just triggers one extra key comparison
///
/// # Implementors
///
/// Standard implementations are provided for [`u8`] (8-bit tags) and [`u16`] (16-bit tags).
/// Smaller tags use less memory but have higher collision rates; choose based on your
/// key distribution and performance requirements.
pub trait Tag: Copy + Eq + PartialEq + Default {
    /// The number of bits in this tag type.
    ///
    /// Used for compile-time layout calculations and determining how much entropy
    /// to extract from the hash.
    const BITS: usize;

    /// Extracts a tag from hash entropy.
    ///
    /// The input `entropy` is assumed to be a well-distributed hash value. This method
    /// truncates it to the tag's bit width. Since the hash is already uniformly
    /// distributed, a simple truncation preserves that distribution.
    fn truncate(entropy: u64) -> Self;
}

/// 8-bit tag implementation.
///
/// Provides the smallest memory footprint (1 byte per entry) at the cost of higher
/// collision probability (1/256 chance of false positive). Suitable for caches where
/// memory efficiency is paramount or key comparison is inexpensive.
impl Tag for u8 {
    const BITS: usize = 8;

    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u8
    }
}

/// 16-bit tag implementation.
///
/// Offers a good balance between memory usage (2 bytes per entry) and collision
/// resistance (1/65536 chance of false positive). Recommended for most use cases
/// where key comparison is expensive relative to the extra byte of storage.
impl Tag for u16 {
    const BITS: usize = 16;

    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u16
    }
}

/// Defines the key/value types and operations required by a set-associative cache.
///
/// This trait allows the cache to be generic over different key-value pairs while
/// providing the necessary operations for hashing and key extraction. Implementors
/// specify their domain-specific types and how to work with them.
///
/// # Design Rationale
///
/// The cache stores values but needs to look them up by key. Rather than storing
/// keys separately (which would double memory usage), this trait requires that
/// keys can be extracted from values. This is natural for many use cases where
/// values contain their own identifiers.
///
/// # Requirements
///
/// - `Key` must be `Copy + Eq` for efficient comparison and storage
/// - `Value` must be `Copy` for efficient cache operations
/// - The hash function must be deterministic (same key always produces same hash)
pub trait SetAssociativeCacheContext {
    /// The key type used for lookups.
    ///
    /// Keys identify cache entries and are used to determine which set an entry
    /// belongs to. Must be cheaply copyable and equality-comparable.
    type Key: Copy + Eq;

    /// The value type stored in the cache.
    ///
    /// Values contain the actual cached data. Must be cheaply copyable since the
    /// cache may return copies during eviction or lookup operations.
    type Value: Copy;

    /// Extracts the key from a cached value.
    ///
    /// This enables the cache to store only values while still being able to
    /// identify them by key. The implementation must ensure that the extracted
    /// key uniquely identifies the value for cache correctness.
    fn key_from_value(value: &Self::Value) -> Self::Key;

    /// Computes a hash of the given key.
    ///
    /// The hash is used to determine which set the key maps to and to generate
    /// the tag for fast mismatch detection. Must be deterministic: the same key
    /// must always produce the same hash value.
    ///
    /// A well-distributed hash function improves cache utilization by spreading
    /// entries evenly across sets.
    fn hash(key: Self::Key) -> u64;
}

/// Tracks cache performance statistics using interior mutability.
///
/// Metrics are updated during cache operations to measure hit rate and identify
/// potential tuning opportunities. Uses [`Cell`] for interior mutability, allowing
/// metrics to be updated through shared references without requiring `&mut self`.
///
/// # Thread Safety
///
/// This type is **not** thread-safe. The use of [`Cell`] provides interior mutability
/// for single-threaded contexts only. For concurrent access, wrap in appropriate
/// synchronization primitives or use atomic counters.
///
/// # Usage
///
/// Metrics are typically owned by or associated with a cache instance. After a
/// period of operation, inspect `hits()` and `misses()` to compute hit rate:
///
/// ```ignore
/// let hit_rate = metrics.hits() as f64 / (metrics.hits() + metrics.misses()) as f64;
/// ```
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
    ///
    /// Call this to start a fresh measurement period, for example after
    /// configuration changes or at regular intervals for time-windowed metrics.
    #[inline]
    pub fn reset(&self) {
        self.hits.set(0);
        self.misses.set(0);
    }

    /// Returns the number of cache hits since the last reset.
    ///
    /// A hit occurs when a lookup finds the requested key in the cache.
    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits.get()
    }

    /// Returns the number of cache misses since the last reset.
    ///
    /// A miss occurs when a lookup does not find the requested key in the cache.
    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses.get()
    }
}

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
