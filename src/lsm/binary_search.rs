//! Binary search implementation optimized for LSM tree operations.
//!
//! This module provides cache-friendly binary search functions designed for
//! sorted key-value stores, with architecture-specific prefetching optimizations.
//!
//! # Key Features
//!
//! - **Branchless implementation**: Eliminates branch mispredictions by using
//!   arithmetic instead of conditional jumps, critical for performance in
//!   tight loops over large datasets.
//!
//! - **Cache prefetching**: Optionally prefetches memory at the 1/4 and 3/4
//!   positions during each iteration, reducing cache misses on large arrays.
//!
//! - **Lower/upper bound modes**: Supports both bound types for range queries
//!   common in LSM tree operations.
//!
//! # When to Use
//!
//! Prefer this over [`slice::binary_search`] when:
//! - Searching large, sorted arrays where cache misses dominate
//! - Performing range queries that need both lower and upper bounds
//! - Working with LSM tree data structures
//!
//! # Example
//!
//! ```no_run
//! use consensus::lsm::binary_search::{binary_search_keys, Config};
//!
//! let keys = [1, 3, 5, 7, 9];
//! let result = binary_search_keys(&keys, 5, Config::default());
//! assert!(result.exact);
//! assert_eq!(result.index, 2);
//! ```

#![allow(dead_code)]

use crate::constants;

/// Search mode controlling behavior when duplicate keys exist or key is not found.
///
/// These modes determine which position is returned when a key appears
/// multiple times in the sorted array, or where an insertion should occur.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Mode {
    /// Returns the index of the first element not less than the key.
    ///
    /// For duplicates, returns the leftmost matching position.
    /// For non-existent keys, returns the insertion point.
    ///
    /// ```text
    /// keys:   [1, 3, 3, 3, 5]
    /// search for 3 -> index 1 (first 3)
    /// search for 2 -> index 1 (insertion point)
    /// search for 6 -> index 5 (past end)
    /// ```
    LowerBound,

    /// Returns the index of the rightmost matching element when the key exists.
    ///
    /// For duplicates, returns the rightmost matching position.
    /// For non-existent keys, returns the insertion point.
    ///
    /// ```text
    /// keys:   [1, 3, 3, 3, 5]
    /// search for 3 -> index 3 (last 3)
    /// search for 2 -> index 1 (insertion point)
    /// search for 6 -> index 5 (past end)
    /// ```
    UpperBound,
}

/// Configuration for binary search operations.
///
/// Controls the search mode and whether cache prefetching is enabled.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Config {
    /// The bound mode for the search. See [`Mode`] for details.
    pub mode: Mode,

    /// Whether to prefetch cache lines during search.
    ///
    /// When `true`, prefetches memory at the 1/4 and 3/4 positions of the
    /// current search range on each iteration. This hides memory latency
    /// for large arrays but adds overhead for small arrays.
    ///
    /// **Recommendation**: Enable for arrays larger than ~1000 elements.
    pub prefetch: bool,
}

impl Default for Config {
    #[inline(always)]
    fn default() -> Self {
        Config {
            mode: Mode::LowerBound,
            prefetch: true,
        }
    }
}

/// Result of a binary search operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinarySearchResult {
    /// Index where the key was found or should be inserted.
    ///
    /// If `exact` is `true`, this is the position of the matching element.
    /// If `exact` is `false`, this is the insertion point to maintain sort order.
    pub index: u32,

    /// Whether an exact match was found.
    pub exact: bool,
}

/// Index bounds for range operations, using upsert semantics.
///
/// Both `start` and `end` represent insertion points, not necessarily
/// positions of existing elements. This is useful for determining where
/// new elements would be inserted while maintaining sort order.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinarySearchRangeUpsertIndexes {
    /// Lower bound (inclusive): first position >= key_min.
    pub start: u32,

    /// Upper bound (exclusive): first position > key_max.
    pub end: u32,
}

/// A range of elements matching a key range query.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinarySearchRange {
    /// Starting index of matching elements.
    ///
    /// If `count` is 0, this is the position where `key_min` would be
    /// inserted, clamped to `len - 1` for non-empty arrays.
    pub start: u32,

    /// Number of elements in the range `[key_min, key_max]` (inclusive).
    pub count: u32,
}

/// Converts `usize` to `u32`, panicking in debug mode if out of range.
#[inline(always)]
fn as_u32(x: usize) -> u32 {
    debug_assert!(x <= u32::MAX as usize);
    x as u32
}

/// Prefetches a cache line into the CPU cache for reading.
///
/// Uses non-temporal hints (`NTA`) to minimize cache pollution, which is
/// appropriate for streaming access patterns where data is accessed once.
///
/// # Safety
///
/// - `addr` must be a valid pointer within mapped memory.
/// - On most architectures, prefetching an invalid address is silently ignored,
///   but this is not guaranteed.
///
/// # Platform Support
///
/// - **x86/x86_64**: Uses `_mm_prefetch` with `_MM_HINT_NTA`
/// - **aarch64**: Uses `prfm pldl1stream` instruction
/// - **Other**: No-op
#[inline(always)]
unsafe fn prefetch_read_data(addr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{_MM_HINT_NTA, _mm_prefetch};
        _mm_prefetch(addr as *const i8, _MM_HINT_NTA);
    }
    #[cfg(target_arch = "x86")]
    {
        use core::arch::x86::{_MM_HINT_NTA, _mm_prefetch};
        _mm_prefetch(addr as *const i8, _MM_HINT_NTA);
    }
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::asm;
        unsafe {
            asm!(
                "prfm pldl1stream, [{0}]",
                in(reg) addr,
                options(nostack, preserves_flags)
            )
        };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        let _ = addr;
    }
}

/// Prefetches all cache lines covering a value of type `T`.
///
/// For values spanning multiple cache lines, prefetches each line
/// sequentially from the start of the value.
///
/// # Safety
///
/// - `ptr` must point to valid (possibly uninitialized) memory of size `size_of::<T>()`.
/// - The memory region `[ptr, ptr + size_of::<T>())` must be mapped.
#[inline(always)]
unsafe fn prefetch_value<T>(ptr: *const T) {
    let lines: usize = (size_of::<T>()).div_ceil(constants::CACHE_LINE_SIZE as usize);
    let base = ptr as *const u8;
    let cache_line_size = constants::CACHE_LINE_SIZE as usize;

    for i in 0..lines {
        unsafe { prefetch_read_data(base.add(i * cache_line_size)) };
    }
}

/// Finds the insertion index for a key within sorted values.
///
/// Returns the index where `key` should be inserted to maintain sort order.
/// The exact position depends on the [`Mode`] configuration:
/// - [`Mode::LowerBound`]: First position where element >= key
/// - [`Mode::UpperBound`]: Rightmost match if any, otherwise insertion point
///
/// # Panics
///
/// Debug-panics if `values.len() > u32::MAX`.
#[inline(always)]
pub fn binary_search_values_upsert_index<Key, Value, F>(
    values: &[Value],
    key: Key,
    config: Config,
    key_from_value: &F,
) -> u32
where
    Key: Ord + Copy,
    F: Fn(&Value) -> Key,
{
    match (config.mode, config.prefetch) {
        (Mode::LowerBound, true) => binary_search_values_upsert_index_impl::<false, true, _, _, _>(
            values,
            key,
            key_from_value,
        ),
        (Mode::LowerBound, false) => {
            binary_search_values_upsert_index_impl::<false, false, _, _, _>(
                values,
                key,
                key_from_value,
            )
        }
        (Mode::UpperBound, true) => binary_search_values_upsert_index_impl::<true, true, _, _, _>(
            values,
            key,
            key_from_value,
        ),
        (Mode::UpperBound, false) => {
            binary_search_values_upsert_index_impl::<true, false, _, _, _>(
                values,
                key,
                key_from_value,
            )
        }
    }
}

/// Core branchless binary search implementation.
///
/// # Algorithm
///
/// Uses a branchless variant that eliminates conditional jumps by computing:
/// ```text
/// offset += half * (condition as usize)
/// ```
/// This converts the branch to an arithmetic operation, avoiding branch
/// misprediction penalties at the cost of slightly more work per iteration.
///
/// # Prefetch Strategy
///
/// When `PREFETCH` is true, prefetches at the 1/4 and 3/4 positions:
/// ```text
/// [.........|....mid....|.........]
///      ^                     ^
///    1/4                   3/4
/// ```
/// This anticipates the next iteration's access regardless of which half
/// is taken, hiding memory latency.
///
/// # Safety Invariants
///
/// Uses `get_unchecked` internally. Safety is guaranteed because:
/// - `mid = offset + half` where `half < length` and `offset + length <= values.len()`
/// - Loop invariant: `offset + length <= values.len()` holds at all times
/// - Final access at `offset` is valid because we reach it only when `length == 1`
#[inline(always)]
fn binary_search_values_upsert_index_impl<const UPPER: bool, const PREFETCH: bool, Key, Value, F>(
    values: &[Value],
    key: Key,
    key_from_value: &F,
) -> u32
where
    Key: Ord + Copy,
    F: Fn(&Value) -> Key,
{
    debug_assert!(values.len() <= u32::MAX as usize);

    if values.is_empty() {
        return 0;
    }

    let mut offset: usize = 0;
    let mut length: usize = values.len();

    while length > 1 {
        let half = length / 2;

        if PREFETCH {
            let one_quarter = offset + (half / 2);
            let three_quarters = one_quarter + half;

            unsafe {
                prefetch_value::<Value>(values.as_ptr().add(one_quarter));
                prefetch_value::<Value>(values.as_ptr().add(three_quarters));
            }
        }

        let mid = offset + half;

        let mid_key = key_from_value(unsafe { values.get_unchecked(mid) });

        let take_upper_half = if UPPER { mid_key <= key } else { mid_key < key };

        offset += half * (take_upper_half as usize);

        length -= half;
    }

    let last_key = key_from_value(unsafe { values.get_unchecked(offset) });
    if last_key < key {
        offset += 1;
    }

    as_u32(offset)
}

/// Finds the insertion index for a key within a sorted key slice.
///
/// Convenience wrapper around [`binary_search_values_upsert_index`] for
/// slices where the values are the keys themselves.
#[inline(always)]
pub fn binary_search_keys_upsert_index<Key>(keys: &[Key], key: Key, config: Config) -> u32
where
    Key: Ord + Copy,
{
    let key_from_key = |k: &Key| *k;
    binary_search_values_upsert_index(keys, key, config, &key_from_key)
}

/// Searches for a value by key, returning a reference if an exact match exists.
///
/// Returns `Some(&value)` if a value with the exact key is found,
/// or `None` if no match exists.
#[inline(always)]
pub fn binary_search_values<'a, Key, Value, F>(
    values: &'a [Value],
    key: Key,
    config: Config,
    key_from_value: &F,
) -> Option<&'a Value>
where
    Key: Ord + Copy,
    F: Fn(&Value) -> Key,
{
    let index = binary_search_values_upsert_index(values, key, config, key_from_value) as usize;

    if index < values.len() && key_from_value(&values[index]) == key {
        Some(&values[index])
    } else {
        None
    }
}

/// Searches for a key, returning both the index and whether an exact match was found.
///
/// Returns a [`BinarySearchResult`] with:
/// - `index`: Position where key was found or should be inserted
/// - `exact`: `true` if the key exists at that index
#[inline(always)]
pub fn binary_search_keys<Key>(keys: &[Key], key: Key, config: Config) -> BinarySearchResult
where
    Key: Ord + Copy,
{
    let index = binary_search_keys_upsert_index(keys, key, config);
    let i = index as usize;
    BinarySearchResult {
        index,
        exact: i < keys.len() && keys[i] == key,
    }
}

/// Finds insertion indexes for a key range within sorted values.
///
/// Returns [`BinarySearchRangeUpsertIndexes`] with:
/// - `start`: Lower bound (first position >= `key_min`)
/// - `end`: Upper bound (first position > `key_max`)
///
/// # Panics
///
/// Debug-panics if `key_min > key_max`.
#[inline(always)]
pub fn binary_search_values_range_upsert_indexes<Key, Value, F>(
    values: &[Value],
    key_min: Key,
    key_max: Key,
    key_from_value: &F,
) -> BinarySearchRangeUpsertIndexes
where
    Key: Ord + Copy,
    F: Fn(&Value) -> Key,
{
    debug_assert!(key_min <= key_max);

    let start = binary_search_values_upsert_index(
        values,
        key_min,
        Config {
            mode: Mode::LowerBound,
            ..Config::default()
        },
        key_from_value,
    );

    if start as usize == values.len() {
        return BinarySearchRangeUpsertIndexes { start, end: start };
    }

    let tail = &values[start as usize..];
    let end_rel = binary_search_values_upsert_index(
        tail,
        key_max,
        Config {
            mode: Mode::UpperBound,
            ..Config::default()
        },
        key_from_value,
    );

    let end_size = (start as usize) + (end_rel as usize);
    BinarySearchRangeUpsertIndexes {
        start,
        end: as_u32(end_size),
    }
}

/// Finds insertion indexes for a key range within a sorted key slice.
///
/// Convenience wrapper around [`binary_search_values_range_upsert_indexes`]
/// for slices where the values are the keys themselves.
#[inline(always)]
pub fn binary_search_keys_range_upsert_indexes<Key>(
    keys: &[Key],
    key_min: Key,
    key_max: Key,
) -> BinarySearchRangeUpsertIndexes
where
    Key: Ord + Copy,
{
    let key_from_key = |k: &Key| *k;
    binary_search_values_range_upsert_indexes(keys, key_min, key_max, &key_from_key)
}

/// Finds all values with keys in the inclusive range `[key_min, key_max]`.
///
/// Returns a [`BinarySearchRange`] with:
/// - `start`: Index of the first matching element
/// - `count`: Number of elements in the range (0 if none match)
///
/// This is the primary range query function for LSM tree lookups.
#[inline(always)]
pub fn binary_search_values_range<Key, Value, F>(
    values: &[Value],
    key_min: Key,
    key_max: Key,
    key_from_value: &F,
) -> BinarySearchRange
where
    Key: Ord + Copy,
    F: Fn(&Value) -> Key,
{
    let upsert =
        binary_search_values_range_upsert_indexes(values, key_min, key_max, key_from_value);
    let len_u32 = as_u32(values.len());

    if upsert.start == len_u32 {
        return BinarySearchRange {
            start: upsert.start.saturating_sub(1),
            count: 0,
        };
    }

    let end = upsert.end as usize;
    let inclusive = if end < values.len() && key_from_value(&values[end]) == key_max {
        1
    } else {
        0
    };

    BinarySearchRange {
        start: upsert.start,
        count: (upsert.end - upsert.start) + inclusive,
    }
}

/// Finds all keys in the inclusive range `[key_min, key_max]`.
///
/// Convenience wrapper around [`binary_search_values_range`] for slices
/// where the values are the keys themselves.
#[inline(always)]
pub fn binary_search_keys_range<Key>(keys: &[Key], key_min: Key, key_max: Key) -> BinarySearchRange
where
    Key: Ord + Copy,
{
    let key_from_key = |k: &Key| *k;
    binary_search_values_range(keys, key_min, key_max, &key_from_key)
}
