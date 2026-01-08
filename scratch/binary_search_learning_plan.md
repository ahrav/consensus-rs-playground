# Binary Search (TigerBeetle-style) First-Principles Learning Plan

## Sources of Truth
- `scratch/binary_search.rs`

## One-Sentence Mental Model
Binary search keeps a shrinking [offset, offset+length) window that preserves ordering invariants and ends with a final adjustment to return the upsert index.

## Glossary
- key: comparable value used for ordering.
- value: struct or scalar that contains or represents the key.
- key_from_value: function that extracts a key from a value.
- lower_bound: first index where key could be inserted (first duplicate).
- upper_bound: last index where key could be inserted (last duplicate).
- upsert index: exact match index if present, otherwise insertion index.
- offset + length: TigerBeetle search window representation.

## Canonical Semantics
Given a sorted slice, `binary_search_values_upsert_index` returns:
- If an exact match exists, an index of a matching element.
- Otherwise, the insertion index where the key should be inserted.
Duplicate handling:
- LowerBound returns the first matching index.
- UpperBound returns the last matching index.
`binary_search_values` and `binary_search_keys` add exact-match checks on top of the upsert index.
Range helpers use two upsert searches to compute an inclusive range.

## Key Invariants
Let `offset` and `length` define the current window `[offset, offset+length)`.

Lower bound:
- if offset > 0, key(values[offset-1]) < key
- if offset+length < len, key <= key(values[offset+length])

Upper bound:
- if offset > 0, key(values[offset-1]) <= key
- if offset+length < len, key < key(values[offset+length])

These invariants are asserted when `constants::VERIFY` is enabled.

## Phase 0: Core Upsert Search (No Prefetch, No Ranges)
Goal:
- Implement the offset+length search and exact-match helpers.
- Deviation from reference: Config has no prefetch flag; range helpers and tests are omitted.

Intuition:
- Keep a valid window and halve it until only one candidate remains.

Algorithm:
- If the slice is empty, return 0.
- Maintain `offset` and `length` for the current window.
- Each iteration: choose a half based on lower or upper bound rules, then shrink length.
- Apply the final adjustment when `key(values[offset]) < key`.

Code (full listing for this phase):
```rust
// binary_search.rs
#![allow(dead_code)]

use crate::constants;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Mode {
    LowerBound,
    UpperBound,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Config {
    pub mode: Mode,
}

impl Default for Config {
    #[inline(always)]
    fn default() -> Self {
        Self {
            mode: Mode::LowerBound,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinarySearchResult {
    pub index: u32,
    pub exact: bool,
}

#[inline(always)]
fn as_u32(x: usize) -> u32 {
    debug_assert!(x <= u32::MAX as usize);
    x as u32
}

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
    match config.mode {
        Mode::LowerBound => binary_search_values_upsert_index_impl::<false, _, _, _>(
            values,
            key,
            key_from_value,
        ),
        Mode::UpperBound => binary_search_values_upsert_index_impl::<true, _, _, _>(
            values,
            key,
            key_from_value,
        ),
    }
}

#[inline(always)]
fn binary_search_values_upsert_index_impl<const UPPER: bool, Key, Value, F>(
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
        if constants::VERIFY {
            // offset == 0 or:
            //   lower_bound: key(values[offset-1]) < key
            //   upper_bound: key(values[offset-1]) <= key
            if offset != 0 {
                let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
                if UPPER {
                    debug_assert!(left_key <= key);
                } else {
                    debug_assert!(left_key < key);
                }
            }

            // offset+length == len or:
            //   lower_bound: key <= key(values[offset+length])
            //   upper_bound: key < key(values[offset+length])
            if offset + length != values.len() {
                let right_key = key_from_value(unsafe { values.get_unchecked(offset + length) });
                if UPPER {
                    debug_assert!(key < right_key);
                } else {
                    debug_assert!(key <= right_key);
                }
            }
        }

        let half = length / 2;
        let mid = offset + half;

        // For exact matches:
        // - lower_bound takes first half  (mid_key < key)
        // - upper_bound takes second half (mid_key <= key)
        let mid_key = key_from_value(unsafe { values.get_unchecked(mid) });
        let take_upper_half = if UPPER { mid_key <= key } else { mid_key < key };

        if take_upper_half {
            // Zig uses @branchHint(.unpredictable) here.
            offset = mid;
        }

        length -= half;
    }

    if constants::VERIFY {
        debug_assert!(length == 1);

        if offset != 0 {
            let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
            if UPPER {
                debug_assert!(left_key <= key);
            } else {
                debug_assert!(left_key < key);
            }
        }

        if offset + length != values.len() {
            let right_key = key_from_value(unsafe { values.get_unchecked(offset + length) });
            if UPPER {
                debug_assert!(key < right_key);
            } else {
                debug_assert!(key <= right_key);
            }
        }
    }

    // Final adjustment exactly like Zig:
    // offset += intFromBool(key(values[offset]) < key)
    let last_key = key_from_value(unsafe { values.get_unchecked(offset) });
    if last_key < key {
        offset += 1;
    }

    if constants::VERIFY {
        if offset != 0 {
            let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
            if UPPER {
                debug_assert!(left_key <= key);
            } else {
                debug_assert!(left_key < key);
            }
        }

        // Zig: offset >= len-1 or:
        //   lower_bound: key <= key(values[offset+1])
        //   upper_bound: key <  key(values[offset+1])
        if offset + 1 < values.len() {
            let next_key = key_from_value(unsafe { values.get_unchecked(offset + 1) });
            if UPPER {
                debug_assert!(key < next_key);
            } else {
                debug_assert!(key <= next_key);
            }
        }

        // Zig: offset == len or key <= key(values[offset])
        if offset != values.len() {
            let at_key = key_from_value(unsafe { values.get_unchecked(offset) });
            debug_assert!(key <= at_key);
        }
    }

    as_u32(offset)
}

#[inline(always)]
pub fn binary_search_keys_upsert_index<Key>(keys: &[Key], key: Key, config: Config) -> u32
where
    Key: Ord + Copy,
{
    let key_from_key = |k: &Key| *k;
    binary_search_values_upsert_index(keys, key, config, &key_from_key)
}

#[inline(always)]
pub fn binary_search_values<Key, Value, F>(
    values: &[Value],
    key: Key,
    config: Config,
    key_from_value: &F,
) -> Option<&Value>
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
```

Example:
- keys: [0, 0, 3, 3, 3, 5, 5]
- key=3, LowerBound -> index 2, exact true

ASCII diagram:
```
values: 0 0 3 3 3 5 5
window: [offset........offset+length)
```

Exit criteria/tests:
- Results match a simple reference scan for both modes.
- Handles empty and single-element slices.

## Phase 1: Prefetch and Specialization
Goal:
- Add prefetch helpers and compile-time specialization by (mode, prefetch).
- Deviation from reference: range helpers and tests are omitted.

Intuition:
- Prefetch likely future reads to hide memory latency.

Algorithm:
- Add a `prefetch` flag to Config and dispatch to specialized impls.
- When PREFETCH is true, prefetch one-quarter and three-quarter positions.

Code (full listing for this phase):
```rust
// binary_search.rs
#![allow(dead_code)]

use core::mem::size_of;

use crate::constants;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Mode {
    LowerBound,
    UpperBound,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Config {
    pub mode: Mode,
    pub prefetch: bool,
}

impl Default for Config {
    #[inline(always)]
    fn default() -> Self {
        Self {
            mode: Mode::LowerBound,
            prefetch: true,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinarySearchResult {
    pub index: u32,
    pub exact: bool,
}

#[inline(always)]
const fn div_ceil_usize(numerator: usize, denominator: usize) -> usize {
    // denominator must be > 0
    if numerator == 0 {
        0
    } else {
        (numerator - 1) / denominator + 1
    }
}

#[inline(always)]
fn as_u32(x: usize) -> u32 {
    debug_assert!(x <= u32::MAX as usize);
    x as u32
}

#[inline(always)]
unsafe fn prefetch_read_data(addr: *const u8) {
    // Zig uses locality=0, read, data-cache.
    // On x86/x86_64, _MM_HINT_NTA is the closest "no temporal locality" hint.
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
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        let _ = addr;
    }
}

#[inline(always)]
unsafe fn prefetch_value<T>(ptr: *const T) {
    const LINES: usize = div_ceil_usize(size_of::<T>(), constants::CACHE_LINE_SIZE as usize);
    // If T is a ZST, size_of == 0, LINES == 0, loop does nothing.
    let base = ptr as *const u8;
    let cache_line_size = constants::CACHE_LINE_SIZE as usize;

    let mut i = 0usize;
    while i < LINES {
        prefetch_read_data(base.add(i * cache_line_size));
        i += 1;
    }
}

/// Port of Zig:
/// binary_search_values_upsert_index(Key, Value, key_from_value, values, key, config) -> u32
///
/// Returns either:
/// - an index of a value equal to `key`, or
/// - if no exact match exists, the insertion index for `key`.
///
/// Duplicates:
/// - Mode::LowerBound => first matching index
/// - Mode::UpperBound => last matching index
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
    // Mirror Zig's comptime config: specialize on (mode, prefetch).
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
        if constants::VERIFY {
            // offset == 0 or:
            //   lower_bound: key(values[offset-1]) < key
            //   upper_bound: key(values[offset-1]) <= key
            if offset != 0 {
                let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
                if UPPER {
                    debug_assert!(left_key <= key);
                } else {
                    debug_assert!(left_key < key);
                }
            }

            // offset+length == len or:
            //   lower_bound: key <= key(values[offset+length])
            //   upper_bound: key < key(values[offset+length])
            if offset + length != values.len() {
                let right_key = key_from_value(unsafe { values.get_unchecked(offset + length) });
                if UPPER {
                    debug_assert!(key < right_key);
                } else {
                    debug_assert!(key <= right_key);
                }
            }
        }

        let half = length / 2;

        if PREFETCH {
            // Prefetch one quarter and three quarters, same as Zig.
            let one_quarter = offset + (half / 2);
            let three_quarters = one_quarter + half;

            if constants::VERIFY {
                debug_assert!(one_quarter < values.len());
                debug_assert!(three_quarters < values.len());
            }

            unsafe {
                prefetch_value::<Value>(values.as_ptr().add(one_quarter));
                prefetch_value::<Value>(values.as_ptr().add(three_quarters));
            }
        }

        let mid = offset + half;

        // For exact matches:
        // - lower_bound takes first half  (mid_key < key)
        // - upper_bound takes second half (mid_key <= key)
        let mid_key = key_from_value(unsafe { values.get_unchecked(mid) });
        let take_upper_half = if UPPER { mid_key <= key } else { mid_key < key };

        if take_upper_half {
            // Zig uses @branchHint(.unpredictable) here.
            offset = mid;
        }

        length -= half;
    }

    if constants::VERIFY {
        debug_assert!(length == 1);

        if offset != 0 {
            let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
            if UPPER {
                debug_assert!(left_key <= key);
            } else {
                debug_assert!(left_key < key);
            }
        }

        if offset + length != values.len() {
            let right_key = key_from_value(unsafe { values.get_unchecked(offset + length) });
            if UPPER {
                debug_assert!(key < right_key);
            } else {
                debug_assert!(key <= right_key);
            }
        }
    }

    // Final adjustment exactly like Zig:
    // offset += intFromBool(key(values[offset]) < key)
    let last_key = key_from_value(unsafe { values.get_unchecked(offset) });
    if last_key < key {
        offset += 1;
    }

    if constants::VERIFY {
        if offset != 0 {
            let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
            if UPPER {
                debug_assert!(left_key <= key);
            } else {
                debug_assert!(left_key < key);
            }
        }

        // Zig: offset >= len-1 or:
        //   lower_bound: key <= key(values[offset+1])
        //   upper_bound: key <  key(values[offset+1])
        if offset + 1 < values.len() {
            let next_key = key_from_value(unsafe { values.get_unchecked(offset + 1) });
            if UPPER {
                debug_assert!(key < next_key);
            } else {
                debug_assert!(key <= next_key);
            }
        }

        // Zig: offset == len or key <= key(values[offset])
        if offset != values.len() {
            let at_key = key_from_value(unsafe { values.get_unchecked(offset) });
            debug_assert!(key <= at_key);
        }
    }

    as_u32(offset)
}

#[inline(always)]
pub fn binary_search_keys_upsert_index<Key>(keys: &[Key], key: Key, config: Config) -> u32
where
    Key: Ord + Copy,
{
    let key_from_key = |k: &Key| *k;
    binary_search_values_upsert_index(keys, key, config, &key_from_key)
}

#[inline(always)]
pub fn binary_search_values<Key, Value, F>(
    values: &[Value],
    key: Key,
    config: Config,
    key_from_value: &F,
) -> Option<&Value>
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
```

Example:
- Same index results as Phase 0; prefetch only affects performance.

ASCII diagram:
```
[...............mid...............]
      ^                     ^
  one quarter         three quarters
```

Exit criteria/tests:
- Output matches Phase 0 for both modes and both prefetch settings.
- Prefetch is a no-op on non-x86 targets.

## Phase 2: Range Queries
Goal:
- Add range upsert helpers and inclusive range computation.
- Deviation from reference: tests are omitted.

Intuition:
- Two upsert searches (min and max) define the inclusive slice.

Algorithm:
- start = lower_bound(key_min)
- end = upper_bound(key_max) on the tail starting at start
- If end points at an exact key_max, count includes it

Code (full listing for this phase):
```rust
// binary_search.rs
#![allow(dead_code)]

use core::mem::size_of;

use crate::constants;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Mode {
    LowerBound,
    UpperBound,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Config {
    pub mode: Mode,
    pub prefetch: bool,
}

impl Default for Config {
    #[inline(always)]
    fn default() -> Self {
        Self {
            mode: Mode::LowerBound,
            prefetch: true,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinarySearchResult {
    pub index: u32,
    pub exact: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinarySearchRangeUpsertIndexes {
    pub start: u32,
    pub end: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinarySearchRange {
    pub start: u32,
    pub count: u32,
}

#[inline(always)]
const fn div_ceil_usize(numerator: usize, denominator: usize) -> usize {
    // denominator must be > 0
    if numerator == 0 {
        0
    } else {
        (numerator - 1) / denominator + 1
    }
}

#[inline(always)]
fn as_u32(x: usize) -> u32 {
    debug_assert!(x <= u32::MAX as usize);
    x as u32
}

#[inline(always)]
unsafe fn prefetch_read_data(addr: *const u8) {
    // Zig uses locality=0, read, data-cache.
    // On x86/x86_64, _MM_HINT_NTA is the closest "no temporal locality" hint.
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
        // prfm pldl1strm = streaming (no temporal locality), read, data.
        asm!(
            "prfm pldl1strm, [{0}]",
            in(reg) addr,
            options(nostack, preserves_flags)
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        let _ = addr;
    }
}

#[inline(always)]
unsafe fn prefetch_value<T>(ptr: *const T) {
    const LINES: usize = div_ceil_usize(size_of::<T>(), constants::CACHE_LINE_SIZE as usize);
    // If T is a ZST, size_of == 0, LINES == 0, loop does nothing.
    let base = ptr as *const u8;
    let cache_line_size = constants::CACHE_LINE_SIZE as usize;

    let mut i = 0usize;
    while i < LINES {
        prefetch_read_data(base.add(i * cache_line_size));
        i += 1;
    }
}

/// Port of Zig:
/// binary_search_values_upsert_index(Key, Value, key_from_value, values, key, config) -> u32
///
/// Returns either:
/// - an index of a value equal to `key`, or
/// - if no exact match exists, the insertion index for `key`.
///
/// Duplicates:
/// - Mode::LowerBound => first matching index
/// - Mode::UpperBound => last matching index
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
    // Mirror Zig's comptime config: specialize on (mode, prefetch).
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
        if constants::VERIFY {
            // offset == 0 or:
            //   lower_bound: key(values[offset-1]) < key
            //   upper_bound: key(values[offset-1]) <= key
            if offset != 0 {
                let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
                if UPPER {
                    debug_assert!(left_key <= key);
                } else {
                    debug_assert!(left_key < key);
                }
            }

            // offset+length == len or:
            //   lower_bound: key <= key(values[offset+length])
            //   upper_bound: key < key(values[offset+length])
            if offset + length != values.len() {
                let right_key = key_from_value(unsafe { values.get_unchecked(offset + length) });
                if UPPER {
                    debug_assert!(key < right_key);
                } else {
                    debug_assert!(key <= right_key);
                }
            }
        }

        let half = length / 2;

        if PREFETCH {
            // Prefetch one quarter and three quarters, same as Zig.
            let one_quarter = offset + (half / 2);
            let three_quarters = one_quarter + half;

            if constants::VERIFY {
                debug_assert!(one_quarter < values.len());
                debug_assert!(three_quarters < values.len());
            }

            unsafe {
                prefetch_value::<Value>(values.as_ptr().add(one_quarter));
                prefetch_value::<Value>(values.as_ptr().add(three_quarters));
            }
        }

        let mid = offset + half;

        // For exact matches:
        // - lower_bound takes first half  (mid_key < key)
        // - upper_bound takes second half (mid_key <= key)
        let mid_key = key_from_value(unsafe { values.get_unchecked(mid) });
        let take_upper_half = if UPPER { mid_key <= key } else { mid_key < key };

        if take_upper_half {
            // Zig uses @branchHint(.unpredictable) here.
            offset = mid;
        }

        length -= half;
    }

    if constants::VERIFY {
        debug_assert!(length == 1);

        if offset != 0 {
            let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
            if UPPER {
                debug_assert!(left_key <= key);
            } else {
                debug_assert!(left_key < key);
            }
        }

        if offset + length != values.len() {
            let right_key = key_from_value(unsafe { values.get_unchecked(offset + length) });
            if UPPER {
                debug_assert!(key < right_key);
            } else {
                debug_assert!(key <= right_key);
            }
        }
    }

    // Final adjustment exactly like Zig:
    // offset += intFromBool(key(values[offset]) < key)
    let last_key = key_from_value(unsafe { values.get_unchecked(offset) });
    if last_key < key {
        offset += 1;
    }

    if constants::VERIFY {
        if offset != 0 {
            let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
            if UPPER {
                debug_assert!(left_key <= key);
            } else {
                debug_assert!(left_key < key);
            }
        }

        // Zig: offset >= len-1 or:
        //   lower_bound: key <= key(values[offset+1])
        //   upper_bound: key <  key(values[offset+1])
        if offset + 1 < values.len() {
            let next_key = key_from_value(unsafe { values.get_unchecked(offset + 1) });
            if UPPER {
                debug_assert!(key < next_key);
            } else {
                debug_assert!(key <= next_key);
            }
        }

        // Zig: offset == len or key <= key(values[offset])
        if offset != values.len() {
            let at_key = key_from_value(unsafe { values.get_unchecked(offset) });
            debug_assert!(key <= at_key);
        }
    }

    as_u32(offset)
}

#[inline(always)]
pub fn binary_search_keys_upsert_index<Key>(keys: &[Key], key: Key, config: Config) -> u32
where
    Key: Ord + Copy,
{
    let key_from_key = |k: &Key| *k;
    binary_search_values_upsert_index(keys, key, config, &key_from_key)
}

#[inline(always)]
pub fn binary_search_values<Key, Value, F>(
    values: &[Value],
    key: Key,
    config: Config,
    key_from_value: &F,
) -> Option<&Value>
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

    let end_usize = (start as usize) + (end_rel as usize);
    BinarySearchRangeUpsertIndexes {
        start,
        end: as_u32(end_usize),
    }
}

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

    // Zig:
    // if (start == values.len) return { start = start -| 1, count = 0 }
    if upsert.start == len_u32 {
        return BinarySearchRange {
            start: upsert.start.saturating_sub(1),
            count: 0,
        };
    }

    // Zig:
    // inclusive = (end < len && key_max == key(values[end]))
    let end = upsert.end as usize;
    let inclusive: u32 = if end < values.len() && key_from_value(&values[end]) == key_max {
        1
    } else {
        0
    };

    BinarySearchRange {
        start: upsert.start,
        count: (upsert.end - upsert.start) + inclusive,
    }
}

#[inline(always)]
pub fn binary_search_keys_range<Key>(keys: &[Key], key_min: Key, key_max: Key) -> BinarySearchRange
where
    Key: Ord + Copy,
{
    let key_from_key = |k: &Key| *k;
    binary_search_values_range(keys, key_min, key_max, &key_from_key)
}
```

Example:
- keys: [3, 4, 10, 15, 20, 25, 30, 100, 1000]
- key_min=15, key_max=100 -> start=3, count=5

ASCII diagram:
```
values: 3 4 10 15 20 25 30 100 1000
          ^-----------range-----------^
start=3, count=5
```

Exit criteria/tests:
- Slicing with start/count yields the expected inclusive range.
- Handles key_min == key_max and out-of-range intervals.

## Phase 3: Tests and Final Form
Goal:
- Add tests and match the reference implementation exactly.

Intuition:
- Lock in duplicate semantics and range behavior with focused tests.

Algorithm:
- Port the duplicate and range tests directly from the Rust file.

Code (full listing for this phase; matches `scratch/binary_search.rs`):
```rust
// binary_search.rs
#![allow(dead_code)]

use core::mem::size_of;

use crate::constants;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Mode {
    LowerBound,
    UpperBound,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Config {
    pub mode: Mode,
    pub prefetch: bool,
}

impl Default for Config {
    #[inline(always)]
    fn default() -> Self {
        Self {
            mode: Mode::LowerBound,
            prefetch: true,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinarySearchResult {
    pub index: u32,
    pub exact: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinarySearchRangeUpsertIndexes {
    pub start: u32,
    pub end: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BinarySearchRange {
    pub start: u32,
    pub count: u32,
}

#[inline(always)]
const fn div_ceil_usize(numerator: usize, denominator: usize) -> usize {
    // denominator must be > 0
    if numerator == 0 {
        0
    } else {
        (numerator - 1) / denominator + 1
    }
}

#[inline(always)]
fn as_u32(x: usize) -> u32 {
    debug_assert!(x <= u32::MAX as usize);
    x as u32
}

#[inline(always)]
unsafe fn prefetch_read_data(addr: *const u8) {
    // Zig uses locality=0, read, data-cache.
    // On x86/x86_64, _MM_HINT_NTA is the closest "no temporal locality" hint.
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
        // prfm pldl1strm = streaming (no temporal locality), read, data.
        asm!(
            "prfm pldl1strm, [{0}]",
            in(reg) addr,
            options(nostack, preserves_flags)
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        let _ = addr;
    }
}

#[inline(always)]
unsafe fn prefetch_value<T>(ptr: *const T) {
    const LINES: usize = div_ceil_usize(size_of::<T>(), constants::CACHE_LINE_SIZE as usize);
    // If T is a ZST, size_of == 0, LINES == 0, loop does nothing.
    let base = ptr as *const u8;
    let cache_line_size = constants::CACHE_LINE_SIZE as usize;

    let mut i = 0usize;
    while i < LINES {
        prefetch_read_data(base.add(i * cache_line_size));
        i += 1;
    }
}

/// Port of Zig:
/// binary_search_values_upsert_index(Key, Value, key_from_value, values, key, config) -> u32
///
/// Returns either:
/// - an index of a value equal to `key`, or
/// - if no exact match exists, the insertion index for `key`.
///
/// Duplicates:
/// - Mode::LowerBound => first matching index
/// - Mode::UpperBound => last matching index
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
    // Mirror Zig's comptime config: specialize on (mode, prefetch).
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
        if constants::VERIFY {
            // offset == 0 or:
            //   lower_bound: key(values[offset-1]) < key
            //   upper_bound: key(values[offset-1]) <= key
            if offset != 0 {
                let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
                if UPPER {
                    debug_assert!(left_key <= key);
                } else {
                    debug_assert!(left_key < key);
                }
            }

            // offset+length == len or:
            //   lower_bound: key <= key(values[offset+length])
            //   upper_bound: key < key(values[offset+length])
            if offset + length != values.len() {
                let right_key = key_from_value(unsafe { values.get_unchecked(offset + length) });
                if UPPER {
                    debug_assert!(key < right_key);
                } else {
                    debug_assert!(key <= right_key);
                }
            }
        }

        let half = length / 2;

        if PREFETCH {
            // Prefetch one quarter and three quarters, same as Zig.
            let one_quarter = offset + (half / 2);
            let three_quarters = one_quarter + half;

            if constants::VERIFY {
                debug_assert!(one_quarter < values.len());
                debug_assert!(three_quarters < values.len());
            }

            unsafe {
                prefetch_value::<Value>(values.as_ptr().add(one_quarter));
                prefetch_value::<Value>(values.as_ptr().add(three_quarters));
            }
        }

        let mid = offset + half;

        // For exact matches:
        // - lower_bound takes first half  (mid_key < key)
        // - upper_bound takes second half (mid_key <= key)
        let mid_key = key_from_value(unsafe { values.get_unchecked(mid) });
        let take_upper_half = if UPPER { mid_key <= key } else { mid_key < key };

        if take_upper_half {
            // Zig uses @branchHint(.unpredictable) here.
            offset = mid;
        }

        length -= half;
    }

    if constants::VERIFY {
        debug_assert!(length == 1);

        if offset != 0 {
            let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
            if UPPER {
                debug_assert!(left_key <= key);
            } else {
                debug_assert!(left_key < key);
            }
        }

        if offset + length != values.len() {
            let right_key = key_from_value(unsafe { values.get_unchecked(offset + length) });
            if UPPER {
                debug_assert!(key < right_key);
            } else {
                debug_assert!(key <= right_key);
            }
        }
    }

    // Final adjustment exactly like Zig:
    // offset += intFromBool(key(values[offset]) < key)
    let last_key = key_from_value(unsafe { values.get_unchecked(offset) });
    if last_key < key {
        offset += 1;
    }

    if constants::VERIFY {
        if offset != 0 {
            let left_key = key_from_value(unsafe { values.get_unchecked(offset - 1) });
            if UPPER {
                debug_assert!(left_key <= key);
            } else {
                debug_assert!(left_key < key);
            }
        }

        // Zig: offset >= len-1 or:
        //   lower_bound: key <= key(values[offset+1])
        //   upper_bound: key <  key(values[offset+1])
        if offset + 1 < values.len() {
            let next_key = key_from_value(unsafe { values.get_unchecked(offset + 1) });
            if UPPER {
                debug_assert!(key < next_key);
            } else {
                debug_assert!(key <= next_key);
            }
        }

        // Zig: offset == len or key <= key(values[offset])
        if offset != values.len() {
            let at_key = key_from_value(unsafe { values.get_unchecked(offset) });
            debug_assert!(key <= at_key);
        }
    }

    as_u32(offset)
}

#[inline(always)]
pub fn binary_search_keys_upsert_index<Key>(keys: &[Key], key: Key, config: Config) -> u32
where
    Key: Ord + Copy,
{
    let key_from_key = |k: &Key| *k;
    binary_search_values_upsert_index(keys, key, config, &key_from_key)
}

#[inline(always)]
pub fn binary_search_values<Key, Value, F>(
    values: &[Value],
    key: Key,
    config: Config,
    key_from_value: &F,
) -> Option<&Value>
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

    let end_usize = (start as usize) + (end_rel as usize);
    BinarySearchRangeUpsertIndexes {
        start,
        end: as_u32(end_usize),
    }
}

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

    // Zig:
    // if (start == values.len) return { start = start -| 1, count = 0 }
    if upsert.start == len_u32 {
        return BinarySearchRange {
            start: upsert.start.saturating_sub(1),
            count: 0,
        };
    }

    // Zig:
    // inclusive = (end < len && key_max == key(values[end]))
    let end = upsert.end as usize;
    let inclusive: u32 = if end < values.len() && key_from_value(&values[end]) == key_max {
        1
    } else {
        0
    };

    BinarySearchRange {
        start: upsert.start,
        count: (upsert.end - upsert.start) + inclusive,
    }
}

#[inline(always)]
pub fn binary_search_keys_range<Key>(keys: &[Key], key_min: Key, key_max: Key) -> BinarySearchRange
where
    Key: Ord + Copy,
{
    let key_from_key = |k: &Key| *k;
    binary_search_values_range(keys, key_min, key_max, &key_from_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicates_lower_upper() {
        let keys = [0u32, 0, 3, 3, 3, 5, 5, 5, 5];

        // lower_bound
        let cfg = Config {
            mode: Mode::LowerBound,
            prefetch: false,
        };
        assert_eq!(
            binary_search_keys(&keys, 0, cfg),
            BinarySearchResult {
                index: 0,
                exact: true
            }
        );
        assert_eq!(
            binary_search_keys(&keys, 3, cfg),
            BinarySearchResult {
                index: 2,
                exact: true
            }
        );
        assert_eq!(
            binary_search_keys(&keys, 5, cfg),
            BinarySearchResult {
                index: 5,
                exact: true
            }
        );
        assert_eq!(
            binary_search_keys(&keys, 1, cfg),
            BinarySearchResult {
                index: 2,
                exact: false
            }
        );

        // upper_bound (returns last duplicate index when exact)
        let cfg = Config {
            mode: Mode::UpperBound,
            prefetch: false,
        };
        assert_eq!(
            binary_search_keys(&keys, 0, cfg),
            BinarySearchResult {
                index: 1,
                exact: true
            }
        );
        assert_eq!(
            binary_search_keys(&keys, 3, cfg),
            BinarySearchResult {
                index: 4,
                exact: true
            }
        );
        assert_eq!(
            binary_search_keys(&keys, 5, cfg),
            BinarySearchResult {
                index: 8,
                exact: true
            }
        );
        assert_eq!(
            binary_search_keys(&keys, 1, cfg),
            BinarySearchResult {
                index: 2,
                exact: false
            }
        );
    }

    #[test]
    fn range_search_basic() {
        let keys = [3u32, 4, 10, 15, 20, 25, 30, 100, 1000];

        // key_min=15, key_max=100 => [15,20,25,30,100] count=5, start=3
        let r = binary_search_keys_range(&keys, 15, 100);
        assert_eq!(r, BinarySearchRange { start: 3, count: 5 });

        let slice = &keys[r.start as usize..][..r.count as usize];
        assert_eq!(slice, &[15, 20, 25, 30, 100]);
    }
}
```

Example:
- duplicates_lower_upper validates both bounds over duplicates.
- range_search_basic validates inclusive range extraction.

ASCII diagram:
```
tests -> expected indexes and ranges over fixed arrays
```

Exit criteria/tests:
- All tests in the module pass.

## How the Phases Fit Together
- Phase 0 establishes the core upsert index and exact-match wrappers.
- Phase 1 adds prefetch support and compile-time specialization.
- Phase 2 composes range queries from two upsert searches.
- Phase 3 adds tests and matches the final reference code.

## Proof Sketch or Correctness Notes
- The window invariants hold initially and are preserved by each halving step.
- The final adjustment moves right only when the last candidate is strictly less than the key.
- For UpperBound, the `<=` comparison selects the last duplicate index.

## Performance Notes
- Prefetch targets quarter points to overlap memory latency with comparison work.
- The offset+length loop keeps a predictable structure for branch hints.
- `constants::CACHE_LINE_SIZE` controls prefetch line stepping.

## Pitfalls and Edge Cases
- Empty slices must return 0 and skip unsafe indexing.
- Duplicate keys require different comparisons for lower vs upper bound.
- Range queries need inclusive handling when key_max is present.
- Prefetch should remain safe on non-x86 targets (no-op path).

## Final Checklist
- [ ] Matches reference semantics
- [ ] Each phase is self-sufficient
- [ ] Tests cover edge cases
- [ ] Final phase includes the complete code listing
