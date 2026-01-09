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
fn as_u32(x: usize) -> u32 {
    debug_assert!(x <= u32::MAX as usize);
    x as u32
}

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

#[inline(always)]
unsafe fn prefetch_value<T>(ptr: *const T) {
    let lines: usize = (size_of::<T>()).div_ceil(constants::CACHE_LINE_SIZE as usize);
    let base = ptr as *const u8;
    let cache_line_size = constants::CACHE_LINE_SIZE as usize;

    for i in 0..lines {
        unsafe { prefetch_read_data(base.add(i * cache_line_size)) };
    }
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

#[inline(always)]
pub fn binary_search_keys_upsert_index<Key>(keys: &[Key], key: Key, config: Config) -> u32
where
    Key: Ord + Copy,
{
    let key_from_key = |k: &Key| *k;
    binary_search_values_upsert_index(keys, key, config, &key_from_key)
}

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
