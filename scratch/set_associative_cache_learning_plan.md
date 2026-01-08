# Set-Associative Cache (CLOCK Nth-Chance) First-Principles Learning Plan

## Sources of Truth
- `scratch/set_associative_cache.rs`
- https://github.com/tigerbeetle/tigerbeetle/blob/main/src/lsm/set_associative_cache.zig

## One-Sentence Mental Model
A key hashes to a fixed-size set, and a per-set clock hand decrements small counters until a slot reaches zero and can be reused.

## Glossary
- set: A group of WAYS slots that a key can map to.
- way: One slot inside a set.
- tag: Truncated hash bits used to filter candidate ways.
- count: Small per-slot counter that tracks recent access.
- clock hand: Per-set pointer that drives CLOCK Nth-Chance eviction.
- fastrange: Multiply-high mapping from hash to [0, sets).
- set offset: set_index * WAYS, base index into flat arrays.

## Canonical Semantics
- associate(key) -> (tag, set_index, set_offset) using tag = truncate(hash(key)) and set_index = fastrange(hash(key), sets).
- get_index(key): on hit, increment hits and saturating-increment the count; on miss, increment misses.
- get(key): returns a reference to the value if present (via get_index).
- upsert(value): update if present; otherwise walk the set clock hand, decrementing counts until a slot reaches 0, then evict and insert.
- remove(key): if present, set count to 0 and return the removed value.
- demote(key): if present, set count to 1 to reduce retention priority.
- reset(): clear tags, counts, clocks, and metrics; values are only valid when count > 0.

## Key Invariants
- count == 0 means the value is logically absent and must not be read.
- counts are in [0, max_count] and saturate on get.
- each clock hand is always in [0, WAYS - 1].
- tags are advisory; final equality is key_from_value(value) == key.
- set offset is set_index * WAYS and always within array bounds.
- value_count_max is a multiple of WAYS and VALUE_COUNT_MAX_MULTIPLE.

## Phase 0: Reference Spec (Simple and Correct)
Goal:
Build a correct, readable CLOCK Nth-Chance set-associative cache using simple storage.

Intuition:
Each set is a tiny array; eviction is local to that set and the clock hand cycles until a slot reaches count 0.

Algorithm:
1) Hash the key, truncate to a tag, and map to a set with fastrange.
2) Scan the set for tag matches and confirm equality by key_from_value.
3) On hit, increment metrics and saturating-increment the count.
4) On upsert miss, decrement counts as the clock hand advances until a count reaches 0, then insert.

Code (full listing for this phase):
```rust
#![allow(dead_code)]

use core::marker::PhantomData;
use std::cell::Cell;

#[inline]
pub fn fastrange(word: u64, p: u64) -> u64 {
    let ln = (word as u128).wrapping_mul(p as u128);
    (ln >> 64) as u64
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateOrInsert {
    Update,
    Insert,
}

pub trait Tag: Copy + Eq + Default {
    const BITS: usize;

    #[inline]
    fn truncate(entropy: u64) -> Self;
}

impl Tag for u8 {
    const BITS: usize = 8;
    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u8
    }
}

impl Tag for u16 {
    const BITS: usize = 16;
    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u16
    }
}

pub trait SetAssociativeCacheContext {
    type Key: Copy + Eq;
    type Value: Copy;

    #[inline]
    fn key_from_value(value: &Self::Value) -> Self::Key;

    #[inline]
    fn hash(key: Self::Key) -> u64;
}

#[derive(Debug)]
pub struct Metrics {
    hits: Cell<u64>,
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
    #[inline]
    pub fn reset(&self) {
        self.hits.set(0);
        self.misses.set(0);
    }

    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits.get()
    }

    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses.get()
    }
}

pub struct Options<'a> {
    pub name: &'a str,
}

pub struct SetAssociativeCache<C, TAG_T, const WAYS: usize, const CLOCK_BITS: usize>
where
    C: SetAssociativeCacheContext,
    TAG_T: Tag,
{
    name: String,
    sets: usize,
    metrics: Metrics,
    tags: Vec<TAG_T>,
    values: Vec<Option<C::Value>>,
    counts: Vec<Cell<u8>>,
    clocks: Vec<u8>,
    _marker: PhantomData<C>,
}

impl<C, TAG_T, const WAYS: usize, const CLOCK_BITS: usize>
    SetAssociativeCache<C, TAG_T, WAYS, CLOCK_BITS>
where
    C: SetAssociativeCacheContext,
    TAG_T: Tag,
{
    #[inline]
    fn max_count() -> u8 {
        debug_assert!(CLOCK_BITS <= 8);
        ((1u16 << CLOCK_BITS) - 1) as u8
    }

    #[inline]
    fn wrap_way(way: usize) -> usize {
        way & (WAYS - 1)
    }

    pub fn new(value_count_max: usize, options: Options<'_>) -> Self {
        match WAYS {
            2 | 4 | 16 => {}
            _ => panic!("WAYS must be 2, 4, or 16"),
        }
        match TAG_T::BITS {
            8 | 16 => {}
            _ => panic!("tag bits must be 8 or 16"),
        }
        match CLOCK_BITS {
            1 | 2 | 4 => {}
            _ => panic!("CLOCK_BITS must be 1, 2, or 4"),
        }

        assert!(WAYS.is_power_of_two());

        assert!(value_count_max > 0);
        assert!(value_count_max >= WAYS);
        assert!(value_count_max % WAYS == 0);

        let sets = value_count_max / WAYS;

        let tags = vec![TAG_T::default(); value_count_max];
        let values = vec![None; value_count_max];
        let counts = (0..value_count_max).map(|_| Cell::new(0)).collect();
        let clocks = vec![0u8; sets];

        let mut sac = Self {
            name: options.name.to_string(),
            sets,
            metrics: Metrics::default(),
            tags,
            values,
            counts,
            clocks,
            _marker: PhantomData,
        };

        sac.reset();
        sac
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    pub fn sets(&self) -> usize {
        self.sets
    }

    #[inline]
    pub fn metrics(&self) -> (u64, u64) {
        (self.metrics.hits(), self.metrics.misses())
    }

    pub fn reset(&mut self) {
        self.tags.fill(TAG_T::default());
        for c in &self.counts {
            c.set(0);
        }
        self.values.fill(None);
        self.clocks.fill(0);
        self.metrics.reset();
    }

    pub fn get_index(&self, key: C::Key) -> Option<usize> {
        let set = self.associate(key);
        if let Some(way) = self.search(set.offset, set.tag, key) {
            self.metrics.hits.set(self.metrics.hits.get() + 1);

            let idx = set.offset + way;
            let count = self.counts_get(idx);
            let next = if count == Self::max_count() {
                count
            } else {
                count + 1
            };
            self.counts_set(idx, next);
            Some(idx)
        } else {
            self.metrics.misses.set(self.metrics.misses.get() + 1);
            None
        }
    }

    pub fn get(&self, key: C::Key) -> Option<&C::Value> {
        let index = self.get_index(key)?;
        self.values[index].as_ref()
    }

    pub fn remove(&mut self, key: C::Key) -> Option<C::Value> {
        let set = self.associate(key);
        let way = self.search(set.offset, set.tag, key)?;

        let idx = set.offset + way;
        let removed = self.values[idx];
        self.values[idx] = None;
        self.counts_set(idx, 0);
        removed
    }

    pub fn demote(&mut self, key: C::Key) {
        let set = self.associate(key);
        let Some(way) = self.search(set.offset, set.tag, key) else { return; };
        let idx = set.offset + way;
        self.counts_set(idx, 1);
    }

    pub struct UpsertResult<V> {
        pub index: usize,
        pub updated: UpdateOrInsert,
        pub evicted: Option<V>,
    }

    pub fn upsert(&mut self, value: &C::Value) -> UpsertResult<C::Value> {
        let key = C::key_from_value(value);
        let set = self.associate(key);

        if let Some(way) = self.search(set.offset, set.tag, key) {
            let idx = set.offset + way;
            self.counts_set(idx, 1);
            let evicted = self.values[idx];
            self.values[idx] = Some(*value);
            return UpsertResult {
                index: idx,
                updated: UpdateOrInsert::Update,
                evicted,
            };
        }

        let clock_index = set.offset / WAYS;
        let mut way = self.clocks_get(clock_index) as usize;
        debug_assert!(way < WAYS);

        let max_count = Self::max_count() as usize;
        let clock_iterations_max = WAYS * (max_count.saturating_sub(1));

        let mut evicted: Option<C::Value> = None;
        let mut safety_count: usize = 0;

        while safety_count <= clock_iterations_max {
            let idx = set.offset + way;
            let mut count = self.counts_get(idx) as usize;
            if count == 0 {
                break;
            }

            count -= 1;
            self.counts_set(idx, count as u8);
            if count == 0 {
                evicted = self.values[idx];
                break;
            }

            safety_count += 1;
            way = Self::wrap_way(way + 1);
        }

        assert!(self.counts_get(set.offset + way) == 0);

        self.tags[set.offset + way] = set.tag;
        self.values[set.offset + way] = Some(*value);
        self.counts_set(set.offset + way, 1);
        self.clocks_set(clock_index, Self::wrap_way(way + 1) as u8);

        UpsertResult {
            index: set.offset + way,
            updated: UpdateOrInsert::Insert,
            evicted,
        }
    }

    #[derive(Clone, Copy)]
    struct Set<T> {
        tag: T,
        offset: usize,
    }

    #[inline]
    fn associate(&self, key: C::Key) -> Set<TAG_T> {
        let entropy = C::hash(key);
        let tag = TAG_T::truncate(entropy);
        let index = fastrange(entropy, self.sets as u64) as usize;
        let offset = index * WAYS;
        Set { tag, offset }
    }

    #[inline]
    fn search(&self, set_offset: usize, tag: TAG_T, key: C::Key) -> Option<usize> {
        let tags = &self.tags[set_offset..set_offset + WAYS];
        let ways_mask = Self::search_tags(tags, tag);
        if ways_mask == 0 {
            return None;
        }

        for way in 0..WAYS {
            if ((ways_mask >> way) & 1) == 1 {
                let idx = set_offset + way;
                if self.counts_get(idx) > 0 {
                    let v = self.values[idx];
                    if let Some(value) = v {
                        if C::key_from_value(&value) == key {
                            return Some(way);
                        }
                    }
                }
            }
        }
        None
    }

    #[inline]
    fn search_tags(tags: &[TAG_T], tag: TAG_T) -> u16 {
        debug_assert_eq!(tags.len(), WAYS);
        let mut bits: u16 = 0;
        for (i, t) in tags.iter().enumerate() {
            if *t == tag {
                bits |= 1u16 << i;
            }
        }
        bits
    }

    #[inline]
    fn counts_get(&self, index: usize) -> u8 {
        self.counts[index].get()
    }

    #[inline]
    fn counts_set(&self, index: usize, value: u8) {
        self.counts[index].set(value)
    }

    #[inline]
    fn clocks_get(&self, index: usize) -> u8 {
        self.clocks[index]
    }

    #[inline]
    fn clocks_set(&mut self, index: usize, value: u8) {
        self.clocks[index] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct IdentityCtx;

    impl SetAssociativeCacheContext for IdentityCtx {
        type Key = u64;
        type Value = u64;

        fn key_from_value(value: &Self::Value) -> Self::Key {
            *value
        }

        fn hash(key: Self::Key) -> u64 {
            key
        }
    }

    type Cache = SetAssociativeCache<IdentityCtx, u8, 4, 2>;

    #[test]
    fn basic_eviction_and_counts() {
        let mut cache = Cache::new(4 * 4, Options { name: "phase0" });

        for i in 0..4u64 {
            let _ = cache.upsert(&i);
        }

        assert!(cache.get(0).is_some());
        assert!(cache.get(1).is_some());

        let _ = cache.upsert(&4);
        assert!(cache.get(0).is_none() || cache.get(1).is_none());

        let (hits, misses) = cache.metrics();
        assert!(hits + misses > 0);
    }
}
```

Example:
WAYS=2, one set. Insert k0 and k1 => counts [1,1]. get(k0) => counts [2,1]. upsert(k2) walks the clock, decrements counts, and evicts k1.

ASCII diagram:
```
set 0:
  way0 [k0:c2]  way1 [k1:c1]  clock->0
upsert k2:
  dec way0 -> c1, advance
  dec way1 -> c0, evict k1, insert k2, clock->0
```

Exit criteria/tests:
- Fill one set and observe deterministic eviction.
- get increments count up to max_count.
- remove clears the slot and subsequent get misses.

## Phase 1: Uninitialized Values and Layout Invariants
Goal:
Move values into an aligned uninitialized buffer and make count the sole validity source, while keeping counts and clocks as plain arrays.

Intuition:
Tags and counts are hot metadata; values can stay uninitialized until count > 0, which saves work and mirrors the final memory layout.

Algorithm:
1) Allocate tags, counts, clocks, and an aligned values buffer; enforce layout invariants at construction.
2) On insert, write tag/value first and then publish count = 1.
3) On remove, set count = 0 and optionally poison the value slot.
4) During search, only read values when count > 0.

Code (full listing for this phase):
```rust
#![allow(dead_code)]

use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ptr::{self, NonNull};
use std::alloc::{alloc, dealloc, Layout as AllocLayout};
use std::cell::Cell;

#[inline]
pub fn fastrange(word: u64, p: u64) -> u64 {
    let ln = (word as u128).wrapping_mul(p as u128);
    (ln >> 64) as u64
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateOrInsert {
    Update,
    Insert,
}

pub trait Tag: Copy + Eq + Default {
    const BITS: usize;

    #[inline]
    fn truncate(entropy: u64) -> Self;
}

impl Tag for u8 {
    const BITS: usize = 8;
    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u8
    }
}

impl Tag for u16 {
    const BITS: usize = 16;
    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u16
    }
}

pub trait SetAssociativeCacheContext {
    type Key: Copy + Eq;
    type Value: Copy;

    #[inline]
    fn key_from_value(value: &Self::Value) -> Self::Key;

    #[inline]
    fn hash(key: Self::Key) -> u64;
}

#[derive(Debug)]
pub struct Metrics {
    hits: Cell<u64>,
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
    #[inline]
    pub fn reset(&self) {
        self.hits.set(0);
        self.misses.set(0);
    }

    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits.get()
    }

    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses.get()
    }
}

#[derive(Debug)]
pub struct AlignedBuf<T> {
    ptr: NonNull<u8>,
    len: usize,
    layout: AllocLayout,
    _marker: PhantomData<T>,
}

impl<T> AlignedBuf<T> {
    pub fn new_uninit(len: usize, alignment: usize) -> Self {
        assert!(len > 0);
        assert!(alignment >= align_of::<T>());
        assert!(size_of::<T>() % alignment == 0);

        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("size overflow");

        let layout = AllocLayout::from_size_align(bytes, alignment).expect("bad layout");

        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).expect("oom");

        Self {
            ptr,
            len,
            layout,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr() as *const T
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr() as *mut T
    }

    #[inline]
    pub fn get_ref(&self, index: usize) -> &T {
        assert!(index < self.len);
        unsafe { &*self.as_ptr().add(index) }
    }

    #[inline]
    pub fn read_copy(&self, index: usize) -> T
    where
        T: Copy,
    {
        assert!(index < self.len);
        unsafe { self.as_ptr().add(index).read() }
    }

    #[inline]
    pub fn write(&mut self, index: usize, value: T) {
        assert!(index < self.len);
        unsafe {
            self.as_mut_ptr().add(index).write(value);
        }
    }

    #[inline]
    pub fn write_uninit(&mut self, index: usize) {
        assert!(index < self.len);
        unsafe {
            ptr::write_bytes(self.ptr.as_ptr().add(index * size_of::<T>()), 0xAA, size_of::<T>());
        }
    }
}

impl<T> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

pub struct Options<'a> {
    pub name: &'a str,
}

pub struct SetAssociativeCache<
    C,
    TAG_T,
    const WAYS: usize,
    const CLOCK_BITS: usize,
    const CACHE_LINE_SIZE: usize,
    const VALUE_ALIGNMENT: usize,
    const CLOCK_HAND_BITS: usize,
> where
    C: SetAssociativeCacheContext,
    TAG_T: Tag,
{
    name: String,
    sets: usize,

    metrics: Metrics,

    tags: Vec<TAG_T>,
    values: AlignedBuf<C::Value>,

    counts: Vec<Cell<u8>>,
    clocks: Vec<u8>,

    _marker: PhantomData<C>,
}

impl<
        C,
        TAG_T,
        const WAYS: usize,
        const CLOCK_BITS: usize,
        const CACHE_LINE_SIZE: usize,
        const VALUE_ALIGNMENT: usize,
        const CLOCK_HAND_BITS: usize,
    > SetAssociativeCache<C, TAG_T, WAYS, CLOCK_BITS, CACHE_LINE_SIZE, VALUE_ALIGNMENT, CLOCK_HAND_BITS>
where
    C: SetAssociativeCacheContext,
    TAG_T: Tag,
{
    pub const VALUE_COUNT_MAX_MULTIPLE: usize = {
        const fn max_u(a: usize, b: usize) -> usize {
            if a > b { a } else { b }
        }
        const fn min_u(a: usize, b: usize) -> usize {
            if a < b { a } else { b }
        }

        let value_size = size_of::<C::Value>();
        let values_term = (max_u(value_size, CACHE_LINE_SIZE) / min_u(value_size, CACHE_LINE_SIZE)) * WAYS;
        let counts_term = (CACHE_LINE_SIZE * 8) / CLOCK_BITS;
        max_u(values_term, counts_term)
    };

    #[inline]
    fn value_alignment() -> usize {
        if VALUE_ALIGNMENT == 0 {
            align_of::<C::Value>()
        } else {
            VALUE_ALIGNMENT
        }
    }

    #[inline]
    fn max_count() -> u8 {
        debug_assert!(CLOCK_BITS <= 8);
        ((1u16 << CLOCK_BITS) - 1) as u8
    }

    #[inline]
    fn wrap_way(way: usize) -> usize {
        way & (WAYS - 1)
    }

    pub fn new(value_count_max: usize, options: Options<'_>) -> Self {
        assert!(size_of::<C::Key>().is_power_of_two());
        assert!(size_of::<C::Value>().is_power_of_two());

        match WAYS {
            2 | 4 | 16 => {}
            _ => panic!("WAYS must be 2, 4, or 16"),
        }
        match TAG_T::BITS {
            8 | 16 => {}
            _ => panic!("tag bits must be 8 or 16"),
        }
        match CLOCK_BITS {
            1 | 2 | 4 => {}
            _ => panic!("CLOCK_BITS must be 1, 2, or 4"),
        }

        let value_alignment = Self::value_alignment();
        assert!(value_alignment >= align_of::<C::Value>());
        assert!(size_of::<C::Value>() % value_alignment == 0);

        assert!(WAYS.is_power_of_two());
        assert!(TAG_T::BITS.is_power_of_two());
        assert!(CLOCK_BITS.is_power_of_two());
        assert!(CACHE_LINE_SIZE.is_power_of_two());

        assert!(size_of::<C::Key>() <= size_of::<C::Value>());
        assert!(size_of::<C::Key>() < CACHE_LINE_SIZE);
        assert!(CACHE_LINE_SIZE % size_of::<C::Key>() == 0);

        if CACHE_LINE_SIZE > size_of::<C::Value>() {
            assert!(CACHE_LINE_SIZE % size_of::<C::Value>() == 0);
        } else {
            assert!(size_of::<C::Value>() % CACHE_LINE_SIZE == 0);
        }

        assert!(CLOCK_HAND_BITS.is_power_of_two());
        assert!((1usize << CLOCK_HAND_BITS) == WAYS);

        assert!(value_count_max > 0);
        assert!(value_count_max >= WAYS);
        assert!(value_count_max % WAYS == 0);

        let sets = value_count_max / WAYS;

        let values_size_max = value_count_max
            .checked_mul(size_of::<C::Value>())
            .expect("values_size_max overflow");
        assert!(values_size_max >= CACHE_LINE_SIZE);
        assert!(values_size_max % CACHE_LINE_SIZE == 0);

        let counts_bits = value_count_max
            .checked_mul(CLOCK_BITS)
            .expect("counts_bits overflow");
        assert!(counts_bits % 8 == 0);
        let counts_size = counts_bits / 8;
        assert!(counts_size >= CACHE_LINE_SIZE);
        assert!(counts_size % CACHE_LINE_SIZE == 0);

        let clocks_bits = sets
            .checked_mul(CLOCK_HAND_BITS)
            .expect("clocks_bits overflow");
        assert!(clocks_bits % 8 == 0);
        let _ = clocks_bits;

        assert!(value_count_max % Self::VALUE_COUNT_MAX_MULTIPLE == 0);

        let tags = vec![TAG_T::default(); value_count_max];
        let values = AlignedBuf::<C::Value>::new_uninit(value_count_max, value_alignment);
        let counts = (0..value_count_max).map(|_| Cell::new(0)).collect();
        let clocks = vec![0u8; sets];

        let mut sac = Self {
            name: options.name.to_string(),
            sets,
            metrics: Metrics::default(),
            tags,
            values,
            counts,
            clocks,
            _marker: PhantomData,
        };

        sac.reset();
        sac
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    pub fn sets(&self) -> usize {
        self.sets
    }

    #[inline]
    pub fn metrics(&self) -> (u64, u64) {
        (self.metrics.hits(), self.metrics.misses())
    }

    pub fn reset(&mut self) {
        self.tags.fill(TAG_T::default());
        for c in &self.counts {
            c.set(0);
        }
        self.clocks.fill(0);
        self.metrics.reset();
    }

    pub fn get_index(&self, key: C::Key) -> Option<usize> {
        let set = self.associate(key);
        if let Some(way) = self.search(set.offset, set.tag, key) {
            self.metrics.hits.set(self.metrics.hits.get() + 1);

            let idx = set.offset + way;
            let count = self.counts_get(idx);
            let next = if count == Self::max_count() {
                count
            } else {
                count + 1
            };
            self.counts_set(idx, next);
            Some(idx)
        } else {
            self.metrics.misses.set(self.metrics.misses.get() + 1);
            None
        }
    }

    pub fn get(&self, key: C::Key) -> Option<&C::Value> {
        let index = self.get_index(key)?;
        Some(self.values.get_ref(index))
    }

    pub fn remove(&mut self, key: C::Key) -> Option<C::Value> {
        let set = self.associate(key);
        let way = self.search(set.offset, set.tag, key)?;

        let idx = set.offset + way;
        let removed = self.values.read_copy(idx);
        self.counts_set(idx, 0);
        self.values.write_uninit(idx);
        Some(removed)
    }

    pub fn demote(&mut self, key: C::Key) {
        let set = self.associate(key);
        let Some(way) = self.search(set.offset, set.tag, key) else { return; };
        let idx = set.offset + way;
        self.counts_set(idx, 1);
    }

    pub struct UpsertResult<V> {
        pub index: usize,
        pub updated: UpdateOrInsert,
        pub evicted: Option<V>,
    }

    pub fn upsert(&mut self, value: &C::Value) -> UpsertResult<C::Value> {
        let key = C::key_from_value(value);
        let set = self.associate(key);

        if let Some(way) = self.search(set.offset, set.tag, key) {
            let idx = set.offset + way;
            self.counts_set(idx, 1);
            let evicted = self.values.read_copy(idx);
            self.values.write(idx, *value);
            return UpsertResult {
                index: idx,
                updated: UpdateOrInsert::Update,
                evicted: Some(evicted),
            };
        }

        let clock_index = set.offset / WAYS;
        let mut way = self.clocks_get(clock_index) as usize;
        debug_assert!(way < WAYS);

        let max_count = Self::max_count() as usize;
        let clock_iterations_max = WAYS * (max_count.saturating_sub(1));

        let mut evicted: Option<C::Value> = None;
        let mut safety_count: usize = 0;

        while safety_count <= clock_iterations_max {
            let idx = set.offset + way;
            let mut count = self.counts_get(idx) as usize;
            if count == 0 {
                break;
            }

            count -= 1;
            self.counts_set(idx, count as u8);
            if count == 0 {
                evicted = Some(self.values.read_copy(idx));
                break;
            }

            safety_count += 1;
            way = Self::wrap_way(way + 1);
        }

        assert!(self.counts_get(set.offset + way) == 0);

        self.tags[set.offset + way] = set.tag;
        self.values.write(set.offset + way, *value);
        self.counts_set(set.offset + way, 1);
        self.clocks_set(clock_index, Self::wrap_way(way + 1) as u8);

        UpsertResult {
            index: set.offset + way,
            updated: UpdateOrInsert::Insert,
            evicted,
        }
    }

    #[derive(Clone, Copy)]
    struct Set<T> {
        tag: T,
        offset: usize,
    }

    #[inline]
    fn associate(&self, key: C::Key) -> Set<TAG_T> {
        let entropy = C::hash(key);
        let tag = TAG_T::truncate(entropy);
        let index = fastrange(entropy, self.sets as u64) as usize;
        let offset = index * WAYS;
        Set { tag, offset }
    }

    #[inline]
    fn search(&self, set_offset: usize, tag: TAG_T, key: C::Key) -> Option<usize> {
        let tags = &self.tags[set_offset..set_offset + WAYS];
        let ways_mask = Self::search_tags(tags, tag);
        if ways_mask == 0 {
            return None;
        }

        for way in 0..WAYS {
            if ((ways_mask >> way) & 1) == 1 {
                let idx = set_offset + way;
                if self.counts_get(idx) > 0 {
                    let v = self.values.get_ref(idx);
                    if C::key_from_value(v) == key {
                        return Some(way);
                    }
                }
            }
        }
        None
    }

    #[inline]
    fn search_tags(tags: &[TAG_T], tag: TAG_T) -> u16 {
        debug_assert_eq!(tags.len(), WAYS);
        let mut bits: u16 = 0;
        for (i, t) in tags.iter().enumerate() {
            if *t == tag {
                bits |= 1u16 << i;
            }
        }
        bits
    }

    #[inline]
    fn counts_get(&self, index: usize) -> u8 {
        self.counts[index].get()
    }

    #[inline]
    fn counts_set(&self, index: usize, value: u8) {
        self.counts[index].set(value)
    }

    #[inline]
    fn clocks_get(&self, index: usize) -> u8 {
        self.clocks[index]
    }

    #[inline]
    fn clocks_set(&mut self, index: usize, value: u8) {
        self.clocks[index] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct IdentityCtx;

    impl SetAssociativeCacheContext for IdentityCtx {
        type Key = u64;
        type Value = u64;

        fn key_from_value(value: &Self::Value) -> Self::Key {
            *value
        }

        fn hash(key: Self::Key) -> u64 {
            key
        }
    }

    type Cache = SetAssociativeCache<IdentityCtx, u8, 4, 2, 64, 0, 2>;

    #[test]
    fn upsert_remove_demote() {
        let mut cache = Cache::new(256, Options { name: "phase1" });

        let _ = cache.upsert(&1);
        assert_eq!(Some(&1u64), cache.get(1));

        cache.demote(1);
        let removed = cache.remove(1);
        assert_eq!(Some(1u64), removed);
        assert!(cache.get(1).is_none());
    }
}
```

Example:
count == 0 marks a slot as invalid even if the underlying bytes still contain old data.

ASCII diagram:
```
index: 0 1 2 3
count: 1 0 2 0
value: v0 ?? v2 ??
```

Exit criteria/tests:
- No reads occur when count == 0.
- remove sets count to 0 and get misses afterwards.
- Layout invariants and VALUE_COUNT_MAX_MULTIPLE assertions pass.

## Phase 2: Packed Counters and Interior Mutability
Goal:
Pack counts and clocks into u64 words and use interior mutability so get/get_index can take &self.

Intuition:
Packing reduces metadata footprint and improves cache locality; UnsafeCell preserves the &self hot-path API.

Algorithm:
1) Replace Vec<u8> counts/clocks with PackedUnsignedIntegerArray and compute word counts.
2) Store packed arrays behind UnsafeCell and expose counts_get/set and clocks_get/set helpers.
3) Preserve eviction logic, but read and write packed counters instead of byte arrays.

Code (full listing for this phase):
```rust
// Port of `set_associative_cache.zig` to Rust.
//
// Goals:
// - Match Zig semantics as closely as possible (CLOCK Nth-Chance, packed counters, fastrange).
// - Keep hot-path APIs `get_index/get` taking `&self` (interior mutability), like Zig's `*const`.
// - Avoid external crates.
//
// Notes:
// - This is intentionally single-threaded / !Sync (like the Zig version).
// - Values are stored in uninitialized memory and are only considered valid when Count > 0.

#![allow(dead_code)]

use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ptr::{self, NonNull};
use std::alloc::{alloc, dealloc, Layout as AllocLayout};
use std::cell::{Cell, UnsafeCell};

/// Fast alternative to modulo reduction (not equivalent to modulo).
///
/// Matches stdx.fastrange():
///   return ((word as u128 * p as u128) >> 64) as u64
#[inline]
pub fn fastrange(word: u64, p: u64) -> u64 {
    let ln = (word as u128).wrapping_mul(p as u128);
    (ln >> 64) as u64
}

#[inline]
pub fn div_ceil_u64(numerator: u64, denominator: u64) -> u64 {
    assert!(denominator > 0);
    if numerator == 0 {
        return 0;
    }
    ((numerator - 1) / denominator) + 1
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateOrInsert {
    Update,
    Insert,
}

/// Tag types supported by the cache (matches Zig's `tag_bits` of 8 or 16).
///
/// Implemented for `u8` and `u16`.
pub trait Tag: Copy + Eq + Default {
    const BITS: usize;

    /// Truncate a 64-bit hash down to the tag width.
    #[inline]
    fn truncate(entropy: u64) -> Self;
}

impl Tag for u8 {
    const BITS: usize = 8;
    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u8
    }
}

impl Tag for u16 {
    const BITS: usize = 16;
    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u16
    }
}

/// Compile-time context for the cache.
///
/// This mirrors Zig's `key_from_value` and `hash` comptime function parameters.
pub trait SetAssociativeCacheContext {
    type Key: Copy + Eq;
    type Value: Copy;

    #[inline]
    fn key_from_value(value: &Self::Value) -> Self::Key;

    #[inline]
    fn hash(key: Self::Key) -> u64;
}

/// Hit/miss metrics (interior-mutable so `get` can take `&self`).
#[derive(Debug)]
pub struct Metrics {
    hits: Cell<u64>,
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
    #[inline]
    pub fn reset(&self) {
        self.hits.set(0);
        self.misses.set(0);
    }

    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits.get()
    }

    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses.get()
    }
}

/// A packed array of unsigned integers with a fixed bit width, stored in little-endian u64 words.
///
/// This is the Rust port of `PackedUnsignedIntegerArrayType(UInt)`.
///
/// - `BITS` must be < 8 and a power of two (1, 2, 4).
/// - Index 0 maps to the least significant bits of word 0.
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

    /// Number of u64 words required to store `len` packed integers.
    #[inline]
    pub const fn words_for_len(len: usize) -> usize {
        // ceil((len * BITS) / 64)
        let bits = len * BITS;
        (bits + (Self::WORD_BITS - 1)) / Self::WORD_BITS
    }

    pub fn new_zeroed(words_len: usize) -> Self {
        assert!(cfg!(target_endian = "little"), "requires little-endian");
        assert!(BITS < 8, "BITS must be < 8");
        assert!(BITS.is_power_of_two(), "BITS must be power-of-two");
        assert!(Self::WORD_BITS % BITS == 0);
        Self {
            words: vec![0u64; words_len].into_boxed_slice(),
        }
    }

    pub fn from_words(words: Vec<u64>) -> Self {
        assert!(cfg!(target_endian = "little"), "requires little-endian");
        assert!(BITS < 8, "BITS must be < 8");
        assert!(BITS.is_power_of_two(), "BITS must be power-of-two");
        assert!(Self::WORD_BITS % BITS == 0);
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

    /// Returns the unsigned integer at `index`.
    #[inline]
    pub fn get(&self, index: usize) -> u8 {
        let uints_per_word = Self::uints_per_word();
        let word_index = index / uints_per_word;
        let within = index % uints_per_word;
        let shift = within * BITS;
        let word = self.words[word_index];
        ((word >> shift) & Self::mask_value()) as u8
    }

    /// Sets the unsigned integer at `index` to `value`.
    #[inline]
    pub fn set(&mut self, index: usize, value: u8) {
        debug_assert!((value as u64) <= Self::mask_value());
        let uints_per_word = Self::uints_per_word();
        let word_index = index / uints_per_word;
        let within = index % uints_per_word;
        let shift = within * BITS;
        let mask = Self::mask_value() << shift;
        let w = &mut self.words[word_index];
        *w &= !mask;
        *w |= (value as u64) << shift;
    }
}

/// Raw aligned buffer of `T` values.
///
/// The memory is allocated but not initialized.
/// We treat entries as valid only when the corresponding Count > 0.
#[derive(Debug)]
pub struct AlignedBuf<T> {
    ptr: NonNull<u8>,
    len: usize,
    layout: AllocLayout,
    _marker: PhantomData<T>,
}

impl<T> AlignedBuf<T> {
    pub fn new_uninit(len: usize, alignment: usize) -> Self {
        assert!(len > 0);
        assert!(alignment >= align_of::<T>());
        // Ensures each element is aligned if we stride by `size_of::<T>()`.
        assert!(size_of::<T>() % alignment == 0);

        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("size overflow");

        let layout = AllocLayout::from_size_align(bytes, alignment).expect("bad layout");

        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).expect("oom");

        Self {
            ptr,
            len,
            layout,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr() as *const T
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr() as *mut T
    }

    #[inline]
    pub fn get_ref(&self, index: usize) -> &T {
        assert!(index < self.len);
        unsafe { &*self.as_ptr().add(index) }
    }

    #[inline]
    pub fn read_copy(&self, index: usize) -> T
    where
        T: Copy,
    {
        assert!(index < self.len);
        unsafe { self.as_ptr().add(index).read() }
    }

    #[inline]
    pub fn write(&mut self, index: usize, value: T) {
        assert!(index < self.len);
        unsafe {
            self.as_mut_ptr().add(index).write(value);
        }
    }

    /// Best-effort "poison" on remove. Not required for correctness.
    #[inline]
    pub fn write_uninit(&mut self, index: usize) {
        assert!(index < self.len);
        unsafe {
            // Write an uninitialized `T` via byte pattern.
            // This is *not* relied upon by the algorithm; counts are the source of truth.
            ptr::write_bytes(self.ptr.as_ptr().add(index * size_of::<T>()), 0xAA, size_of::<T>());
        }
    }
}

impl<T> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

pub struct Options<'a> {
    pub name: &'a str,
}

/// Rust port of `SetAssociativeCacheType(...)`.
///
/// Layout parameters (mirroring Zig's comptime `Layout`):
/// - `WAYS`: 2, 4, or 16
/// - `TAG_T`: u8 or u16
/// - `CLOCK_BITS`: 1, 2, or 4
/// - `CACHE_LINE_SIZE`: usually 64
/// - `VALUE_ALIGNMENT`: 0 means "use align_of::<Value>()", else an explicit alignment override
/// - `CLOCK_HAND_BITS`: must satisfy (1 << CLOCK_HAND_BITS) == WAYS and be power-of-two
///
/// The cache is intentionally !Sync (Cell/UnsafeCell) to match the Zig design.
pub struct SetAssociativeCache<
    C,
    TAG_T,
    const WAYS: usize,
    const CLOCK_BITS: usize,
    const CACHE_LINE_SIZE: usize,
    const VALUE_ALIGNMENT: usize,
    const CLOCK_HAND_BITS: usize,
> where
    C: SetAssociativeCacheContext,
    TAG_T: Tag,
{
    name: String,
    sets: usize,

    metrics: Metrics,

    tags: Vec<TAG_T>,
    values: AlignedBuf<C::Value>,

    counts: UnsafeCell<PackedUnsignedIntegerArray<CLOCK_BITS>>,
    clocks: UnsafeCell<PackedUnsignedIntegerArray<CLOCK_HAND_BITS>>,

    _marker: PhantomData<C>,
}

impl<
        C,
        TAG_T,
        const WAYS: usize,
        const CLOCK_BITS: usize,
        const CACHE_LINE_SIZE: usize,
        const VALUE_ALIGNMENT: usize,
        const CLOCK_HAND_BITS: usize,
    > SetAssociativeCache<C, TAG_T, WAYS, CLOCK_BITS, CACHE_LINE_SIZE, VALUE_ALIGNMENT, CLOCK_HAND_BITS>
where
    C: SetAssociativeCacheContext,
    TAG_T: Tag,
{
    /// Mirrors Zig's `value_count_max_multiple`.
    pub const VALUE_COUNT_MAX_MULTIPLE: usize = {
        const fn max_u(a: usize, b: usize) -> usize {
            if a > b { a } else { b }
        }
        const fn min_u(a: usize, b: usize) -> usize {
            if a < b { a } else { b }
        }

        let value_size = size_of::<C::Value>();
        let values_term = (max_u(value_size, CACHE_LINE_SIZE) / min_u(value_size, CACHE_LINE_SIZE)) * WAYS;
        let counts_term = (CACHE_LINE_SIZE * 8) / CLOCK_BITS;
        max_u(values_term, counts_term)
    };

    #[inline]
    fn value_alignment() -> usize {
        if VALUE_ALIGNMENT == 0 {
            align_of::<C::Value>()
        } else {
            VALUE_ALIGNMENT
        }
    }

    #[inline]
    fn max_count() -> u8 {
        debug_assert!(CLOCK_BITS <= 8);
        ((1u16 << CLOCK_BITS) - 1) as u8
    }

    #[inline]
    fn wrap_way(way: usize) -> usize {
        // WAYS is power-of-two.
        way & (WAYS - 1)
    }

    /// Creates a new cache.
    ///
    /// This is the Rust analogue of Zig's `init(allocator, value_count_max, options)`.
    pub fn new(value_count_max: usize, options: Options<'_>) -> Self {
        // --- Zig comptime asserts (runtime in Rust) ---
        assert!(size_of::<C::Key>().is_power_of_two());
        assert!(size_of::<C::Value>().is_power_of_two());

        match WAYS {
            2 | 4 | 16 => {}
            _ => panic!("WAYS must be 2, 4, or 16"),
        }
        match TAG_T::BITS {
            8 | 16 => {}
            _ => panic!("tag bits must be 8 or 16"),
        }
        match CLOCK_BITS {
            1 | 2 | 4 => {}
            _ => panic!("CLOCK_BITS must be 1, 2, or 4"),
        }

        let value_alignment = Self::value_alignment();
        assert!(value_alignment >= align_of::<C::Value>());
        assert!(size_of::<C::Value>() % value_alignment == 0);

        assert!(WAYS.is_power_of_two());
        assert!(TAG_T::BITS.is_power_of_two());
        assert!(CLOCK_BITS.is_power_of_two());
        assert!(CACHE_LINE_SIZE.is_power_of_two());

        assert!(size_of::<C::Key>() <= size_of::<C::Value>());
        assert!(size_of::<C::Key>() < CACHE_LINE_SIZE);
        assert!(CACHE_LINE_SIZE % size_of::<C::Key>() == 0);

        if CACHE_LINE_SIZE > size_of::<C::Value>() {
            assert!(CACHE_LINE_SIZE % size_of::<C::Value>() == 0);
        } else {
            assert!(size_of::<C::Value>() % CACHE_LINE_SIZE == 0);
        }

        // clock hand width checks
        assert!(CLOCK_HAND_BITS.is_power_of_two());
        assert!((1usize << CLOCK_HAND_BITS) == WAYS);

        // --- Zig init-time asserts ---
        assert!(value_count_max > 0);
        assert!(value_count_max >= WAYS);
        assert!(value_count_max % WAYS == 0);

        let sets = value_count_max / WAYS;

        let values_size_max = value_count_max
            .checked_mul(size_of::<C::Value>())
            .expect("values_size_max overflow");
        assert!(values_size_max >= CACHE_LINE_SIZE);
        assert!(values_size_max % CACHE_LINE_SIZE == 0);

        // counts_size bytes = value_count_max * CLOCK_BITS / 8
        let counts_bits = value_count_max
            .checked_mul(CLOCK_BITS)
            .expect("counts_bits overflow");
        assert!(counts_bits % 8 == 0);
        let counts_size = counts_bits / 8;
        assert!(counts_size >= CACHE_LINE_SIZE);
        assert!(counts_size % CACHE_LINE_SIZE == 0);
        assert!(counts_size % 8 == 0);
        let counts_words_len = counts_size / 8;

        // clocks_size bytes = sets * CLOCK_HAND_BITS / 8
        let clocks_bits = sets
            .checked_mul(CLOCK_HAND_BITS)
            .expect("clocks_bits overflow");
        assert!(clocks_bits % 8 == 0);
        let clocks_size = clocks_bits / 8;
        // In Zig they `maybe(...)` these, i.e. documentation-only.
        let _ = clocks_size;
        let clocks_words_len = div_ceil_u64(clocks_bits as u64, 64) as usize;

        // value_count_max_multiple (matches Zig's associated const)
        assert!(value_count_max % Self::VALUE_COUNT_MAX_MULTIPLE == 0);

        // Allocate.
        let tags = vec![TAG_T::default(); value_count_max];
        let values = AlignedBuf::<C::Value>::new_uninit(value_count_max, value_alignment);
        let counts = PackedUnsignedIntegerArray::<CLOCK_BITS>::new_zeroed(counts_words_len);
        let clocks = PackedUnsignedIntegerArray::<CLOCK_HAND_BITS>::new_zeroed(clocks_words_len);

        let mut sac = Self {
            name: options.name.to_string(),
            sets,
            metrics: Metrics::default(),
            tags,
            values,
            counts: UnsafeCell::new(counts),
            clocks: UnsafeCell::new(clocks),
            _marker: PhantomData,
        };

        sac.reset();
        sac
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    pub fn sets(&self) -> usize {
        self.sets
    }

    #[inline]
    pub fn metrics(&self) -> (u64, u64) {
        (self.metrics.hits(), self.metrics.misses())
    }

    pub fn reset(&mut self) {
        self.tags.fill(TAG_T::default());
        unsafe {
            (*self.counts.get()).words_mut().fill(0);
            (*self.clocks.get()).words_mut().fill(0);
        }
        self.metrics.reset();
    }

    /// Returns the index of `key` if present.
    ///
    /// Semantics match Zig's `get_index(self: *const, key) ?usize`:
    /// - increments metrics
    /// - increments the entry's Count with saturation
    pub fn get_index(&self, key: C::Key) -> Option<usize> {
        let set = self.associate(key);
        if let Some(way) = self.search(set.offset, set.tag, key) {
            self.metrics.hits.set(self.metrics.hits.get() + 1);

            let idx = set.offset + way;
            let count = self.counts_get(idx);
            let next = if count == Self::max_count() {
                count
            } else {
                count + 1
            };
            self.counts_set(idx, next);
            Some(idx)
        } else {
            self.metrics.misses.set(self.metrics.misses.get() + 1);
            None
        }
    }

    /// Returns a reference to the value for `key` if present.
    pub fn get(&self, key: C::Key) -> Option<&C::Value> {
        let index = self.get_index(key)?;
        Some(self.values.get_ref(index))
    }

    /// Remove `key` if present.
    /// Returns the removed value.
    pub fn remove(&mut self, key: C::Key) -> Option<C::Value> {
        let set = self.associate(key);
        let way = self.search(set.offset, set.tag, key)?;

        let idx = set.offset + way;
        let removed = self.values.read_copy(idx);
        self.counts_set(idx, 0);
        self.values.write_uninit(idx);
        Some(removed)
    }

    /// Hint that `key` is less likely to be accessed in the future.
    pub fn demote(&mut self, key: C::Key) {
        let set = self.associate(key);
        let Some(way) = self.search(set.offset, set.tag, key) else { return; };
        let idx = set.offset + way;
        self.counts_set(idx, 1);
    }

    pub struct UpsertResult<V> {
        pub index: usize,
        pub updated: UpdateOrInsert,
        pub evicted: Option<V>,
    }

    /// Upsert a value, evicting an older entry if needed.
    pub fn upsert(&mut self, value: &C::Value) -> UpsertResult<C::Value> {
        let key = C::key_from_value(value);
        let set = self.associate(key);

        if let Some(way) = self.search(set.offset, set.tag, key) {
            let idx = set.offset + way;
            // Zig sets count=1 before overwriting.
            self.counts_set(idx, 1);
            let evicted = self.values.read_copy(idx);
            self.values.write(idx, *value);
            return UpsertResult {
                index: idx,
                updated: UpdateOrInsert::Update,
                evicted: Some(evicted),
            };
        }

        let clock_index = set.offset / WAYS;
        let mut way = self.clocks_get(clock_index) as usize;
        debug_assert!(way < WAYS);

        let max_count = Self::max_count() as usize;
        let clock_iterations_max = WAYS * (max_count.saturating_sub(1));

        let mut evicted: Option<C::Value> = None;
        let mut safety_count: usize = 0;

        while safety_count <= clock_iterations_max {
            let idx = set.offset + way;
            let mut count = self.counts_get(idx) as usize;
            if count == 0 {
                break; // free
            }

            count -= 1;
            self.counts_set(idx, count as u8);
            if count == 0 {
                evicted = Some(self.values.read_copy(idx));
                break;
            }

            safety_count += 1;
            way = Self::wrap_way(way + 1);
        }

        assert!(self.counts_get(set.offset + way) == 0);

        // Write tag/value, then publish Count=1.
        self.tags[set.offset + way] = set.tag;
        self.values.write(set.offset + way, *value);
        self.counts_set(set.offset + way, 1);
        self.clocks_set(clock_index, Self::wrap_way(way + 1) as u8);

        UpsertResult {
            index: set.offset + way,
            updated: UpdateOrInsert::Insert,
            evicted,
        }
    }

    /// Debug: prints derived layout parameters.
    pub fn inspect_layout() {
        let clock_hand_bits = CLOCK_HAND_BITS;
        let tags_per_line = (CACHE_LINE_SIZE * 8) / (WAYS * TAG_T::BITS);
        let clocks_per_line = (CACHE_LINE_SIZE * 8) / (WAYS * CLOCK_BITS);
        let clock_hands_per_line = (CACHE_LINE_SIZE * 8) / clock_hand_bits;

        println!(
            "Key_bits={} Value_size={} ways={} tag_bits={} clock_bits={} clock_hand_bits={} tags_per_line={} clocks_per_line={} clock_hands_per_line={}",
            size_of::<C::Key>() * 8,
            size_of::<C::Value>(),
            WAYS,
            TAG_T::BITS,
            CLOCK_BITS,
            clock_hand_bits,
            tags_per_line,
            clocks_per_line,
            clock_hands_per_line,
        );
    }

    // ----- Internals -----

    #[derive(Clone, Copy)]
    struct Set<T> {
        tag: T,
        offset: usize,
    }

    #[inline]
    fn associate(&self, key: C::Key) -> Set<TAG_T> {
        let entropy = C::hash(key);
        let tag = TAG_T::truncate(entropy);
        let index = fastrange(entropy, self.sets as u64) as usize;
        let offset = index * WAYS;
        Set { tag, offset }
    }

    /// Returns the way if `key` is present in the set.
    #[inline]
    fn search(&self, set_offset: usize, tag: TAG_T, key: C::Key) -> Option<usize> {
        let tags = &self.tags[set_offset..set_offset + WAYS];
        let ways_mask = Self::search_tags(tags, tag);
        if ways_mask == 0 {
            return None;
        }

        for way in 0..WAYS {
            if ((ways_mask >> way) & 1) == 1 {
                let idx = set_offset + way;
                if self.counts_get(idx) > 0 {
                    let v = self.values.get_ref(idx);
                    if C::key_from_value(v) == key {
                        return Some(way);
                    }
                }
            }
        }
        None
    }

    /// Bitmask of ways whose tag matches.
    /// Lowest bit corresponds to way 0.
    #[inline]
    fn search_tags(tags: &[TAG_T], tag: TAG_T) -> u16 {
        debug_assert_eq!(tags.len(), WAYS);
        let mut bits: u16 = 0;
        for (i, t) in tags.iter().enumerate() {
            if *t == tag {
                bits |= 1u16 << i;
            }
        }
        bits
    }

    #[inline]
    fn counts_get(&self, index: usize) -> u8 {
        unsafe { (*self.counts.get()).get(index) }
    }

    #[inline]
    fn counts_set(&self, index: usize, value: u8) {
        unsafe { (*self.counts.get()).set(index, value) }
    }

    #[inline]
    fn clocks_get(&self, index: usize) -> u8 {
        unsafe { (*self.clocks.get()).get(index) }
    }

    #[inline]
    fn clocks_set(&self, index: usize, value: u8) {
        unsafe { (*self.clocks.get()).set(index, value) }
    }

    // Exposed for tests/debug.
    #[inline]
    pub fn raw_tags(&self) -> &[TAG_T] {
        &self.tags
    }

    #[inline]
    pub fn counts_words_snapshot(&self) -> Vec<u64> {
        // NOTE: we return an owned snapshot to avoid exposing interior-mutable storage.
        unsafe { (*self.counts.get()).words().to_vec() }
    }

    #[inline]
    pub fn clocks_words_snapshot(&self) -> Vec<u64> {
        // NOTE: we return an owned snapshot to avoid exposing interior-mutable storage.
        unsafe { (*self.clocks.get()).words().to_vec() }
    }
}
```

Example:
CLOCK_BITS=2 packs four 2-bit counters into one byte, and 32 counters into a u64 word.

ASCII diagram:
```
counts word (little-endian bits): [c0 c1 c2 c3 ...] as 2-bit fields
clocks word (log2(WAYS) bits):    [h0 h1 h2 h3 ...]
```

Exit criteria/tests:
- PackedUnsignedIntegerArray get/set round-trips for BITS in {1,2,4}.
- get_index increments counts and metrics with &self.
- upsert eviction behavior matches Phase 1.

## Phase N: Final Form
Goal:
Match the reference implementation exactly, including the full test suite.

Intuition:
All layout checks, packed metadata, aligned values, and tests are in place to mirror the Zig behavior.

Algorithm:
Same as Phase 2, plus the complete test coverage for packed arrays, eviction, and tag matching.

Code (full listing for this phase; must match reference if provided):
```rust
// Port of `set_associative_cache.zig` to Rust.
//
// Goals:
// - Match Zig semantics as closely as possible (CLOCK Nth-Chance, packed counters, fastrange).
// - Keep hot-path APIs `get_index/get` taking `&self` (interior mutability), like Zig's `*const`.
// - Avoid external crates.
//
// Notes:
// - This is intentionally single-threaded / !Sync (like the Zig version).
// - Values are stored in uninitialized memory and are only considered valid when Count > 0.

#![allow(dead_code)]

use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ptr::{self, NonNull};
use std::alloc::{alloc, dealloc, Layout as AllocLayout};
use std::cell::{Cell, UnsafeCell};

/// Fast alternative to modulo reduction (not equivalent to modulo).
///
/// Matches stdx.fastrange():
///   return ((word as u128 * p as u128) >> 64) as u64
#[inline]
pub fn fastrange(word: u64, p: u64) -> u64 {
    let ln = (word as u128).wrapping_mul(p as u128);
    (ln >> 64) as u64
}

#[inline]
pub fn div_ceil_u64(numerator: u64, denominator: u64) -> u64 {
    assert!(denominator > 0);
    if numerator == 0 {
        return 0;
    }
    ((numerator - 1) / denominator) + 1
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateOrInsert {
    Update,
    Insert,
}

/// Tag types supported by the cache (matches Zig's `tag_bits` of 8 or 16).
///
/// Implemented for `u8` and `u16`.
pub trait Tag: Copy + Eq + Default {
    const BITS: usize;

    /// Truncate a 64-bit hash down to the tag width.
    #[inline]
    fn truncate(entropy: u64) -> Self;
}

impl Tag for u8 {
    const BITS: usize = 8;
    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u8
    }
}

impl Tag for u16 {
    const BITS: usize = 16;
    #[inline]
    fn truncate(entropy: u64) -> Self {
        entropy as u16
    }
}

/// Compile-time context for the cache.
///
/// This mirrors Zig's `key_from_value` and `hash` comptime function parameters.
pub trait SetAssociativeCacheContext {
    type Key: Copy + Eq;
    type Value: Copy;

    #[inline]
    fn key_from_value(value: &Self::Value) -> Self::Key;

    #[inline]
    fn hash(key: Self::Key) -> u64;
}

/// Hit/miss metrics (interior-mutable so `get` can take `&self`).
#[derive(Debug)]
pub struct Metrics {
    hits: Cell<u64>,
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
    #[inline]
    pub fn reset(&self) {
        self.hits.set(0);
        self.misses.set(0);
    }

    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits.get()
    }

    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses.get()
    }
}

/// A packed array of unsigned integers with a fixed bit width, stored in little-endian u64 words.
///
/// This is the Rust port of `PackedUnsignedIntegerArrayType(UInt)`.
///
/// - `BITS` must be < 8 and a power of two (1, 2, 4).
/// - Index 0 maps to the least significant bits of word 0.
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

    /// Number of u64 words required to store `len` packed integers.
    #[inline]
    pub const fn words_for_len(len: usize) -> usize {
        // ceil((len * BITS) / 64)
        let bits = len * BITS;
        (bits + (Self::WORD_BITS - 1)) / Self::WORD_BITS
    }

    pub fn new_zeroed(words_len: usize) -> Self {
        assert!(cfg!(target_endian = "little"), "requires little-endian");
        assert!(BITS < 8, "BITS must be < 8");
        assert!(BITS.is_power_of_two(), "BITS must be power-of-two");
        assert!(Self::WORD_BITS % BITS == 0);
        Self {
            words: vec![0u64; words_len].into_boxed_slice(),
        }
    }

    pub fn from_words(words: Vec<u64>) -> Self {
        assert!(cfg!(target_endian = "little"), "requires little-endian");
        assert!(BITS < 8, "BITS must be < 8");
        assert!(BITS.is_power_of_two(), "BITS must be power-of-two");
        assert!(Self::WORD_BITS % BITS == 0);
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

    /// Returns the unsigned integer at `index`.
    #[inline]
    pub fn get(&self, index: usize) -> u8 {
        let uints_per_word = Self::uints_per_word();
        let word_index = index / uints_per_word;
        let within = index % uints_per_word;
        let shift = within * BITS;
        let word = self.words[word_index];
        ((word >> shift) & Self::mask_value()) as u8
    }

    /// Sets the unsigned integer at `index` to `value`.
    #[inline]
    pub fn set(&mut self, index: usize, value: u8) {
        debug_assert!((value as u64) <= Self::mask_value());
        let uints_per_word = Self::uints_per_word();
        let word_index = index / uints_per_word;
        let within = index % uints_per_word;
        let shift = within * BITS;
        let mask = Self::mask_value() << shift;
        let w = &mut self.words[word_index];
        *w &= !mask;
        *w |= (value as u64) << shift;
    }
}

/// Raw aligned buffer of `T` values.
///
/// The memory is allocated but not initialized.
/// We treat entries as valid only when the corresponding Count > 0.
#[derive(Debug)]
pub struct AlignedBuf<T> {
    ptr: NonNull<u8>,
    len: usize,
    layout: AllocLayout,
    _marker: PhantomData<T>,
}

impl<T> AlignedBuf<T> {
    pub fn new_uninit(len: usize, alignment: usize) -> Self {
        assert!(len > 0);
        assert!(alignment >= align_of::<T>());
        // Ensures each element is aligned if we stride by `size_of::<T>()`.
        assert!(size_of::<T>() % alignment == 0);

        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("size overflow");

        let layout = AllocLayout::from_size_align(bytes, alignment).expect("bad layout");

        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).expect("oom");

        Self {
            ptr,
            len,
            layout,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr() as *const T
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr() as *mut T
    }

    #[inline]
    pub fn get_ref(&self, index: usize) -> &T {
        assert!(index < self.len);
        unsafe { &*self.as_ptr().add(index) }
    }

    #[inline]
    pub fn read_copy(&self, index: usize) -> T
    where
        T: Copy,
    {
        assert!(index < self.len);
        unsafe { self.as_ptr().add(index).read() }
    }

    #[inline]
    pub fn write(&mut self, index: usize, value: T) {
        assert!(index < self.len);
        unsafe {
            self.as_mut_ptr().add(index).write(value);
        }
    }

    /// Best-effort "poison" on remove. Not required for correctness.
    #[inline]
    pub fn write_uninit(&mut self, index: usize) {
        assert!(index < self.len);
        unsafe {
            // Write an uninitialized `T` via byte pattern.
            // This is *not* relied upon by the algorithm; counts are the source of truth.
            ptr::write_bytes(self.ptr.as_ptr().add(index * size_of::<T>()), 0xAA, size_of::<T>());
        }
    }
}

impl<T> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

pub struct Options<'a> {
    pub name: &'a str,
}

/// Rust port of `SetAssociativeCacheType(...)`.
///
/// Layout parameters (mirroring Zig's comptime `Layout`):
/// - `WAYS`: 2, 4, or 16
/// - `TAG_T`: u8 or u16
/// - `CLOCK_BITS`: 1, 2, or 4
/// - `CACHE_LINE_SIZE`: usually 64
/// - `VALUE_ALIGNMENT`: 0 means "use align_of::<Value>()", else an explicit alignment override
/// - `CLOCK_HAND_BITS`: must satisfy (1 << CLOCK_HAND_BITS) == WAYS and be power-of-two
///
/// The cache is intentionally !Sync (Cell/UnsafeCell) to match the Zig design.
pub struct SetAssociativeCache<
    C,
    TAG_T,
    const WAYS: usize,
    const CLOCK_BITS: usize,
    const CACHE_LINE_SIZE: usize,
    const VALUE_ALIGNMENT: usize,
    const CLOCK_HAND_BITS: usize,
> where
    C: SetAssociativeCacheContext,
    TAG_T: Tag,
{
    name: String,
    sets: usize,

    metrics: Metrics,

    tags: Vec<TAG_T>,
    values: AlignedBuf<C::Value>,

    counts: UnsafeCell<PackedUnsignedIntegerArray<CLOCK_BITS>>,
    clocks: UnsafeCell<PackedUnsignedIntegerArray<CLOCK_HAND_BITS>>,

    _marker: PhantomData<C>,
}

impl<
        C,
        TAG_T,
        const WAYS: usize,
        const CLOCK_BITS: usize,
        const CACHE_LINE_SIZE: usize,
        const VALUE_ALIGNMENT: usize,
        const CLOCK_HAND_BITS: usize,
    > SetAssociativeCache<C, TAG_T, WAYS, CLOCK_BITS, CACHE_LINE_SIZE, VALUE_ALIGNMENT, CLOCK_HAND_BITS>
where
    C: SetAssociativeCacheContext,
    TAG_T: Tag,
{
    /// Mirrors Zig's `value_count_max_multiple`.
    pub const VALUE_COUNT_MAX_MULTIPLE: usize = {
        const fn max_u(a: usize, b: usize) -> usize {
            if a > b { a } else { b }
        }
        const fn min_u(a: usize, b: usize) -> usize {
            if a < b { a } else { b }
        }

        let value_size = size_of::<C::Value>();
        let values_term = (max_u(value_size, CACHE_LINE_SIZE) / min_u(value_size, CACHE_LINE_SIZE)) * WAYS;
        let counts_term = (CACHE_LINE_SIZE * 8) / CLOCK_BITS;
        max_u(values_term, counts_term)
    };

    #[inline]
    fn value_alignment() -> usize {
        if VALUE_ALIGNMENT == 0 {
            align_of::<C::Value>()
        } else {
            VALUE_ALIGNMENT
        }
    }

    #[inline]
    fn max_count() -> u8 {
        debug_assert!(CLOCK_BITS <= 8);
        ((1u16 << CLOCK_BITS) - 1) as u8
    }

    #[inline]
    fn wrap_way(way: usize) -> usize {
        // WAYS is power-of-two.
        way & (WAYS - 1)
    }

    /// Creates a new cache.
    ///
    /// This is the Rust analogue of Zig's `init(allocator, value_count_max, options)`.
    pub fn new(value_count_max: usize, options: Options<'_>) -> Self {
        // --- Zig comptime asserts (runtime in Rust) ---
        assert!(size_of::<C::Key>().is_power_of_two());
        assert!(size_of::<C::Value>().is_power_of_two());

        match WAYS {
            2 | 4 | 16 => {}
            _ => panic!("WAYS must be 2, 4, or 16"),
        }
        match TAG_T::BITS {
            8 | 16 => {}
            _ => panic!("tag bits must be 8 or 16"),
        }
        match CLOCK_BITS {
            1 | 2 | 4 => {}
            _ => panic!("CLOCK_BITS must be 1, 2, or 4"),
        }

        let value_alignment = Self::value_alignment();
        assert!(value_alignment >= align_of::<C::Value>());
        assert!(size_of::<C::Value>() % value_alignment == 0);

        assert!(WAYS.is_power_of_two());
        assert!(TAG_T::BITS.is_power_of_two());
        assert!(CLOCK_BITS.is_power_of_two());
        assert!(CACHE_LINE_SIZE.is_power_of_two());

        assert!(size_of::<C::Key>() <= size_of::<C::Value>());
        assert!(size_of::<C::Key>() < CACHE_LINE_SIZE);
        assert!(CACHE_LINE_SIZE % size_of::<C::Key>() == 0);

        if CACHE_LINE_SIZE > size_of::<C::Value>() {
            assert!(CACHE_LINE_SIZE % size_of::<C::Value>() == 0);
        } else {
            assert!(size_of::<C::Value>() % CACHE_LINE_SIZE == 0);
        }

        // clock hand width checks
        assert!(CLOCK_HAND_BITS.is_power_of_two());
        assert!((1usize << CLOCK_HAND_BITS) == WAYS);

        // --- Zig init-time asserts ---
        assert!(value_count_max > 0);
        assert!(value_count_max >= WAYS);
        assert!(value_count_max % WAYS == 0);

        let sets = value_count_max / WAYS;

        let values_size_max = value_count_max
            .checked_mul(size_of::<C::Value>())
            .expect("values_size_max overflow");
        assert!(values_size_max >= CACHE_LINE_SIZE);
        assert!(values_size_max % CACHE_LINE_SIZE == 0);

        // counts_size bytes = value_count_max * CLOCK_BITS / 8
        let counts_bits = value_count_max
            .checked_mul(CLOCK_BITS)
            .expect("counts_bits overflow");
        assert!(counts_bits % 8 == 0);
        let counts_size = counts_bits / 8;
        assert!(counts_size >= CACHE_LINE_SIZE);
        assert!(counts_size % CACHE_LINE_SIZE == 0);
        assert!(counts_size % 8 == 0);
        let counts_words_len = counts_size / 8;

        // clocks_size bytes = sets * CLOCK_HAND_BITS / 8
        let clocks_bits = sets
            .checked_mul(CLOCK_HAND_BITS)
            .expect("clocks_bits overflow");
        assert!(clocks_bits % 8 == 0);
        let clocks_size = clocks_bits / 8;
        // In Zig they `maybe(...)` these, i.e. documentation-only.
        let _ = clocks_size;
        let clocks_words_len = div_ceil_u64(clocks_bits as u64, 64) as usize;

        // value_count_max_multiple (matches Zig's associated const)
        assert!(value_count_max % Self::VALUE_COUNT_MAX_MULTIPLE == 0);

        // Allocate.
        let tags = vec![TAG_T::default(); value_count_max];
        let values = AlignedBuf::<C::Value>::new_uninit(value_count_max, value_alignment);
        let counts = PackedUnsignedIntegerArray::<CLOCK_BITS>::new_zeroed(counts_words_len);
        let clocks = PackedUnsignedIntegerArray::<CLOCK_HAND_BITS>::new_zeroed(clocks_words_len);

        let mut sac = Self {
            name: options.name.to_string(),
            sets,
            metrics: Metrics::default(),
            tags,
            values,
            counts: UnsafeCell::new(counts),
            clocks: UnsafeCell::new(clocks),
            _marker: PhantomData,
        };

        sac.reset();
        sac
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    pub fn sets(&self) -> usize {
        self.sets
    }

    #[inline]
    pub fn metrics(&self) -> (u64, u64) {
        (self.metrics.hits(), self.metrics.misses())
    }

    pub fn reset(&mut self) {
        self.tags.fill(TAG_T::default());
        unsafe {
            (*self.counts.get()).words_mut().fill(0);
            (*self.clocks.get()).words_mut().fill(0);
        }
        self.metrics.reset();
    }

    /// Returns the index of `key` if present.
    ///
    /// Semantics match Zig's `get_index(self: *const, key) ?usize`:
    /// - increments metrics
    /// - increments the entry's Count with saturation
    pub fn get_index(&self, key: C::Key) -> Option<usize> {
        let set = self.associate(key);
        if let Some(way) = self.search(set.offset, set.tag, key) {
            self.metrics.hits.set(self.metrics.hits.get() + 1);

            let idx = set.offset + way;
            let count = self.counts_get(idx);
            let next = if count == Self::max_count() {
                count
            } else {
                count + 1
            };
            self.counts_set(idx, next);
            Some(idx)
        } else {
            self.metrics.misses.set(self.metrics.misses.get() + 1);
            None
        }
    }

    /// Returns a reference to the value for `key` if present.
    pub fn get(&self, key: C::Key) -> Option<&C::Value> {
        let index = self.get_index(key)?;
        Some(self.values.get_ref(index))
    }

    /// Remove `key` if present.
    /// Returns the removed value.
    pub fn remove(&mut self, key: C::Key) -> Option<C::Value> {
        let set = self.associate(key);
        let way = self.search(set.offset, set.tag, key)?;

        let idx = set.offset + way;
        let removed = self.values.read_copy(idx);
        self.counts_set(idx, 0);
        self.values.write_uninit(idx);
        Some(removed)
    }

    /// Hint that `key` is less likely to be accessed in the future.
    pub fn demote(&mut self, key: C::Key) {
        let set = self.associate(key);
        let Some(way) = self.search(set.offset, set.tag, key) else { return; };
        let idx = set.offset + way;
        self.counts_set(idx, 1);
    }

    pub struct UpsertResult<V> {
        pub index: usize,
        pub updated: UpdateOrInsert,
        pub evicted: Option<V>,
    }

    /// Upsert a value, evicting an older entry if needed.
    pub fn upsert(&mut self, value: &C::Value) -> UpsertResult<C::Value> {
        let key = C::key_from_value(value);
        let set = self.associate(key);

        if let Some(way) = self.search(set.offset, set.tag, key) {
            let idx = set.offset + way;
            // Zig sets count=1 before overwriting.
            self.counts_set(idx, 1);
            let evicted = self.values.read_copy(idx);
            self.values.write(idx, *value);
            return UpsertResult {
                index: idx,
                updated: UpdateOrInsert::Update,
                evicted: Some(evicted),
            };
        }

        let clock_index = set.offset / WAYS;
        let mut way = self.clocks_get(clock_index) as usize;
        debug_assert!(way < WAYS);

        let max_count = Self::max_count() as usize;
        let clock_iterations_max = WAYS * (max_count.saturating_sub(1));

        let mut evicted: Option<C::Value> = None;
        let mut safety_count: usize = 0;

        while safety_count <= clock_iterations_max {
            let idx = set.offset + way;
            let mut count = self.counts_get(idx) as usize;
            if count == 0 {
                break; // free
            }

            count -= 1;
            self.counts_set(idx, count as u8);
            if count == 0 {
                evicted = Some(self.values.read_copy(idx));
                break;
            }

            safety_count += 1;
            way = Self::wrap_way(way + 1);
        }

        assert!(self.counts_get(set.offset + way) == 0);

        // Write tag/value, then publish Count=1.
        self.tags[set.offset + way] = set.tag;
        self.values.write(set.offset + way, *value);
        self.counts_set(set.offset + way, 1);
        self.clocks_set(clock_index, Self::wrap_way(way + 1) as u8);

        UpsertResult {
            index: set.offset + way,
            updated: UpdateOrInsert::Insert,
            evicted,
        }
    }

    /// Debug: prints derived layout parameters.
    pub fn inspect_layout() {
        let clock_hand_bits = CLOCK_HAND_BITS;
        let tags_per_line = (CACHE_LINE_SIZE * 8) / (WAYS * TAG_T::BITS);
        let clocks_per_line = (CACHE_LINE_SIZE * 8) / (WAYS * CLOCK_BITS);
        let clock_hands_per_line = (CACHE_LINE_SIZE * 8) / clock_hand_bits;

        println!(
            "Key_bits={} Value_size={} ways={} tag_bits={} clock_bits={} clock_hand_bits={} tags_per_line={} clocks_per_line={} clock_hands_per_line={}",
            size_of::<C::Key>() * 8,
            size_of::<C::Value>(),
            WAYS,
            TAG_T::BITS,
            CLOCK_BITS,
            clock_hand_bits,
            tags_per_line,
            clocks_per_line,
            clock_hands_per_line,
        );
    }

    // ----- Internals -----

    #[derive(Clone, Copy)]
    struct Set<T> {
        tag: T,
        offset: usize,
    }

    #[inline]
    fn associate(&self, key: C::Key) -> Set<TAG_T> {
        let entropy = C::hash(key);
        let tag = TAG_T::truncate(entropy);
        let index = fastrange(entropy, self.sets as u64) as usize;
        let offset = index * WAYS;
        Set { tag, offset }
    }

    /// Returns the way if `key` is present in the set.
    #[inline]
    fn search(&self, set_offset: usize, tag: TAG_T, key: C::Key) -> Option<usize> {
        let tags = &self.tags[set_offset..set_offset + WAYS];
        let ways_mask = Self::search_tags(tags, tag);
        if ways_mask == 0 {
            return None;
        }

        for way in 0..WAYS {
            if ((ways_mask >> way) & 1) == 1 {
                let idx = set_offset + way;
                if self.counts_get(idx) > 0 {
                    let v = self.values.get_ref(idx);
                    if C::key_from_value(v) == key {
                        return Some(way);
                    }
                }
            }
        }
        None
    }

    /// Bitmask of ways whose tag matches.
    /// Lowest bit corresponds to way 0.
    #[inline]
    fn search_tags(tags: &[TAG_T], tag: TAG_T) -> u16 {
        debug_assert_eq!(tags.len(), WAYS);
        let mut bits: u16 = 0;
        for (i, t) in tags.iter().enumerate() {
            if *t == tag {
                bits |= 1u16 << i;
            }
        }
        bits
    }

    #[inline]
    fn counts_get(&self, index: usize) -> u8 {
        unsafe { (*self.counts.get()).get(index) }
    }

    #[inline]
    fn counts_set(&self, index: usize, value: u8) {
        unsafe { (*self.counts.get()).set(index, value) }
    }

    #[inline]
    fn clocks_get(&self, index: usize) -> u8 {
        unsafe { (*self.clocks.get()).get(index) }
    }

    #[inline]
    fn clocks_set(&self, index: usize, value: u8) {
        unsafe { (*self.clocks.get()).set(index, value) }
    }

    // Exposed for tests/debug.
    #[inline]
    pub fn raw_tags(&self) -> &[TAG_T] {
        &self.tags
    }

    #[inline]
    pub fn counts_words_snapshot(&self) -> Vec<u64> {
        // NOTE: we return an owned snapshot to avoid exposing interior-mutable storage.
        unsafe { (*self.counts.get()).words().to_vec() }
    }

    #[inline]
    pub fn clocks_words_snapshot(&self) -> Vec<u64> {
        // NOTE: we return an owned snapshot to avoid exposing interior-mutable storage.
        unsafe { (*self.clocks.get()).words().to_vec() }
    }
}

// ------------------ Tests ------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal deterministic PRNG (SplitMix64).
    #[derive(Clone)]
    struct Prng {
        state: u64,
    }

    impl Prng {
        fn from_seed(seed: u64) -> Self {
            Self { state: seed }
        }

        #[inline]
        fn next_u64(&mut self) -> u64 {
            // splitmix64
            self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        fn gen_usize(&mut self, bound: usize) -> usize {
            assert!(bound > 0);
            // Use fastrange-style multiply-high to map u64 -> [0, bound).
            let x = self.next_u64();
            fastrange(x, bound as u64) as usize
        }

        fn gen_usize_inclusive(&mut self, max_inclusive: usize) -> usize {
            self.gen_usize(max_inclusive + 1)
        }

        fn shuffle<T>(&mut self, slice: &mut [T]) {
            for i in (1..slice.len()).rev() {
                let j = self.gen_usize(i + 1);
                slice.swap(i, j);
            }
        }
    }

    #[test]
    fn packed_unsigned_integer_array_unit() {
        // Mirrors Zig test "PackedUnsignedIntegerArray: unit" for u2.
        let mut words = vec![0u64; 8];
        words[1] = 0b1011_0010;

        let mut p = PackedUnsignedIntegerArray::<2>::from_words(words);

        assert_eq!(0b10u8, p.get(32 + 0));
        assert_eq!(0b00u8, p.get(32 + 1));
        assert_eq!(0b11u8, p.get(32 + 2));
        assert_eq!(0b10u8, p.get(32 + 3));

        p.set(0, 0b01);
        assert_eq!(0b00000001u64, p.words()[0]);
        assert_eq!(0b01u8, p.get(0));

        p.set(1, 0b10);
        assert_eq!(0b00001001u64, p.words()[0]);
        assert_eq!(0b10u8, p.get(1));

        p.set(2, 0b11);
        assert_eq!(0b00111001u64, p.words()[0]);
        assert_eq!(0b11u8, p.get(2));

        p.set(3, 0b11);
        assert_eq!(0b11111001u64, p.words()[0]);
        assert_eq!(0b11u8, p.get(3));

        p.set(3, 0b01);
        assert_eq!(0b01111001u64, p.words()[0]);
        assert_eq!(0b01u8, p.get(3));

        p.set(3, 0b00);
        assert_eq!(0b00111001u64, p.words()[0]);
        assert_eq!(0b00u8, p.get(3));

        p.set(4, 0b11);
        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000001100111001u64,
            p.words()[0]
        );

        p.set(31, 0b11);
        assert_eq!(
            0b1100000000000000000000000000000000000000000000000000001100111001u64,
            p.words()[0]
        );
    }

    fn fuzz_packed<const BITS: usize>() {
        let mut prng = Prng::from_seed(42);
        let len = 1024usize;

        let words_len = PackedUnsignedIntegerArray::<BITS>::words_for_len(len);
        let mut array = PackedUnsignedIntegerArray::<BITS>::new_zeroed(words_len);
        let mut reference = vec![0u8; len];

        for _ in 0..10_000 {
            let index = prng.gen_usize(len);
            let value = (prng.next_u64() & ((1u64 << BITS) - 1)) as u8;

            array.set(index, value);
            reference[index] = value;

            for (i, expected) in reference.iter().enumerate() {
                assert_eq!(*expected, array.get(i));
            }
        }
    }

    #[test]
    fn packed_unsigned_integer_array_fuzz() {
        fuzz_packed::<1>();
        fuzz_packed::<2>();
        fuzz_packed::<4>();
    }

    struct IdentityCtx;

    impl SetAssociativeCacheContext for IdentityCtx {
        type Key = u64;
        type Value = u64;

        #[inline]
        fn key_from_value(value: &Self::Value) -> Self::Key {
            *value
        }

        #[inline]
        fn hash(key: Self::Key) -> u64 {
            key
        }
    }

    struct CollisionCtx;

    impl SetAssociativeCacheContext for CollisionCtx {
        type Key = u64;
        type Value = u64;

        #[inline]
        fn key_from_value(value: &Self::Value) -> Self::Key {
            *value
        }

        #[inline]
        fn hash(_key: Self::Key) -> u64 {
            0
        }
    }

    type DefaultSAC<C> = SetAssociativeCache<C, u8, 16, 2, 64, 0, 4>;

    fn run_eviction_test<C: SetAssociativeCacheContext>() {
        let mut sac = DefaultSAC::<C>::new(16 * 16 * 8, Options { name: "test" });

        // Ensure tags/counts/clocks are zeroed.
        assert!(sac.raw_tags().iter().all(|t| *t == 0));
        assert!(sac.counts_words_snapshot().iter().all(|w| *w == 0));
        assert!(sac.clocks_words_snapshot().iter().all(|w| *w == 0));

        // Fill up the first set entirely.
        for i in 0..16usize {
            assert_eq!(i as u8, sac.clocks_get(0));

            let key = (i * sac.sets()) as u64;
            let _ = sac.upsert(&key);

            assert_eq!(1u8, sac.counts_get(i));
            assert_eq!(key, *sac.get(key).unwrap());
            assert_eq!(2u8, sac.counts_get(i));
        }
        assert_eq!(0u8, sac.clocks_get(0));

        // Insert another element into the first set, causing key 0 to be evicted.
        {
            let key = (16 * sac.sets()) as u64;
            let _ = sac.upsert(&key);
            assert_eq!(1u8, sac.counts_get(0));
            assert_eq!(key, *sac.get(key).unwrap());
            assert_eq!(2u8, sac.counts_get(0));

            assert!(sac.get(0).is_none());

            for i in 1..16usize {
                assert_eq!(1u8, sac.counts_get(i));
            }
        }

        // Ensure removal works.
        {
            let key = (5 * sac.sets()) as u64;
            assert_eq!(key, *sac.get(key).unwrap());
            assert_eq!(2u8, sac.counts_get(5));

            let _ = sac.remove(key);
            assert!(sac.get(key).is_none());
            assert_eq!(0u8, sac.counts_get(5));
        }

        sac.reset();

        assert!(sac.raw_tags().iter().all(|t| *t == 0));
        assert!(sac.counts_words_snapshot().iter().all(|w| *w == 0));
        assert!(sac.clocks_words_snapshot().iter().all(|w| *w == 0));

        // Fill up the first set entirely, maxing out the count for each slot.
        for i in 0..16usize {
            assert_eq!(i as u8, sac.clocks_get(0));
            let key = (i * sac.sets()) as u64;
            let _ = sac.upsert(&key);
            assert_eq!(1u8, sac.counts_get(i));

            // CLOCK_BITS=2 => max count is 3.
            for expected in 2..=3u8 {
                assert_eq!(key, *sac.get(key).unwrap());
                assert_eq!(expected, sac.counts_get(i));
            }

            // Saturation.
            assert_eq!(key, *sac.get(key).unwrap());
            assert_eq!(3u8, sac.counts_get(i));
        }
        assert_eq!(0u8, sac.clocks_get(0));

        // Insert another element into the first set, causing key 0 to be evicted.
        {
            let key = (16 * sac.sets()) as u64;
            let _ = sac.upsert(&key);
            assert_eq!(1u8, sac.counts_get(0));
            assert_eq!(key, *sac.get(key).unwrap());
            assert_eq!(2u8, sac.counts_get(0));

            assert!(sac.get(0).is_none());

            for i in 1..16usize {
                assert_eq!(1u8, sac.counts_get(i));
            }
        }
    }

    #[test]
    fn set_associative_cache_eviction() {
        run_eviction_test::<IdentityCtx>();
    }

    #[test]
    fn set_associative_cache_hash_collision() {
        run_eviction_test::<CollisionCtx>();
    }

    fn search_tags_reference<TagT: Tag, const WAYS: usize>(tags: &[TagT], tag: TagT) -> u16 {
        let mut bits: u16 = 0;
        let mut count = 0u32;
        for (i, t) in tags.iter().enumerate() {
            if *t == tag {
                bits |= 1u16 << i;
                count += 1;
            }
        }
        assert_eq!(count, bits.count_ones());
        bits
    }

    fn run_search_tags_fuzz<TagT: Tag, const WAYS: usize>() {
        let mut prng = Prng::from_seed(42);

        for _ in 0..10_000 {
            let mut tags = vec![TagT::default(); WAYS];
            for t in tags.iter_mut() {
                *t = TagT::truncate(prng.next_u64());
            }

            let tag = TagT::truncate(prng.next_u64());

            let mut indexes: Vec<usize> = (0..WAYS).collect();
            prng.shuffle(&mut indexes);

            let matches_count_min = prng.gen_usize_inclusive(WAYS);
            for &idx in &indexes[..matches_count_min] {
                tags[idx] = tag;
            }

            let expected = search_tags_reference::<TagT, WAYS>(&tags, tag);
            let actual = match WAYS {
                2 => SetAssociativeCache::<IdentityCtx, TagT, 2, 2, 64, 0, 1>::search_tags(&tags, tag),
                4 => SetAssociativeCache::<IdentityCtx, TagT, 4, 2, 64, 0, 2>::search_tags(&tags, tag),
                16 => SetAssociativeCache::<IdentityCtx, TagT, 16, 2, 64, 0, 4>::search_tags(&tags, tag),
                _ => unreachable!("unsupported WAYS"),
            };

            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn set_associative_cache_search_tags_fuzz() {
        run_search_tags_fuzz::<u8, 2>();
        run_search_tags_fuzz::<u8, 4>();
        run_search_tags_fuzz::<u8, 16>();

        run_search_tags_fuzz::<u16, 2>();
        run_search_tags_fuzz::<u16, 4>();
        run_search_tags_fuzz::<u16, 16>();
    }
}
```

Example:
Run the eviction tests with both IdentityCtx and CollisionCtx to confirm tag collisions do not break correctness.

ASCII diagram:
```
set 0 (WAYS=16):
  tags  [t0 t1 ... t15]
  count [c0 c1 ... c15]
  clock hand -> way k
```

Exit criteria/tests:
- packed_unsigned_integer_array_unit and packed_unsigned_integer_array_fuzz pass.
- set_associative_cache_eviction and set_associative_cache_hash_collision pass.
- set_associative_cache_search_tags_fuzz passes for u8 and u16 tags.

## How the Phases Fit Together
- Phase 0 establishes correctness and the CLOCK eviction logic with simple storage.
- Phase 1 introduces uninitialized aligned values and layout invariants while keeping counters simple.
- Phase 2 packs counts/clocks and adds interior mutability without changing the external semantics.
- Phase N adds the full test suite and matches the reference code exactly.

## Proof Sketch or Correctness Notes
- Search correctness: tags only filter candidates; final equality is key_from_value(value) == key, so false positives are eliminated.
- Safety of reads: values are only read when count > 0, and count is only set after writing the value.
- Eviction progress: each miss decrements counts; with bounded counts, a zero is reached within WAYS * (max_count - 1) steps.
- Saturation: counts never exceed max_count, so the clock walk bound holds.

## Performance Notes
- fastrange avoids modulo and is fast for 64-bit entropy; it is not identical to modulo.
- Packed counts/clocks reduce metadata footprint and improve cache locality.
- Structure-of-arrays keeps tags/counts hot and avoids loading full values during search.
- Aligned values reduce false sharing with cache lines when values are large.

## Pitfalls and Edge Cases
- Reading a value when count == 0 can observe uninitialized memory.
- Forgetting saturating increment can overflow counts and break eviction bounds.
- CLOCK_HAND_BITS must satisfy (1 << CLOCK_HAND_BITS) == WAYS.
- Tag collisions are expected; always verify keys.
- value_count_max must be a multiple of VALUE_COUNT_MAX_MULTIPLE.

## Final Checklist
- [ ] Matches reference semantics
- [ ] Each phase is self-sufficient
- [ ] Tests cover edge cases
- [ ] Final phase includes the complete code listing
