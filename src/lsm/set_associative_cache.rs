#![allow(dead_code)]

use std::{
    alloc::{Layout as AllocLayout, alloc, dealloc},
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::NonNull,
};

use crate::stdx::fastrange::fast_range;

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

/// A heap-allocated buffer with custom alignment for cache-line-aligned storage.
///
/// Elements are stored as `MaybeUninit<T>`; callers must track initialization state.
/// `Drop` deallocates memory but does NOT run element destructors—intended for `Copy` types.
#[derive(Debug)]
#[allow(clippy::len_without_is_empty)] // Buffer is never empty (len > 0 enforced at construction)
pub struct AlignedBuf<T> {
    ptr: NonNull<MaybeUninit<T>>,
    len: usize,
    layout: AllocLayout,
    _marker: PhantomData<T>,
}

impl<T> AlignedBuf<T> {
    /// Allocates an uninitialized buffer with the specified length and alignment.
    ///
    /// # Panics
    ///
    /// Panics if `len == 0`, `alignment < align_of::<T>()`, `alignment` is not a power of two,
    /// `size_of::<T>()` is not a multiple of `alignment`, or allocation fails.
    pub fn new_uninit(len: usize, alignment: usize) -> Self {
        assert!(len > 0);
        assert!(alignment >= align_of::<T>());
        assert!(alignment.is_power_of_two());
        assert!(size_of::<T>().is_multiple_of(alignment));

        let bytes = len.checked_mul(size_of::<T>()).expect("size overflow");

        let layout = AllocLayout::from_size_align(bytes, alignment).expect("bad layout");

        // SAFETY: Layout is valid (size > 0, alignment is power of two, size fits in isize).
        // We check for null below.
        let raw = unsafe { alloc(layout) } as *mut MaybeUninit<T>;
        let ptr = NonNull::new(raw).expect("oom");

        Self {
            ptr,
            len,
            layout,
            _marker: PhantomData,
        }
    }

    /// Returns the number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns a raw pointer to the buffer's first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr() as *const T
    }

    /// Returns a mutable raw pointer to the buffer's first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr() as *mut T
    }

    /// Returns a mutable pointer to the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    #[inline]
    pub fn get_ptr(&self, index: usize) -> *mut T {
        assert!(index < self.len);
        self.ptr.as_ptr().wrapping_add(index) as *mut T
    }

    /// Returns a reference to the element at `index`.
    ///
    /// # Safety
    ///
    /// The slot at `index` must have been initialized via [`write`](Self::write).
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    #[inline]
    pub unsafe fn get_ref(&self, index: usize) -> &T {
        assert!(index < self.len);
        // SAFETY: Caller guarantees the slot is initialized. Pointer is valid and aligned.
        unsafe { (&*self.ptr.as_ptr().add(index)).assume_init_ref() }
    }

    /// Reads a copy of the element at `index`.
    ///
    /// # Safety
    ///
    /// The slot at `index` must have been initialized via [`write`](Self::write).
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    #[inline]
    pub unsafe fn read_copy(&self, index: usize) -> T
    where
        T: Copy,
    {
        assert!(index < self.len);
        // SAFETY: Caller guarantees the slot is initialized. T: Copy prevents double-drop.
        unsafe { (&*self.ptr.as_ptr().add(index)).assume_init_read() }
    }

    /// Writes a value to the slot at `index`, initializing it.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    #[inline]
    pub fn write(&mut self, index: usize, value: T) {
        assert!(index < self.len);
        // SAFETY: Index is bounds-checked. Pointer is valid and properly aligned.
        unsafe {
            self.ptr.as_ptr().add(index).write(MaybeUninit::new(value));
        }
    }

    /// Marks the slot at `index` as uninitialized.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    #[inline]
    pub fn write_uninit(&mut self, index: usize) {
        assert!(index < self.len);
        // SAFETY: Index is bounds-checked. Pointer is valid and properly aligned.
        unsafe {
            self.ptr.as_ptr().add(index).write(MaybeUninit::uninit());
        }
    }
}

impl<T> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        // SAFETY: Layout matches the one used in new_uninit. Pointer is valid.
        // Note: Element destructors are NOT run—this is intentional for Copy types.
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

/// Configuration options for initializing a cache instance.
pub struct Options<'a> {
    /// Human-readable name used for diagnostics.
    pub name: &'a str,
}

/// Result of inserting or updating a cache entry.
pub struct UpsertResult<V> {
    /// Index of the slot written within the cache storage.
    pub index: usize,
    /// Whether the operation replaced an existing entry or inserted a new one.
    pub updated: UpdateOrInsert,
    /// The evicted value, if an insertion displaced an older entry.
    pub evicted: Option<V>,
}

/// Per-set view used during lookups and insertions.
///
/// Each key maps to a set of `WAYS` consecutive slots; this bundles the derived
/// tag and pointers into the backing tag/value arrays for that set. The
/// `offset` is the base index used to address the set's ways.
#[derive(Clone, Copy)]
struct Set<TagT, ValueT, const WAYS: usize> {
    /// Tag derived from the lookup key's hash entropy.
    tag: TagT,
    /// Base index for this set in the tag/value arrays.
    offset: u64,
    /// Tag storage for the `WAYS` slots in this set.
    tags: *mut [TagT; WAYS],
    /// Value storage for the `WAYS` slots in this set.
    values: *mut [ValueT; WAYS],
}

/// N-way set-associative cache with CLOCK Nth-Chance eviction.
///
/// Each key maps to one set of `WAYS` consecutive slots that may contain its
/// value. Tags provide a compact hash prefix to avoid full key comparisons on
/// most misses, while counts/clocks drive the replacement policy.
pub struct SetAssociativeCache<
    'a,
    C,
    TagT,
    const WAYS: usize,
    const CLOCK_BITS: usize,
    const CACHE_LINE_SIZE: usize,
    const VALUE_ALIGNMENT: usize,
    const CLOCK_HAND_BITS: usize,
> where
    C: SetAssociativeCacheContext,
    TagT: Tag,
{
    /// Human-readable cache name for diagnostics.
    name: &'a str,
    /// Number of sets in the cache.
    sets: u64,

    /// Hit/miss counters stored behind interior mutability.
    metrics: Box<UnsafeCell<Metrics>>,

    /// Short, partial hashes of keys stored alongside cached values.
    ///
    /// Because the tag is small, collisions are possible: `tag(k1) == tag(k2)`
    /// does not imply `k1 == k2`. However, most of the time, where the tag
    /// differs, a full key comparison can be avoided. Since tags are 16-32x
    /// smaller than keys, they can also be kept hot in cache.
    tags: Vec<TagT>,

    /// Cache values; a slot is present when its count is non-zero.
    values: AlignedBuf<C::Value>,

    /// Per-slot access counts, tracking recent reads.
    ///
    /// * A count is incremented when a value is accessed by `get`.
    /// * A count is decremented when a cache write to the value's set misses.
    /// * A value is evicted when its count reaches zero.
    counts: UnsafeCell<PackedUnsignedIntegerArray<CLOCK_BITS>>,

    /// Per-set clock hand that rotates across ways to find eviction candidates.
    ///
    /// On cache write, entries are checked for occupancy (or eviction) beginning
    /// from the clock's position, wrapping around. The algorithm implemented is
    /// CLOCK Nth-Chance, where each way has more than one bit to give entries
    /// multiple chances before eviction. A similar algorithm called "RRIParoo"
    /// is described in "Kangaroo: Caching Billions of Tiny Objects on Flash".
    /// For general background, see:
    /// https://en.wikipedia.org/wiki/Page_replacement_algorithm.
    clocks: UnsafeCell<PackedUnsignedIntegerArray<CLOCK_HAND_BITS>>,

    /// Marker for the cache context's key/value types.
    _marker: PhantomData<C>,
}

impl<
    'a,
    C,
    TagT,
    const WAYS: usize,
    const CLOCK_BITS: usize,
    const CACHE_LINE_SIZE: usize,
    const VALUE_ALIGNMENT: usize,
    const CLOCK_HAND_BITS: usize,
>
    SetAssociativeCache<
        'a,
        C,
        TagT,
        WAYS,
        CLOCK_BITS,
        CACHE_LINE_SIZE,
        VALUE_ALIGNMENT,
        CLOCK_HAND_BITS,
    >
where
    C: SetAssociativeCacheContext,
    TagT: Tag + core::simd::SimdElement,
    core::simd::LaneCount<WAYS>: core::simd::SupportedLaneCount,
    core::simd::Simd<TagT, WAYS>: core::simd::cmp::SimdPartialEq<
            Mask = core::simd::Mask<<TagT as core::simd::SimdElement>::Mask, WAYS>,
        >,
{
    pub const VALUE_COUNT_MAX_MULTIPLE: u64 = {
        const fn max_u(a: u64, b: u64) -> u64 {
            if a > b { a } else { b }
        }

        const fn min_u(a: u64, b: u64) -> u64 {
            if a < b { a } else { b }
        }

        let value_size = size_of::<C::Value>() as u64;
        let cache_line = CACHE_LINE_SIZE as u64;
        let ways = WAYS as u64;
        let values_term = (max_u(value_size, cache_line) / min_u(value_size, cache_line)) * ways;
        let counts_term = (cache_line * 8) / CLOCK_BITS as u64;
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

    #[inline]
    fn metric_ref(&self) -> &Metrics {
        unsafe { &*self.metrics.as_ref().get() }
    }

    #[inline]
    fn metrics_mut(&self) -> &mut Metrics {
        unsafe { &mut *self.metrics.as_ref().get() }
    }

    #[inline]
    fn index_usize(index: u64) -> usize {
        let idx = index as usize;
        debug_assert_eq!(idx as u64, index);
        idx
    }

    /// Initializes a cache sized for `value_count_max` values.
    ///
    /// `value_count_max` must be a multiple of `WAYS` and `VALUE_COUNT_MAX_MULTIPLE` so that
    /// tags, values, counts, and clocks stay cache-line aligned. This allocates the backing
    /// arrays and zeroes the tag/count/clock state via `reset`.
    ///
    /// # Panics
    ///
    /// Panics if any layout invariant is violated (ways/tag bits/clock bits, cache-line or value
    /// alignment constraints) or if computed sizes overflow.
    pub fn init(value_count_max: u64, options: Options<'a>) -> Self {
        assert!(size_of::<C::Key>().is_power_of_two());
        assert!(size_of::<C::Value>().is_power_of_two());

        match WAYS {
            2 | 4 | 16 => {}
            _ => panic!("Invalid number of ways"),
        }

        match TagT::BITS {
            8 | 16 => {}
            _ => panic!("tag bits must be 8 or 16"),
        }

        match CLOCK_BITS {
            1 | 2 | 4 => {}
            _ => panic!("CLOCK_BITS must be 1, 2, or 4"),
        }

        let value_alignment = Self::value_alignment();
        assert!(value_alignment >= align_of::<C::Value>());
        assert!(size_of::<C::Value>().is_multiple_of(value_alignment));

        assert!(WAYS.is_power_of_two());
        assert!(TagT::BITS.is_power_of_two());
        assert!(CLOCK_BITS.is_power_of_two());
        assert!(CACHE_LINE_SIZE.is_power_of_two());

        assert!(size_of::<C::Key>() <= size_of::<C::Value>());
        assert!(size_of::<C::Key>() < CACHE_LINE_SIZE);
        assert!(CACHE_LINE_SIZE.is_multiple_of(size_of::<C::Key>()));

        if CACHE_LINE_SIZE > size_of::<C::Value>() {
            assert!(CACHE_LINE_SIZE.is_multiple_of(size_of::<C::Value>()));
        } else {
            assert!(size_of::<C::Value>().is_multiple_of(CACHE_LINE_SIZE));
        }

        assert!(CLOCK_HAND_BITS.is_power_of_two());
        assert_eq!((1usize << CLOCK_HAND_BITS), WAYS);

        let ways_u64 = WAYS as u64;
        let cache_line_u64 = CACHE_LINE_SIZE as u64;
        let tag_bits_u64 = TagT::BITS as u64;
        let clock_bits_u64 = CLOCK_BITS as u64;
        let clock_hand_bits = CLOCK_HAND_BITS as u64;

        let tags_divisor = ways_u64 * tag_bits_u64;
        assert!(tags_divisor > 0);
        assert_eq!((cache_line_u64 * 8) % tags_divisor, 0);
        let _tags_per_line = (cache_line_u64 * 8) / tags_divisor;
        assert!(_tags_per_line > 0);

        let clock_divisor = ways_u64 * clock_bits_u64;
        assert!(clock_divisor > 0);
        assert_eq!((cache_line_u64 * 8) % clock_divisor, 0);
        let _clocks_per_line = (cache_line_u64 * 8) / clock_divisor;
        assert!(_clocks_per_line > 0);

        assert_eq!((cache_line_u64 * 8) % clock_bits_u64, 0);
        let _clock_hand_per_line = (cache_line_u64 * 8) / clock_hand_bits;
        assert!(_clock_hand_per_line > 0);

        assert!(value_count_max > 0);
        assert!(value_count_max >= ways_u64);
        assert_eq!(value_count_max % ways_u64, 0);

        let sets = value_count_max / ways_u64;

        let value_size = size_of::<C::Value>() as u64;
        let values_size_max = value_count_max
            .checked_mul(value_size)
            .expect("values_size_max overflow");
        assert!(values_size_max >= cache_line_u64);
        assert_eq!(values_size_max % cache_line_u64, 0);

        let counts_bits = value_count_max
            .checked_mul(clock_bits_u64)
            .expect("counts_bits overflow");
        assert_eq!(counts_bits % 8, 0);
        let counts_size = counts_bits / 8;
        assert!(counts_size >= cache_line_u64);
        assert_eq!(counts_size % cache_line_u64, 0);
        assert_eq!(counts_size % 8, 0);
        let counts_words_len = counts_size / 8;

        let clocks_bits = sets
            .checked_mul(clock_hand_bits)
            .expect("clocks_bits overflow");
        assert_eq!(clocks_bits % 8, 0);
        let clocks_size = clocks_bits / 8;
        let _ = clocks_size;
        let clocks_words_len = clocks_bits.div_ceil(64);

        assert_eq!(value_count_max % Self::VALUE_COUNT_MAX_MULTIPLE, 0);
        assert!(value_count_max <= usize::MAX as u64);
        let value_count_max_usize =
            usize::try_from(value_count_max).expect("value_count_max overflow usize");
        let counts_words_len_usize =
            usize::try_from(counts_words_len).expect("counts_words_len overflow usize");
        let clocks_words_len_usize =
            usize::try_from(clocks_words_len).expect("clocks_words_len overflow usize");

        let tags = vec![TagT::default(); value_count_max_usize];
        let values = AlignedBuf::<C::Value>::new_uninit(value_count_max_usize, value_alignment);
        let counts = PackedUnsignedIntegerArray::<CLOCK_BITS>::new_zeroed(counts_words_len_usize);
        let clocks =
            PackedUnsignedIntegerArray::<CLOCK_HAND_BITS>::new_zeroed(clocks_words_len_usize);

        let mut sac = Self {
            name: options.name,
            sets,
            metrics: Box::new(UnsafeCell::new(Metrics::default())),
            tags,
            values,
            counts: UnsafeCell::new(counts),
            clocks: UnsafeCell::new(clocks),
            _marker: PhantomData,
        };

        sac.reset();
        sac
    }

    pub fn deinit(mut self) {
        assert!(self.sets > 0);
        self.sets = 0;
    }

    pub fn reset(&mut self) {
        self.tags.fill(TagT::default());
        unsafe {
            (*self.counts.get()).words_mut().fill(0);
            (*self.clocks.get()).words_mut().fill(0);
        }
        self.metrics_mut().reset();
    }

    pub fn get_index(&self, key: C::Key) -> Option<usize> {
        let set = self.associate(key);
        if let Some(way) = self.search(set, key) {
            let metrics = self.metrics_mut();
            metrics.hits.set(metrics.hits.get() + 1);

            let idx = set.offset + way as u64;
            let count = self.counts_get(idx);
            let next = count.saturating_add(1);
            self.counts_set(idx, next);
            Some(Self::index_usize(idx))
        } else {
            let metrics = self.metrics_mut();
            metrics.misses.set(metrics.misses.get() + 1);
            None
        }
    }

    pub fn get(&self, key: C::Key) -> Option<*mut C::Value> {
        let index = self.get_index(key)?;
        Some(self.values.get_ptr(index))
    }

    // ----- Internals -----

    /// Computes the set metadata for `key` (tag, offset, and set-local pointers).
    #[inline]
    fn associate(&self, key: C::Key) -> Set<TagT, C::Value, WAYS> {
        let entropy = C::hash(key);
        let tag = TagT::truncate(entropy);
        let index = fast_range(entropy, self.sets);
        let offset = index * WAYS as u64;

        let offset_usize = Self::index_usize(offset);
        debug_assert!(offset_usize + WAYS <= self.tags.len());
        debug_assert!(offset_usize + WAYS <= self.values.len());

        let tags = unsafe { self.tags.as_ptr().add(offset_usize) as *mut [TagT; WAYS] };
        let values = unsafe { self.values.as_ptr().add(offset_usize) as *mut [C::Value; WAYS] };

        Set {
            tag,
            offset,
            tags,
            values,
        }
    }

    /// If the key is present in the set, returns the way index; otherwise `None`.
    #[inline]
    fn search(&self, set: Set<TagT, C::Value, WAYS>, key: C::Key) -> Option<u16> {
        let tags = unsafe { &*set.tags };
        let ways_mask = Self::search_tags(tags, set.tag);
        if ways_mask == 0 {
            return None;
        }

        for way in 0..WAYS {
            if ((ways_mask >> way) & 1) == 1 && self.counts_get(set.offset + way as u64) > 0 {
                let v = unsafe { &(*set.values)[way] };
                if C::key_from_value(v) == key {
                    return Some(way as u16);
                }
            }
        }
        None
    }

    /// Bitmask of ways whose tag matches `tag` (bit i corresponds to way i).
    #[inline]
    fn search_tags(tags: &[TagT; WAYS], tag: TagT) -> u16 {
        use core::simd::cmp::SimdPartialEq;
        use core::simd::{Mask, Simd};

        let x = Simd::<TagT, WAYS>::from_array(*tags);
        let y = Simd::<TagT, WAYS>::splat(tag);
        let mask: Mask<<TagT as core::simd::SimdElement>::Mask, WAYS> = x.simd_eq(y);
        mask.to_bitmask() as u16
    }

    /// Reads the CLOCK count for a slot at `index`.
    #[inline]
    fn counts_get(&self, index: u64) -> u8 {
        unsafe { (*self.counts.get()).get(index) }
    }

    /// Writes the CLOCK count for a slot at `index`.
    #[inline]
    fn counts_set(&self, index: u64, value: u8) {
        unsafe {
            (*self.counts.get()).set(index, value);
        }
    }

    /// Reads the clock hand value for the set at `index`.
    #[inline]
    fn clocks_get(&self, index: u64) -> u8 {
        unsafe { (*self.clocks.get()).get(index) }
    }

    /// Writes the clock hand value for the set at `index`.
    #[inline]
    fn clocks_set(&self, index: u64, value: u8) {
        unsafe {
            (*self.clocks.get()).set(index, value);
        }
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
