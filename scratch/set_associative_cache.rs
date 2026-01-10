// // Port of `set_associative_cache.zig` to Rust.
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
#![feature(portable_simd)]

use core::marker::PhantomData;
use core::mem::{align_of, size_of, MaybeUninit};
use core::ptr::NonNull;
use std::alloc::{alloc, dealloc, Layout as AllocLayout};
use std::cell::UnsafeCell;

/// Fast alternative to modulo reduction (not equivalent to modulo).
///
/// Matches stdx.fastrange():
///   return ((word as u128 * p as u128) >> 64) as u64
#[inline]
pub fn fastrange(word: u64, p: u64) -> u64 {
    debug_assert!(p > 0);
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

#[derive(Debug, Default)]
pub struct Metrics {
    hits: u64,
    misses: u64,
}

impl Metrics {
    #[inline]
    pub fn reset(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }

    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses
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
    pub fn get(&self, index: u64) -> u8 {
        let uints_per_word = Self::uints_per_word() as u64;
        let word_index = index / uints_per_word;
        let within = index % uints_per_word;
        let shift = (within as usize) * BITS;
        debug_assert!(word_index < self.words.len() as u64);
        let word = self.words[word_index as usize];
        ((word >> shift) & Self::mask_value()) as u8
    }

    /// Sets the unsigned integer at `index` to `value`.
    #[inline]
    pub fn set(&mut self, index: u64, value: u8) {
        debug_assert!((value as u64) <= Self::mask_value());
        let uints_per_word = Self::uints_per_word() as u64;
        let word_index = index / uints_per_word;
        let within = index % uints_per_word;
        let shift = (within as usize) * BITS;
        let mask = Self::mask_value() << shift;
        debug_assert!(word_index < self.words.len() as u64);
        let w = &mut self.words[word_index as usize];
        *w &= !mask;
        *w |= (value as u64) << shift;
    }
}

/// Raw aligned buffer of `T` values stored as `MaybeUninit<T>`.
///
/// The memory is allocated but not initialized.
/// We treat entries as valid only when the corresponding Count > 0.
#[derive(Debug)]
pub struct AlignedBuf<T> {
    ptr: NonNull<MaybeUninit<T>>,
    len: usize,
    layout: AllocLayout,
    _marker: PhantomData<T>,
}

impl<T> AlignedBuf<T> {
    pub fn new_uninit(len: usize, alignment: usize) -> Self {
        assert!(len > 0);
        assert!(alignment >= align_of::<T>());
        assert!(alignment.is_power_of_two());
        // Ensures each element is aligned if we stride by `size_of::<T>()`.
        assert!(size_of::<T>() % alignment == 0);

        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("size overflow");

        let layout = AllocLayout::from_size_align(bytes, alignment).expect("bad layout");

        let raw = unsafe { alloc(layout) } as *mut MaybeUninit<T>;
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
    pub fn get_ptr(&self, index: usize) -> *mut T {
        assert!(index < self.len);
        self.ptr.as_ptr().wrapping_add(index) as *mut T
    }

    #[inline]
    pub fn get_ref(&self, index: usize) -> &T {
        assert!(index < self.len);
        // SAFETY: Callers ensure the entry is initialized (Count > 0).
        unsafe { (&*self.ptr.as_ptr().add(index)).assume_init_ref() }
    }

    #[inline]
    pub fn read_copy(&self, index: usize) -> T
    where
        T: Copy,
    {
        assert!(index < self.len);
        // SAFETY: Callers ensure the entry is initialized (Count > 0).
        unsafe { (&*self.ptr.as_ptr().add(index)).assume_init_read() }
    }

    #[inline]
    pub fn write(&mut self, index: usize, value: T) {
        assert!(index < self.len);
        unsafe {
            self.ptr
                .as_ptr()
                .add(index)
                .write(MaybeUninit::new(value));
        }
    }

    /// Best-effort "poison" on remove. Not required for correctness.
    #[inline]
    pub fn write_uninit(&mut self, index: usize) {
        assert!(index < self.len);
        unsafe {
            // Write an uninitialized `T`. This is not relied upon for correctness;
            // counts are the source of truth.
            self.ptr.as_ptr().add(index).write(MaybeUninit::uninit());
        }
    }
}

impl<T> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

pub struct Options<'a> {
    pub name: &'a str,
}

pub struct UpsertResult<V> {
    pub index: usize,
    pub updated: UpdateOrInsert,
    pub evicted: Option<V>,
}

#[derive(Clone, Copy)]
struct Set<TagT, ValueT, const WAYS: usize> {
    tag: TagT,
    offset: u64,
    tags: *mut [TagT; WAYS],
    values: *mut [ValueT; WAYS],
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
    'a,
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
    name: &'a str,
    sets: u64,

    // TODO Expose these as metrics.
    // Explicitly allocated so that get/get_index can take `&self`.
    metrics: Box<UnsafeCell<Metrics>>,

    /// A short, partial hash of a Key, corresponding to a Value.
    /// Because the tag is small, collisions are possible: `tag(v1) == tag(v2)` does not
    /// imply `v1 == v2`. However, most of the time, where the tag differs, a full key
    /// comparison can be avoided. Since tags are 16-32x smaller than keys, they can also
    /// be kept hot in cache.
    tags: Vec<TAG_T>,

    /// When the corresponding Count is zero, the Value is absent.
    values: AlignedBuf<C::Value>,

    /// Each value has a Count, which tracks the number of recent reads.
    ///
    /// * A Count is incremented when the value is accessed by `get`.
    /// * A Count is decremented when a cache write to the value's Set misses.
    /// * The value is evicted when its Count reaches zero.
    counts: UnsafeCell<PackedUnsignedIntegerArray<CLOCK_BITS>>,

    /// Each set has a Clock: a counter that cycles between each of the set's ways (i.e. slots).
    ///
    /// On cache write, entries are checked for occupancy (or eviction) beginning from the
    /// clock's position, wrapping around.
    ///
    /// The algorithm implemented is \"CLOCK Nth-Chance\" - each way has more than one bit,
    /// to give ways more than one chance before eviction.
    ///
    /// * A similar algorithm called \"RRIParoo\" is described in
    ///   \"Kangaroo: Caching Billions of Tiny Objects on Flash\".
    /// * For more general information on CLOCK algorithms, see:
    ///   https://en.wikipedia.org/wiki/Page_replacement_algorithm.
    clocks: UnsafeCell<PackedUnsignedIntegerArray<CLOCK_HAND_BITS>>,

    _marker: PhantomData<C>,
}

impl<
        'a,
        C,
        TAG_T,
        const WAYS: usize,
        const CLOCK_BITS: usize,
        const CACHE_LINE_SIZE: usize,
        const VALUE_ALIGNMENT: usize,
        const CLOCK_HAND_BITS: usize,
    > SetAssociativeCache<
        'a,
        C,
        TAG_T,
        WAYS,
        CLOCK_BITS,
        CACHE_LINE_SIZE,
        VALUE_ALIGNMENT,
        CLOCK_HAND_BITS,
    >
where
    C: SetAssociativeCacheContext,
    TAG_T: Tag,
{
    /// Mirrors Zig's `value_count_max_multiple`.
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
        // WAYS is power-of-two.
        way & (WAYS - 1)
    }

    #[inline]
    fn metrics_ref(&self) -> &Metrics {
        // SAFETY: `metrics` is only mutated through this cache.
        unsafe { &*self.metrics.as_ref().get() }
    }

    #[inline]
    fn metrics_mut(&self) -> &mut Metrics {
        // SAFETY: `metrics` is only mutated through this cache.
        unsafe { &mut *self.metrics.as_ref().get() }
    }

    #[inline]
    fn index_usize(index: u64) -> usize {
        let idx = index as usize;
        debug_assert_eq!(idx as u64, index);
        idx
    }

    /// Creates a new cache.
    ///
    /// This is the Rust analogue of Zig's `init(allocator, value_count_max, options)`.
    pub fn init(value_count_max: u64, options: Options<'a>) -> Self {
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

        let ways_u64 = WAYS as u64;
        let cache_line_u64 = CACHE_LINE_SIZE as u64;
        let tag_bits_u64 = TAG_T::BITS as u64;
        let clock_bits_u64 = CLOCK_BITS as u64;
        let clock_hand_bits_u64 = CLOCK_HAND_BITS as u64;

        let tags_divisor = ways_u64 * tag_bits_u64;
        assert!(tags_divisor > 0);
        assert!((cache_line_u64 * 8) % tags_divisor == 0);
        let _tags_per_line = (cache_line_u64 * 8) / tags_divisor;
        assert!(_tags_per_line > 0);

        let clocks_divisor = ways_u64 * clock_bits_u64;
        assert!(clocks_divisor > 0);
        assert!((cache_line_u64 * 8) % clocks_divisor == 0);
        let _clocks_per_line = (cache_line_u64 * 8) / clocks_divisor;
        assert!(_clocks_per_line > 0);

        assert!((cache_line_u64 * 8) % clock_hand_bits_u64 == 0);
        let _clock_hands_per_line = (cache_line_u64 * 8) / clock_hand_bits_u64;
        assert!(_clock_hands_per_line > 0);

        // --- Zig init-time asserts ---
        assert!(value_count_max > 0);
        assert!(value_count_max >= ways_u64);
        assert!(value_count_max % ways_u64 == 0);

        let sets = value_count_max / ways_u64;

        let value_size = size_of::<C::Value>() as u64;
        let values_size_max = value_count_max
            .checked_mul(value_size)
            .expect("values_size_max overflow");
        assert!(values_size_max >= cache_line_u64);
        assert!(values_size_max % cache_line_u64 == 0);

        // counts_size bytes = value_count_max * CLOCK_BITS / 8
        let counts_bits = value_count_max
            .checked_mul(clock_bits_u64)
            .expect("counts_bits overflow");
        assert!(counts_bits % 8 == 0);
        let counts_size = counts_bits / 8;
        assert!(counts_size >= cache_line_u64);
        assert!(counts_size % cache_line_u64 == 0);
        assert!(counts_size % 8 == 0);
        let counts_words_len = counts_size / 8;

        // clocks_size bytes = sets * CLOCK_HAND_BITS / 8
        let clocks_bits = sets
            .checked_mul(clock_hand_bits_u64)
            .expect("clocks_bits overflow");
        assert!(clocks_bits % 8 == 0);
        let clocks_size = clocks_bits / 8;
        // In Zig they `maybe(...)` these, i.e. documentation-only.
        let _ = clocks_size;
        let clocks_words_len = div_ceil_u64(clocks_bits, 64);

        // value_count_max_multiple (matches Zig's associated const)
        assert!(value_count_max % Self::VALUE_COUNT_MAX_MULTIPLE == 0);

        // Allocate.
        assert!(value_count_max <= usize::MAX as u64);
        let value_count_max_usize =
            usize::try_from(value_count_max).expect("value_count_max overflow usize");
        let counts_words_len_usize =
            usize::try_from(counts_words_len).expect("counts_words_len overflow usize");
        let clocks_words_len_usize =
            usize::try_from(clocks_words_len).expect("clocks_words_len overflow usize");

        let tags = vec![TAG_T::default(); value_count_max_usize];
        let values = AlignedBuf::<C::Value>::new_uninit(value_count_max_usize, value_alignment);
        let counts = PackedUnsignedIntegerArray::<CLOCK_BITS>::new_zeroed(counts_words_len_usize);
        let clocks = PackedUnsignedIntegerArray::<CLOCK_HAND_BITS>::new_zeroed(clocks_words_len_usize);

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
        self.tags.fill(TAG_T::default());
        unsafe {
            (*self.counts.get()).words_mut().fill(0);
            (*self.clocks.get()).words_mut().fill(0);
        }
        self.metrics_mut().reset();
    }

    /// Returns the index of `key` if present.
    ///
    /// Semantics match Zig's `get_index(self: *const, key) ?usize`:
    /// - increments metrics
    /// - increments the entry's Count with saturation
    pub fn get_index(&self, key: C::Key) -> Option<usize> {
        let set = self.associate(key);
        if let Some(way) = self.search(set, key) {
            self.metrics_mut().hits += 1;

            let idx = set.offset + way as u64;
            let count = self.counts_get(idx);
            let next = count.saturating_add(1);
            self.counts_set(idx, next);
            Some(Self::index_usize(idx))
        } else {
            self.metrics_mut().misses += 1;
            None
        }
    }

    /// Returns a pointer to the value for `key` if present.
    /// The pointer is aligned to `value_alignment`.
    pub fn get(&self, key: C::Key) -> Option<*mut C::Value> {
        let index = self.get_index(key)?;
        Some(self.values.get_ptr(index))
    }

    /// Remove `key` if present.
    /// Returns the removed value.
    pub fn remove(&mut self, key: C::Key) -> Option<C::Value> {
        let set = self.associate(key);
        let way = self.search(set, key)?;

        let idx = set.offset + way as u64;
        let idx_usize = Self::index_usize(idx);
        let removed = self.values.read_copy(idx_usize);
        self.counts_set(idx, 0);
        self.values.write_uninit(idx_usize);
        Some(removed)
    }

    /// Hint that `key` is less likely to be accessed in the future.
    pub fn demote(&mut self, key: C::Key) {
        let set = self.associate(key);
        let Some(way) = self.search(set, key) else { return; };
        let idx = set.offset + way as u64;
        self.counts_set(idx, 1);
    }

    /// Upsert a value, evicting an older entry if needed.
    pub fn upsert(&mut self, value: &C::Value) -> UpsertResult<C::Value> {
        let key = C::key_from_value(value);
        let set = self.associate(key);
        let offset_usize = Self::index_usize(set.offset);

        if let Some(way) = self.search(set, key) {
            let way_usize = way as usize;
            let idx = set.offset + way as u64;
            let idx_usize = offset_usize + way_usize;
            // Zig sets count=1 before overwriting.
            self.counts_set(idx, 1);
            let evicted = self.values.read_copy(idx_usize);
            self.values.write(idx_usize, *value);
            return UpsertResult {
                index: idx_usize,
                updated: UpdateOrInsert::Update,
                evicted: Some(evicted),
            };
        }

        let clock_index = set.offset / WAYS as u64;
        let mut way = self.clocks_get(clock_index) as usize;
        debug_assert!(way < WAYS);

        let max_count = Self::max_count() as usize;
        // The maximum iterations happens when all counts are at the maximum.
        let clock_iterations_max = WAYS * (max_count.saturating_sub(1));

        let mut evicted: Option<C::Value> = None;
        let mut safety_count: usize = 0;

        while safety_count <= clock_iterations_max {
            let idx = set.offset + way as u64;
            let idx_usize = offset_usize + way;
            let mut count = self.counts_get(idx) as usize;
            if count == 0 {
                break; // Way is already free.
            }

            count -= 1;
            self.counts_set(idx, count as u8);
            if count == 0 {
                // Way has become free.
                evicted = Some(self.values.read_copy(idx_usize));
                break;
            }

            safety_count += 1;
            way = Self::wrap_way(way + 1);
        }
        if safety_count > clock_iterations_max {
            unreachable!("clock eviction exceeded maximum iterations");
        }

        assert!(self.counts_get(set.offset + way as u64) == 0);

        // Write tag/value, then publish Count=1.
        self.tags[offset_usize + way] = set.tag;
        self.values.write(offset_usize + way, *value);
        self.counts_set(set.offset + way as u64, 1);
        self.clocks_set(clock_index, Self::wrap_way(way + 1) as u8);

        UpsertResult {
            index: offset_usize + way,
            updated: UpdateOrInsert::Insert,
            evicted,
        }
    }

    /// Debug: prints derived layout parameters.
    pub fn inspect() {
        let clock_hand_bits = CLOCK_HAND_BITS as u64;
        let tags_per_line = (CACHE_LINE_SIZE as u64 * 8) / (WAYS as u64 * TAG_T::BITS as u64);
        let clocks_per_line = (CACHE_LINE_SIZE as u64 * 8) / (WAYS as u64 * CLOCK_BITS as u64);
        let clock_hands_per_line = (CACHE_LINE_SIZE as u64 * 8) / clock_hand_bits;

        println!(
            "\nKey={} Value={} ways={} tag_bits={} clock_bits={} clock_hand_bits={} tags_per_line={} clocks_per_line={} clock_hands_per_line={}",
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

    #[inline]
    fn associate(&self, key: C::Key) -> Set<TAG_T, C::Value, WAYS> {
        let entropy = C::hash(key);
        let tag = TAG_T::truncate(entropy);
        let index = fastrange(entropy, self.sets);
        let offset = index * WAYS as u64;

        let offset_usize = Self::index_usize(offset);
        debug_assert!(offset_usize + WAYS <= self.tags.len());
        debug_assert!(offset_usize + WAYS <= self.values.len());

        let tags = unsafe { self.tags.as_ptr().add(offset_usize) as *mut [TAG_T; WAYS] };
        let values = unsafe { self.values.as_ptr().add(offset_usize) as *mut [C::Value; WAYS] };

        Set {
            tag,
            offset,
            tags,
            values,
        }
    }

    /// Returns the way if `key` is present in the set.
    #[inline]
    fn search(&self, set: Set<TAG_T, C::Value, WAYS>, key: C::Key) -> Option<u16> {
        let tags = unsafe { &*set.tags };
        let ways_mask = Self::search_tags(tags, set.tag);
        if ways_mask == 0 {
            return None;
        }

        // Iterate over all ways to help the OOO execution.
        for way in 0..WAYS {
            if ((ways_mask >> way) & 1) == 1
                && self.counts_get(set.offset + way as u64) > 0
            {
                // SAFETY: count > 0 means the value is initialized.
                let v = unsafe { &(*set.values)[way] };
                if C::key_from_value(v) == key {
                    return Some(way as u16);
                }
            }
        }
        None
    }

    /// Where each set bit represents the index of a way that has the same tag.
    #[inline]
    fn search_tags(tags: &[TAG_T; WAYS], tag: TAG_T) -> u16 {
        use core::simd::prelude::SimdPartialEq;
        use core::simd::Simd;

        let x = Simd::<TAG_T, WAYS>::from_array(*tags);
        let y = Simd::<TAG_T, WAYS>::splat(tag);
        let mask = x.simd_eq(y);
        mask.to_bitmask() as u16
    }

    #[inline]
    fn counts_get(&self, index: u64) -> u8 {
        unsafe { (*self.counts.get()).get(index) }
    }

    #[inline]
    fn counts_set(&self, index: u64, value: u8) {
        unsafe { (*self.counts.get()).set(index, value) }
    }

    #[inline]
    fn clocks_get(&self, index: u64) -> u8 {
        unsafe { (*self.clocks.get()).get(index) }
    }

    #[inline]
    fn clocks_set(&self, index: u64, value: u8) {
        unsafe { (*self.clocks.get()).set(index, value) }
    }

    // Exposed for tests/debug.
    #[cfg(test)]
    #[inline]
    fn raw_tags(&self) -> &[TAG_T] {
        &self.tags
    }

    #[cfg(test)]
    #[inline]
    fn counts_words_snapshot(&self) -> Vec<u64> {
        // NOTE: we return an owned snapshot to avoid exposing interior-mutable storage.
        unsafe { (*self.counts.get()).words().to_vec() }
    }

    #[cfg(test)]
    #[inline]
    fn clocks_words_snapshot(&self) -> Vec<u64> {
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

            array.set(index as u64, value);
            reference[index] = value;

            for (i, expected) in reference.iter().enumerate() {
                assert_eq!(*expected, array.get(i as u64));
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

    type DefaultSAC<C> = SetAssociativeCache<'static, C, u8, 16, 2, 64, 0, 4>;

    fn run_eviction_test<C: SetAssociativeCacheContext>() {
        let mut sac = DefaultSAC::<C>::init(16 * 16 * 8, Options { name: "test" });

        // Ensure tags/counts/clocks are zeroed.
        assert!(sac.raw_tags().iter().all(|t| *t == 0));
        assert!(sac.counts_words_snapshot().iter().all(|w| *w == 0));
        assert!(sac.clocks_words_snapshot().iter().all(|w| *w == 0));

        // Fill up the first set entirely.
        for i in 0..16usize {
            assert_eq!(i as u8, sac.clocks_get(0));

            let key = (i as u64) * sac.sets;
            let _ = sac.upsert(&key);

            assert_eq!(1u8, sac.counts_get(i as u64));
            assert_eq!(key, unsafe { *sac.get(key).unwrap() });
            assert_eq!(2u8, sac.counts_get(i as u64));
        }
        assert_eq!(0u8, sac.clocks_get(0));

        // Insert another element into the first set, causing key 0 to be evicted.
        {
            let key = 16u64 * sac.sets;
            let _ = sac.upsert(&key);
            assert_eq!(1u8, sac.counts_get(0));
            assert_eq!(key, unsafe { *sac.get(key).unwrap() });
            assert_eq!(2u8, sac.counts_get(0));

            assert!(sac.get(0).is_none());

            for i in 1..16usize {
                assert_eq!(1u8, sac.counts_get(i as u64));
            }
        }

        // Ensure removal works.
        {
            let key = 5u64 * sac.sets;
            assert_eq!(key, unsafe { *sac.get(key).unwrap() });
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
            let key = (i as u64) * sac.sets;
            let _ = sac.upsert(&key);
            assert_eq!(1u8, sac.counts_get(i as u64));

            // CLOCK_BITS=2 => max count is 3.
            for expected in 2..=3u8 {
                assert_eq!(key, unsafe { *sac.get(key).unwrap() });
                assert_eq!(expected, sac.counts_get(i as u64));
            }

            // Saturation.
            assert_eq!(key, unsafe { *sac.get(key).unwrap() });
            assert_eq!(3u8, sac.counts_get(i as u64));
        }
        assert_eq!(0u8, sac.clocks_get(0));

        // Insert another element into the first set, causing key 0 to be evicted.
        {
            let key = 16u64 * sac.sets;
            let _ = sac.upsert(&key);
            assert_eq!(1u8, sac.counts_get(0));
            assert_eq!(key, unsafe { *sac.get(key).unwrap() });
            assert_eq!(2u8, sac.counts_get(0));

            assert!(sac.get(0).is_none());

            for i in 1..16usize {
                assert_eq!(1u8, sac.counts_get(i as u64));
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

    fn search_tags_reference<TagT: Tag, const WAYS: usize>(tags: &[TagT; WAYS], tag: TagT) -> u16 {
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
            let mut tags = [TagT::default(); WAYS];
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
                2 => SetAssociativeCache::<'static, IdentityCtx, TagT, 2, 2, 64, 0, 1>::search_tags(&tags, tag),
                4 => SetAssociativeCache::<'static, IdentityCtx, TagT, 4, 2, 64, 0, 2>::search_tags(&tags, tag),
                16 => SetAssociativeCache::<'static, IdentityCtx, TagT, 16, 2, 64, 0, 4>::search_tags(&tags, tag),
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
