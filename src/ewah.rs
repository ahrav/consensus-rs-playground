//! EWAH (Enhanced Word-Aligned Hybrid) bitmap compression codec.
//!
//! EWAH is a word-aligned compression scheme for sparse bitmaps that achieves
//! good compression ratios while enabling efficient logical operations directly
//! on compressed data. This implementation is optimized for use in the VSR
//! protocol's journal free-set tracking.
//!
//! # Wire Format
//!
//! EWAH compresses bitmaps by alternating between two types of words:
//!
//! - **Marker words**: Encode metadata about subsequent data
//! - **Literal words**: Uncompressed bitmap data
//!
//! ## Marker Word Layout
//!
//! For a word size of W bits, a marker word is structured as:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  Bit 0   │  Bits 1..(W/2)  │           Bits (W/2)..W               │
//! ├──────────┼─────────────────┼───────────────────────────────────────┤
//! │ uniform  │  uniform word   │         literal word count            │
//! │   bit    │     count       │                                       │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! - **uniform_bit**: The fill value (0 or 1) for uniform runs
//! - **uniform_word_count**: Number of words filled with uniform_bit
//! - **literal_word_count**: Number of literal words following this marker
//!
//! # Chunk-Based Streaming
//!
//! Both [`Encoder`] and [`Decoder`] support chunk-based processing for
//! streaming scenarios where data arrives incrementally (e.g., from disk I/O).
//!
//! # Endianness
//!
//! This codec requires a little-endian target architecture. Attempting to
//! compile on a big-endian platform will produce a compile-time error.
//!
//! # References
//!
//! - Lemire et al., "Sorting improves word-aligned bitmap indexes"

use core::{
    cmp::min,
    convert::TryFrom,
    marker::PhantomData,
    mem::size_of,
};

// EWAH uses native word layout in memory. Little-endian is required because:
// 1. The marker word bit layout assumes LSB-first ordering
// 2. Cross-platform encoded data interoperability depends on consistent byte order
#[cfg(not(target_endian = "little"))]
compile_error!("EWAH codec requires a little-endian target");

/// A word type suitable for EWAH bitmap compression.
///
/// This trait abstracts over the word size used for compression, allowing
/// the same algorithm to work with different word widths (8, 16, 32, 64 bits).
/// Larger word sizes provide better compression for sparse bitmaps but require
/// correspondingly aligned memory.
///
/// # Implementer Requirements
///
/// Implementations must satisfy:
/// - `zero()` returns a word with all bits cleared
/// - `ones()` returns a word with all bits set
/// - `to_u128()` and `from_u128()` perform lossless round-trip conversion
///   for values within the word's range
///
/// # Provided Implementations
///
/// This trait is implemented for `u8`, `u16`, `u32`, `u64`, and `usize`.
pub trait EwahWord: Copy + Eq + core::fmt::Debug + 'static {
    /// The number of bits in this word type.
    const BITS: u32;

    /// Returns a word with all bits set to zero.
    fn zero() -> Self;

    /// Returns a word with all bits set to one.
    fn ones() -> Self;

    /// Converts this word to a `u128` for uniform arithmetic operations.
    fn to_u128(self) -> u128;

    /// Constructs a word from a `u128` value, truncating to the word's width.
    fn from_u128(value: u128) -> Self;
}

macro_rules! impl_ewah_word {
    ($t:ty) => {
        impl EwahWord for $t {
            const BITS: u32 = <$t>::BITS;

            #[inline]
            fn zero() -> Self {
                0
            }

            #[inline]
            fn ones() -> Self {
                !0
            }

            #[inline]
            fn to_u128(self) -> u128 {
                self as u128
            }

            #[inline]
            fn from_u128(value: u128) -> Self {
                value as $t
            }
        }
    };
}

impl_ewah_word!(u8);
impl_ewah_word!(u16);
impl_ewah_word!(u32);
impl_ewah_word!(u64);
impl_ewah_word!(usize);

/// Metadata for an EWAH run-length encoded segment.
///
/// A marker describes a sequence of words in the uncompressed bitmap:
/// first, a run of `uniform_word_count` words all filled with `uniform_bit`,
/// followed by `literal_word_count` words of literal (uncompressed) data.
///
/// # Encoding Limits
///
/// The maximum values for `uniform_word_count` and `literal_word_count`
/// depend on the word size. For a 64-bit word, each can be up to 2^30 - 1.
/// See [`Ewah::MARKER_UNIFORM_WORD_COUNT_MAX`] and
/// [`Ewah::MARKER_LITERAL_WORD_COUNT_MAX`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Marker {
    /// The fill value for the uniform run (0 or 1).
    pub uniform_bit: u8,
    /// Number of words filled with `uniform_bit`.
    pub uniform_word_count: u64,
    /// Number of literal (uncompressed) words following the uniform run.
    pub literal_word_count: u64,
}

/// EWAH codec utilities parameterized by word type.
///
/// This zero-sized type provides associated constants and methods for
/// EWAH encoding and decoding operations. The word type `W` determines
/// the alignment requirements and marker field sizes.
pub struct Ewah<W: EwahWord>(PhantomData<W>);

impl<W: EwahWord> Ewah<W> {
    /// Bits per word.
    pub const WORD_BITS: u32 = W::BITS;

    /// Bits in half a word, used for marker field sizing.
    pub const HALF_BITS: u32 = Self::WORD_BITS / 2;

    /// Bits allocated for the uniform word count in a marker.
    pub const MARKER_UNIFORM_COUNT_BITS: u32 = Self::HALF_BITS - 1;

    /// Bits allocated for the literal word count in a marker.
    pub const MARKER_LITERAL_COUNT_BITS: u32 = Self::HALF_BITS - 1;

    /// Maximum number of uniform words encodable in a single marker.
    pub const MARKER_UNIFORM_WORD_COUNT_MAX: u64 = (1 << Self::MARKER_UNIFORM_COUNT_BITS) - 1;

    /// Maximum number of literal words encodable in a single marker.
    pub const MARKER_LITERAL_WORD_COUNT_MAX: u64 = (1 << Self::MARKER_LITERAL_COUNT_BITS) - 1;

    /// Creates a bitmask with the lowest `bits` bits set.
    #[inline]
    fn mask(bits: u32) -> u128 {
        if bits == 0 { 0 } else { (1u128 << bits) - 1 }
    }

    /// Encodes a [`Marker`] into its wire-format word representation.
    ///
    /// # Panics
    ///
    /// Debug-asserts that:
    /// - `uniform_bit` is 0 or 1
    /// - `uniform_word_count` does not exceed [`Self::MARKER_UNIFORM_WORD_COUNT_MAX`]
    /// - `literal_word_count` does not exceed [`Self::MARKER_LITERAL_WORD_COUNT_MAX`]
    #[inline]
    pub fn marker_word(mark: Marker) -> W {
        debug_assert!(mark.uniform_bit <= 1);
        debug_assert!(mark.uniform_word_count <= Self::MARKER_UNIFORM_WORD_COUNT_MAX);
        debug_assert!(mark.literal_word_count <= Self::MARKER_LITERAL_WORD_COUNT_MAX);

        let word = (mark.uniform_bit as u128)
            | ((mark.uniform_word_count as u128) << 1)
            | ((mark.literal_word_count as u128) << Self::HALF_BITS);
        W::from_u128(word)
    }

    /// Decodes a wire-format marker word into a [`Marker`].
    ///
    /// This is the inverse of [`marker_word`](Self::marker_word).
    #[inline]
    pub fn unpack_marker(word: W) -> Marker {
        let word = word.to_u128();
        let uniform_bit = (word & 1) as u8;
        let uniform_word_count = ((word >> 1) & Self::mask(Self::MARKER_UNIFORM_COUNT_BITS)) as u64;
        let literal_word_count =
            ((word >> Self::HALF_BITS) & Self::mask(Self::MARKER_LITERAL_COUNT_BITS)) as u64;
        Marker {
            uniform_bit,
            uniform_word_count,
            literal_word_count,
        }
    }

    /// Returns `true` if `word` is a "literal" word (not all-zeros or all-ones).
    ///
    /// Uniform words (all-zeros or all-ones) can be run-length encoded;
    /// literal words must be stored verbatim.
    #[inline]
    fn is_literal(word: W) -> bool {
        word != W::zero() && word != W::ones()
    }

    /// Returns the maximum encoded size in bytes for a bitmap of `word_count` words.
    ///
    /// Use this to allocate a buffer guaranteed to hold the encoded output.
    /// The actual encoded size may be smaller if the bitmap compresses well.
    #[inline]
    pub fn encode_size_max(word_count: usize) -> usize {
        let marker_count = (word_count as u64).div_ceil(Self::MARKER_LITERAL_WORD_COUNT_MAX);
        (marker_count as usize + word_count) * size_of::<W>()
    }

    /// Creates a new [`Encoder`] for chunk-based compression.
    ///
    /// # Arguments
    ///
    /// * `source_words` - Slice of bitmap words to compress
    #[inline]
    pub fn encode_chunk<'a>(source_words: &'a [W]) -> Encoder<'a, W> {
        Encoder {
            source_words,
            source_index: 0,
            literal_word_count: 0,
            trailing_zero_runs_count: 0,
        }
    }

    /// Creates a new [`Decoder`] for chunk-based decompression.
    ///
    /// # Arguments
    ///
    /// * `target_words` - Mutable slice to receive decoded words
    /// * `source_size` - Total size in bytes of the compressed data
    #[inline]
    pub fn decode_chunks<'a>(target_words: &'a mut [W], source_size: usize) -> Decoder<'a, W> {
        Decoder {
            source_size_remaining: source_size,
            target_words,
            target_index: 0,
            source_literal_words: 0,
        }
    }
}

/// Returns `true` if two byte ranges do not overlap.
///
/// Used to verify source and target buffers are disjoint, which is
/// required for safe memory operations in the encoder/decoder.
#[inline]
fn ranges_disjoint(a_ptr: usize, a_len: usize, b_ptr: usize, b_len: usize) -> bool {
    let a_end = a_ptr
        .checked_add(a_len)
        .expect("overflow computing range end");
    let b_end = b_ptr
        .checked_add(b_len)
        .expect("overflow computing range end");
    a_end <= b_ptr || b_end <= a_ptr
}

/// Reinterprets a byte slice as a slice of words.
///
/// # Panics
///
/// - If `bytes.len()` is not a multiple of the word size
/// - If the byte slice is not properly aligned for `W`
#[inline]
fn as_aligned_words<W: EwahWord>(bytes: &[u8]) -> &[W] {
    assert!(bytes.len().is_multiple_of(size_of::<W>()));

    // SAFETY: `align_to` is safe because:
    // 1. The input slice is valid for reads (guaranteed by the borrow)
    // 2. We assert that prefix and suffix are empty, meaning the input was
    //    already word-aligned. Callers must ensure proper alignment.
    let (prefix, words, suffix) = unsafe { bytes.align_to::<W>() };
    assert!(prefix.is_empty(), "input buffer not word-aligned");
    assert!(suffix.is_empty(), "input buffer not word-aligned");
    words
}

/// Reinterprets a mutable byte slice as a mutable slice of words.
///
/// # Panics
///
/// - If `bytes.len()` is not a multiple of the word size
/// - If the byte slice is not properly aligned for `W`
#[inline]
fn as_aligned_words_mut<W: EwahWord>(bytes: &mut [u8]) -> &mut [W] {
    assert!(bytes.len().is_multiple_of(size_of::<W>()));

    // SAFETY: Same rationale as `as_aligned_words`. Additionally:
    // - The input is borrowed mutably, ensuring exclusive access
    // - The returned mutable slice is valid for writes through the
    //   lifetime of the borrow
    let (prefix, words, suffix) = unsafe { bytes.align_to_mut::<W>() };
    assert!(prefix.is_empty(), "input buffer not word-aligned");
    assert!(suffix.is_empty(), "input buffer not word-aligned");
    words
}

/// Streaming decoder for EWAH-compressed bitmaps.
///
/// Processes compressed data in chunks, allowing integration with
/// async I/O or fixed-size read buffers. Maintains internal state
/// across chunk boundaries.
///
/// # Chunk Processing
///
/// Call [`decode_chunk`](Self::decode_chunk) repeatedly with successive chunks of
/// compressed data. The decoder handles markers that span chunk
/// boundaries transparently.
///
/// # Completion
///
/// After all chunks have been processed, [`done`](Self::done) returns `true`.
/// At this point, `target_words` contains the fully decompressed bitmap.
pub struct Decoder<'a, W: EwahWord> {
    /// Remaining bytes of compressed source data not yet processed.
    pub source_size_remaining: usize,
    /// Target buffer for decompressed words.
    pub target_words: &'a mut [W],
    /// Current write position in `target_words`.
    pub target_index: usize,
    /// Number of literal words expected from a previous marker but not yet read.
    ///
    /// Non-zero when a marker's literal words span a chunk boundary.
    pub source_literal_words: usize,
}

impl<'a, W: EwahWord> Decoder<'a, W> {
    /// Decodes one chunk of compressed data into the target buffer.
    ///
    /// # Arguments
    ///
    /// * `source_chunk` - A chunk of EWAH-compressed bytes. Must be
    ///   word-aligned (length divisible by `size_of::<W>()`).
    ///
    /// # Returns
    ///
    /// The number of words written to the target buffer in this call.
    ///
    /// # Panics
    ///
    /// - If `source_chunk.len()` is not a multiple of the word size
    /// - If `source_chunk.len()` exceeds `source_size_remaining`
    pub fn decode_chunk(&mut self, source_chunk: &[u8]) -> usize {
        assert!(source_chunk.len().is_multiple_of(size_of::<W>()));
        assert!(self.source_size_remaining >= source_chunk.len());

        self.source_size_remaining -= source_chunk.len();

        let source_words = as_aligned_words::<W>(source_chunk);
        debug_assert!(ranges_disjoint(
            source_chunk.as_ptr() as usize,
            source_chunk.len(),
            self.target_words.as_ptr() as usize,
            core::mem::size_of_val(self.target_words)
        ));

        let mut source_index: usize = 0;
        let start_target_index = self.target_index;
        let mut target_index = self.target_index;

        if self.source_literal_words > 0 {
            let literal_word_count_chunk = min(self.source_literal_words, source_words.len());
            self.target_words[target_index..target_index + literal_word_count_chunk]
                .copy_from_slice(
                    &source_words[source_index..source_index + literal_word_count_chunk],
                );

            source_index += literal_word_count_chunk;
            target_index += literal_word_count_chunk;
            self.source_literal_words -= literal_word_count_chunk;
        };

        while source_index < source_words.len() {
            assert_eq!(self.source_literal_words, 0);

            let marker_word = source_words[source_index];
            source_index += 1;
            let marker = Ewah::<W>::unpack_marker(marker_word);

            let uniform_word_count = usize::try_from(marker.uniform_word_count)
                .expect("uniform word count does not fit in usize");
            let uniform_values = if marker.uniform_bit == 1 {
                W::ones()
            } else {
                W::zero()
            };

            self.target_words[target_index..target_index + uniform_word_count].fill(uniform_values);
            target_index += uniform_word_count;

            let literal_word_count = usize::try_from(marker.literal_word_count)
                .expect("literal word count does not fit in usize");
            let literal_word_count_chunk =
                min(literal_word_count, source_words.len() - source_index);

            self.target_words[target_index..target_index + literal_word_count_chunk]
                .copy_from_slice(
                    &source_words[source_index..source_index + literal_word_count_chunk],
                );
            source_index += literal_word_count_chunk;
            target_index += literal_word_count_chunk;

            self.source_literal_words = literal_word_count - literal_word_count_chunk;
        }

        assert!(source_index <= source_words.len());
        assert!(target_index <= self.target_words.len());

        self.target_index = target_index;

        target_index - start_target_index
    }

    /// Returns `true` if all compressed data has been processed.
    ///
    /// # Panics
    ///
    /// Debug-asserts internal consistency (e.g., no pending literal words
    /// when source is exhausted).
    #[inline]
    pub fn done(&self) -> bool {
        assert!(self.target_index <= self.target_words.len());

        if self.source_size_remaining == 0 {
            assert_eq!(self.source_literal_words, 0);
            true
        } else {
            false
        }
    }
}

/// Streaming encoder for EWAH bitmap compression.
///
/// Compresses bitmap data in chunks, enabling integration with
/// fixed-size write buffers or streaming I/O.
pub struct Encoder<'a, W: EwahWord> {
    /// Source bitmap words to compress.
    pub source_words: &'a [W],
    /// Current read position in `source_words`.
    pub source_index: usize,
    /// Number of literal words pending from a previous chunk.
    ///
    /// Non-zero when literal words span a chunk boundary.
    pub literal_word_count: usize,
    /// Count of consecutive trailing zero-fill-only markers.
    ///
    /// Used for optimization: trailing zero runs can often be omitted
    /// when the consumer initializes the target buffer to zero.
    pub trailing_zero_runs_count: usize,
}

impl<'a, W: EwahWord> Encoder<'a, W> {
    /// Encodes source words into a target chunk.
    ///
    /// # Arguments
    ///
    /// * `target_chunk` - Mutable byte buffer for encoded output.
    ///   Must be word-aligned (length divisible by `size_of::<W>()`).
    ///
    /// # Returns
    ///
    /// The number of bytes written to `target_chunk`.
    ///
    /// # Panics
    ///
    /// - If `target_chunk.len()` is not a multiple of the word size
    pub fn encode_chunk(&mut self, target_chunk: &mut [u8]) -> usize {
        let source_words = self.source_words;

        assert!(target_chunk.len().is_multiple_of(size_of::<W>()));
        assert!(self.source_index <= source_words.len());
        assert!(self.literal_word_count <= source_words.len());

        debug_assert!(ranges_disjoint(
            source_words.as_ptr() as usize,
            core::mem::size_of_val(source_words),
            target_chunk.as_ptr() as usize,
            target_chunk.len()
        ));

        let target_words = as_aligned_words_mut::<W>(target_chunk);
        target_words.fill(W::zero());

        let mut target_index: usize = 0;
        let mut source_index: usize = self.source_index;

        if self.literal_word_count > 0 {
            let literal_word_count_chunk = min(self.literal_word_count, target_words.len());

            target_words[target_index..target_index + literal_word_count_chunk].copy_from_slice(
                &source_words[source_index..source_index + literal_word_count_chunk],
            );

            target_index += literal_word_count_chunk;
            source_index += literal_word_count_chunk;
            self.literal_word_count -= literal_word_count_chunk;
        }

        while source_index < source_words.len() && target_index < target_words.len() {
            assert_eq!(self.literal_word_count, 0);

            let word = source_words[source_index];

            let uniform_word_count: usize = if Ewah::<W>::is_literal(word) {
                0
            } else {
                let remaining = (source_words.len() - source_index) as u64;
                let uniform_max_u64 = remaining.min(Ewah::<W>::MARKER_UNIFORM_WORD_COUNT_MAX);

                let slice = &source_words[source_index..source_index + uniform_max_u64 as usize];
                let uniform_value = word;

                match slice.iter().position(|&w| w != uniform_value) {
                    Some(pos) => pos,
                    None => slice.len(),
                }
            };

            source_index += uniform_word_count;

            let uniform_bit: u8 = if uniform_word_count == 0 {
                0
            } else {
                (word.to_u128() & 1) as u8
            };

            let literal_word_count: usize = {
                let literal_max_u64 = (source_words.len() - source_index) as u64;
                let literal_max_u64 = literal_max_u64.min(Ewah::<W>::MARKER_LITERAL_WORD_COUNT_MAX);

                let slice = &source_words[source_index..source_index + literal_max_u64 as usize];

                let mut count = literal_max_u64 as usize;
                for (i, &w) in slice.iter().enumerate() {
                    if !Ewah::<W>::is_literal(w) {
                        count = i;
                        break;
                    }
                }
                count
            };

            target_words[target_index] = Ewah::<W>::marker_word(Marker {
                uniform_bit,
                uniform_word_count: uniform_word_count as u64,
                literal_word_count: literal_word_count as u64,
            });
            target_index += 1;

            let literal_word_count_chunk =
                min(literal_word_count, target_words.len() - target_index);

            target_words[target_index..target_index + literal_word_count_chunk].copy_from_slice(
                &source_words[source_index..source_index + literal_word_count_chunk],
            );
            target_index += literal_word_count_chunk;
            source_index += literal_word_count_chunk;

            self.literal_word_count = literal_word_count - literal_word_count_chunk;

            if uniform_bit == 0 && literal_word_count == 0 {
                assert!(uniform_word_count > 0);
                self.trailing_zero_runs_count += 1;
            } else {
                self.trailing_zero_runs_count = 0
            }
        }

        assert!(source_index <= source_words.len());
        self.source_index = source_index;

        target_index * size_of::<W>()
    }

    /// Returns `true` if all source words have been encoded.
    #[inline]
    pub fn done(&self) -> bool {
        assert!(self.source_index <= self.source_words.len());
        self.source_index == self.source_words.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;
    use core::slice;
    use proptest::prelude::*;

    fn words_as_bytes<W: EwahWord>(words: &[W]) -> &[u8] {
        let byte_len = core::mem::size_of_val(words);
        // Safety: Vec<W> (or slice from it) is aligned and sized for W.
        unsafe { slice::from_raw_parts(words.as_ptr() as *const u8, byte_len) }
    }

    fn words_as_bytes_mut<W: EwahWord>(words: &mut [W]) -> &mut [u8] {
        let byte_len = core::mem::size_of_val(words);
        // Safety: Vec<W> (or slice from it) is aligned and sized for W.
        unsafe {
            slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, byte_len)
        }
    }

    fn encode_all<W: EwahWord>(source: &[W], chunk_words: usize) -> Vec<W> {
        let mut encoder = Encoder {
            source_words: source,
            source_index: 0,
            literal_word_count: 0,
            trailing_zero_runs_count: 0,
        };
        let chunk_words = chunk_words.max(1);
        let mut encoded = Vec::new();

        while !encoder.done() {
            let mut chunk = vec![W::zero(); chunk_words];
            let bytes_written = encoder.encode_chunk(words_as_bytes_mut(&mut chunk));
            let words_written = bytes_written / size_of::<W>();
            assert!(words_written > 0, "encoder made no progress");
            encoded.extend_from_slice(&chunk[..words_written]);
        }

        encoded
    }

    fn decode_all<W: EwahWord>(encoded: &[W], target_len: usize, chunk_words: usize) -> Vec<W> {
        let mut target = vec![W::zero(); target_len];
        let mut decoder = Ewah::<W>::decode_chunks(&mut target, core::mem::size_of_val(encoded));
        let chunk_words = chunk_words.max(1);

        let mut index = 0;
        while index < encoded.len() {
            let end = (index + chunk_words).min(encoded.len());
            decoder.decode_chunk(words_as_bytes(&encoded[index..end]));
            index = end;
        }

        assert!(decoder.done());
        assert_eq!(decoder.target_index, target_len);
        target
    }

    macro_rules! arb_word {
        ($t:ty) => {
            prop_oneof![
                4 => Just(0 as $t),
                4 => Just(<$t>::MAX),
                2 => any::<$t>(),
            ]
        };
    }

    macro_rules! ewah_roundtrip_props {
        ($name:ident, $t:ty) => {
            proptest! {
                #![proptest_config(ProptestConfig {
                    cases: 64,
                    ..ProptestConfig::default()
                })]
                #[test]
                fn $name(
                    words in proptest::collection::vec(arb_word!($t), 0..256),
                    encode_chunk in 1usize..=16,
                    decode_chunk in 1usize..=16,
                ) {
                    let encoded = encode_all::<$t>(&words, encode_chunk);
                    let encoded_bytes = core::mem::size_of_val(encoded.as_slice());
                    prop_assert!(encoded_bytes <= Ewah::<$t>::encode_size_max(words.len()));
                    let decoded = decode_all::<$t>(&encoded, words.len(), decode_chunk);
                    prop_assert_eq!(decoded, words);
                }
            }
        };
    }

    macro_rules! ewah_marker_props {
        ($name:ident, $t:ty) => {
            proptest! {
                #![proptest_config(ProptestConfig {
                    cases: 64,
                    ..ProptestConfig::default()
                })]
                #[test]
                fn $name(
                    uniform_bit in 0u8..=1,
                    uniform_word_count in 0u64..=Ewah::<$t>::MARKER_UNIFORM_WORD_COUNT_MAX,
                    literal_word_count in 0u64..=Ewah::<$t>::MARKER_LITERAL_WORD_COUNT_MAX,
                ) {
                    let marker = Marker {
                        uniform_bit,
                        uniform_word_count,
                        literal_word_count,
                    };
                    let packed = Ewah::<$t>::marker_word(marker);
                    let unpacked = Ewah::<$t>::unpack_marker(packed);
                    prop_assert_eq!(unpacked, marker);
                }
            }
        };
    }

    ewah_roundtrip_props!(roundtrip_u8, u8);
    ewah_roundtrip_props!(roundtrip_u16, u16);
    ewah_roundtrip_props!(roundtrip_u32, u32);
    ewah_roundtrip_props!(roundtrip_u64, u64);

    ewah_marker_props!(marker_roundtrip_u8, u8);
    ewah_marker_props!(marker_roundtrip_u16, u16);
    ewah_marker_props!(marker_roundtrip_u32, u32);
    ewah_marker_props!(marker_roundtrip_u64, u64);

    #[test]
    fn fuzz_roundtrip_u64() {
        use proptest::strategy::{Strategy, ValueTree};
        use proptest::test_runner::{Config as ProptestConfig, TestRunner};

        let mut runner = TestRunner::new(ProptestConfig {
            cases: 256,
            ..ProptestConfig::default()
        });
        let strat = (
            proptest::collection::vec(arb_word!(u64), 0..512),
            1usize..=32,
            1usize..=32,
        );

        for _ in 0..256 {
            let (words, encode_chunk, decode_chunk) = strat
                .new_tree(&mut runner)
                .expect("strategy build should succeed")
                .current();
            let encoded = encode_all::<u64>(&words, encode_chunk);
            let decoded = decode_all::<u64>(&encoded, words.len(), decode_chunk);
            assert_eq!(decoded, words);
        }
    }
}
