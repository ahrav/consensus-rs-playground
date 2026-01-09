/// Fast alternative to modulo reduction (note: it is not the same as modulo).
///
/// The returned value is always in `[0, p)` when `p > 0`. This is best used with
/// uniformly distributed 64-bit inputs, such as hashes or PRNG output.
///
/// See: https://github.com/lemire/fastrange/ and
/// https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
///
/// # Examples
/// ```
/// use consensus::stdx::fastrange::fast_range;
///
/// assert_eq!(fast_range(u64::MAX, 8), 7);
/// ```
#[inline]
pub fn fast_range(word: u64, p: u64) -> u64 {
    let ln = (word as u128).wrapping_mul(p as u128);
    (ln >> 64) as u64
}

#[cfg(test)]
mod tests {
    use super::fast_range;
    use proptest::prelude::*;
    use proptest::test_runner::{RngAlgorithm, TestRng};

    const PRNG_SEED: [u8; 32] = *b"fastrange-test-seed-000000000000";

    fn test_rng() -> TestRng {
        TestRng::from_seed(RngAlgorithm::ChaCha, &PRNG_SEED)
    }

    #[test]
    fn distribution_matches_reference_rng() {
        let mut prng = test_rng();
        let mut distribution = [0u32; 8];
        for _ in 0..10_000 {
            let key = prng.next_u64();
            distribution[fast_range(key, 8) as usize] += 1;
        }
        assert_eq!(
            distribution,
            [1253, 1264, 1299, 1243, 1268, 1204, 1266, 1203]
        );
    }

    #[test]
    fn fastrange_not_modulo() {
        let mut distribution = [0u32; 8];
        for key in 0..10_000u64 {
            distribution[fast_range(key, 8) as usize] += 1;
        }
        assert_eq!(distribution, [10_000, 0, 0, 0, 0, 0, 0, 0]);
    }

    proptest! {
        #[test]
        fn output_is_within_range(word in any::<u64>(), p in 1u64..u64::MAX) {
            let value = fast_range(word, p);
            prop_assert!(value < p);
        }

        #[test]
        fn output_is_zero_when_p_is_one(word in any::<u64>()) {
            prop_assert_eq!(fast_range(word, 1), 0);
        }

        #[test]
        fn power_of_two_matches_high_bits(word in any::<u64>(), shift in 1u32..64) {
            let p = 1u64 << shift;
            let expected = word >> (64 - shift);
            prop_assert_eq!(fast_range(word, p), expected);
        }
    }
}
