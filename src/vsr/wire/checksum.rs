//! Deterministic AEGIS-128L MAC used as the wire-format checksum.
//!
//! Mirrors the Zig reference implementation: both key and nonce are zeroed so tags
//! are reproducible across implementations. This detects corruption but is not a
//! secret-key authenticator.
use aegis::aegis128l::Aegis128L;

pub type Checksum128 = u128;

/// Computes the deterministic AEGIS-128L tag for `data`.
///
/// Input bytes are fed as associated data with an empty payload; the zero key and
/// nonce keep the result stable and language-agnostic. The tag is returned as a
/// little-endian `u128` to match the wire format.
///
/// # Examples
/// ```
/// use consensus::vsr::wire::checksum;
///
/// let tag = checksum::checksum(b"message");
/// assert_eq!(tag, checksum::checksum(b"message"));
/// ```
#[inline]
pub fn checksum(data: &[u8]) -> u128 {
    let key = [0u8; 16];
    let nonce = [0u8; 16];
    let aegis = Aegis128L::new(&key, &nonce);

    let (_ciphertext, tag) = aegis.encrypt(&[], data);
    u128::from_le_bytes(tag)
}

#[cfg(test)]
mod tests {
    use super::checksum;
    use proptest::prelude::*;

    // =========================================================================
    // Unit Tests: Determinism
    // =========================================================================

    #[test]
    fn determinism_empty() {
        let c1 = checksum(b"");
        let c2 = checksum(b"");
        assert_eq!(c1, c2, "Empty input must produce consistent MAC");
    }

    #[test]
    fn determinism_single_byte() {
        let c1 = checksum(b"a");
        let c2 = checksum(b"a");
        assert_eq!(c1, c2, "Single byte must produce consistent MAC");
    }

    #[test]
    fn determinism_large_input() {
        let data = vec![0xAB; 10_000];
        let c1 = checksum(&data);
        let c2 = checksum(&data);
        assert_eq!(c1, c2, "Large input must produce consistent MAC");
    }

    // =========================================================================
    // Unit Tests: Collision Resistance
    // =========================================================================

    #[test]
    fn different_inputs_different_macs() {
        let c1 = checksum(b"message1");
        let c2 = checksum(b"message2");
        assert_ne!(c1, c2, "Different inputs must produce different MACs");
    }

    #[test]
    fn single_bit_flip_changes_mac() {
        let data1 = b"message";
        let mut data2 = data1.to_vec();
        data2[0] ^= 0x01;

        let c1 = checksum(data1);
        let c2 = checksum(&data2);
        assert_ne!(c1, c2, "Single bit flip must change MAC (avalanche effect)");
    }

    #[test]
    fn trailing_byte_changes_mac() {
        let c1 = checksum(b"message");
        let c2 = checksum(b"message\0");
        assert_ne!(c1, c2, "Trailing byte must change MAC");
    }

    #[test]
    fn reordered_bytes_different_mac() {
        let c1 = checksum(b"ab");
        let c2 = checksum(b"ba");
        assert_ne!(c1, c2, "Byte order must affect MAC");
    }

    // =========================================================================
    // Unit Tests: Edge Cases
    // =========================================================================

    #[test]
    fn empty_input_no_panic() {
        let data: [u8; 0] = [];
        let _ = checksum(&data);
    }

    #[test]
    fn all_zeros_deterministic() {
        let data = vec![0u8; 1024];
        let c1 = checksum(&data);
        let c2 = checksum(&data);
        assert_eq!(c1, c2, "All-zeros input must be deterministic");
    }

    #[test]
    fn all_ones_deterministic() {
        let data = vec![0xFFu8; 1024];
        let c1 = checksum(&data);
        let c2 = checksum(&data);
        assert_eq!(c1, c2, "All-ones input must be deterministic");
    }

    #[test]
    fn boundary_sizes() {
        // Test sizes around cipher block boundaries (AEGIS-128L uses 32-byte blocks)
        for size in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129] {
            let data = vec![0x42u8; size];
            let c1 = checksum(&data);
            let c2 = checksum(&data);
            assert_eq!(c1, c2, "Size {} must produce deterministic MAC", size);
        }
    }

    #[test]
    fn large_realistic_message() {
        // VSR messages might be several KB
        let data = vec![0x5A; 64 * 1024]; // 64KB
        let _ = checksum(&data); // Should not panic
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    proptest! {
        #[test]
        fn prop_deterministic(data: Vec<u8>) {
            let c1 = checksum(&data);
            let c2 = checksum(&data);
            prop_assert_eq!(c1, c2, "MAC must be deterministic");
        }

        #[test]
        fn prop_collision_resistance(data1: Vec<u8>, data2: Vec<u8>) {
            prop_assume!(data1 != data2);
            let c1 = checksum(&data1);
            let c2 = checksum(&data2);
            prop_assert_ne!(c1, c2, "Different inputs produced same MAC");
        }

        #[test]
        fn prop_bit_sensitivity(data: Vec<u8>, byte_idx in 0usize..256, bit_idx in 0u8..8) {
            prop_assume!(!data.is_empty());
            let byte_idx = byte_idx % data.len();

            let original = checksum(&data);
            let mut modified = data.clone();
            modified[byte_idx] ^= 1 << bit_idx;
            let modified_mac = checksum(&modified);

            prop_assert_ne!(original, modified_mac, "Bit flip must change MAC");
        }

        #[test]
        fn prop_length_sensitivity(data: Vec<u8>) {
            prop_assume!(!data.is_empty());

            let original = checksum(&data);

            // Truncate
            let truncated = checksum(&data[..data.len() - 1]);
            prop_assert_ne!(original, truncated);

            // Extend
            let mut extended = data.clone();
            extended.push(0x00);
            let extended_mac = checksum(&extended);
            prop_assert_ne!(original, extended_mac);
        }

        #[test]
        fn prop_order_sensitivity(data: Vec<u8>, idx1 in 0usize..256, idx2 in 0usize..256) {
            prop_assume!(data.len() >= 2);
            let idx1 = idx1 % data.len();
            let idx2 = idx2 % data.len();
            prop_assume!(idx1 != idx2);
            prop_assume!(data[idx1] != data[idx2]);

            let original = checksum(&data);
            let mut swapped = data.clone();
            swapped.swap(idx1, idx2);
            let swapped_mac = checksum(&swapped);

            prop_assert_ne!(original, swapped_mac, "Byte swap must change MAC");
        }
    }
}
