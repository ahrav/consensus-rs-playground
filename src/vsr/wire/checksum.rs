//! Deterministic AEGIS-128L tag used as the wire-format checksum.
//!
//! Mirrors the Zig reference implementation: both key and nonce are zeroed so tags
//! are reproducible across implementations. This detects corruption but is not a
//! secret-key authenticator.

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
pub fn checksum(data: &[u8]) -> u128 {
    use aegis::aegis128l::Aegis128L;

    let key = [0u8; 16];
    let nonce = [0u8; 16];
    let aegis = Aegis128L::new(&key, &nonce);

    let (_ciphertext, tag) = aegis.encrypt(&[], data);
    u128::from_le_bytes(tag)
}

/// Computes the AEGISMAC-128L tag for `data`.
///
/// This uses the AEGIS-MAC construction from the `aegis` crate and produces
/// different tags than [`checksum`]. It is only available on Linux because it
/// requires the non-`pure-rust` `aegis` backend.
#[cfg(target_os = "linux")]
pub fn checksum_aegis_mac(data: &[u8]) -> u128 {
    use aegis::aegis128l::Aegis128LMac;

    // Being explicit about the aliases avoids any type mismatch surprises.
    let key: aegis::aegis128l::Key = [0u8; 16];
    let nonce: aegis::aegis128l::Nonce = [0u8; 16];

    let mut mac = Aegis128LMac::<16>::new_with_nonce(&key, &nonce);
    mac.update(data);
    let tag = mac.finalize();
    u128::from_le_bytes(tag)
}

#[cfg(test)]
mod tests {
    use super::checksum;
    use proptest::prelude::*;

    // =========================================================================
    // Unit Tests: Determinism
    // =========================================================================

    // =========================================================================
    // Unit Tests: Collision Resistance
    // =========================================================================

    // =========================================================================
    // Unit Tests: Edge Cases
    // =========================================================================

    #[test]
    fn large_realistic_message() {
        // VSR messages might be several KB
        let data = vec![0x5A; 64 * 1024]; // 64KB
        let _ = checksum(&data); // Should not panic
    }

    #[test]
    fn checksum_known_values() {
        assert_eq!(checksum(&[]), 0x49f174618255402de6e7e3c40d60cc83);
        assert_eq!(checksum(&[0u8; 16]), 0x263abed41c10336165d15dd08dd42af7);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn checksum_aegis_mac_known_values() {
        assert_eq!(
            super::checksum_aegis_mac(&[]),
            0x635037dbb2b81e9bf114d2092a5aa1ad
        );
        assert_eq!(
            super::checksum_aegis_mac(&[0u8; 16]),
            0x9cb583709438f5be0c1866de93ad9818
        );
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
