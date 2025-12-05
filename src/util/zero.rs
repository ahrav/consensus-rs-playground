pub fn is_all_zeros(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

#[cfg(test)]
mod tests {
    use super::is_all_zeros;
    use proptest::prelude::*;

    #[test]
    fn empty_slice_is_considered_all_zeros() {
        assert!(is_all_zeros(&[]));
    }

    #[test]
    fn slice_with_only_zeros_returns_true() {
        assert!(is_all_zeros(&[0, 0, 0, 0]));
    }

    #[test]
    fn any_non_zero_byte_causes_false() {
        assert!(!is_all_zeros(&[1]));
        assert!(!is_all_zeros(&[0, 2, 0]));
        assert!(!is_all_zeros(&[0, 0, 3]));
    }

    proptest! {
        #[test]
        fn zero_filled_vectors_pass(len in 0usize..65) {
            let bytes = vec![0u8; len];
            prop_assert!(is_all_zeros(&bytes));
        }

        #[test]
        fn vectors_with_any_non_zero_fail(bytes in proptest::collection::vec(any::<u8>(), 1..65)) {
            prop_assume!(bytes.iter().any(|&b| b != 0));
            prop_assert!(!is_all_zeros(&bytes));
        }
    }
}
