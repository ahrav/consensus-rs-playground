pub const VSR_OPERATIONS_RESERVED: u8 = 128;

pub const OPERATION_COUNT: u8 = 18;

pub const OPERATION_MIN: u8 = VSR_OPERATIONS_RESERVED;

pub const OPERATION_MAX: u8 = VSR_OPERATIONS_RESERVED + OPERATION_COUNT - 1;

const _: () = {
    assert!(VSR_OPERATIONS_RESERVED == 128);
    assert!(OPERATION_COUNT > 0);
    assert!(OPERATION_COUNT <= 128);

    assert!(VSR_OPERATIONS_RESERVED as u16 + OPERATION_COUNT as u16 <= u8::MAX as u16 + 1);

    assert!(OPERATION_MIN == 128);
    assert!(OPERATION_MAX == 145);
    assert!(OPERATION_MAX - OPERATION_MIN + 1 == OPERATION_COUNT);
};

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Operation {
    /// `constants.vsr_operations_reserved + 0`
    Pulse = 128,

    /// `constants.vsr_operations_reserved + 1`
    DeprecatedCreateAccountsUnbatched = 129,
    /// `constants.vsr_operations_reserved + 2`
    DeprecatedCreateTransfersUnbatched = 130,
    /// `constants.vsr_operations_reserved + 3`
    DeprecatedLookupAccountsUnbatched = 131,
    /// `constants.vsr_operations_reserved + 4`
    DeprecatedLookupTransfersUnbatched = 132,
    /// `constants.vsr_operations_reserved + 5`
    DeprecatedGetAccountTransfersUnbatched = 133,
    /// `constants.vsr_operations_reserved + 6`
    DeprecatedGetAccountBalancesUnbatched = 134,

    /// `constants.vsr_operations_reserved + 7`
    CreateAccounts = 135,
    /// `constants.vsr_operations_reserved + 8`
    CreateTransfers = 136,
    /// `constants.vsr_operations_reserved + 9`
    LookupAccounts = 137,
    /// `constants.vsr_operations_reserved + 10`
    LookupTransfers = 138,
    /// `constants.vsr_operations_reserved + 11`
    GetAccountTransfers = 139,
    /// `constants.vsr_operations_reserved + 12`
    GetAccountBalances = 140,
    /// `constants.vsr_operations_reserved + 13`
    QueryAccounts = 141,
    /// `constants.vsr_operations_reserved + 14`
    QueryTransfers = 142,
    /// `constants.vsr_operations_reserved + 15`
    GetChangeEvents = 143,

    /// `constants.vsr_operations_reserved + 16`
    ///
    /// Intentionally reserved/unused slot (kept to preserve numeric stability).
    UnusedReserved = 144,

    /// `constants.vsr_operations_reserved + 17`
    ///
    /// Internal op used by TB (keep the numeric value stable).
    QueryTransfersOp = 145,
}

const _: () = {
    // Verify first and last variants match expected values
    assert!(Operation::Pulse as u8 == OPERATION_MIN);
    assert!(Operation::QueryTransfersOp as u8 == OPERATION_MAX);

    // Verify each discriminant equals VSR_OPERATIONS_RESERVED + offset
    assert!(Operation::Pulse as u8 == VSR_OPERATIONS_RESERVED);
    assert!(Operation::DeprecatedCreateAccountsUnbatched as u8 == VSR_OPERATIONS_RESERVED + 1);
    assert!(Operation::DeprecatedCreateTransfersUnbatched as u8 == VSR_OPERATIONS_RESERVED + 2);
    assert!(Operation::DeprecatedLookupAccountsUnbatched as u8 == VSR_OPERATIONS_RESERVED + 3);
    assert!(Operation::DeprecatedLookupTransfersUnbatched as u8 == VSR_OPERATIONS_RESERVED + 4);
    assert!(Operation::DeprecatedGetAccountTransfersUnbatched as u8 == VSR_OPERATIONS_RESERVED + 5);
    assert!(Operation::DeprecatedGetAccountBalancesUnbatched as u8 == VSR_OPERATIONS_RESERVED + 6);
    assert!(Operation::CreateAccounts as u8 == VSR_OPERATIONS_RESERVED + 7);
    assert!(Operation::CreateTransfers as u8 == VSR_OPERATIONS_RESERVED + 8);
    assert!(Operation::LookupAccounts as u8 == VSR_OPERATIONS_RESERVED + 9);
    assert!(Operation::LookupTransfers as u8 == VSR_OPERATIONS_RESERVED + 10);
    assert!(Operation::GetAccountTransfers as u8 == VSR_OPERATIONS_RESERVED + 11);
    assert!(Operation::GetAccountBalances as u8 == VSR_OPERATIONS_RESERVED + 12);
    assert!(Operation::QueryAccounts as u8 == VSR_OPERATIONS_RESERVED + 13);
    assert!(Operation::QueryTransfers as u8 == VSR_OPERATIONS_RESERVED + 14);
    assert!(Operation::GetChangeEvents as u8 == VSR_OPERATIONS_RESERVED + 15);
    assert!(Operation::UnusedReserved as u8 == VSR_OPERATIONS_RESERVED + 16);
    assert!(Operation::QueryTransfersOp as u8 == VSR_OPERATIONS_RESERVED + 17);

    // Verify enum is exactly one byte (wire format requirement)
    assert!(core::mem::size_of::<Operation>() == 1);
    assert!(core::mem::align_of::<Operation>() == 1);
};

impl Operation {
    #[inline]
    pub fn as_u8(self) -> u8 {
        let value = self as u8;

        #[cfg(debug_assertions)]
        {
            debug_assert!(value >= OPERATION_MIN);
            debug_assert!(value <= OPERATION_MAX);
        }

        value
    }

    #[inline]
    pub const fn is_deprecated(self) -> bool {
        matches!(
            self,
            Operation::DeprecatedCreateAccountsUnbatched
                | Operation::DeprecatedCreateTransfersUnbatched
                | Operation::DeprecatedLookupAccountsUnbatched
                | Operation::DeprecatedLookupTransfersUnbatched
                | Operation::DeprecatedGetAccountTransfersUnbatched
                | Operation::DeprecatedGetAccountBalancesUnbatched
        )
    }

    #[inline]
    pub const fn is_client_facing(self) -> bool {
        matches!(
            self,
            Self::CreateAccounts
                | Self::CreateTransfers
                | Self::LookupAccounts
                | Self::LookupTransfers
                | Self::GetAccountTransfers
                | Self::GetAccountBalances
                | Self::QueryAccounts
                | Self::QueryTransfers
                | Self::GetChangeEvents
        )
    }

    #[inline]
    pub const fn is_valid_client_request(self) -> bool {
        let valid = self.is_client_facing();

        #[cfg(debug_assertions)]
        {
            if valid {
                debug_assert!(!self.is_deprecated());
            }
        }

        valid
    }

    /// Checks if a raw `u8` value is in the valid operation range.
    #[inline]
    pub const fn is_valid_discriminant(v: u8) -> bool {
        v >= OPERATION_MIN && v <= OPERATION_MAX
    }
}

impl From<Operation> for u8 {
    #[inline]
    fn from(op: Operation) -> u8 {
        let value = op as u8;

        debug_assert!(
            (OPERATION_MIN..=OPERATION_MAX).contains(&value),
            "operation discriminant out of range"
        );

        value
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct InvalidOperation(pub u8);

impl InvalidOperation {
    /// Returns the invalid value that caused this error.
    #[inline]
    pub const fn value(self) -> u8 {
        self.0
    }

    /// Returns true if the value was below the valid range.
    #[inline]
    pub const fn is_below_range(self) -> bool {
        self.0 < OPERATION_MIN
    }

    /// Returns true if the value was above the valid range.
    #[inline]
    pub const fn is_above_range(self) -> bool {
        self.0 > OPERATION_MAX
    }
}

impl core::fmt::Display for InvalidOperation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "invalid operation: {} (valid range: {}..={})",
            self.0, OPERATION_MIN, OPERATION_MAX
        )
    }
}

impl std::error::Error for InvalidOperation {}

impl TryFrom<u8> for Operation {
    type Error = InvalidOperation;

    #[inline]
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        // Tiger Style: early rejection with clear bounds check
        if !(OPERATION_MIN..=OPERATION_MAX).contains(&v) {
            return Err(InvalidOperation(v));
        }

        let result = match v {
            128 => Self::Pulse,

            129 => Self::DeprecatedCreateAccountsUnbatched,
            130 => Self::DeprecatedCreateTransfersUnbatched,
            131 => Self::DeprecatedLookupAccountsUnbatched,
            132 => Self::DeprecatedLookupTransfersUnbatched,
            133 => Self::DeprecatedGetAccountTransfersUnbatched,
            134 => Self::DeprecatedGetAccountBalancesUnbatched,

            135 => Self::CreateAccounts,
            136 => Self::CreateTransfers,
            137 => Self::LookupAccounts,
            138 => Self::LookupTransfers,
            139 => Self::GetAccountTransfers,
            140 => Self::GetAccountBalances,
            141 => Self::QueryAccounts,
            142 => Self::QueryTransfers,
            143 => Self::GetChangeEvents,

            144 => Self::UnusedReserved,
            145 => Self::QueryTransfersOp,

            _ => {
                debug_assert!(false, "bounds check should have caught value {}", v);
                return Err(InvalidOperation(v));
            }
        };

        debug_assert_eq!(
            result as u8, v,
            "roundtrip failed: {:?} as u8 = {}, expected {}",
            result, result as u8, v
        );

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Tiger Style: compile-time test of constants
    const _: () = {
        assert!(OPERATION_MIN == 128);
        assert!(OPERATION_MAX == 145);
        assert!(OPERATION_COUNT == 18);
    };

    // All operation variants in discriminant order for exhaustive testing.
    const ALL_OPERATIONS: [Operation; OPERATION_COUNT as usize] = [
        Operation::Pulse,
        Operation::DeprecatedCreateAccountsUnbatched,
        Operation::DeprecatedCreateTransfersUnbatched,
        Operation::DeprecatedLookupAccountsUnbatched,
        Operation::DeprecatedLookupTransfersUnbatched,
        Operation::DeprecatedGetAccountTransfersUnbatched,
        Operation::DeprecatedGetAccountBalancesUnbatched,
        Operation::CreateAccounts,
        Operation::CreateTransfers,
        Operation::LookupAccounts,
        Operation::LookupTransfers,
        Operation::GetAccountTransfers,
        Operation::GetAccountBalances,
        Operation::QueryAccounts,
        Operation::QueryTransfers,
        Operation::GetChangeEvents,
        Operation::UnusedReserved,
        Operation::QueryTransfersOp,
    ];

    // Compile-time: validates ALL_OPERATIONS array has correct length
    const _: () = assert!(ALL_OPERATIONS.len() == OPERATION_COUNT as usize);

    // ==========================================================================
    // PropTest Strategy for Operation Enum
    // ==========================================================================

    impl Arbitrary for Operation {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (OPERATION_MIN..=OPERATION_MAX)
                .prop_map(|b| Operation::try_from(b).unwrap())
                .boxed()
        }
    }

    // ==========================================================================
    // Property-Based Tests: Core Invariants
    // ==========================================================================

    proptest! {
        #[test]
        fn prop_roundtrip_all_variants(op in any::<Operation>()) {
            let byte = op.as_u8();
            let recovered = Operation::try_from(byte)
                .expect("Valid operation must convert back from its byte");
            prop_assert_eq!(op, recovered);
        }

        #[test]
        fn prop_inverse_for_valid_bytes(byte in OPERATION_MIN..=OPERATION_MAX) {
            let op = Operation::try_from(byte)
                .expect("Byte in valid range must convert");
            prop_assert_eq!(op.as_u8(), byte);
        }

        #[test]
        fn prop_all_variants_have_unique_discriminants(
            v1 in any::<Operation>(),
            v2 in any::<Operation>()
        ) {
            if v1 != v2 {
                prop_assert_ne!(v1.as_u8(), v2.as_u8());
            }
        }

        #[test]
        fn prop_from_trait_matches_as_u8(op in any::<Operation>()) {
            let byte: u8 = op.into();
            prop_assert_eq!(byte, op.as_u8());
        }

        #[test]
        fn prop_is_valid_discriminant_consistent(byte: u8) {
            let is_valid = Operation::is_valid_discriminant(byte);
            let try_from_result = Operation::try_from(byte);
            prop_assert_eq!(is_valid, try_from_result.is_ok());
        }

        #[test]
        fn prop_as_u8_always_in_range(op in any::<Operation>()) {
            let byte = op.as_u8();
            prop_assert!(byte >= OPERATION_MIN);
            prop_assert!(byte <= OPERATION_MAX);
        }

        #[test]
        fn prop_client_request_invariant(op in any::<Operation>()) {
            // If an operation is a valid client request,
            // it must be client-facing and not deprecated
            if op.is_valid_client_request() {
                prop_assert!(op.is_client_facing());
                prop_assert!(!op.is_deprecated());
            }
        }
    }

    // ==========================================================================
    // Unit Tests: Wire Protocol Stability
    // ==========================================================================

    #[test]
    fn discriminants_match_tb() {
        // Tiger Style: split assertions (one per check)
        // These values are part of the wire protocol and MUST NOT change
        assert_eq!(VSR_OPERATIONS_RESERVED, 128);

        assert_eq!(Operation::Pulse as u8, 128);
        assert_eq!(Operation::DeprecatedCreateAccountsUnbatched as u8, 129);
        assert_eq!(Operation::DeprecatedCreateTransfersUnbatched as u8, 130);
        assert_eq!(Operation::DeprecatedLookupAccountsUnbatched as u8, 131);
        assert_eq!(Operation::DeprecatedLookupTransfersUnbatched as u8, 132);
        assert_eq!(Operation::DeprecatedGetAccountTransfersUnbatched as u8, 133);
        assert_eq!(Operation::DeprecatedGetAccountBalancesUnbatched as u8, 134);
        assert_eq!(Operation::CreateAccounts as u8, 135);
        assert_eq!(Operation::CreateTransfers as u8, 136);
        assert_eq!(Operation::LookupAccounts as u8, 137);
        assert_eq!(Operation::LookupTransfers as u8, 138);
        assert_eq!(Operation::GetAccountTransfers as u8, 139);
        assert_eq!(Operation::GetAccountBalances as u8, 140);
        assert_eq!(Operation::QueryAccounts as u8, 141);
        assert_eq!(Operation::QueryTransfers as u8, 142);
        assert_eq!(Operation::GetChangeEvents as u8, 143);
        assert_eq!(Operation::UnusedReserved as u8, 144);
        assert_eq!(Operation::QueryTransfersOp as u8, 145);
    }

    // ==========================================================================
    // Unit Tests: Conversion Correctness
    // ==========================================================================

    #[test]
    fn try_from_u8_roundtrip() {
        // Tiger Style: bounded loop with explicit limits
        for v in OPERATION_MIN..=OPERATION_MAX {
            let op = Operation::try_from(v).expect("valid range should succeed");

            // Paired assertions: roundtrip and range check
            assert_eq!(op as u8, v, "roundtrip failed for {}", v);
            assert!(
                Operation::is_valid_discriminant(op as u8),
                "result should be valid discriminant"
            );
        }
    }

    #[test]
    fn try_from_rejects_below_range() {
        // Tiger Style: test boundary conditions explicitly
        for v in 0..OPERATION_MIN {
            let err = Operation::try_from(v).expect_err("below range should fail");

            // Verify error contains correct value
            assert_eq!(err.0, v);
            assert!(err.is_below_range());
            assert!(!err.is_above_range());
        }
    }

    #[test]
    fn try_from_rejects_above_range() {
        // Test values above range up to u8::MAX
        for v in (OPERATION_MAX + 1)..=u8::MAX {
            let err = Operation::try_from(v).expect_err("above range should fail");

            assert_eq!(err.0, v);
            assert!(!err.is_below_range());
            assert!(err.is_above_range());
        }
    }

    #[test]
    fn boundary_values() {
        // Tiger Style: explicit boundary testing

        // Just below valid range
        assert!(Operation::try_from(OPERATION_MIN - 1).is_err());

        // First valid value
        assert!(Operation::try_from(OPERATION_MIN).is_ok());
        assert_eq!(
            Operation::try_from(OPERATION_MIN).unwrap(),
            Operation::Pulse
        );

        // Last valid value
        assert!(Operation::try_from(OPERATION_MAX).is_ok());
        assert_eq!(
            Operation::try_from(OPERATION_MAX).unwrap(),
            Operation::QueryTransfersOp
        );

        // Just above valid range
        assert!(Operation::try_from(OPERATION_MAX + 1).is_err());
    }

    #[test]
    fn contiguous_discriminants() {
        // Tiger Style: verify no gaps in discriminant range
        let mut count = 0u8;
        for v in OPERATION_MIN..=OPERATION_MAX {
            assert!(
                Operation::try_from(v).is_ok(),
                "gap in discriminants at {}",
                v
            );
            count += 1;
        }
        assert_eq!(count, OPERATION_COUNT);
    }

    #[test]
    fn all_operations_array_matches_try_from() {
        for (i, &op) in ALL_OPERATIONS.iter().enumerate() {
            let byte = OPERATION_MIN + i as u8;
            let expected = Operation::try_from(byte).unwrap();
            assert_eq!(
                op, expected,
                "ALL_OPERATIONS[{}] doesn't match try_from({})",
                i, byte
            );
        }
    }

    // ==========================================================================
    // Unit Tests: as_u8() Method
    // ==========================================================================

    #[test]
    fn as_u8_returns_correct_values() {
        // Tiger Style: test all variants explicitly
        assert_eq!(Operation::Pulse.as_u8(), 128);
        assert_eq!(Operation::DeprecatedCreateAccountsUnbatched.as_u8(), 129);
        assert_eq!(Operation::DeprecatedCreateTransfersUnbatched.as_u8(), 130);
        assert_eq!(Operation::DeprecatedLookupAccountsUnbatched.as_u8(), 131);
        assert_eq!(Operation::DeprecatedLookupTransfersUnbatched.as_u8(), 132);
        assert_eq!(
            Operation::DeprecatedGetAccountTransfersUnbatched.as_u8(),
            133
        );
        assert_eq!(
            Operation::DeprecatedGetAccountBalancesUnbatched.as_u8(),
            134
        );
        assert_eq!(Operation::CreateAccounts.as_u8(), 135);
        assert_eq!(Operation::CreateTransfers.as_u8(), 136);
        assert_eq!(Operation::LookupAccounts.as_u8(), 137);
        assert_eq!(Operation::LookupTransfers.as_u8(), 138);
        assert_eq!(Operation::GetAccountTransfers.as_u8(), 139);
        assert_eq!(Operation::GetAccountBalances.as_u8(), 140);
        assert_eq!(Operation::QueryAccounts.as_u8(), 141);
        assert_eq!(Operation::QueryTransfers.as_u8(), 142);
        assert_eq!(Operation::GetChangeEvents.as_u8(), 143);
        assert_eq!(Operation::UnusedReserved.as_u8(), 144);
        assert_eq!(Operation::QueryTransfersOp.as_u8(), 145);
    }

    #[test]
    fn as_u8_always_in_valid_range() {
        // Tiger Style: verify postcondition for all operations
        for op in ALL_OPERATIONS {
            let byte = op.as_u8();
            assert!(byte >= OPERATION_MIN);
            assert!(byte <= OPERATION_MAX);
        }
    }

    // ==========================================================================
    // Unit Tests: is_deprecated() - Exhaustive Coverage
    // ==========================================================================

    #[test]
    fn deprecated_operations_exhaustive() {
        // Tiger Style: test ALL variants explicitly

        // Deprecated operations (6 total)
        assert!(Operation::DeprecatedCreateAccountsUnbatched.is_deprecated());
        assert!(Operation::DeprecatedCreateTransfersUnbatched.is_deprecated());
        assert!(Operation::DeprecatedLookupAccountsUnbatched.is_deprecated());
        assert!(Operation::DeprecatedLookupTransfersUnbatched.is_deprecated());
        assert!(Operation::DeprecatedGetAccountTransfersUnbatched.is_deprecated());
        assert!(Operation::DeprecatedGetAccountBalancesUnbatched.is_deprecated());

        // Non-deprecated operations (12 total)
        assert!(!Operation::Pulse.is_deprecated());
        assert!(!Operation::CreateAccounts.is_deprecated());
        assert!(!Operation::CreateTransfers.is_deprecated());
        assert!(!Operation::LookupAccounts.is_deprecated());
        assert!(!Operation::LookupTransfers.is_deprecated());
        assert!(!Operation::GetAccountTransfers.is_deprecated());
        assert!(!Operation::GetAccountBalances.is_deprecated());
        assert!(!Operation::QueryAccounts.is_deprecated());
        assert!(!Operation::QueryTransfers.is_deprecated());
        assert!(!Operation::GetChangeEvents.is_deprecated());
        assert!(!Operation::UnusedReserved.is_deprecated());
        assert!(!Operation::QueryTransfersOp.is_deprecated());
    }

    #[test]
    fn deprecated_count_matches_expected() {
        let deprecated_count = ALL_OPERATIONS
            .iter()
            .filter(|op| op.is_deprecated())
            .count();
        assert_eq!(deprecated_count, 6);
    }

    // ==========================================================================
    // Unit Tests: is_client_facing() - Exhaustive Coverage
    // ==========================================================================

    #[test]
    fn client_facing_operations_exhaustive() {
        // Client-facing ops (9 total)
        assert!(Operation::CreateAccounts.is_client_facing());
        assert!(Operation::CreateTransfers.is_client_facing());
        assert!(Operation::LookupAccounts.is_client_facing());
        assert!(Operation::LookupTransfers.is_client_facing());
        assert!(Operation::GetAccountTransfers.is_client_facing());
        assert!(Operation::GetAccountBalances.is_client_facing());
        assert!(Operation::QueryAccounts.is_client_facing());
        assert!(Operation::QueryTransfers.is_client_facing());
        assert!(Operation::GetChangeEvents.is_client_facing());

        // Non-client-facing (9 total)
        assert!(!Operation::Pulse.is_client_facing());
        assert!(!Operation::DeprecatedCreateAccountsUnbatched.is_client_facing());
        assert!(!Operation::DeprecatedCreateTransfersUnbatched.is_client_facing());
        assert!(!Operation::DeprecatedLookupAccountsUnbatched.is_client_facing());
        assert!(!Operation::DeprecatedLookupTransfersUnbatched.is_client_facing());
        assert!(!Operation::DeprecatedGetAccountTransfersUnbatched.is_client_facing());
        assert!(!Operation::DeprecatedGetAccountBalancesUnbatched.is_client_facing());
        assert!(!Operation::UnusedReserved.is_client_facing());
        assert!(!Operation::QueryTransfersOp.is_client_facing());
    }

    #[test]
    fn client_facing_count_matches_expected() {
        let client_facing_count = ALL_OPERATIONS
            .iter()
            .filter(|op| op.is_client_facing())
            .count();
        assert_eq!(client_facing_count, 9);
    }

    // ==========================================================================
    // Unit Tests: is_valid_client_request() - Exhaustive Coverage
    // ==========================================================================

    #[test]
    fn valid_client_request_exhaustive() {
        // Valid client requests: client-facing AND not deprecated (9 total)
        assert!(Operation::CreateAccounts.is_valid_client_request());
        assert!(Operation::CreateTransfers.is_valid_client_request());
        assert!(Operation::LookupAccounts.is_valid_client_request());
        assert!(Operation::LookupTransfers.is_valid_client_request());
        assert!(Operation::GetAccountTransfers.is_valid_client_request());
        assert!(Operation::GetAccountBalances.is_valid_client_request());
        assert!(Operation::QueryAccounts.is_valid_client_request());
        assert!(Operation::QueryTransfers.is_valid_client_request());
        assert!(Operation::GetChangeEvents.is_valid_client_request());

        // NOT valid client requests (9 total)
        assert!(!Operation::Pulse.is_valid_client_request());
        assert!(!Operation::DeprecatedCreateAccountsUnbatched.is_valid_client_request());
        assert!(!Operation::DeprecatedCreateTransfersUnbatched.is_valid_client_request());
        assert!(!Operation::DeprecatedLookupAccountsUnbatched.is_valid_client_request());
        assert!(!Operation::DeprecatedLookupTransfersUnbatched.is_valid_client_request());
        assert!(!Operation::DeprecatedGetAccountTransfersUnbatched.is_valid_client_request());
        assert!(!Operation::DeprecatedGetAccountBalancesUnbatched.is_valid_client_request());
        assert!(!Operation::UnusedReserved.is_valid_client_request());
        assert!(!Operation::QueryTransfersOp.is_valid_client_request());
    }

    #[test]
    fn valid_client_request_invariants() {
        // Invariant: valid client requests must be client-facing and not deprecated
        for op in ALL_OPERATIONS {
            if op.is_valid_client_request() {
                assert!(
                    op.is_client_facing(),
                    "{:?} is valid client request but not client-facing",
                    op
                );
                assert!(
                    !op.is_deprecated(),
                    "{:?} is valid client request but is deprecated",
                    op
                );
            }
        }
    }

    #[test]
    fn valid_client_request_count_matches_expected() {
        let valid_count = ALL_OPERATIONS
            .iter()
            .filter(|op| op.is_valid_client_request())
            .count();
        assert_eq!(valid_count, 9);
    }

    // ==========================================================================
    // Unit Tests: is_valid_discriminant()
    // ==========================================================================

    #[test]
    fn is_valid_discriminant_boundary_values() {
        // Below range
        assert!(!Operation::is_valid_discriminant(0));
        assert!(!Operation::is_valid_discriminant(127));

        // Exact boundaries
        assert!(Operation::is_valid_discriminant(OPERATION_MIN));
        assert!(Operation::is_valid_discriminant(OPERATION_MAX));

        // Above range
        assert!(!Operation::is_valid_discriminant(146));
        assert!(!Operation::is_valid_discriminant(255));
    }

    #[test]
    fn is_valid_discriminant_matches_try_from() {
        // Tiger Style: verify two ways of checking validity agree
        for byte in 0..=255u8 {
            let is_valid = Operation::is_valid_discriminant(byte);
            let try_from_ok = Operation::try_from(byte).is_ok();
            assert_eq!(
                is_valid, try_from_ok,
                "Mismatch at byte {}: is_valid={}, try_from.is_ok()={}",
                byte, is_valid, try_from_ok
            );
        }
    }

    // ==========================================================================
    // Unit Tests: Memory Layout
    // ==========================================================================

    #[test]
    fn enum_size_and_alignment() {
        // Tiger Style: verify wire format requirements
        assert_eq!(core::mem::size_of::<Operation>(), 1);
        assert_eq!(core::mem::align_of::<Operation>(), 1);
        assert_eq!(core::mem::size_of::<Option<Operation>>(), 1);
    }

    // ==========================================================================
    // Unit Tests: Derived Traits
    // ==========================================================================

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn clone_works() {
        for op in ALL_OPERATIONS {
            let cloned = op.clone();
            assert_eq!(op, cloned);
        }
    }

    #[test]
    fn copy_works() {
        let op = Operation::CreateAccounts;
        let copied = op; // Uses Copy
        assert_eq!(op, copied);
        // Both should still be usable
        assert_eq!(op.as_u8(), copied.as_u8());
    }

    #[test]
    fn debug_formatting() {
        // Tiger Style: test Debug impl shows variant names
        assert_eq!(format!("{:?}", Operation::Pulse), "Pulse");
        assert_eq!(format!("{:?}", Operation::CreateAccounts), "CreateAccounts");
        assert_eq!(
            format!("{:?}", Operation::DeprecatedCreateAccountsUnbatched),
            "DeprecatedCreateAccountsUnbatched"
        );
    }

    #[test]
    fn equality_works() {
        // Tiger Style: test both positive and negative cases
        assert_eq!(Operation::Pulse, Operation::Pulse);
        assert_ne!(Operation::Pulse, Operation::CreateAccounts);

        // Test roundtrip equality
        for v in OPERATION_MIN..=OPERATION_MAX {
            let op1 = Operation::try_from(v).unwrap();
            let op2 = Operation::try_from(v).unwrap();
            assert_eq!(op1, op2);
        }
    }

    #[test]
    fn hash_uniqueness() {
        // Tiger Style: verify all operations hash to unique values in a set
        use std::collections::HashSet;
        let mut set = HashSet::new();

        for op in ALL_OPERATIONS {
            set.insert(op);
        }

        assert_eq!(set.len(), OPERATION_COUNT as usize);
    }

    #[test]
    fn hash_deterministic() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Same operation should hash to same value
        for op in ALL_OPERATIONS {
            let mut hasher1 = DefaultHasher::new();
            op.hash(&mut hasher1);
            let hash1 = hasher1.finish();

            let mut hasher2 = DefaultHasher::new();
            op.hash(&mut hasher2);
            let hash2 = hasher2.finish();

            assert_eq!(hash1, hash2, "{:?} hash is not deterministic", op);
        }
    }

    // ==========================================================================
    // Unit Tests: Trait Implementations
    // ==========================================================================

    #[test]
    fn from_operation_for_u8_works() {
        for op in ALL_OPERATIONS {
            let byte: u8 = op.into();
            assert_eq!(byte, op.as_u8());
        }
    }

    #[test]
    fn try_from_u8_for_operation_works() {
        // Valid bytes
        for byte in OPERATION_MIN..=OPERATION_MAX {
            let result: Result<Operation, InvalidOperation> = byte.try_into();
            assert!(result.is_ok());
            assert_eq!(result.unwrap().as_u8(), byte);
        }

        // Invalid bytes
        let result: Result<Operation, InvalidOperation> = 127u8.try_into();
        assert_eq!(result, Err(InvalidOperation(127)));

        let result: Result<Operation, InvalidOperation> = 146u8.try_into();
        assert_eq!(result, Err(InvalidOperation(146)));
    }

    // ==========================================================================
    // Unit Tests: InvalidOperation Error Type
    // ==========================================================================

    #[test]
    fn invalid_operation_value() {
        let err = InvalidOperation(42);
        assert_eq!(err.value(), 42);
    }

    #[test]
    fn invalid_operation_range_checks() {
        // Below range
        let below = InvalidOperation(100);
        assert!(below.is_below_range());
        assert!(!below.is_above_range());

        // Above range
        let above = InvalidOperation(200);
        assert!(!above.is_below_range());
        assert!(above.is_above_range());

        // Boundary cases
        let just_below = InvalidOperation(OPERATION_MIN - 1);
        assert!(just_below.is_below_range());

        let just_above = InvalidOperation(OPERATION_MAX + 1);
        assert!(just_above.is_above_range());
    }

    #[test]
    fn invalid_operation_display() {
        let err = InvalidOperation(127);
        let msg = format!("{}", err);
        assert!(msg.contains("127"));
        assert!(msg.contains("128"));
        assert!(msg.contains("145"));
    }

    #[test]
    fn invalid_operation_is_error() {
        // Verify Error trait is implemented
        fn assert_is_error<T: std::error::Error>() {}
        assert_is_error::<InvalidOperation>();

        // Test source() returns None (default implementation)
        use std::error::Error;
        let err = InvalidOperation(127);
        assert!(err.source().is_none());
    }

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn invalid_operation_clone() {
        let err1 = InvalidOperation(255);
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    #[test]
    fn invalid_operation_debug() {
        let err = InvalidOperation(99);
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("99"));
    }

    // ==========================================================================
    // Fuzz Tests: Robustness Against Untrusted Input
    // ==========================================================================

    #[test]
    fn fuzz_try_from_never_panics() {
        // Tiger Style: verify all u8 values handled gracefully
        for byte in 0..=255u8 {
            let result = std::panic::catch_unwind(|| Operation::try_from(byte));
            assert!(
                result.is_ok(),
                "try_from({}) panicked - must handle all bytes gracefully",
                byte
            );
        }
    }

    #[test]
    fn fuzz_as_u8_never_panics() {
        for op in ALL_OPERATIONS {
            let result = std::panic::catch_unwind(|| op.as_u8());
            assert!(result.is_ok(), "{:?}.as_u8() panicked", op);
        }
    }

    #[test]
    fn fuzz_is_valid_discriminant_never_panics() {
        for byte in 0..=255u8 {
            let result = std::panic::catch_unwind(|| Operation::is_valid_discriminant(byte));
            assert!(result.is_ok());
        }
    }

    #[test]
    fn fuzz_is_deprecated_never_panics() {
        for op in ALL_OPERATIONS {
            let result = std::panic::catch_unwind(|| op.is_deprecated());
            assert!(result.is_ok(), "{:?}.is_deprecated() panicked", op);
        }
    }

    #[test]
    fn fuzz_is_client_facing_never_panics() {
        for op in ALL_OPERATIONS {
            let result = std::panic::catch_unwind(|| op.is_client_facing());
            assert!(result.is_ok(), "{:?}.is_client_facing() panicked", op);
        }
    }

    #[test]
    fn fuzz_is_valid_client_request_never_panics() {
        for op in ALL_OPERATIONS {
            let result = std::panic::catch_unwind(|| op.is_valid_client_request());
            assert!(
                result.is_ok(),
                "{:?}.is_valid_client_request() panicked",
                op
            );
        }
    }
}
