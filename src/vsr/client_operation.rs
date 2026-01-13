//! Wire-format operation codes for client requests.
//!
//! Operations occupy the reserved range `128..=145`. Discriminants are part of the wire
//! protocol and must never change. Use [`Operation::try_from`] to parse from bytes.

/// First byte value reserved for user-defined operations.
pub const VSR_OPERATIONS_RESERVED: u8 = 128;

/// Total number of defined operations.
pub const OPERATION_COUNT: u8 = 18;

/// Minimum valid operation discriminant.
pub const OPERATION_MIN: u8 = VSR_OPERATIONS_RESERVED;

/// Maximum valid operation discriminant.
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

/// Client request operation codes with stable wire-format discriminants.
///
/// Each variant maps to a fixed `u8` value in the range [`OPERATION_MIN`]..=[`OPERATION_MAX`].
/// Discriminants are part of the protocol specification and must never change.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Operation {
    /// Internal heartbeat/keepalive operation.
    Pulse = 128,

    // Deprecated unbatched operations (129-134). Retained for wire compatibility.
    #[doc(hidden)]
    DeprecatedCreateAccountsUnbatched = 129,
    #[doc(hidden)]
    DeprecatedCreateTransfersUnbatched = 130,
    #[doc(hidden)]
    DeprecatedLookupAccountsUnbatched = 131,
    #[doc(hidden)]
    DeprecatedLookupTransfersUnbatched = 132,
    #[doc(hidden)]
    DeprecatedGetAccountTransfersUnbatched = 133,
    #[doc(hidden)]
    DeprecatedGetAccountBalancesUnbatched = 134,

    /// Batch create accounts.
    CreateAccounts = 135,
    /// Batch create transfers.
    CreateTransfers = 136,
    /// Lookup accounts by ID.
    LookupAccounts = 137,
    /// Lookup transfers by ID.
    LookupTransfers = 138,
    /// Get transfers for a specific account.
    GetAccountTransfers = 139,
    /// Get balance history for a specific account.
    GetAccountBalances = 140,
    /// Query accounts with filters.
    QueryAccounts = 141,
    /// Query transfers with filters.
    QueryTransfers = 142,
    /// Subscribe to change events.
    GetChangeEvents = 143,

    /// Reserved slot for future use. Preserves discriminant stability.
    UnusedReserved = 144,

    /// Internal query operation. Not client-facing.
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
    /// Returns the wire-format byte value.
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

    /// Returns `true` for legacy operations no longer accepted by clients.
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

    /// Returns `true` for operations exposed to external clients.
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

    /// Returns `true` if this operation can be requested by clients.
    ///
    /// Equivalent to `is_client_facing() && !is_deprecated()`.
    #[inline]
    pub const fn is_valid_client_request(self) -> bool {
        self.is_client_facing() && !self.is_deprecated()
    }

    /// Checks if a raw byte is within the valid operation range.
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

/// Error returned when parsing an invalid operation byte.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct InvalidOperation(pub u8);

impl InvalidOperation {
    /// Returns the invalid byte value.
    #[inline]
    pub const fn value(self) -> u8 {
        self.0
    }

    /// Returns `true` if the value was below [`OPERATION_MIN`].
    #[inline]
    pub const fn is_below_range(self) -> bool {
        self.0 < OPERATION_MIN
    }

    /// Returns `true` if the value was above [`OPERATION_MAX`].
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

    const PROPTEST_CASES: u32 = 16;

    const _: () = {
        assert!(OPERATION_MIN == 128);
        assert!(OPERATION_MAX == 145);
        assert!(OPERATION_COUNT == 18);
    };

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

    const _: () = assert!(ALL_OPERATIONS.len() == OPERATION_COUNT as usize);

    impl Arbitrary for Operation {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (OPERATION_MIN..=OPERATION_MAX)
                .prop_map(|b| Operation::try_from(b).unwrap())
                .boxed()
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(
            crate::test_utils::proptest_cases(PROPTEST_CASES)
        ))]

        #[test]
        fn prop_roundtrip(op in any::<Operation>()) {
            let byte = op.as_u8();
            let recovered = Operation::try_from(byte)
                .expect("Valid operation must convert back from its byte");
            prop_assert_eq!(op, recovered);
        }

        #[test]
        fn prop_inverse_roundtrip(byte in OPERATION_MIN..=OPERATION_MAX) {
            let op = Operation::try_from(byte)
                .expect("Byte in valid range must convert");
            prop_assert_eq!(op.as_u8(), byte);
        }

        #[test]
        fn prop_unique_discriminants(
            v1 in any::<Operation>(),
            v2 in any::<Operation>()
        ) {
            if v1 != v2 {
                prop_assert_ne!(v1.as_u8(), v2.as_u8());
            }
        }

        #[test]
        fn prop_into_u8(op in any::<Operation>()) {
            let byte: u8 = op.into();
            prop_assert_eq!(byte, op.as_u8());
        }

        #[test]
        fn prop_is_valid_discriminant(byte: u8) {
            let is_valid = Operation::is_valid_discriminant(byte);
            let try_from_result = Operation::try_from(byte);
            prop_assert_eq!(is_valid, try_from_result.is_ok());
        }

        #[test]
        fn prop_as_u8_range(op in any::<Operation>()) {
            let byte = op.as_u8();
            prop_assert!(byte >= OPERATION_MIN);
            prop_assert!(byte <= OPERATION_MAX);
        }

        #[test]
        fn prop_client_request(op in any::<Operation>()) {
            if op.is_valid_client_request() {
                prop_assert!(op.is_client_facing());
                prop_assert!(!op.is_deprecated());
            }
        }
    }

    #[test]
    fn discriminants() {
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

    #[test]
    fn try_from_roundtrip() {
        for v in OPERATION_MIN..=OPERATION_MAX {
            let op = Operation::try_from(v).expect("valid range should succeed");

            assert_eq!(op as u8, v, "roundtrip failed for {}", v);
            assert!(
                Operation::is_valid_discriminant(op as u8),
                "result should be valid discriminant"
            );
        }
    }

    #[test]
    fn try_from_below_min() {
        for v in 0..OPERATION_MIN {
            let err = Operation::try_from(v).expect_err("below range should fail");

            assert_eq!(err.0, v);
            assert!(err.is_below_range());
            assert!(!err.is_above_range());
        }
    }

    #[test]
    fn try_from_above_max() {
        for v in (OPERATION_MAX + 1)..=u8::MAX {
            let err = Operation::try_from(v).expect_err("above range should fail");

            assert_eq!(err.0, v);
            assert!(!err.is_below_range());
            assert!(err.is_above_range());
        }
    }

    #[test]
    fn boundaries() {
        assert!(Operation::try_from(OPERATION_MIN - 1).is_err());

        assert!(Operation::try_from(OPERATION_MIN).is_ok());
        assert_eq!(
            Operation::try_from(OPERATION_MIN).unwrap(),
            Operation::Pulse
        );

        assert!(Operation::try_from(OPERATION_MAX).is_ok());
        assert_eq!(
            Operation::try_from(OPERATION_MAX).unwrap(),
            Operation::QueryTransfersOp
        );

        assert!(Operation::try_from(OPERATION_MAX + 1).is_err());
    }

    #[test]
    fn contiguous() {
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
    fn all_operations_array() {
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

    #[test]
    fn as_u8_values() {
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
    fn deprecated() {
        assert!(Operation::DeprecatedCreateAccountsUnbatched.is_deprecated());
        assert!(Operation::DeprecatedCreateTransfersUnbatched.is_deprecated());
        assert!(Operation::DeprecatedLookupAccountsUnbatched.is_deprecated());
        assert!(Operation::DeprecatedLookupTransfersUnbatched.is_deprecated());
        assert!(Operation::DeprecatedGetAccountTransfersUnbatched.is_deprecated());
        assert!(Operation::DeprecatedGetAccountBalancesUnbatched.is_deprecated());

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
    fn deprecated_count() {
        let count = ALL_OPERATIONS
            .iter()
            .filter(|op| op.is_deprecated())
            .count();
        assert_eq!(count, 6);
    }

    #[test]
    fn client_facing() {
        assert!(Operation::CreateAccounts.is_client_facing());
        assert!(Operation::CreateTransfers.is_client_facing());
        assert!(Operation::LookupAccounts.is_client_facing());
        assert!(Operation::LookupTransfers.is_client_facing());
        assert!(Operation::GetAccountTransfers.is_client_facing());
        assert!(Operation::GetAccountBalances.is_client_facing());
        assert!(Operation::QueryAccounts.is_client_facing());
        assert!(Operation::QueryTransfers.is_client_facing());
        assert!(Operation::GetChangeEvents.is_client_facing());

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
    fn client_facing_count() {
        let count = ALL_OPERATIONS
            .iter()
            .filter(|op| op.is_client_facing())
            .count();
        assert_eq!(count, 9);
    }

    #[test]
    fn valid_client_request() {
        assert!(Operation::CreateAccounts.is_valid_client_request());
        assert!(Operation::CreateTransfers.is_valid_client_request());
        assert!(Operation::LookupAccounts.is_valid_client_request());
        assert!(Operation::LookupTransfers.is_valid_client_request());
        assert!(Operation::GetAccountTransfers.is_valid_client_request());
        assert!(Operation::GetAccountBalances.is_valid_client_request());
        assert!(Operation::QueryAccounts.is_valid_client_request());
        assert!(Operation::QueryTransfers.is_valid_client_request());
        assert!(Operation::GetChangeEvents.is_valid_client_request());

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
    fn valid_client_request_count() {
        let count = ALL_OPERATIONS
            .iter()
            .filter(|op| op.is_valid_client_request())
            .count();
        assert_eq!(count, 9);
    }

    #[test]
    fn valid_client_request_definition() {
        for op in ALL_OPERATIONS {
            let expected = op.is_client_facing() && !op.is_deprecated();
            assert_eq!(
                op.is_valid_client_request(),
                expected,
                "{:?} validity mismatch",
                op
            );
        }
    }

    #[test]
    fn is_valid_discriminant_boundaries() {
        assert!(!Operation::is_valid_discriminant(0));
        assert!(!Operation::is_valid_discriminant(127));

        assert!(Operation::is_valid_discriminant(OPERATION_MIN));
        assert!(Operation::is_valid_discriminant(OPERATION_MAX));

        assert!(!Operation::is_valid_discriminant(146));
        assert!(!Operation::is_valid_discriminant(255));
    }

    #[test]
    fn layout() {
        assert_eq!(core::mem::size_of::<Operation>(), 1);
        assert_eq!(core::mem::align_of::<Operation>(), 1);
        assert_eq!(core::mem::size_of::<Option<Operation>>(), 1);
    }

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn clone() {
        for op in ALL_OPERATIONS {
            let cloned = op.clone();
            assert_eq!(op, cloned);
        }
    }

    #[test]
    fn copy() {
        let op = Operation::CreateAccounts;
        let copied = op; // Uses Copy
        assert_eq!(op, copied);
        assert_eq!(op.as_u8(), copied.as_u8());
    }

    #[test]
    fn debug() {
        assert_eq!(format!("{:?}", Operation::Pulse), "Pulse");
        assert_eq!(format!("{:?}", Operation::CreateAccounts), "CreateAccounts");
    }

    #[test]
    fn equality() {
        assert_eq!(Operation::Pulse, Operation::Pulse);
        assert_ne!(Operation::Pulse, Operation::CreateAccounts);
    }

    #[test]
    fn hash() {
        use std::collections::HashSet;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut set = HashSet::new();
        for op in ALL_OPERATIONS {
            set.insert(op);
        }
        assert_eq!(set.len(), OPERATION_COUNT as usize);

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

    #[test]
    fn error_value() {
        let err = InvalidOperation(42);
        assert_eq!(err.value(), 42);
    }

    #[test]
    fn error_range() {
        let below = InvalidOperation(100);
        assert!(below.is_below_range());
        assert!(!below.is_above_range());

        let above = InvalidOperation(200);
        assert!(!above.is_below_range());
        assert!(above.is_above_range());

        let just_below = InvalidOperation(OPERATION_MIN - 1);
        assert!(just_below.is_below_range());

        let just_above = InvalidOperation(OPERATION_MAX + 1);
        assert!(just_above.is_above_range());
    }

    #[test]
    fn error_display() {
        let err = InvalidOperation(127);
        let msg = format!("{}", err);
        assert!(msg.contains("127"));
        assert!(msg.contains("128"));
        assert!(msg.contains("145"));
    }

    #[test]
    fn error_trait() {
        fn assert_is_error<T: std::error::Error>() {}
        assert_is_error::<InvalidOperation>();

        use std::error::Error;
        let err = InvalidOperation(127);
        assert!(err.source().is_none());
    }

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn error_clone() {
        let err1 = InvalidOperation(255);
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    #[test]
    fn error_debug() {
        let err = InvalidOperation(99);
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("99"));
    }

    #[test]
    fn fuzz_try_from() {
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
    fn fuzz_as_u8() {
        for op in ALL_OPERATIONS {
            let result = std::panic::catch_unwind(|| op.as_u8());
            assert!(result.is_ok(), "{:?}.as_u8() panicked", op);
        }
    }

    #[test]
    fn fuzz_is_valid_discriminant() {
        for byte in 0..=255u8 {
            let result = std::panic::catch_unwind(|| Operation::is_valid_discriminant(byte));
            assert!(result.is_ok());
        }
    }

    #[test]
    fn fuzz_is_deprecated() {
        for op in ALL_OPERATIONS {
            let result = std::panic::catch_unwind(|| op.is_deprecated());
            assert!(result.is_ok(), "{:?}.is_deprecated() panicked", op);
        }
    }

    #[test]
    fn fuzz_is_client_facing() {
        for op in ALL_OPERATIONS {
            let result = std::panic::catch_unwind(|| op.is_client_facing());
            assert!(result.is_ok(), "{:?}.is_client_facing() panicked", op);
        }
    }

    #[test]
    fn fuzz_is_valid_client_request() {
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
