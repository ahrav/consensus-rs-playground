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
    assert!(Operation::Pulse as u8 == VSR_OPERATIONS_RESERVED + 0);
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
        match self {
            Operation::DeprecatedCreateAccountsUnbatched
            | Operation::DeprecatedCreateTransfersUnbatched
            | Operation::DeprecatedLookupAccountsUnbatched
            | Operation::DeprecatedLookupTransfersUnbatched
            | Operation::DeprecatedGetAccountTransfersUnbatched
            | Operation::DeprecatedGetAccountBalancesUnbatched => true,
            _ => false,
        }
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
            value >= OPERATION_MIN && value <= OPERATION_MAX,
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
        if v < OPERATION_MIN || v > OPERATION_MAX {
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

    // Tiger Style: compile-time test of constants
    const _: () = {
        assert!(OPERATION_MIN == 128);
        assert!(OPERATION_MAX == 145);
        assert!(OPERATION_COUNT == 18);
    };

    #[test]
    fn discriminants_match_tb() {
        // Tiger Style: split assertions (one per check)
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
    fn deprecated_operations() {
        // Verify deprecated classification
        assert!(Operation::DeprecatedCreateAccountsUnbatched.is_deprecated());
        assert!(Operation::DeprecatedCreateTransfersUnbatched.is_deprecated());
        assert!(Operation::DeprecatedLookupAccountsUnbatched.is_deprecated());
        assert!(Operation::DeprecatedLookupTransfersUnbatched.is_deprecated());
        assert!(Operation::DeprecatedGetAccountTransfersUnbatched.is_deprecated());
        assert!(Operation::DeprecatedGetAccountBalancesUnbatched.is_deprecated());

        // Non-deprecated should return false
        assert!(!Operation::Pulse.is_deprecated());
        assert!(!Operation::CreateAccounts.is_deprecated());
        assert!(!Operation::UnusedReserved.is_deprecated());
    }

    #[test]
    fn client_facing_operations() {
        // Client-facing ops
        assert!(Operation::CreateAccounts.is_client_facing());
        assert!(Operation::CreateTransfers.is_client_facing());
        assert!(Operation::LookupAccounts.is_client_facing());
        assert!(Operation::LookupTransfers.is_client_facing());
        assert!(Operation::GetAccountTransfers.is_client_facing());
        assert!(Operation::GetAccountBalances.is_client_facing());
        assert!(Operation::QueryAccounts.is_client_facing());
        assert!(Operation::QueryTransfers.is_client_facing());
        assert!(Operation::GetChangeEvents.is_client_facing());

        // Non-client-facing
        assert!(!Operation::Pulse.is_client_facing());
        assert!(!Operation::DeprecatedCreateAccountsUnbatched.is_client_facing());
        assert!(!Operation::UnusedReserved.is_client_facing());
        assert!(!Operation::QueryTransfersOp.is_client_facing());
    }

    #[test]
    fn enum_size_and_alignment() {
        // Tiger Style: verify wire format requirements
        assert_eq!(core::mem::size_of::<Operation>(), 1);
        assert_eq!(core::mem::align_of::<Operation>(), 1);
        assert_eq!(core::mem::size_of::<Option<Operation>>(), 1);
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
    fn invalid_operation_display() {
        let err = InvalidOperation(127);
        let msg = format!("{}", err);
        assert!(msg.contains("127"));
        assert!(msg.contains("128"));
        assert!(msg.contains("145"));
    }

    #[test]
    fn valid_client_requests() {
        // Only non-deprecated, client-facing ops should be valid
        assert!(Operation::CreateAccounts.is_valid_client_request());
        assert!(!Operation::DeprecatedCreateAccountsUnbatched.is_valid_client_request());
        assert!(!Operation::Pulse.is_valid_client_request());

        // Invariant: If it is a valid client request, it must not be deprecated
        for v in OPERATION_MIN..=OPERATION_MAX {
            if let Ok(op) = Operation::try_from(v) {
                if op.is_valid_client_request() {
                    assert!(!op.is_deprecated(), "{:?} should not be deprecated", op);
                    assert!(op.is_client_facing(), "{:?} should be client facing", op);
                }
            }
        }
    }
}
