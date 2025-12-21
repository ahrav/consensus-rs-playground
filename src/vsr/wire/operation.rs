//! Wire-level operation codes for VSR protocol messages.
//!
//! Operations are split into two ranges:
//! - **VSR-reserved** (0–127): Protocol-level operations handled by the replication layer.
//! - **User-defined** (128–255): Application-specific operations via [`StateMachineOperation`].
//!
//! This separation allows VSR to manage consensus mechanics (view changes, recovery)
//! independently of application logic.

use core::fmt;

use crate::constants;

/// Highest VSR-reserved operation with a defined meaning.
/// Operations 7–127 are reserved for future protocol use.
const VSR_RESERVED_MAX: u8 = 6;

// Compile-time validation of operation discriminants.
// These values are part of the wire protocol and must never change.
const _: () = {
    assert!(Operation::RESERVED.0 == 0);
    assert!(Operation::ROOT.0 == 1);
    assert!(Operation::REGISTER.0 == 2);
    assert!(Operation::RECONFIGURE.0 == 3);
    assert!(Operation::PULSE.0 == 4);
    assert!(Operation::UPGRADE.0 == 5);
    assert!(Operation::NOOP.0 == 6);

    assert!(constants::VSR_OPERATIONS_RESERVED > VSR_RESERVED_MAX);
    assert!(constants::VSR_OPERATIONS_RESERVED <= 128);
};

/// Trait for application-defined state machine operations.
///
/// Implementors define the operations their state machine supports. Each operation
/// must have a stable `u8` discriminant in the range `128..=255`.
///
/// # Example
///
/// ```ignore
/// #[derive(Copy, Clone, Eq, PartialEq)]
/// #[repr(u8)]
/// enum MyOp {
///     CreateAccount = 128,
///     Transfer = 129,
/// }
///
/// impl StateMachineOperation for MyOp {
///     fn from_u8(v: u8) -> Option<Self> {
///         match v {
///             128 => Some(Self::CreateAccount),
///             129 => Some(Self::Transfer),
///             _ => None,
///         }
///     }
///     fn as_u8(&self) -> u8 { *self as u8 }
///     fn tag_name(self) -> &'static str {
///         match self {
///             Self::CreateAccount => "create_account",
///             Self::Transfer => "transfer",
///         }
///     }
/// }
/// ```
pub trait StateMachineOperation: Copy + Eq {
    /// Converts a raw byte to this operation type.
    /// Returns `None` if the value is not a valid discriminant.
    fn from_u8(v: u8) -> Option<Self>;

    /// Returns the wire discriminant for this operation.
    fn as_u8(&self) -> u8;

    /// Returns a human-readable name for logging/debugging.
    fn tag_name(self) -> &'static str;

    /// Converts from wire [`Operation`], returning `None` for VSR-reserved operations.
    #[inline]
    fn from_vsr(op: Operation) -> Option<Self> {
        if op.vsr_reserved() {
            return None;
        }
        Self::from_u8(op.as_u8())
    }
}

/// Wire-level operation code carried in message headers.
///
/// A thin wrapper over `u8` that distinguishes VSR-reserved operations from
/// user-defined state machine operations. Use [`Operation::from`] and [`Operation::to`]
/// to convert between this type and a [`StateMachineOperation`].
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Operation(u8);

impl Operation {
    /// Invalid/uninitialized operation. Used as a sentinel in zeroed headers.
    pub const RESERVED: Self = Self(0);

    /// Bootstraps the commit log with an initial entry at op-number 0.
    pub const ROOT: Self = Self(1);

    /// Registers a new client session with the cluster.
    pub const REGISTER: Self = Self(2);

    /// Reconfigures cluster membership (add/remove replicas).
    pub const RECONFIGURE: Self = Self(3);

    /// Periodic heartbeat to drive liveness and view-change detection.
    pub const PULSE: Self = Self(4);

    /// Upgrades the cluster to a new protocol version.
    pub const UPGRADE: Self = Self(5);

    /// No-operation used for padding or leader assertions.
    pub const NOOP: Self = Self(6);

    /// Wraps a raw byte without validation.
    #[inline]
    pub const fn from_u8(v: u8) -> Self {
        Self(v)
    }

    /// Returns the underlying byte value.
    #[inline]
    pub const fn as_u8(&self) -> u8 {
        self.0
    }

    /// Wraps a state machine operation for wire transmission.
    ///
    /// # Panics
    ///
    /// Panics if the operation's discriminant is in the VSR-reserved range.
    #[inline]
    pub fn from<SM: StateMachineOperation>(operation: SM) -> Self {
        let v = operation.as_u8();

        assert!(v >= constants::VSR_OPERATIONS_RESERVED);
        let roundtrip = SM::from_u8(v).expect("state machine operation must round-trip");
        assert!(roundtrip.as_u8() == v);

        Self(v)
    }

    /// Extracts a state machine operation from this wire operation.
    ///
    /// # Panics
    ///
    /// Panics if this operation is VSR-reserved or invalid for `SM`.
    #[inline]
    pub fn to<SM: StateMachineOperation>(operation: Self) -> SM {
        assert!(
            operation.valid::<SM>(),
            "invalid operation for this state machine"
        );
        assert!(
            !operation.vsr_reserved(),
            "reserved operations cannot be converted to state machine operations"
        );

        let result = SM::from_u8(operation.0).expect("valid non-reserved op must map to SM");
        assert!(result.as_u8() == operation.0);

        result
    }

    /// Converts to a state machine operation, panicking on failure.
    ///
    /// Unlike [`Operation::to`], this is a method taking `self`.
    ///
    /// # Panics
    ///
    /// Panics if this operation cannot be converted to `SM`.
    #[inline]
    pub fn cast<SM: StateMachineOperation>(self) -> SM {
        let result =
            SM::from_vsr(self).expect("operation not covertible to state machine operation");
        assert!(result.as_u8() == self.0 || self.vsr_reserved());

        result
    }

    /// Returns `true` if this operation is a defined VSR operation (0-6) or a
    /// user-defined `SM` operation in the `128..=255` range that round-trips.
    #[inline]
    pub fn valid<SM: StateMachineOperation>(self) -> bool {
        if self.0 <= VSR_RESERVED_MAX {
            true
        } else if self.0 < constants::VSR_OPERATIONS_RESERVED {
            false
        } else {
            SM::from_u8(self.0)
                .map(|op| op.as_u8() == self.0)
                .unwrap_or(false)
        }
    }

    /// Returns `true` if this operation is in the VSR-reserved range (0–127).
    ///
    /// Reserved operations are handled by the protocol layer, not the state machine.
    #[inline]
    pub const fn vsr_reserved(&self) -> bool {
        self.0 < constants::VSR_OPERATIONS_RESERVED
    }

    /// Returns the human-readable name for this operation.
    ///
    /// # Panics
    ///
    /// Panics if `self.valid::<SM>()` is false.
    #[inline]
    pub fn tag_name<SM: StateMachineOperation>(self) -> &'static str {
        assert!(self.valid::<SM>());

        if self.0 <= VSR_RESERVED_MAX {
            Self::vsr_reserved_tag_name(self.0)
        } else {
            SM::from_u8(self.0)
                .expect("valid non-reserved op must map to SM")
                .tag_name()
        }
    }

    #[inline]
    const fn vsr_reserved_tag_name(opcode: u8) -> &'static str {
        assert!(opcode <= VSR_RESERVED_MAX);

        match opcode {
            0 => "reserved",
            1 => "root",
            2 => "register",
            3 => "reconfigure",
            4 => "pulse",
            5 => "upgrade",
            6 => "noop",
            _ => unreachable!(),
        }
    }
}

impl Default for Operation {
    /// Defaults to [`Operation::RESERVED`].
    #[inline]
    fn default() -> Self {
        Operation::RESERVED
    }
}

impl From<u8> for Operation {
    #[inline]
    fn from(v: u8) -> Self {
        Operation::from_u8(v)
    }
}

impl From<Operation> for u8 {
    #[inline]
    fn from(op: Operation) -> u8 {
        op.as_u8()
    }
}

impl fmt::Debug for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 <= VSR_RESERVED_MAX {
            write!(
                f,
                "Operation({}={})",
                Self::vsr_reserved_tag_name(self.0),
                self.0
            )
        } else {
            write!(f, "Operation({})", self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use core::hash::{Hash, Hasher};

    use super::*;

    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    #[repr(u8)]
    enum TestOp {
        CreateAccount = 128,
        Transfer = 129,
    }

    impl StateMachineOperation for TestOp {
        fn from_u8(v: u8) -> Option<Self> {
            match v {
                128 => Some(Self::CreateAccount),
                129 => Some(Self::Transfer),
                _ => None,
            }
        }

        fn as_u8(&self) -> u8 {
            *self as u8
        }

        fn tag_name(self) -> &'static str {
            match self {
                Self::CreateAccount => "create_account",
                Self::Transfer => "transfer",
            }
        }
    }

    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    struct InconsistentOp(u8);

    impl StateMachineOperation for InconsistentOp {
        fn from_u8(v: u8) -> Option<Self> {
            if v == 128 { Some(Self(129)) } else { None }
        }

        fn as_u8(&self) -> u8 {
            self.0
        }

        fn tag_name(self) -> &'static str {
            "inconsistent"
        }
    }

    // =========================================================================
    // Basic validity tests
    // =========================================================================

    #[test]
    fn test_vsr_constants() {
        // All named VSR constants must be valid and reserved.
        let constants = [
            Operation::RESERVED,
            Operation::ROOT,
            Operation::REGISTER,
            Operation::RECONFIGURE,
            Operation::PULSE,
            Operation::UPGRADE,
            Operation::NOOP,
        ];

        for op in constants {
            assert!(op.valid::<TestOp>());
            assert!(op.vsr_reserved());
        }
    }

    #[test]
    fn test_invalid_gap() {
        // Ensure operations in the gap 7..128 are invalid
        for op_code in (VSR_RESERVED_MAX + 1)..constants::VSR_OPERATIONS_RESERVED {
            let op = Operation::from_u8(op_code);
            assert!(!op.valid::<TestOp>(), "Op {} should be invalid", op_code);
            assert!(op.vsr_reserved(), "Op {} should be reserved", op_code);
        }
    }

    #[test]
    fn test_reserved_range_not_valid_for_sm() {
        #[derive(Copy, Clone, Eq, PartialEq, Debug)]
        struct ReservedRangeOp;

        impl StateMachineOperation for ReservedRangeOp {
            fn from_u8(v: u8) -> Option<Self> {
                if v == 42 { Some(Self) } else { None }
            }

            fn as_u8(&self) -> u8 {
                42
            }

            fn tag_name(self) -> &'static str {
                "reserved_range"
            }
        }

        let op = Operation::from_u8(42);
        assert!(op.vsr_reserved());
        assert!(!op.valid::<ReservedRangeOp>());
    }

    #[test]
    fn test_sm_roundtrip() {
        let op = Operation::from(TestOp::CreateAccount);
        assert!(op.as_u8() == 128);
        assert!(op.valid::<TestOp>());

        let sm_op: TestOp = Operation::to(op);
        assert!(sm_op == TestOp::CreateAccount);
    }

    #[test]
    fn test_invalid_opcode() {
        let op = Operation::from_u8(200); // Not a TestOp
        assert!(!op.valid::<TestOp>());
    }

    #[test]
    fn test_inconsistent_sm_not_valid() {
        let op = Operation::from_u8(128);
        assert!(!op.valid::<InconsistentOp>());
    }

    #[test]
    fn test_sm_not_reserved() {
        let create = Operation::from(TestOp::CreateAccount);
        let transfer = Operation::from(TestOp::Transfer);

        assert!(!create.vsr_reserved());
        assert!(!transfer.vsr_reserved());
    }

    #[test]
    fn test_sm_valid() {
        let create = Operation::from(TestOp::CreateAccount);
        let transfer = Operation::from(TestOp::Transfer);

        assert!(create.valid::<TestOp>());
        assert!(transfer.valid::<TestOp>());
    }

    // =========================================================================
    // Boundary tests (127/128 boundary between VSR reserved and user ops)
    // =========================================================================

    #[test]
    fn test_reserved_boundary() {
        // All 0-127 should be vsr_reserved
        for opcode in 0..128 {
            assert!(
                Operation::from_u8(opcode).vsr_reserved(),
                "opcode {opcode} should be vsr_reserved"
            );
        }

        // All 128-255 should NOT be vsr_reserved
        for opcode in 128..=255 {
            assert!(
                !Operation::from_u8(opcode).vsr_reserved(),
                "opcode {opcode} should NOT be vsr_reserved"
            );
        }
    }

    // =========================================================================
    // from_vsr trait method tests
    // =========================================================================

    #[test]
    fn test_from_vsr_reserved() {
        for opcode in 0..128 {
            let op = Operation::from_u8(opcode);
            assert!(
                TestOp::from_vsr(op).is_none(),
                "from_vsr should return None for VSR reserved opcode {opcode}"
            );
        }
    }

    #[test]
    fn test_from_vsr_sm() {
        let create_op = Operation::from_u8(128);
        let transfer_op = Operation::from_u8(129);

        assert_eq!(TestOp::from_vsr(create_op), Some(TestOp::CreateAccount));
        assert_eq!(TestOp::from_vsr(transfer_op), Some(TestOp::Transfer));
    }

    #[test]
    fn test_from_vsr_invalid() {
        let invalid_op = Operation::from_u8(200);
        assert!(TestOp::from_vsr(invalid_op).is_none());
    }

    // =========================================================================
    // cast() method tests
    // =========================================================================

    #[test]
    fn test_cast_sm() {
        let create_op = Operation::from_u8(128);
        let transfer_op = Operation::from_u8(129);

        let create: TestOp = create_op.cast();
        let transfer: TestOp = transfer_op.cast();

        assert_eq!(create, TestOp::CreateAccount);
        assert_eq!(transfer, TestOp::Transfer);
    }

    #[test]
    #[should_panic(expected = "operation not covertible to state machine operation")]
    fn test_cast_reserved_panic() {
        Operation::ROOT.cast::<TestOp>();
    }

    #[test]
    #[should_panic(expected = "operation not covertible to state machine operation")]
    fn test_cast_invalid_panic() {
        let invalid = Operation::from_u8(200);
        let _: TestOp = invalid.cast();
    }

    #[test]
    fn test_cast_remapping_logic() {
        // Define a custom SM that maps ROOT(1) to a user op.
        #[derive(Copy, Clone, Eq, PartialEq, Debug)]
        struct RemappingOp(u8);
        impl StateMachineOperation for RemappingOp {
            fn from_u8(v: u8) -> Option<Self> {
                if v == 128 { Some(Self(128)) } else { None }
            }
            fn as_u8(&self) -> u8 {
                self.0
            }
            fn tag_name(self) -> &'static str {
                "remapping"
            }
            fn from_vsr(op: Operation) -> Option<Self> {
                if op == Operation::ROOT {
                    Some(Self(128))
                } else {
                    Self::from_u8(op.as_u8())
                }
            }
        }

        let op = Operation::ROOT;
        // cast allows the mapping because from_vsr handles it
        let sm = op.cast::<RemappingOp>();
        assert_eq!(sm.0, 128);
    }

    // =========================================================================
    // tag_name() tests
    // =========================================================================

    #[test]
    fn test_tag_name_reserved() {
        assert_eq!(Operation::RESERVED.tag_name::<TestOp>(), "reserved");
        assert_eq!(Operation::ROOT.tag_name::<TestOp>(), "root");
        assert_eq!(Operation::REGISTER.tag_name::<TestOp>(), "register");
        assert_eq!(Operation::RECONFIGURE.tag_name::<TestOp>(), "reconfigure");
        assert_eq!(Operation::PULSE.tag_name::<TestOp>(), "pulse");
        assert_eq!(Operation::UPGRADE.tag_name::<TestOp>(), "upgrade");
        assert_eq!(Operation::NOOP.tag_name::<TestOp>(), "noop");
    }

    #[test]
    fn test_tag_name_sm() {
        let create = Operation::from(TestOp::CreateAccount);
        let transfer = Operation::from(TestOp::Transfer);

        assert_eq!(create.tag_name::<TestOp>(), "create_account");
        assert_eq!(transfer.tag_name::<TestOp>(), "transfer");
    }

    #[test]
    #[should_panic]
    fn test_tag_name_panic() {
        let invalid = Operation::from_u8(200);
        let _ = invalid.tag_name::<TestOp>();
    }

    // =========================================================================
    // Panic behavior tests for from/to
    // =========================================================================

    /// A malformed state machine operation that returns a VSR-reserved discriminant.
    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    struct MalformedOp;

    impl StateMachineOperation for MalformedOp {
        fn from_u8(v: u8) -> Option<Self> {
            if v == 5 { Some(MalformedOp) } else { None }
        }

        fn as_u8(&self) -> u8 {
            5 // VSR-reserved range - this is invalid!
        }

        fn tag_name(self) -> &'static str {
            "malformed"
        }
    }

    #[test]
    #[should_panic]
    fn test_from_panic_reserved() {
        // MalformedOp has discriminant 5, which is in VSR-reserved range
        let _ = Operation::from(MalformedOp);
    }

    #[test]
    #[should_panic]
    fn test_from_panic_inconsistent_roundtrip() {
        let _ = Operation::from(InconsistentOp(128));
    }

    #[test]
    #[should_panic(
        expected = "reserved operations cannot be converted to state machine operations"
    )]
    fn test_to_panic_reserved() {
        Operation::to::<TestOp>(Operation::ROOT);
    }

    #[test]
    #[should_panic]
    fn test_to_panic_invalid() {
        let invalid = Operation::from_u8(200);
        let _: TestOp = Operation::to(invalid);
    }

    // =========================================================================
    // Default trait tests
    // =========================================================================

    #[test]
    fn test_default_is_reserved() {
        assert_eq!(Operation::default(), Operation::RESERVED);
        assert_eq!(Operation::default().as_u8(), 0);
    }

    // =========================================================================
    // From trait tests
    // =========================================================================

    #[test]
    fn test_from_trait_roundtrip() {
        for byte in 0..=255u8 {
            let op: Operation = byte.into();
            assert_eq!(op.as_u8(), byte);

            let back: u8 = op.into();
            assert_eq!(byte, back);
        }
    }

    // =========================================================================
    // Debug formatting tests
    // =========================================================================

    #[test]
    fn test_debug_reserved() {
        let debug_str = format!("{:?}", Operation::RESERVED);
        assert_eq!(debug_str, "Operation(reserved=0)");

        let debug_str = format!("{:?}", Operation::ROOT);
        assert_eq!(debug_str, "Operation(root=1)");

        let debug_str = format!("{:?}", Operation::NOOP);
        assert_eq!(debug_str, "Operation(noop=6)");
    }

    #[test]
    fn test_debug_sm() {
        let op = Operation::from_u8(128);
        let debug_str = format!("{:?}", op);
        assert_eq!(debug_str, "Operation(128)");

        let op = Operation::from_u8(255);
        let debug_str = format!("{:?}", op);
        assert_eq!(debug_str, "Operation(255)");
    }

    #[test]
    fn test_debug_undefined() {
        // Operations 7-127 are VSR reserved but undefined, so they show as raw numbers
        let op = Operation::from_u8(50);
        let debug_str = format!("{:?}", op);
        assert_eq!(debug_str, "Operation(50)");
    }

    // =========================================================================
    // Ordering tests
    // =========================================================================

    #[test]
    fn test_ordering() {
        // Verify ordering matches underlying u8 ordering
        for i in 0..255u8 {
            let op1 = Operation::from_u8(i);
            let op2 = Operation::from_u8(i + 1);
            assert!(op1 < op2);
            assert!(op2 > op1);
            assert!(op1 <= op1);
            assert!(op1 >= op1);
        }
    }

    // =========================================================================
    // Hash tests
    // =========================================================================

    /// Simple hasher for testing that collects bytes.
    #[derive(Default)]
    struct TestHasher {
        bytes: Vec<u8>,
    }

    impl Hasher for TestHasher {
        fn finish(&self) -> u64 {
            // Simple hash: treat first 8 bytes as u64
            let mut result = 0u64;
            for (i, &b) in self.bytes.iter().take(8).enumerate() {
                result |= (b as u64) << (i * 8);
            }
            result
        }

        fn write(&mut self, bytes: &[u8]) {
            self.bytes.extend_from_slice(bytes);
        }
    }

    #[test]
    fn test_operation_hash_consistency() {
        // Same operations should hash the same
        let op1 = Operation::from_u8(128);
        let op2 = Operation::from_u8(128);

        let mut h1 = TestHasher::default();
        let mut h2 = TestHasher::default();
        op1.hash(&mut h1);
        op2.hash(&mut h2);

        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_different_operations_hash_differently() {
        let op1 = Operation::from_u8(128);
        let op2 = Operation::from_u8(129);

        let mut h1 = TestHasher::default();
        let mut h2 = TestHasher::default();
        op1.hash(&mut h1);
        op2.hash(&mut h2);

        assert_ne!(h1.finish(), h2.finish());
    }

    // =========================================================================
    // Full roundtrip tests for all TestOp variants
    // =========================================================================

    #[test]
    fn test_all_test_ops_roundtrip() {
        for test_op in [TestOp::CreateAccount, TestOp::Transfer] {
            // SM -> Operation -> SM
            let wire_op = Operation::from(test_op);
            let recovered: TestOp = Operation::to(wire_op);
            assert_eq!(test_op, recovered);

            // Also test cast
            let cast_recovered: TestOp = wire_op.cast();
            assert_eq!(test_op, cast_recovered);

            // Also test from_vsr
            let vsr_recovered = TestOp::from_vsr(wire_op);
            assert_eq!(Some(test_op), vsr_recovered);
        }
    }
}

#[cfg(test)]
mod proptests {
    use proptest::prelude::*;

    use super::*;

    /// Test state machine with operations spanning the full user range.
    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    struct WideOp(u8);

    impl StateMachineOperation for WideOp {
        fn from_u8(v: u8) -> Option<Self> {
            if v >= 128 { Some(WideOp(v)) } else { None }
        }

        fn as_u8(&self) -> u8 {
            self.0
        }

        fn tag_name(self) -> &'static str {
            "wide_op"
        }
    }

    proptest! {
        #[test]
        fn proptest_u8_roundtrip(byte: u8) {
            let op = Operation::from_u8(byte);
            prop_assert_eq!(op.as_u8(), byte);

            // Also test via From traits
            let op2: Operation = byte.into();
            let back: u8 = op2.into();
            prop_assert_eq!(back, byte);
        }

        #[test]
        fn proptest_vsr_reserved_boundary(byte: u8) {
            let op = Operation::from_u8(byte);
            if byte < 128 {
                prop_assert!(op.vsr_reserved());
            } else {
                prop_assert!(!op.vsr_reserved());
            }
        }

        #[test]
        fn proptest_wide_op_roundtrip(byte in 128u8..=255) {
            let sm_op = WideOp(byte);
            let wire_op = Operation::from(sm_op);
            let recovered: WideOp = Operation::to(wire_op);
            prop_assert_eq!(sm_op, recovered);
        }

        #[test]
        fn proptest_valid_for_defined_vsr_or_sm(byte: u8) {
            let op = Operation::from_u8(byte);
            let is_valid = op.valid::<WideOp>();

            // Valid if: (0-6) OR (128-255)
            let expected_valid = byte <= 6 || byte >= 128;
            prop_assert_eq!(is_valid, expected_valid);
        }

        #[test]
        fn proptest_ordering_consistent_with_u8(a: u8, b: u8) {
            let op_a = Operation::from_u8(a);
            let op_b = Operation::from_u8(b);

            prop_assert_eq!(op_a.cmp(&op_b), a.cmp(&b));
            prop_assert_eq!(op_a.partial_cmp(&op_b), a.partial_cmp(&b));
        }

        #[test]
        fn proptest_equality_consistent_with_u8(a: u8, b: u8) {
            let op_a = Operation::from_u8(a);
            let op_b = Operation::from_u8(b);

            prop_assert_eq!(op_a == op_b, a == b);
            prop_assert_eq!(op_a != op_b, a != b);
        }

        #[test]
        fn proptest_from_vsr_never_returns_for_reserved(byte in 0u8..128) {
            let op = Operation::from_u8(byte);
            prop_assert!(WideOp::from_vsr(op).is_none());
        }

        #[test]
        fn proptest_from_vsr_returns_for_user_range(byte in 128u8..=255) {
            let op = Operation::from_u8(byte);
            let result = WideOp::from_vsr(op);
            prop_assert!(result.is_some());
            prop_assert_eq!(result.unwrap().0, byte);
        }
    }
}
