//! Wire-level VSR command identifiers.
//!
//! This enum is parsed from the on-the-wire `Header.command` byte.
//! Discriminants are stable protocol bytes and MUST NOT change.

#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Command {
    Reserved = 0,

    Ping = 1,
    Pong = 2,

    PingClient = 3,
    PongClient = 4,

    Request = 5,
    Prepare = 6,
    PrepareOk = 7,
    Reply = 8,
    Commit = 9,

    StartViewChange = 10,
    DoViewChange = 11,

    Deprecated12 = 12,

    RequestStartView = 13,
    RequestHeaders = 14,
    RequestPrepare = 15,
    RequestReply = 16,
    Headers = 17,

    Eviction = 18,

    RequestBlocks = 19,
    Block = 20,

    Deprecated21 = 21,
    Deprecated22 = 22,
    Deprecated23 = 23,

    StartView = 24,
}

impl Command {
    pub const MIN: u8 = 0;
    pub const MAX: u8 = 24;
    pub const COUNT: u8 = Self::MAX - Self::MIN + 1;

    /// All command variants in discriminant order (0..=24).
    pub const ALL: [Self; Self::COUNT as usize] = [
        Self::Reserved,
        Self::Ping,
        Self::Pong,
        Self::PingClient,
        Self::PongClient,
        Self::Request,
        Self::Prepare,
        Self::PrepareOk,
        Self::Reply,
        Self::Commit,
        Self::StartViewChange,
        Self::DoViewChange,
        Self::Deprecated12,
        Self::RequestStartView,
        Self::RequestHeaders,
        Self::RequestPrepare,
        Self::RequestReply,
        Self::Headers,
        Self::Eviction,
        Self::RequestBlocks,
        Self::Block,
        Self::Deprecated21,
        Self::Deprecated22,
        Self::Deprecated23,
        Self::StartView,
    ];

    const _RESERVED: () = assert!(Self::Reserved as u8 == Self::MIN);
    const _START_VIEW: () = assert!(Self::StartView as u8 == Self::MAX);
    const _COUNT: () = assert!(Self::COUNT == 25);
    const _ALL_LEN: () = assert!(Self::ALL.len() == Self::COUNT as usize);
    const _ALL_CONTIGUOUS: () = {
        let mut i = 0u8;
        while i < Self::COUNT {
            assert!(Self::ALL[i as usize] as u8 == i);
            i += 1;
        }
    };

    /// Raw on-the-wire byte for this command.
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Returns true iff this is a reserved deprecated wire value.
    #[inline]
    pub const fn is_deprecated(self) -> bool {
        matches!(
            self,
            Self::Deprecated12 | Self::Deprecated21 | Self::Deprecated22 | Self::Deprecated23
        )
    }

    /// Fast decoding from a wire byte.
    #[inline]
    pub fn try_from_u8(b: u8) -> Option<Self> {
        if b <= Self::MAX {
            Some(Self::ALL[b as usize])
        } else {
            None
        }
    }
}

/// Error returned when converting an invalid byte to a [`Command`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidCommand(pub u8);

impl core::fmt::Display for InvalidCommand {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid command byte: {}", self.0)
    }
}

impl std::error::Error for InvalidCommand {}

impl TryFrom<u8> for Command {
    type Error = InvalidCommand;

    #[inline]
    fn try_from(b: u8) -> Result<Self, Self::Error> {
        Self::try_from_u8(b).ok_or(InvalidCommand(b))
    }
}

impl From<Command> for u8 {
    #[inline]
    fn from(cmd: Command) -> Self {
        cmd.as_u8()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ==========================================================================
    // PropTest Strategy for Command Enum
    // ==========================================================================

    impl Arbitrary for Command {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            (Command::MIN..=Command::MAX)
                .prop_map(|b| Command::try_from_u8(b).unwrap())
                .boxed()
        }
    }

    // ==========================================================================
    // Property-Based Tests: Core Invariants
    // ==========================================================================

    proptest! {
        #[test]
        fn roundtrip_all_variants(variant in any::<Command>()) {
            let byte = variant.as_u8();
            let recovered = Command::try_from_u8(byte)
                .expect("Valid command must convert back from its byte");
            prop_assert_eq!(variant, recovered);
        }

        #[test]
        fn inverse_property_for_valid_bytes(byte in Command::MIN..=Command::MAX) {
            let cmd = Command::try_from_u8(byte)
                .expect("Byte in valid range must convert");
            prop_assert_eq!(cmd.as_u8(), byte);
        }

        #[test]
        fn all_variants_have_unique_discriminants(
            v1 in any::<Command>(),
            v2 in any::<Command>()
        ) {
            if v1 != v2 {
                prop_assert_ne!(v1.as_u8(), v2.as_u8());
            }
        }

        #[test]
        fn try_from_trait_matches_try_from_u8(byte: u8) {
            let option_result = Command::try_from_u8(byte);
            let try_from_result = Command::try_from(byte);

            match (option_result, try_from_result) {
                (Some(cmd1), Ok(cmd2)) => prop_assert_eq!(cmd1, cmd2),
                (None, Err(InvalidCommand(b))) => prop_assert_eq!(b, byte),
                _ => prop_assert!(false, "Mismatch between try_from_u8 and TryFrom"),
            }
        }
    }

    // ==========================================================================
    // Unit Tests: Wire Protocol Stability
    // ==========================================================================

    #[test]
    fn discriminants_are_stable_protocol_values() {
        assert_eq!(Command::Reserved.as_u8(), 0);
        assert_eq!(Command::Ping.as_u8(), 1);
        assert_eq!(Command::Pong.as_u8(), 2);
        assert_eq!(Command::PingClient.as_u8(), 3);
        assert_eq!(Command::PongClient.as_u8(), 4);
        assert_eq!(Command::Request.as_u8(), 5);
        assert_eq!(Command::Prepare.as_u8(), 6);
        assert_eq!(Command::PrepareOk.as_u8(), 7);
        assert_eq!(Command::Reply.as_u8(), 8);
        assert_eq!(Command::Commit.as_u8(), 9);
        assert_eq!(Command::StartViewChange.as_u8(), 10);
        assert_eq!(Command::DoViewChange.as_u8(), 11);
        assert_eq!(Command::Deprecated12.as_u8(), 12);
        assert_eq!(Command::RequestStartView.as_u8(), 13);
        assert_eq!(Command::RequestHeaders.as_u8(), 14);
        assert_eq!(Command::RequestPrepare.as_u8(), 15);
        assert_eq!(Command::RequestReply.as_u8(), 16);
        assert_eq!(Command::Headers.as_u8(), 17);
        assert_eq!(Command::Eviction.as_u8(), 18);
        assert_eq!(Command::RequestBlocks.as_u8(), 19);
        assert_eq!(Command::Block.as_u8(), 20);
        assert_eq!(Command::Deprecated21.as_u8(), 21);
        assert_eq!(Command::Deprecated22.as_u8(), 22);
        assert_eq!(Command::Deprecated23.as_u8(), 23);
        assert_eq!(Command::StartView.as_u8(), 24);
    }

    // ==========================================================================
    // Unit Tests: Deprecated Commands
    // ==========================================================================

    #[test]
    fn deprecated_commands_are_identified() {
        assert!(Command::Deprecated12.is_deprecated());
        assert!(Command::Deprecated21.is_deprecated());
        assert!(Command::Deprecated22.is_deprecated());
        assert!(Command::Deprecated23.is_deprecated());

        assert!(!Command::Reserved.is_deprecated());
        assert!(!Command::Ping.is_deprecated());
        assert!(!Command::StartView.is_deprecated());
    }

    // ==========================================================================
    // Unit Tests: Conversion Correctness
    // ==========================================================================

    #[test]
    fn all_array_matches_try_from_u8() {
        for (i, &cmd) in Command::ALL.iter().enumerate() {
            let expected = Command::try_from_u8(i as u8).unwrap();
            assert_eq!(cmd, expected, "ALL[{}] doesn't match try_from_u8({})", i, i);
        }
    }

    // ==========================================================================
    // Unit Tests: Memory Layout
    // ==========================================================================

    #[test]
    fn repr_u8_guarantees() {
        assert_eq!(std::mem::size_of::<Command>(), 1);
        assert_eq!(std::mem::align_of::<Command>(), 1);
    }

    // ==========================================================================
    // Unit Tests: Trait Implementations
    // ==========================================================================

    #[test]
    fn from_command_for_u8_works() {
        for cmd in Command::ALL {
            let byte: u8 = cmd.into();
            assert_eq!(byte, cmd.as_u8());
        }
    }

    #[test]
    fn invalid_command_display() {
        let err = InvalidCommand(42);
        assert_eq!(format!("{}", err), "invalid command byte: 42");
    }

    #[test]
    fn command_is_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for cmd in Command::ALL {
            set.insert(cmd);
        }
        assert_eq!(set.len(), Command::COUNT as usize);
    }

    // ==========================================================================
    // Fuzz Tests: Robustness Against Untrusted Input
    // ==========================================================================

    #[test]
    fn fuzz_try_from_u8_never_panics() {
        for byte in 0..=255u8 {
            let result = std::panic::catch_unwind(|| Command::try_from_u8(byte));
            assert!(
                result.is_ok(),
                "try_from_u8({}) panicked - must handle all bytes gracefully",
                byte
            );
        }
    }

    #[test]
    fn fuzz_as_u8_never_panics() {
        for variant in Command::ALL {
            let result = std::panic::catch_unwind(|| variant.as_u8());
            assert!(result.is_ok(), "{:?}.as_u8() panicked", variant);
        }
    }
}
