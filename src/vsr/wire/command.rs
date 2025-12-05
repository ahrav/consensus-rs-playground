//! Wire-level command identifiers; discriminants are stable protocol bytes.
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
    StartView = 12,
    RequestStartView = 13,
    RequestHeaders = 14,
    RequestPrepare = 15,
    RequestReply = 16,
    Headers = 17,
    Eviction = 18,
    RequestBlocks = 19,
    Block = 20,
}

impl Command {
    pub const MIN: u8 = 0;
    pub const MAX: u8 = 20;
    pub const COUNT: u8 = Self::MAX - Self::MIN + 1;

    /// All command variants in discriminant order.
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
        Self::StartView,
        Self::RequestStartView,
        Self::RequestHeaders,
        Self::RequestPrepare,
        Self::RequestReply,
        Self::Headers,
        Self::Eviction,
        Self::RequestBlocks,
        Self::Block,
    ];

    // Compile-time: catches constant drift
    const _RESERVED: () = assert!(Self::Reserved as u8 == Self::MIN);
    const _BLOCK: () = assert!(Self::Block as u8 == Self::MAX);
    const _COUNT: () = assert!(Self::COUNT == 21);

    // Compile-time: validates ALL array has correct length and contiguous discriminants
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
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Returns the `Command` for a wire byte, or `None` if the byte is unused.
    ///
    /// The match table is kept in sync with discriminants via the post-match
    /// assertion, so newly added variants must be listed here.
    pub fn try_from_u8(b: u8) -> Option<Self> {
        let cmd = match b {
            0 => Self::Reserved,
            1 => Self::Ping,
            2 => Self::Pong,
            3 => Self::PingClient,
            4 => Self::PongClient,
            5 => Self::Request,
            6 => Self::Prepare,
            7 => Self::PrepareOk,
            8 => Self::Reply,
            9 => Self::Commit,
            10 => Self::StartViewChange,
            11 => Self::DoViewChange,
            12 => Self::StartView,
            13 => Self::RequestStartView,
            14 => Self::RequestHeaders,
            15 => Self::RequestPrepare,
            16 => Self::RequestReply,
            17 => Self::Headers,
            18 => Self::Eviction,
            19 => Self::RequestBlocks,
            20 => Self::Block,
            _ => return None,
        };

        // Pair assertion: match logic agrees with discriminant
        assert!(cmd.as_u8() == b);

        Some(cmd)
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

    fn try_from(b: u8) -> Result<Self, Self::Error> {
        Self::try_from_u8(b).ok_or(InvalidCommand(b))
    }
}

impl From<Command> for u8 {
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
        // These values are part of the wire protocol and MUST NOT change
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
        assert_eq!(Command::StartView.as_u8(), 12);
        assert_eq!(Command::RequestStartView.as_u8(), 13);
        assert_eq!(Command::RequestHeaders.as_u8(), 14);
        assert_eq!(Command::RequestPrepare.as_u8(), 15);
        assert_eq!(Command::RequestReply.as_u8(), 16);
        assert_eq!(Command::Headers.as_u8(), 17);
        assert_eq!(Command::Eviction.as_u8(), 18);
        assert_eq!(Command::RequestBlocks.as_u8(), 19);
        assert_eq!(Command::Block.as_u8(), 20);
    }

    // ==========================================================================
    // Unit Tests: Conversion Correctness
    // ==========================================================================

    #[test]
    fn try_from_u8_covers_all_valid_bytes() {
        for byte in Command::MIN..=Command::MAX {
            let cmd = Command::try_from_u8(byte)
                .unwrap_or_else(|| panic!("Byte {} must map to a Command variant", byte));
            assert_eq!(cmd.as_u8(), byte);
        }
    }

    #[test]
    fn try_from_u8_rejects_invalid_bytes() {
        assert_eq!(Command::try_from_u8(Command::MAX + 1), None);
        assert_eq!(Command::try_from_u8(255), None);

        for byte in [21, 50, 100, 128, 200, 254, 255] {
            assert_eq!(Command::try_from_u8(byte), None);
        }
    }

    #[test]
    fn numbering_is_dense_no_gaps() {
        let mut found_variants = vec![false; Command::COUNT as usize];

        for byte in Command::MIN..=Command::MAX {
            if let Some(cmd) = Command::try_from_u8(byte) {
                found_variants[cmd.as_u8() as usize] = true;
            }
        }

        for (i, found) in found_variants.iter().enumerate() {
            assert!(*found, "Gap in numbering at discriminant {}", i);
        }
    }

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
    fn try_from_u8_for_command_works() {
        // Valid bytes
        for byte in Command::MIN..=Command::MAX {
            let result: Result<Command, InvalidCommand> = byte.try_into();
            assert!(result.is_ok());
            assert_eq!(result.unwrap().as_u8(), byte);
        }

        // Invalid bytes
        let result: Result<Command, InvalidCommand> = 21u8.try_into();
        assert_eq!(result, Err(InvalidCommand(21)));
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
