//! View change message handling for the VSR protocol.
//!
//! During view changes, replicas exchange [`HeaderPrepare`] sequences to synchronize
//! their operation logs. This module provides [`ViewChangeSlice`] as a validated,
//! borrowed view into these header sequences.

#[allow(unused_imports)]
use crate::{
    constants,
    stdx::bounded_array::BoundedArray,
    vsr::{header_prepare::HeaderPrepare, wire::Command},
};

// Sanity bounds: VIEW_HEADERS_MAX must be reasonable and total memory bounded to 1MB.
const _: () = {
    assert!(constants::VIEW_HEADERS_MAX >= 1);
    assert!(constants::VIEW_HEADERS_MAX <= 1024);

    assert!(std::mem::size_of::<HeaderPrepare>() * constants::VIEW_HEADERS_MAX <= 1024 * 1024);
};

/// Commands valid in view change context.
///
/// Only [`Command::StartView`] and [`Command::DoViewChange`] are permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewChangeCommand {
    StartView,
    DoViewChange,
}

impl TryFrom<Command> for ViewChangeCommand {
    type Error = ();

    fn try_from(cmd: Command) -> Result<Self, Self::Error> {
        match cmd {
            Command::StartView => Ok(Self::StartView),
            Command::DoViewChange => Ok(Self::DoViewChange),
            _ => Err(()),
        }
    }
}

impl From<ViewChangeCommand> for Command {
    fn from(cmd: ViewChangeCommand) -> Self {
        match cmd {
            ViewChangeCommand::StartView => Command::StartView,
            ViewChangeCommand::DoViewChange => Command::DoViewChange,
        }
    }
}

/// A validated, borrowed slice of [`HeaderPrepare`] entries for view change messages.
///
/// # Invariants
///
/// Maintained by construction ([`init`](Self::init)) and verified by [`verify`](Self::verify):
/// - `command` is [`ViewChangeCommand::StartView`] or [`ViewChangeCommand::DoViewChange`]
/// - `slice` is non-empty and at most [`VIEW_HEADERS_MAX`](constants::VIEW_HEADERS_MAX) entries
/// - All headers have `command == Command::Prepare` (checked in [`verify`](Self::verify))
#[derive(Copy, Clone, Debug)]
pub struct ViewChangeSlice<'a> {
    command: ViewChangeCommand,
    slice: &'a [HeaderPrepare],
}

impl<'a> ViewChangeSlice<'a> {
    /// Creates a new `ViewChangeSlice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is empty or exceeds [`VIEW_HEADERS_MAX`](constants::VIEW_HEADERS_MAX).
    #[inline]
    pub fn init(command: ViewChangeCommand, slice: &'a [HeaderPrepare]) -> Self {
        assert!(!slice.is_empty());
        assert!(slice.len() <= constants::VIEW_HEADERS_MAX);

        Self { command, slice }
    }

    #[inline]
    pub fn command(&self) -> ViewChangeCommand {
        self.command
    }

    #[inline]
    pub fn slice(&self) -> &'a [HeaderPrepare] {
        self.slice
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Asserts all invariants hold, including that every header is a `Prepare` command.
    ///
    /// Call after receiving from an untrusted source to validate contents.
    ///
    /// # Panics
    ///
    /// Panics if any invariant is violated.
    #[inline]
    pub fn verify(self) {
        assert!(!self.slice.is_empty());
        assert!(self.slice.len() <= constants::VIEW_HEADERS_MAX);

        assert!(
            self.slice()
                .iter()
                .all(|header| header.command == Command::Prepare)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsr::header_prepare::HeaderPrepare;

    fn header_prepare() -> HeaderPrepare {
        let mut h = HeaderPrepare::new();
        h.command = Command::Prepare;
        h
    }

    #[test]
    fn test_init_start_view() {
        let headers = [header_prepare()];
        let slice = ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);

        assert_eq!(slice.command(), ViewChangeCommand::StartView);
        assert_eq!(slice.len(), 1);
        assert!(!slice.is_empty());
        assert_eq!(slice.slice().len(), 1);
        slice.verify();
    }

    #[test]
    fn test_init_do_view_change() {
        let headers = [header_prepare(), header_prepare()];
        let slice = ViewChangeSlice::init(ViewChangeCommand::DoViewChange, &headers);

        assert_eq!(slice.command(), ViewChangeCommand::DoViewChange);
        assert_eq!(slice.len(), 2);
        slice.verify();
    }

    #[test]
    fn test_copy_clone() {
        let headers = [header_prepare()];
        let s1 = ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);
        let s2 = s1; // Copy
        let s3 = s1; // Copy

        assert_eq!(s1.command(), s2.command());
        assert_eq!(s1.len(), s3.len());
    }

    #[test]
    fn test_bounds_max() {
        let headers = [header_prepare(); constants::VIEW_HEADERS_MAX];
        let slice = ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);
        assert_eq!(slice.len(), constants::VIEW_HEADERS_MAX);
        slice.verify();
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_init_panic_empty() {
        let headers: [HeaderPrepare; 0] = [];
        ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_init_panic_oversized() {
        let headers = [header_prepare(); constants::VIEW_HEADERS_MAX + 1];
        ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_verify_panic_inner_cmd() {
        let mut invalid = header_prepare();
        invalid.command = Command::StartView; // Inner must be Prepare
        let headers = [invalid];

        ViewChangeSlice::init(ViewChangeCommand::StartView, &headers).verify();
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_verify_panic_mixed_cmds() {
        let valid = header_prepare();
        let mut invalid = header_prepare();
        invalid.command = Command::Ping;
        let headers = [valid, invalid];

        ViewChangeSlice::init(ViewChangeCommand::StartView, &headers).verify();
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_verify_panic_empty_slice_direct() {
        let headers: [HeaderPrepare; 0] = [];
        let slice = ViewChangeSlice {
            command: ViewChangeCommand::StartView,
            slice: &headers,
        };

        slice.verify();
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_verify_panic_oversized_slice_direct() {
        let headers = vec![header_prepare(); constants::VIEW_HEADERS_MAX + 1];
        let slice = ViewChangeSlice {
            command: ViewChangeCommand::StartView,
            slice: &headers,
        };

        slice.verify();
    }

    #[test]
    fn test_try_from_command() {
        assert_eq!(
            ViewChangeCommand::try_from(Command::StartView),
            Ok(ViewChangeCommand::StartView)
        );
        assert_eq!(
            ViewChangeCommand::try_from(Command::DoViewChange),
            Ok(ViewChangeCommand::DoViewChange)
        );

        assert!(ViewChangeCommand::try_from(Command::Prepare).is_err());
        assert!(ViewChangeCommand::try_from(Command::Commit).is_err());
        assert!(ViewChangeCommand::try_from(Command::Ping).is_err());
    }

    #[test]
    fn test_try_from_command_accepts_only_view_change_commands() {
        for cmd in Command::ALL {
            let is_view_change = matches!(cmd, Command::StartView | Command::DoViewChange);
            assert_eq!(ViewChangeCommand::try_from(cmd).is_ok(), is_view_change);
        }
    }

    #[test]
    fn test_into_command() {
        let cmd: Command = ViewChangeCommand::StartView.into();
        assert_eq!(cmd, Command::StartView);
        let cmd: Command = ViewChangeCommand::DoViewChange.into();
        assert_eq!(cmd, Command::DoViewChange);
    }

    #[test]
    fn test_verify_idempotent() {
        let headers = [header_prepare()];
        let slice = ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);
        slice.verify();
        slice.verify();
    }

    #[test]
    fn test_sanity_checks() {
        const {
            assert!(constants::VIEW_HEADERS_MAX >= 1);
            assert!(constants::VIEW_HEADERS_MAX <= 1024);
            assert!(
                std::mem::size_of::<HeaderPrepare>() * constants::VIEW_HEADERS_MAX <= 1024 * 1024
            );
        }
    }
}

// ==========================================================================
// Property-Based Tests
// ==========================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn valid_view_change_command_strategy() -> impl Strategy<Value = ViewChangeCommand> {
        prop_oneof![
            Just(ViewChangeCommand::StartView),
            Just(ViewChangeCommand::DoViewChange),
        ]
    }

    proptest! {
        /// Property: init succeeds for valid commands and non-empty slices within bounds.
        #[test]
        fn prop_init_valid_commands_succeed(
            cmd in valid_view_change_command_strategy(),
            len in 1usize..=constants::VIEW_HEADERS_MAX,
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; len];

            let slice = ViewChangeSlice::init(cmd, &headers);
            prop_assert_eq!(slice.command(), cmd);
            prop_assert_eq!(slice.len(), len);
            prop_assert!(!slice.is_empty());
        }

        /// Property: verify passes when all headers are Command::Prepare.
        #[test]
        fn prop_verify_all_prepare_headers(
            cmd in valid_view_change_command_strategy(),
            len in 1usize..=constants::VIEW_HEADERS_MAX,
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; len];

            let slice = ViewChangeSlice::init(cmd, &headers);
            slice.verify(); // Should not panic
        }

        /// Property: len() returns the same value as the input slice length.
        #[test]
        fn prop_len_matches_input(
            len in 1usize..=constants::VIEW_HEADERS_MAX,
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; len];

            let slice = ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);
            prop_assert_eq!(slice.len(), len);
            prop_assert_eq!(slice.slice().len(), len);
        }

        /// Property: is_empty() is always false (since empty slices are rejected).
        #[test]
        fn prop_is_empty_always_false(
            len in 1usize..=constants::VIEW_HEADERS_MAX,
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; len];

            let slice = ViewChangeSlice::init(ViewChangeCommand::DoViewChange, &headers);
            prop_assert!(!slice.is_empty());
        }

        /// Property: slice() returns a reference to the original data.
        #[test]
        fn prop_slice_returns_original_reference(
            len in 1usize..=constants::VIEW_HEADERS_MAX,
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; len];

            let slice = ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);
            let returned = slice.slice();

            prop_assert_eq!(returned.as_ptr(), headers.as_ptr());
            prop_assert_eq!(returned.len(), headers.len());
        }

        /// Property: Copy and Clone produce equivalent instances.
        #[test]
        fn prop_copy_clone_equivalent(
            len in 1usize..=constants::VIEW_HEADERS_MAX,
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; len];

            let slice1 = ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);
            let slice2 = slice1;
            let slice3 = slice1;

            prop_assert_eq!(slice1.command(), slice2.command());
            prop_assert_eq!(slice1.len(), slice2.len());
            prop_assert_eq!(slice1.command(), slice3.command());
            prop_assert_eq!(slice1.len(), slice3.len());
        }

        /// Property: Both valid view change commands are accepted.
        #[test]
        fn prop_both_valid_commands_accepted(
            cmd in valid_view_change_command_strategy(),
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; 5];

            let slice = ViewChangeSlice::init(cmd, &headers);
            prop_assert_eq!(slice.command(), cmd);
        }

        /// Property: verify is idempotent - can be called multiple times.
        #[test]
        fn prop_verify_idempotent(
            len in 1usize..=constants::VIEW_HEADERS_MAX,
            call_count in 1usize..10,
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; len];

            let slice = ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);

            for _ in 0..call_count {
                slice.verify();
            }
        }
    }
}
