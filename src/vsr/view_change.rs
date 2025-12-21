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

/// An owned, stack-allocated collection of [`HeaderPrepare`] entries for view change messages.
///
/// Unlike [`ViewChangeSlice`], this type owns its data and supports mutation via [`push`](Self::push).
/// Use this when building view change messages incrementally.
///
/// # Invariants
///
/// - Collection is never empty after construction
/// - Length never exceeds [`VIEW_HEADERS_MAX`](constants::VIEW_HEADERS_MAX)
/// - All headers have `command == Command::Prepare`
#[derive(Clone, Copy)]
pub struct ViewChangeArray {
    command: ViewChangeCommand,
    array: BoundedArray<HeaderPrepare, { constants::VIEW_HEADERS_MAX }>,
}

impl ViewChangeArray {
    /// Creates a new array containing only the root prepare header for `cluster`.
    ///
    /// The root header represents op 0 and serves as the initial state for a new view.
    /// Uses [`ViewChangeCommand::StartView`] as the command.
    #[inline]
    pub fn root(cluster: u128) -> Self {
        let root_header = HeaderPrepare::root(cluster);
        assert!(root_header.command == Command::Prepare);

        let mut array = BoundedArray::<HeaderPrepare, { constants::VIEW_HEADERS_MAX }>::new();
        assert!(array.len() <= constants::VIEW_HEADERS_MAX);
        array.push(root_header);
        assert!(array.len() == 1);

        let result = Self {
            command: ViewChangeCommand::StartView,
            array,
        };
        result.verify();
        result
    }

    /// Creates a new array from a slice of headers.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is empty or exceeds [`VIEW_HEADERS_MAX`](constants::VIEW_HEADERS_MAX).
    /// Panics if any header command is not [`Command::Prepare`].
    #[inline]
    pub fn init(command: ViewChangeCommand, slice: &[HeaderPrepare]) -> Self {
        assert!(!slice.is_empty());
        assert!(slice.len() <= constants::VIEW_HEADERS_MAX);
        assert!(slice.iter().all(|h| h.command == Command::Prepare));

        let mut array = BoundedArray::<HeaderPrepare, { constants::VIEW_HEADERS_MAX }>::new();
        array.extend_from_slice(slice);
        assert!(array.len() == slice.len());

        Self { command, array }
    }

    /// Appends a prepare header to the array.
    ///
    /// # Panics
    ///
    /// Panics if the array is full or `header.command` is not [`Command::Prepare`].
    #[inline]
    pub fn push(&mut self, header: HeaderPrepare) {
        assert!(self.array.len() < constants::VIEW_HEADERS_MAX);
        assert!(header.command == Command::Prepare);

        let len_before = self.array.len();
        self.array.push(header);

        assert!(self.array.len() == len_before + 1);
    }

    /// Returns the view change command type.
    #[inline]
    pub fn command(&self) -> ViewChangeCommand {
        self.command
    }

    /// Returns the number of headers in the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Returns `true` if the array contains no headers.
    ///
    /// Note: This should never return `true` for a valid `ViewChangeArray`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    /// Returns how many more headers can be added before reaching capacity.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        constants::VIEW_HEADERS_MAX - self.array.len()
    }

    /// Returns a borrowed [`ViewChangeSlice`] view of this array.
    ///
    /// # Panics
    ///
    /// Panics if the array is empty (indicates a bug, as this violates invariants).
    #[inline]
    pub fn as_slice(&self) -> ViewChangeSlice<'_> {
        assert!(!self.array.is_empty());

        ViewChangeSlice::init(self.command, self.array.const_slice())
    }

    /// Asserts all invariants hold.
    ///
    /// Call after receiving from an untrusted source or after complex mutations.
    ///
    /// # Panics
    ///
    /// Panics if any invariant is violated.
    #[inline]
    pub fn verify(&self) {
        assert!(!self.array.is_empty());
        assert!(self.array.len() <= constants::VIEW_HEADERS_MAX);
        self.as_slice().verify()
    }
}

impl std::fmt::Debug for ViewChangeArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewChangeArray")
            .field("command", &self.command)
            .field("len", &self.array.len())
            .field("capacity", &constants::VIEW_HEADERS_MAX)
            .finish()
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

    // ViewChangeArray Tests

    #[test]
    fn test_array_root() {
        let cluster = 0x1234;
        let array = ViewChangeArray::root(cluster);
        assert_eq!(array.command(), ViewChangeCommand::StartView);
        assert_eq!(array.len(), 1);
        assert!(!array.is_empty());
        assert_eq!(array.remaining_capacity(), constants::VIEW_HEADERS_MAX - 1);

        let root = &array.as_slice().slice()[0];
        assert_eq!(root.cluster, cluster);
        assert_eq!(root.operation, crate::vsr::wire::Operation::ROOT);
        assert_eq!(root.command, Command::Prepare);

        array.verify();
    }

    #[test]
    fn test_array_init_valid() {
        let headers = [header_prepare()];
        let array = ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
        assert_eq!(array.len(), 1);
        assert_eq!(array.command(), ViewChangeCommand::StartView);
        array.verify();
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_init_panic_invalid_command() {
        let mut invalid = header_prepare();
        invalid.command = Command::Ping;
        let headers = [invalid];
        ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_init_panic_empty() {
        let headers: [HeaderPrepare; 0] = [];
        ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_init_panic_oversized() {
        let headers = [header_prepare(); constants::VIEW_HEADERS_MAX + 1];
        ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
    }

    #[test]
    fn test_array_push() {
        let mut array = ViewChangeArray::root(0);
        let len_initial = array.len();

        let header = header_prepare();
        array.push(header);

        assert_eq!(array.len(), len_initial + 1);
        assert_eq!(
            array.as_slice().slice().last().unwrap().command,
            Command::Prepare
        );
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_push_panic_invalid_command() {
        let mut array = ViewChangeArray::root(0);
        let mut invalid = header_prepare();
        invalid.command = Command::Ping;
        array.push(invalid);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_push_panic_full() {
        let headers = [header_prepare(); constants::VIEW_HEADERS_MAX];
        let mut array = ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
        array.push(header_prepare());
    }

    #[test]
    fn test_array_as_slice() {
        let headers = [header_prepare(), header_prepare()];
        let array = ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
        let slice = array.as_slice();

        assert_eq!(slice.len(), 2);
        assert_eq!(slice.command(), ViewChangeCommand::StartView);
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

        // ViewChangeArray Properties

        /// Property: root() creates a valid, non-empty array with StartView command.
        #[test]
        fn prop_array_root_always_valid(cluster in any::<u128>()) {
            let array = ViewChangeArray::root(cluster);
            prop_assert_eq!(array.command(), ViewChangeCommand::StartView);
            prop_assert_eq!(array.len(), 1);
            prop_assert!(!array.is_empty());
            prop_assert!(array.remaining_capacity() < constants::VIEW_HEADERS_MAX);
            array.verify(); // Should not panic
        }

        /// Property: init() succeeds for valid Prepare headers and view commands.
        #[test]
        fn prop_array_init_valid(
            cmd in valid_view_change_command_strategy(),
            len in 1usize..=constants::VIEW_HEADERS_MAX,
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; len];

            let array = ViewChangeArray::init(cmd, &headers);
            prop_assert_eq!(array.command(), cmd);
            prop_assert_eq!(array.len(), len);
            array.verify();
        }

        /// Property: pushing a valid Prepare header increments length and maintains invariants.
        #[test]
        fn prop_array_push_valid(
            mut array in valid_view_change_array_strategy(),
        ) {
            if array.len() < constants::VIEW_HEADERS_MAX {
                let old_len = array.len();
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;

                array.push(h);
                prop_assert_eq!(array.len(), old_len + 1);
                array.verify();
            }
        }
    }

    fn valid_view_change_array_strategy() -> impl Strategy<Value = ViewChangeArray> {
        (
            valid_view_change_command_strategy(),
            1usize..=constants::VIEW_HEADERS_MAX,
        )
            .prop_map(|(cmd, len)| {
                let headers = vec![
                    {
                        let mut h = HeaderPrepare::new();
                        h.command = Command::Prepare;
                        h
                    };
                    len
                ];
                ViewChangeArray::init(cmd, &headers)
            })
    }
}
