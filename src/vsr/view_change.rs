//! View change message handling for the VSR protocol.
//!
//! During view changes, replicas exchange [`HeaderPrepare`] sequences to synchronize
//! their operation logs. This module provides [`ViewChangeSlice`] as a validated,
//! borrowed view into these header sequences.

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
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewChangeCommand {
    StartView,
    DoViewChange,
}

const _: () = {
    assert!(ViewChangeCommand::StartView as u8 == 0);
};

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
    #[inline]
    fn zeroed() -> Self {
        // SAFETY: ViewChangeCommand is repr(u8) with StartView == 0, and BoundedArray
        // can be all-zero. Zeroing keeps padding/unused slots initialized for hashing.
        unsafe { core::mem::zeroed() }
    }

    /// Creates a new array containing only the root prepare header for `cluster`.
    ///
    /// The root header represents op 0 and serves as the initial state for a new view.
    /// Uses [`ViewChangeCommand::StartView`] as the command.
    #[inline]
    pub fn root(cluster: u128) -> Self {
        let root_header = HeaderPrepare::root(cluster);
        assert!(root_header.command == Command::Prepare);

        let mut result = Self::zeroed();
        assert!(result.array.len() <= constants::VIEW_HEADERS_MAX);
        result.command = ViewChangeCommand::StartView;
        result.array.push(root_header);
        assert!(result.array.len() == 1);

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

        let mut result = Self::zeroed();
        result.command = command;
        result.array.extend_from_slice(slice);
        assert!(result.array.len() == slice.len());

        result
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

    /// Converts to a fixed-size array for wire-format storage.
    ///
    /// Returns a `[HeaderPrepare; VIEW_HEADERS_MAX]` with valid headers copied
    /// to the front and remaining slots zero-filled. Use [`Self::len()`] to
    /// determine how many entries are valid.
    ///
    /// This method discards the [`ViewChangeCommand`]; callers storing the result
    /// should preserve the command separately or reconstruct it from VSR state.
    #[inline]
    pub fn into_array(self) -> [HeaderPrepare; constants::VIEW_HEADERS_MAX] {
        let mut result = [HeaderPrepare::zeroed(); constants::VIEW_HEADERS_MAX];
        let slice = self.array.const_slice();
        result[..slice.len()].copy_from_slice(slice);
        result
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

    // --- ViewChangeSlice Tests ---

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_slice_init_empty_panics() {
        let headers: [HeaderPrepare; 0] = [];
        ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_slice_init_oversized_panics() {
        let headers = [header_prepare(); constants::VIEW_HEADERS_MAX + 1];
        ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_slice_verify_invalid_cmd_panics() {
        let mut invalid = header_prepare();
        invalid.command = Command::StartView; // Inner must be Prepare
        let headers = [invalid];

        ViewChangeSlice::init(ViewChangeCommand::StartView, &headers).verify();
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_slice_verify_mixed_cmds_panics() {
        let valid = header_prepare();
        let mut invalid = header_prepare();
        invalid.command = Command::Ping;
        let headers = [valid, invalid];

        ViewChangeSlice::init(ViewChangeCommand::StartView, &headers).verify();
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_slice_verify_empty_panics() {
        let headers: [HeaderPrepare; 0] = [];
        let slice = ViewChangeSlice {
            command: ViewChangeCommand::StartView,
            slice: &headers,
        };

        slice.verify();
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_slice_verify_oversized_panics() {
        let headers = vec![header_prepare(); constants::VIEW_HEADERS_MAX + 1];
        let slice = ViewChangeSlice {
            command: ViewChangeCommand::StartView,
            slice: &headers,
        };

        slice.verify();
    }

    // --- ViewChangeCommand Tests ---

    #[test]
    fn test_command_try_from() {
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
    fn test_command_try_from_exhaustiveness() {
        for cmd in Command::ALL {
            let is_view_change = matches!(cmd, Command::StartView | Command::DoViewChange);
            assert_eq!(ViewChangeCommand::try_from(cmd).is_ok(), is_view_change);
        }
    }

    #[test]
    fn test_command_into() {
        let cmd: Command = ViewChangeCommand::StartView.into();
        assert_eq!(cmd, Command::StartView);
        let cmd: Command = ViewChangeCommand::DoViewChange.into();
        assert_eq!(cmd, Command::DoViewChange);
    }

    // --- ViewChangeArray Tests ---

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_init_invalid_cmd_panics() {
        let mut invalid = header_prepare();
        invalid.command = Command::Ping;
        let headers = [invalid];
        ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_init_empty_panics() {
        let headers: [HeaderPrepare; 0] = [];
        ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_init_oversized_panics() {
        let headers = [header_prepare(); constants::VIEW_HEADERS_MAX + 1];
        ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_push_invalid_cmd_panics() {
        let mut array = ViewChangeArray::root(0);
        let mut invalid = header_prepare();
        invalid.command = Command::Ping;
        array.push(invalid);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_push_full_panics() {
        let headers = [header_prepare(); constants::VIEW_HEADERS_MAX];
        let mut array = ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
        array.push(header_prepare());
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_as_slice_panic_empty_direct() {
        let array = ViewChangeArray {
            command: ViewChangeCommand::StartView,
            array: BoundedArray::<HeaderPrepare, { constants::VIEW_HEADERS_MAX }>::new(),
        };

        let _ = array.as_slice();
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_verify_panic_empty_direct() {
        let array = ViewChangeArray {
            command: ViewChangeCommand::StartView,
            array: BoundedArray::<HeaderPrepare, { constants::VIEW_HEADERS_MAX }>::new(),
        };

        array.verify();
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_array_verify_panic_invalid_header_direct() {
        let mut invalid = header_prepare();
        invalid.command = Command::Ping;

        let mut inner = BoundedArray::<HeaderPrepare, { constants::VIEW_HEADERS_MAX }>::new();
        inner.push(invalid);

        let array = ViewChangeArray {
            command: ViewChangeCommand::StartView,
            array: inner,
        };

        array.verify();
    }

    #[test]
    fn test_array_as_slice_reflects_pushes() {
        let mut array = ViewChangeArray::root(0);

        assert_eq!(array.as_slice().len(), 1);

        array.push(header_prepare());
        assert_eq!(array.as_slice().len(), 2);
    }

    #[test]
    fn test_array_debug() {
        let headers = [header_prepare(); 5];
        let array = ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
        let debug_str = format!("{:?}", array);

        assert!(debug_str.contains("ViewChangeArray"));
        assert!(debug_str.contains("command"));
        assert!(debug_str.contains("len: 5"));
        assert!(debug_str.contains("capacity"));
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

    // --- ViewChangeArray::into_array Tests ---

    #[test]
    fn test_into_array_single_element() {
        let array = ViewChangeArray::root(42);
        assert_eq!(array.len(), 1);

        let raw = array.into_array();

        // First element should be the root header.
        assert_eq!(raw[0].cluster, 42);
        assert_eq!(raw[0].command, Command::Prepare);

        // Remaining elements should be zeroed.
        for header in &raw[1..] {
            assert_eq!(header.checksum, 0);
            assert_eq!(header.cluster, 0);
        }
    }

    #[test]
    fn test_into_array_multiple_elements() {
        let headers: Vec<HeaderPrepare> = (0..5)
            .map(|i| {
                let mut h = header_prepare();
                h.cluster = i as u128;
                h
            })
            .collect();

        let array = ViewChangeArray::init(ViewChangeCommand::StartView, &headers);
        assert_eq!(array.len(), 5);

        let raw = array.into_array();

        // First 5 elements should match input.
        for (i, header) in raw.iter().take(5).enumerate() {
            assert_eq!(header.cluster, i as u128);
        }

        // Remaining elements should be zeroed.
        for header in &raw[5..] {
            assert_eq!(header.checksum, 0);
        }
    }

    #[test]
    fn test_into_array_full_capacity() {
        let headers: Vec<HeaderPrepare> = (0..constants::VIEW_HEADERS_MAX)
            .map(|i| {
                let mut h = header_prepare();
                h.cluster = i as u128;
                h
            })
            .collect();

        let array = ViewChangeArray::init(ViewChangeCommand::DoViewChange, &headers);
        assert_eq!(array.len(), constants::VIEW_HEADERS_MAX);

        let raw = array.into_array();

        // All elements should match input.
        for (i, header) in raw.iter().enumerate() {
            assert_eq!(header.cluster, i as u128);
        }
    }

    #[test]
    fn test_into_array_preserves_header_data() {
        let mut h = header_prepare();
        h.cluster = 12345;
        h.view = 99;
        h.op = 1000;
        h.commit = 500;

        let array = ViewChangeArray::init(ViewChangeCommand::StartView, &[h]);
        let raw = array.into_array();

        assert_eq!(raw[0].cluster, 12345);
        assert_eq!(raw[0].view, 99);
        assert_eq!(raw[0].op, 1000);
        assert_eq!(raw[0].commit, 500);
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
        // --- ViewChangeSlice Properties ---

        /// Property: init succeeds for valid commands and non-empty slices within bounds.
        #[test]
        fn prop_slice_init_valid(
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
        fn prop_slice_verify_valid(
            cmd in valid_view_change_command_strategy(),
            len in 1usize..=constants::VIEW_HEADERS_MAX,
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; len];

            let slice = ViewChangeSlice::init(cmd, &headers);
            slice.verify();
        }

        /// Property: len() returns the same value as the input slice length.
        #[test]
        fn prop_slice_len(
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
        fn prop_slice_is_empty_false(
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
        fn prop_slice_as_slice_ref(
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
        fn prop_slice_copy(
            len in 1usize..=constants::VIEW_HEADERS_MAX,
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; len];

            let s1 = ViewChangeSlice::init(ViewChangeCommand::StartView, &headers);
            let s2 = s1; // Copy
            #[allow(clippy::clone_on_copy)]
            let s3 = s1.clone(); // Clone

            prop_assert_eq!(s1.command(), s2.command());
            prop_assert_eq!(s1.len(), s2.len());
            prop_assert_eq!(s1.command(), s3.command());
            prop_assert_eq!(s1.len(), s3.len());
        }

        /// Property: verify is idempotent.
        #[test]
        fn prop_slice_verify_idempotent(
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

        // --- ViewChangeArray Properties ---

        /// Property: root() creates a valid, non-empty array with StartView command.
        #[test]
        fn prop_array_root_always_valid(cluster in any::<u128>()) {
            let array = ViewChangeArray::root(cluster);
            prop_assert_eq!(array.command(), ViewChangeCommand::StartView);
            prop_assert_eq!(array.len(), 1);
            prop_assert!(!array.is_empty());
            prop_assert!(array.remaining_capacity() < constants::VIEW_HEADERS_MAX);
            array.verify();
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

        /// Property: pushing a valid Prepare header increments length.
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

        /// Property: remaining_capacity is always VIEW_HEADERS_MAX - len.
        #[test]
        fn prop_array_capacity(
            array in valid_view_change_array_strategy(),
        ) {
            prop_assert_eq!(
                array.remaining_capacity(),
                constants::VIEW_HEADERS_MAX - array.len()
            );
        }

        /// Property: as_slice returns a slice with matching command and length.
        #[test]
        fn prop_array_as_slice(
            array in valid_view_change_array_strategy(),
        ) {
            let slice = array.as_slice();
            prop_assert_eq!(slice.command(), array.command());
            prop_assert_eq!(slice.len(), array.len());
        }

        /// Property: is_empty is always false for arrays constructed via public API.
        #[test]
        fn prop_array_is_empty_false(
            array in valid_view_change_array_strategy(),
        ) {
            prop_assert!(!array.is_empty());
        }

        /// Property: Copy/Clone produces equivalent arrays.
                #[test]
                fn prop_array_copy(
                    array in valid_view_change_array_strategy(),
                ) {
                    let copied = array;
                    #[allow(clippy::clone_on_copy)]
                    let cloned = array.clone();

                    prop_assert_eq!(array.command(), copied.command());            prop_assert_eq!(array.len(), copied.len());
            prop_assert_eq!(array.remaining_capacity(), copied.remaining_capacity());

            prop_assert_eq!(array.command(), cloned.command());
            prop_assert_eq!(array.len(), cloned.len());
            prop_assert_eq!(array.remaining_capacity(), cloned.remaining_capacity());
        }

        /// Property: Can push exactly remaining_capacity headers.
        #[test]
        fn prop_array_push_to_capacity(
            initial_len in 1usize..constants::VIEW_HEADERS_MAX,
        ) {
            let headers = vec![{
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h
            }; initial_len];
            let mut array = ViewChangeArray::init(ViewChangeCommand::StartView, &headers);

            let to_push = array.remaining_capacity();
            for _ in 0..to_push {
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                array.push(h);
            }

            prop_assert_eq!(array.len(), constants::VIEW_HEADERS_MAX);
            prop_assert_eq!(array.remaining_capacity(), 0);
            array.verify();
        }

        /// Property: verify is idempotent for arrays.
        #[test]
        fn prop_array_verify_idempotent(
            array in valid_view_change_array_strategy(),
            call_count in 1usize..10,
        ) {
            for _ in 0..call_count {
                array.verify();
            }
        }

        /// Property: Debug output contains current length.
        #[test]
        fn prop_array_debug_contains_len(
            array in valid_view_change_array_strategy(),
        ) {
            let debug_str = format!("{:?}", array);
            let expected = format!("len: {}", array.len());
            prop_assert!(debug_str.contains(&expected));
        }

        /// Property: root cluster ID is preserved in the header.
        #[test]
        fn prop_array_root_preserves_cluster(cluster in any::<u128>()) {
            let array = ViewChangeArray::root(cluster);
            let root_header = &array.as_slice().slice()[0];
            prop_assert_eq!(root_header.cluster, cluster);
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
