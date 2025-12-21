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

/// Alias for commands valid in view change context.
///
/// Only [`Command::StartView`] and [`Command::DoViewChange`] are permitted.
pub type ViewChangeCommand = Command;

/// Returns `true` if `cmd` is valid for view change messages.
#[inline]
const fn is_valid_view_change_command(cmd: Command) -> bool {
    matches!(cmd, Command::StartView | Command::DoViewChange)
}

/// A validated, borrowed slice of [`HeaderPrepare`] entries for view change messages.
///
/// # Invariants
///
/// Maintained by construction ([`init`](Self::init)) and verified by [`verify`](Self::verify):
/// - `command` is [`Command::StartView`] or [`Command::DoViewChange`]
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
    /// Panics if `command` is not a valid view change command, or if `slice` is empty
    /// or exceeds [`VIEW_HEADERS_MAX`](constants::VIEW_HEADERS_MAX).
    #[inline]
    pub fn init(command: ViewChangeCommand, slice: &'a [HeaderPrepare]) -> Self {
        assert!(is_valid_view_change_command(command));
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
        assert!(is_valid_view_change_command(self.command()));
        assert!(!self.slice.is_empty());
        assert!(self.slice.len() <= constants::VIEW_HEADERS_MAX);

        assert!(
            self.slice()
                .iter()
                .all(|header| header.command == Command::Prepare)
        )
    }
}
