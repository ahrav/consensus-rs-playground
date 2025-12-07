//! Wire-format message container for VSR protocol communication.
//!
//! A [`Message`] owns a 16-byte aligned buffer that holds a [`Header`] followed by an
//! optional body. Messages are designed for zero-copy network I/O—[`as_bytes`] returns
//! the exact bytes to send on the wire.
//!
//! # Lifecycle
//!
//! 1. Create with [`Message::new_zeroed`]
//! 2. Initialize header via [`reset`]
//! 3. Set body via [`set_body`] (computes checksums)
//! 4. Send via [`as_bytes`]
//!
//! [`as_bytes`]: Message::as_bytes
//! [`reset`]: Message::reset
//! [`set_body`]: Message::set_body

use super::{Command, Header};
use crate::constants::{
    HEADER_SIZE, HEADER_SIZE_USIZE, MESSAGE_BODY_SIZE_MAX, MESSAGE_SIZE_MAX, MESSAGE_SIZE_MAX_USIZE,
};
use core::cell::Cell;
use std::alloc::{Layout, alloc_zeroed, handle_alloc_error};

// Compile-time: verify alignment requirements
const _: () = assert!(align_of::<Header>() <= 16);
const _: () = assert!(align_of::<u128>() <= 16);

// Compile-time: verify size relationships
const _: () = assert!(HEADER_SIZE_USIZE <= MESSAGE_SIZE_MAX_USIZE);
const _: () = assert!(MESSAGE_BODY_SIZE_MAX == MESSAGE_SIZE_MAX - HEADER_SIZE);

#[repr(C, align(16))]
pub(crate) struct AlignedBuffer {
    pub(crate) bytes: [u8; MESSAGE_SIZE_MAX_USIZE],
}

const _: () = assert!(align_of::<AlignedBuffer>() >= align_of::<Header>());
const _: () = assert!(size_of::<AlignedBuffer>() == MESSAGE_SIZE_MAX_USIZE);

impl AlignedBuffer {
    fn new_zeroed() -> Box<Self> {
        let layout = Layout::new::<Self>();
        let ptr = unsafe { alloc_zeroed(layout) } as *mut Self;

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        // SAFETY: Memory was allocated with the layout of Self and zeroed.
        let buffer = unsafe { Box::from_raw(ptr) };

        assert!((buffer.bytes.as_ptr() as usize).is_multiple_of(align_of::<Header>()));
        buffer
    }
}

/// A protocol message containing a [`Header`] and optional body.
///
/// The internal buffer is 16-byte aligned to allow direct casting to [`Header`].
/// Use [`ref_acquire`]/[`ref_release`] for reference counting in message pools.
///
/// [`ref_acquire`]: Message::ref_acquire
/// [`ref_release`]: Message::ref_release
pub struct Message {
    /// Reference count for pool management. Not thread-safe.
    pub references: Cell<u32>,
    pub(crate) buffer: Box<AlignedBuffer>,
    /// Cached length of valid bytes (header + body).
    pub(crate) len: u32,
}

impl Message {
    /// Minimum message length (header only, no body).
    pub const LEN_MIN: u32 = HEADER_SIZE;
    /// Maximum message length (header + max body).
    pub const LEN_MAX: u32 = MESSAGE_SIZE_MAX;

    /// Creates a message with zeroed buffer. Call [`reset`](Self::reset) before use.
    pub fn new_zeroed() -> Self {
        let buffer = AlignedBuffer::new_zeroed();
        assert!((buffer.bytes.as_ptr() as usize).is_multiple_of(align_of::<Header>()));

        let msg = Message {
            references: Cell::new(0),
            buffer,
            len: HEADER_SIZE,
        };

        assert!(msg.len >= Self::LEN_MIN);
        assert!(msg.len <= Self::LEN_MAX);
        assert!(msg.references.get() == 0);

        msg
    }

    #[inline]
    pub fn len(&self) -> u32 {
        assert!(self.len >= Self::LEN_MIN);
        assert!(self.len <= Self::LEN_MAX);
        self.len
    }

    /// Returns `true` if the message has no body (header only).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == Self::LEN_MIN
    }

    #[inline]
    pub fn body_len(&self) -> u32 {
        assert!(self.len >= Self::LEN_MIN);
        assert!(self.len <= Self::LEN_MAX);
        self.len - Self::LEN_MIN
    }

    #[inline]
    pub fn header(&self) -> &Header {
        assert!(self.len >= Self::LEN_MIN);
        assert!((self.buffer.bytes.as_ptr() as usize).is_multiple_of(align_of::<Header>()));

        // SAFETY:
        // - Buffer is aligned to 16 bytes (AlignedBuffer)
        // - Buffer is at least HEADER_SIZE bytes
        // - Header is repr(C) with known layout
        // - Lifetime tied to &self
        unsafe { &*(self.buffer.bytes.as_ptr() as *const Header) }
    }

    #[inline]
    pub fn header_mut(&mut self) -> &mut Header {
        assert!(self.len >= Self::LEN_MIN);
        assert!((self.buffer.bytes.as_ptr() as usize).is_multiple_of(align_of::<Header>()));

        // SAFETY:
        // - Buffer is aligned to 16 bytes (AlignedBuffer)
        // - Buffer is at least HEADER_SIZE bytes
        // - Header is repr(C) with known layout
        // - Lifetime tied to &mut self
        unsafe { &mut *(self.buffer.bytes.as_mut_ptr() as *mut Header) }
    }

    #[inline]
    pub fn body(&self) -> &[u8] {
        let header = self.header();
        let total_len = self.len();

        assert!(total_len >= Self::LEN_MIN);
        assert!(total_len <= Self::LEN_MAX);
        assert!(total_len == self.len);

        let body = &self.buffer.bytes[HEADER_SIZE_USIZE..total_len as usize];
        assert!(body.len() as u32 == header.body_len());

        body
    }

    /// Returns a mutable slice of the body. Modifications invalidate the body checksum.
    #[inline]
    pub fn body_mut(&mut self) -> &mut [u8] {
        let total_len = self.header().total_len();

        assert!(total_len >= Self::LEN_MIN);
        assert!(total_len <= Self::LEN_MAX);
        assert!(total_len == self.len);

        let body_len = total_len - Self::LEN_MIN;
        let body = &mut self.buffer.bytes[HEADER_SIZE_USIZE..total_len as usize];
        assert!(body.len() as u32 == body_len);

        body
    }

    /// Initializes the header and clears any existing body.
    ///
    /// Must be called before [`body`](Self::body), [`body_mut`](Self::body_mut),
    /// or [`set_body`](Self::set_body).
    pub fn reset(&mut self, command: Command, cluster: u128, replica: u8) {
        let header = Header::new(command, cluster, replica);
        let header_bytes = header.as_bytes();

        assert!(header_bytes.len() == HEADER_SIZE_USIZE);

        self.buffer.bytes[..HEADER_SIZE_USIZE].copy_from_slice(header_bytes);
        self.len = Self::LEN_MIN;

        assert!(self.header().command == command);
        assert!(self.header().cluster == cluster);
        assert!(self.header().replica == replica);
        assert!(self.header().size == Self::LEN_MIN);
    }

    /// Copies `body` into the message and computes both checksums.
    ///
    /// Checksums are computed in the required order: body checksum first, then header.
    /// Returns `Err` if body exceeds [`MESSAGE_BODY_SIZE_MAX`].
    pub fn set_body(&mut self, body: &[u8]) -> Result<(), &'static str> {
        // Preconditions
        assert!(body.len() <= u32::MAX as usize);
        let body_len = body.len() as u32;

        if body_len > MESSAGE_BODY_SIZE_MAX {
            return Err("body too large");
        }

        let total_len = Self::LEN_MIN + body_len;

        // Verify bounds
        assert!(total_len >= Self::LEN_MIN);
        assert!(total_len <= Self::LEN_MAX);

        // Copy body into buffer
        self.buffer.bytes[HEADER_SIZE_USIZE..total_len as usize].copy_from_slice(body);
        self.len = total_len;

        // Update header
        {
            let h = self.header_mut();
            h.size = total_len;

            // set_checksum_body then set_checksum
            h.set_checksum_body(body);
            h.set_checksum();
        }

        // Postconditions
        assert!(self.len == total_len);
        assert!(self.header().size == total_len);
        assert!(self.header().body_len() == body_len);
        assert!(self.header().is_valid_checksum());
        assert!(self.header().is_valid_checksum_body(body));

        assert!(self.body() == body);

        Ok(())
    }

    /// Returns the wire-format bytes: header followed by body.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        assert!(self.len() >= Self::LEN_MIN);
        assert!(self.len() <= Self::LEN_MAX);

        let bytes = &self.buffer.bytes[..self.len as usize];
        assert!(bytes.len() as u32 == self.len);

        bytes
    }

    /// Increments reference count. Panics at `u32::MAX`.
    #[inline]
    pub fn ref_acquire(&self) {
        let old = self.references.get();

        assert!(old < u32::MAX);

        let new = old + 1;
        self.references.set(new);

        assert!(self.references.get() == new);
        assert!(self.references.get() == old + 1);
    }

    /// Decrements reference count. Returns `true` when count reaches zero. Panics on underflow.
    #[inline]
    pub fn ref_release(&self) -> bool {
        let old = self.references.get();

        assert!(old > 0, "reference count underflow");

        let new = old - 1;
        self.references.set(new);

        assert!(self.references.get() == new);
        assert!(self.references.get() == old - 1);

        new == 0
    }

    #[inline]
    pub fn ref_count(&self) -> u32 {
        self.references.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_alignment() {
        let msg = Message::new_zeroed();
        let ptr = msg.buffer.bytes.as_ptr() as usize;

        assert!(ptr.is_multiple_of(align_of::<Header>()));
        assert!(ptr.is_multiple_of(16));
    }

    #[test]
    fn new_zeroed_initial_state() {
        let msg = Message::new_zeroed();

        assert!(msg.len() == Message::LEN_MIN);
        assert!(msg.body_len() == 0);
        assert!(msg.ref_count() == 0);
    }

    #[test]
    fn reset_initializes_header() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Ping, 42, 3);

        assert!(msg.header().command == Command::Ping);
        assert!(msg.header().cluster == 42);
        assert!(msg.header().replica == 3);
        assert!(msg.len() == Message::LEN_MIN);
    }

    #[test]
    fn set_body_updates_message() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let body = b"hello world";
        assert!(msg.set_body(body).is_ok());

        assert!(msg.len() == Message::LEN_MIN + body.len() as u32);
        assert!(msg.body() == body);
        assert!(msg.header().is_valid_checksum());
        assert!(msg.header().is_valid_checksum_body(body));
    }

    #[test]
    fn set_body_rejects_oversized() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let big_body = vec![0u8; MESSAGE_BODY_SIZE_MAX as usize + 1];
        assert!(msg.set_body(&big_body).is_err());
    }

    #[test]
    fn reference_counting() {
        let msg = Message::new_zeroed();

        assert!(msg.ref_count() == 0);

        msg.ref_acquire();
        assert!(msg.ref_count() == 1);

        msg.ref_acquire();
        assert!(msg.ref_count() == 2);

        assert!(!msg.ref_release());
        assert!(msg.ref_count() == 1);

        assert!(msg.ref_release());
        assert!(msg.ref_count() == 0);
    }

    #[test]
    fn as_bytes_length() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Ping, 1, 0);

        // Header only
        assert!(msg.as_bytes().len() == HEADER_SIZE_USIZE);

        // With body
        let body = b"test body";
        msg.set_body(body).unwrap();
        assert!(msg.as_bytes().len() == HEADER_SIZE_USIZE + body.len());
    }

    #[test]
    fn body_roundtrip() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let original = b"some important data";
        msg.set_body(original).unwrap();

        assert!(msg.body() == original);
    }

    #[test]
    #[should_panic(expected = "SIZE_MIN")]
    fn body_panics_without_reset() {
        let msg = Message::new_zeroed();
        let _ = msg.body();
    }

    // =========================================================================
    // Boundary Tests
    // =========================================================================

    #[test]
    fn set_body_empty() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let empty: &[u8] = &[];
        assert!(msg.set_body(empty).is_ok());

        assert!(msg.len() == Message::LEN_MIN);
        assert!(msg.body_len() == 0);
        assert!(msg.body().is_empty());
        assert!(msg.header().is_valid_checksum());
        assert!(msg.header().is_valid_checksum_body(empty));
    }

    #[test]
    fn set_body_single_byte() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let body = &[0x42u8];
        assert!(msg.set_body(body).is_ok());

        assert!(msg.len() == Message::LEN_MIN + 1);
        assert!(msg.body_len() == 1);
        assert!(msg.body() == body);
    }

    #[test]
    fn set_body_max_size() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let max_body = vec![0xAAu8; MESSAGE_BODY_SIZE_MAX as usize];
        assert!(msg.set_body(&max_body).is_ok());

        // Length invariants
        assert!(msg.len() == Message::LEN_MAX);
        assert!(msg.body_len() == MESSAGE_BODY_SIZE_MAX);

        // Content preserved
        assert!(msg.body() == max_body.as_slice());

        // Checksums valid
        assert!(msg.header().is_valid_checksum());
        assert!(msg.header().is_valid_checksum_body(&max_body));

        // Passes basic validation
        assert!(
            msg.header().validate_basic().is_ok(),
            "validate_basic must pass at maximum body size"
        );
    }

    // =========================================================================
    // body_mut() Tests
    // =========================================================================

    #[test]
    fn body_mut_modification() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let initial = b"hello world";
        msg.set_body(initial).unwrap();

        // Modify via body_mut()
        let body = msg.body_mut();
        body[0] = b'H';
        body[6] = b'W';

        assert!(msg.body() == b"Hello World");
    }

    #[test]
    fn body_mut_invalidates_checksum() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let body = b"valid body";
        msg.set_body(body).unwrap();

        assert!(msg.header().is_valid_checksum_body(body));

        // Modify body directly
        msg.body_mut()[0] = b'X'; // "Xalid body"

        // Checksum is now invalid for the current body content
        assert!(!msg.header().is_valid_checksum_body(msg.body()));
    }

    #[test]
    fn body_mut_length() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let body_data = b"test data";
        msg.set_body(body_data).unwrap();

        assert!(msg.body_mut().len() == msg.body().len());
        assert!(msg.body_mut().len() == body_data.len());
    }

    #[test]
    #[should_panic(expected = "SIZE_MIN")]
    fn body_mut_panics_without_reset() {
        let mut msg = Message::new_zeroed();
        let _ = msg.body_mut();
    }

    // =========================================================================
    // Reference Counting Edge Cases
    // =========================================================================

    #[test]
    #[should_panic(expected = "reference count underflow")]
    fn ref_release_underflow_panics() {
        let msg = Message::new_zeroed();
        // ref_count is 0, release should panic
        msg.ref_release();
    }

    #[test]
    fn ref_count_many_cycles() {
        let msg = Message::new_zeroed();

        // Acquire many references
        for i in 1..=1000 {
            msg.ref_acquire();
            assert!(msg.ref_count() == i);
        }

        // Release all but one
        for i in (1..1000).rev() {
            assert!(!msg.ref_release());
            assert!(msg.ref_count() == i);
        }

        // Final release returns true
        assert!(msg.ref_release());
        assert!(msg.ref_count() == 0);
    }

    // =========================================================================
    // Sequential Operations Tests
    // =========================================================================

    #[test]
    fn multiple_reset_calls() {
        let mut msg = Message::new_zeroed();

        for &cmd in Command::ALL.iter() {
            msg.reset(cmd, 42, 3);
            assert!(msg.header().command == cmd);
            assert!(msg.header().cluster == 42);
            assert!(msg.header().replica == 3);
            assert!(msg.len() == Message::LEN_MIN);
        }
    }

    #[test]
    fn set_body_shrink_then_grow() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        // Set large body
        let large = vec![0xAAu8; 1000];
        msg.set_body(&large).unwrap();
        assert!(msg.body_len() == 1000);
        assert!(msg.body() == large.as_slice());

        // Shrink to small body
        let small = b"tiny";
        msg.set_body(small).unwrap();
        assert!(msg.body_len() == 4);
        assert!(msg.body() == small);

        // Grow again
        let medium = vec![0xBBu8; 500];
        msg.set_body(&medium).unwrap();
        assert!(msg.body_len() == 500);
        assert!(msg.body() == medium.as_slice());
    }

    #[test]
    fn as_bytes_content() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 42, 3);

        let body = b"payload data";
        msg.set_body(body).unwrap();

        let bytes = msg.as_bytes();

        // Verify header portion
        let header_bytes = &bytes[..HEADER_SIZE_USIZE];
        assert!(header_bytes == msg.header().as_bytes());

        // Verify body portion
        let body_bytes = &bytes[HEADER_SIZE_USIZE..];
        assert!(body_bytes == body);
    }

    #[test]
    fn reset_clears_previous_state() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 100, 5);

        let body = b"some data";
        msg.set_body(body).unwrap();
        assert!(msg.body_len() > 0);

        // Reset should clear body
        msg.reset(Command::Ping, 200, 10);
        assert!(msg.len() == Message::LEN_MIN);
        assert!(msg.body_len() == 0);
        assert!(msg.header().command == Command::Ping);
        assert!(msg.header().cluster == 200);
        assert!(msg.header().replica == 10);
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        fn arb_command() -> impl Strategy<Value = Command> {
            (0u8..=Command::MAX).prop_map(|b| Command::try_from_u8(b).unwrap())
        }

        fn arb_body() -> impl Strategy<Value = Vec<u8>> {
            // Use smaller bodies for faster property tests; boundary tests cover max size
            prop::collection::vec(any::<u8>(), 0..=1024)
        }

        proptest! {
            #[test]
            fn prop_body_roundtrip(body in arb_body()) {
                let mut msg = Message::new_zeroed();
                msg.reset(Command::Request, 1, 0);

                msg.set_body(&body).unwrap();

                prop_assert_eq!(msg.body(), body.as_slice());
                prop_assert_eq!(msg.body_len() as usize, body.len());
                prop_assert_eq!(msg.len(), Message::LEN_MIN + body.len() as u32);
            }

            #[test]
            fn prop_checksum_valid_after_set_body(body in arb_body()) {
                let mut msg = Message::new_zeroed();
                msg.reset(Command::Request, 1, 0);

                msg.set_body(&body).unwrap();

                prop_assert!(msg.header().is_valid_checksum());
                prop_assert!(msg.header().is_valid_checksum_body(&body));
            }

            #[test]
            fn prop_len_invariants(
                cmd in arb_command(),
                cluster in any::<u128>(),
                replica in any::<u8>(),
                body in arb_body(),
            ) {
                let mut msg = Message::new_zeroed();
                msg.reset(cmd, cluster, replica);
                msg.set_body(&body).unwrap();

                prop_assert!(msg.len() >= Message::LEN_MIN);
                prop_assert!(msg.len() <= Message::LEN_MAX);
                prop_assert_eq!(msg.len(), Message::LEN_MIN + msg.body_len());
                prop_assert_eq!(msg.body_len() as usize, body.len());
            }

            #[test]
            fn prop_as_bytes_length_consistent(body in arb_body()) {
                let mut msg = Message::new_zeroed();
                msg.reset(Command::Request, 1, 0);
                msg.set_body(&body).unwrap();

                prop_assert_eq!(msg.as_bytes().len() as u32, msg.len());
                prop_assert_eq!(msg.as_bytes().len(), HEADER_SIZE_USIZE + body.len());
            }

            #[test]
            fn prop_header_fields_preserved(
                cmd in arb_command(),
                cluster in any::<u128>(),
                replica in any::<u8>(),
            ) {
                let mut msg = Message::new_zeroed();
                msg.reset(cmd, cluster, replica);

                prop_assert_eq!(msg.header().command, cmd);
                prop_assert_eq!(msg.header().cluster, cluster);
                prop_assert_eq!(msg.header().replica, replica);
            }

            #[test]
            fn prop_body_mut_same_as_body(body in arb_body()) {
                let mut msg = Message::new_zeroed();
                msg.reset(Command::Request, 1, 0);
                msg.set_body(&body).unwrap();

                // body_mut() should return the same content as body()
                let body_ref = msg.body().to_vec();
                let body_mut_ref = msg.body_mut().to_vec();
                prop_assert_eq!(body_ref, body_mut_ref);
            }
        }
    }

    // =========================================================================
    // Invariant Preservation Tests
    // =========================================================================

    #[test]
    fn set_body_error_preserves_state() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        // Set initial valid body
        let initial_body = b"initial data";
        msg.set_body(initial_body).unwrap();

        // Capture state before error
        let len_before = msg.len();
        let header_size_before = msg.header().size;
        let body_before = msg.body().to_vec();
        let checksum_before = msg.header().checksum;
        let checksum_body_before = msg.header().checksum_body;

        // Attempt to set oversized body (should fail)
        let oversized = vec![0xFFu8; MESSAGE_BODY_SIZE_MAX as usize + 1];
        assert!(msg.set_body(&oversized).is_err());

        // Verify state is completely unchanged
        assert!(msg.len() == len_before, "len must be unchanged after error");
        assert!(
            msg.header().size == header_size_before,
            "header.size must be unchanged after error"
        );
        assert!(
            msg.body() == body_before.as_slice(),
            "body content must be unchanged after error"
        );
        assert!(
            msg.header().checksum == checksum_before,
            "checksum must be unchanged after error"
        );
        assert!(
            msg.header().checksum_body == checksum_body_before,
            "checksum_body must be unchanged after error"
        );
        assert!(
            msg.header().is_valid_checksum(),
            "checksum must still be valid after error"
        );
    }

    #[test]
    fn ref_acquire_succeeds_near_max() {
        let msg = Message::new_zeroed();

        // Set reference count near maximum
        msg.references.set(u32::MAX - 1);
        assert!(msg.ref_count() == u32::MAX - 1);

        // One more acquire should succeed
        msg.ref_acquire();
        assert!(msg.ref_count() == u32::MAX);
    }

    #[test]
    #[should_panic(expected = "u32::MAX")]
    fn ref_acquire_at_max_panics() {
        let msg = Message::new_zeroed();

        // Set reference count to maximum
        msg.references.set(u32::MAX);

        // This acquire must panic
        msg.ref_acquire();
    }

    #[test]
    fn validate_basic_after_reset() {
        let mut msg = Message::new_zeroed();

        for &cmd in Command::ALL.iter() {
            msg.reset(cmd, 12345, 7);

            // Header must pass basic validation after reset
            assert!(
                msg.header().validate_basic().is_ok(),
                "validate_basic must pass after reset with command {:?}",
                cmd
            );
        }
    }

    #[test]
    fn validate_basic_after_set_body() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 42, 3);

        let body_sizes = [0, 1, 100, 1024, 4096];

        for &size in &body_sizes {
            let body = vec![0xABu8; size];
            msg.set_body(&body).unwrap();

            // Header must pass basic validation after set_body
            assert!(
                msg.header().validate_basic().is_ok(),
                "validate_basic must pass after set_body with size {}",
                size
            );
        }
    }

    #[test]
    fn set_body_near_max_sizes() {
        // Test checksum at various large sizes near the maximum
        // (max size is covered by set_body_max_size)
        let large_sizes = [
            MESSAGE_BODY_SIZE_MAX as usize - 4096,
            MESSAGE_BODY_SIZE_MAX as usize - 1,
        ];

        for &size in &large_sizes {
            let mut msg = Message::new_zeroed();
            msg.reset(Command::Request, 1, 0);

            // Use non-zero pattern to ensure checksum is meaningful
            let body: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            msg.set_body(&body).unwrap();

            assert!(
                msg.header().is_valid_checksum(),
                "header checksum must be valid at size {}",
                size
            );
            assert!(
                msg.header().is_valid_checksum_body(&body),
                "body checksum must be valid at size {}",
                size
            );

            // Verify body content preserved
            assert!(
                msg.body() == body.as_slice(),
                "body content must match at size {}",
                size
            );
        }
    }

    #[test]
    fn is_empty() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Ping, 1, 0);

        // Header-only message is "empty" (no body)
        assert!(msg.is_empty());
        assert!(msg.body_len() == 0);

        // With body, no longer empty
        msg.set_body(b"data").unwrap();
        assert!(!msg.is_empty());
        assert!(msg.body_len() > 0);

        // Reset clears body, back to empty
        msg.reset(Command::Pong, 2, 1);
        assert!(msg.is_empty());
    }
}
