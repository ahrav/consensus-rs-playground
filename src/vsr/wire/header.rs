//! VSR wire protocol header.
//!
//! The [`Header`] is a fixed-size 256-byte structure prepended to every message.
//! It contains checksums for integrity verification, routing metadata (cluster, replica),
//! and protocol versioning.
//!
//! # Wire layout
//!
//! ```text
//! Bytes 0-15:   checksum (covers bytes 16-255)
//! Bytes 16-31:  checksum_padding
//! Bytes 32-47:  checksum_body (covers message body)
//! Bytes 48-63:  checksum_body_padding
//! Bytes 64-79:  nonce_reserved
//! Bytes 80-95:  cluster (u128)
//! Bytes 96-99:  size (u32)
//! Bytes 100-103: epoch (u32)
//! Bytes 104-107: view (u32)
//! Bytes 108-111: release (u32)
//! Bytes 112-113: protocol (u16)
//! Byte 114:      command (u8)
//! Byte 115:      replica (u8)
//! Bytes 116-127: reserved_frame
//! Bytes 128-255: reserved_command
//! ```

use super::constants::{ClusterId, HEADER_SIZE, HEADER_SIZE_USIZE, MESSAGE_SIZE_MAX, VSR_VERSION};
use super::{Checksum128, Command, checksum};
use crate::util::{Pod, zero::is_all_zeros};

/// Byte offset where checksummed content begins (after the checksum field itself).
const CHECKSUM_SIZE: u32 = 16;
const _: () = assert!(CHECKSUM_SIZE as usize == size_of::<Checksum128>());
const _: () = assert!(HEADER_SIZE > CHECKSUM_SIZE);

/// Software release version for coordinating rolling upgrades.
///
/// Not currently used; reserved for future protocol evolution.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Release(pub u32);

// SAFETY: Release is repr(transparent) over u32, which is Pod.
// No padding bytes exist; all bytes are initialized for any valid value.
unsafe impl Pod for Release {}

impl Release {
    /// Zero release indicates no specific version requirement.
    pub const ZERO: Release = Release(0);

    #[inline]
    pub const fn value(self) -> u32 {
        self.0
    }
}

/// Fixed-size 256-byte header prepended to every VSR message.
///
/// # Checksum coverage
///
/// - `checksum`: Covers header bytes 16-255 (everything after itself)
/// - `checksum_body`: Covers the message body only (not the header)
///
/// Always set `checksum_body` before `checksum` since the header checksum
/// covers the `checksum_body` field.
///
/// # Reserved fields
///
/// Padding and reserved fields must be zero. This enables:
/// - Forward compatibility with future protocol versions
/// - Detection of corrupted or malformed messages
/// - Deterministic checksums across implementations
#[repr(C)]
pub struct Header {
    /// 128-bit checksum of header bytes 16-255.
    pub checksum: Checksum128,
    /// Reserved; must be zero.
    pub checksum_padding: Checksum128,
    /// 128-bit checksum of the message body.
    pub checksum_body: Checksum128,
    /// Reserved; must be zero.
    pub checksum_body_padding: Checksum128,

    /// Reserved for nonce; must be zero.
    pub nonce_reserved: Checksum128,
    /// Cluster identifier for routing.
    pub cluster: ClusterId,

    /// Total message size in bytes (header + body).
    pub size: u32,
    /// Epoch number for reconfiguration.
    pub epoch: u32,
    /// View number for leader election.
    pub view: u32,

    /// Software release version.
    pub release: Release,
    /// Protocol version; must equal [`VSR_VERSION`].
    pub protocol: u16,
    /// Message type discriminator.
    pub command: Command,
    /// Sender's replica index.
    pub replica: u8,

    /// Reserved for frame-level extensions; must be zero.
    pub reserved_frame: [u8; 12],
    /// Reserved for command-specific extensions; must be zero.
    pub reserved_command: [u8; 128],
}

const _: () = assert!(size_of::<Header>() == HEADER_SIZE_USIZE);

impl core::fmt::Debug for Header {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Header")
            .field("checksum", &format_args!("{:#x}", self.checksum))
            .field("checksum_body", &format_args!("{:#x}", self.checksum_body))
            .field("cluster", &self.cluster)
            .field("size", &self.size)
            .field("epoch", &self.epoch)
            .field("view", &self.view)
            .field("release", &self.release.value())
            .field("protocol", &self.protocol)
            .field("command", &self.command)
            .field("replica", &self.replica)
            .finish()
    }
}

impl Header {
    /// Minimum valid message size (header only, no body).
    pub const SIZE_MIN: u32 = HEADER_SIZE;
    /// Maximum valid message size (header + max body).
    pub const SIZE_MAX: u32 = MESSAGE_SIZE_MAX;

    /// Creates a header-only message with no body.
    ///
    /// Initializes `size` to [`Self::SIZE_MIN`], `protocol` to [`VSR_VERSION`],
    /// and all reserved/padding fields to zero.
    ///
    /// # Panics
    ///
    /// Panics if `command` exceeds [`Command::MAX`].
    pub fn new(command: Command, cluster: ClusterId, replica: u8) -> Self {
        assert!(command.as_u8() <= Command::MAX);

        let header = Header {
            checksum: Checksum128::default(),
            checksum_padding: Checksum128::default(),
            checksum_body: Checksum128::default(),
            checksum_body_padding: Checksum128::default(),
            nonce_reserved: Checksum128::default(),

            cluster,
            size: Self::SIZE_MIN,
            epoch: 0,
            view: 0,

            release: Release::ZERO,
            protocol: VSR_VERSION,
            command,
            replica,

            reserved_frame: [0u8; 12],
            reserved_command: [0u8; 128],
        };

        assert!(header.size >= Self::SIZE_MIN);
        assert!(header.size <= Self::SIZE_MAX);
        assert!(header.protocol == VSR_VERSION);
        assert!(is_all_zeros(&header.reserved_frame));
        assert!(is_all_zeros(&header.reserved_command));

        header
    }

    /// Returns the body length (`size - SIZE_MIN`).
    ///
    /// # Panics
    ///
    /// Panics if `size` is outside `[SIZE_MIN, SIZE_MAX]`.
    #[inline]
    pub fn body_len(&self) -> u32 {
        assert!(self.size >= Self::SIZE_MIN);
        assert!(self.size <= Self::SIZE_MAX);

        let body_len = self.size - Self::SIZE_MIN;
        assert!(body_len <= Self::SIZE_MAX - Self::SIZE_MIN);
        body_len
    }

    /// Returns total message length (equal to `size`).
    ///
    /// # Panics
    ///
    /// Panics if `size` is outside `[SIZE_MIN, SIZE_MAX]`.
    #[inline]
    pub fn total_len(&self) -> u32 {
        assert!(self.size >= Self::SIZE_MIN);
        assert!(self.size <= Self::SIZE_MAX);

        self.size
    }

    /// Reinterprets the header as a byte array.
    ///
    /// # Safety
    ///
    /// Safe because `Header` is `#[repr(C)]` with fixed 256-byte layout.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; HEADER_SIZE_USIZE] {
        // SAFETY: Header is repr(C) with compile-time size assertion.
        unsafe { &*(self as *const Self as *const [u8; HEADER_SIZE_USIZE]) }
    }

    /// Computes checksum over header bytes 16-255 (excludes `checksum` field).
    pub fn calculate_checksum(&self) -> Checksum128 {
        let bytes = self.as_bytes();
        let checksum_input = &bytes[CHECKSUM_SIZE as usize..];
        assert!(checksum_input.len() == HEADER_SIZE_USIZE - CHECKSUM_SIZE as usize);

        checksum(checksum_input)
    }

    /// Computes checksum over the message body.
    ///
    /// # Panics
    ///
    /// Panics if `body.len()` doesn't match [`Self::body_len()`].
    pub fn calculate_checksum_body(&self, body: &[u8]) -> Checksum128 {
        assert!(body.len() <= u32::MAX as usize);
        let body_len = body.len() as u32;

        assert!(self.size >= Self::SIZE_MIN);
        assert!(body_len == self.body_len());

        assert!(self.size == Self::SIZE_MIN + body_len);

        checksum(body)
    }

    /// Returns `true` if `checksum` matches computed value.
    #[inline]
    pub fn is_valid_checksum(&self) -> bool {
        self.checksum == self.calculate_checksum()
    }

    /// Returns `true` if `checksum_body` matches computed value.
    ///
    /// # Panics
    ///
    /// Panics if `body.len()` doesn't match [`Self::body_len()`].
    #[inline]
    pub fn is_valid_checksum_body(&self, body: &[u8]) -> bool {
        assert!(body.len() as u32 == self.body_len());
        self.checksum_body == self.calculate_checksum_body(body)
    }

    /// Computes and stores the header checksum.
    ///
    /// Call this *after* [`Self::set_checksum_body()`] since the header checksum
    /// covers the `checksum_body` field.
    pub fn set_checksum(&mut self) {
        let calculated_checksum = self.calculate_checksum();
        self.checksum = calculated_checksum;

        assert!(self.is_valid_checksum());
        assert!(self.checksum == calculated_checksum);
    }

    /// Computes and stores the body checksum.
    ///
    /// # Panics
    ///
    /// Panics if `body.len()` doesn't match [`Self::body_len()`].
    pub fn set_checksum_body(&mut self, body: &[u8]) {
        assert!(body.len() as u32 == self.body_len());

        self.checksum_body = self.calculate_checksum_body(body);

        assert!(self.is_valid_checksum_body(body));
    }

    /// Validates size bounds, protocol version, and reserved field invariants.
    ///
    /// Does *not* verify checksums; use [`Self::is_valid_checksum()`] separately.
    pub fn validate_basic(&self) -> Result<(), &'static str> {
        if self.size < Self::SIZE_MIN {
            return Err("size < SIZE_MIN");
        }
        if self.size > Self::SIZE_MAX {
            return Err("size > SIZE_MAX");
        }
        if self.protocol != VSR_VERSION {
            return Err("protocol mismatch");
        }
        if !is_all_zeros(&self.reserved_frame) {
            return Err("reserved_frame not zero");
        }
        if !is_all_zeros(&self.reserved_command) {
            return Err("reserved_command not zero");
        }

        assert!(self.size >= Self::SIZE_MIN);
        assert!(self.size <= Self::SIZE_MAX);

        Ok(())
    }

    /// Deserializes a header from raw bytes.
    ///
    /// The resulting header may contain invalid data; call [`Self::validate_basic()`]
    /// and [`Self::is_valid_checksum()`] before trusting the contents.
    ///
    /// # Safety
    ///
    /// Safe because `Header` is `#[repr(C)]` and all bit patterns are valid
    /// (no padding bytes with invariants, no enums with restricted values).
    pub fn from_bytes(bytes: &[u8; HEADER_SIZE_USIZE]) -> Self {
        let header = {
            let mut h = core::mem::MaybeUninit::<Header>::uninit();

            // SAFETY: Header is repr(C), properly sized, and all bit patterns
            // are valid for the field types.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    h.as_mut_ptr() as *mut u8,
                    HEADER_SIZE_USIZE,
                );
                h.assume_init()
            }
        };

        assert!(header.as_bytes() == bytes);

        header
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type FieldCorruptor = (&'static str, fn(&mut Header));

    #[test]
    fn new_header_defaults() {
        let h = Header::new(Command::Ping, 1, 0);

        assert!(h.command == Command::Ping);
        assert!(h.cluster == 1);
        assert!(h.replica == 0);
        assert!(h.size == Header::SIZE_MIN);
        assert!(h.body_len() == 0);
        assert!(h.protocol == VSR_VERSION);
        assert!(is_all_zeros(&h.reserved_frame));
        assert!(is_all_zeros(&h.reserved_command));
    }

    #[test]
    fn body_len_consistency() {
        let mut h = Header::new(Command::Request, 1, 0);

        assert!(h.body_len() == 0);
        assert!(h.total_len() == Header::SIZE_MIN);

        h.size = Header::SIZE_MIN + 100;
        assert!(h.body_len() == 100);
        assert!(h.total_len() == Header::SIZE_MIN + 100);
    }

    #[test]
    fn checksum_roundtrip() {
        let mut h = Header::new(Command::Ping, 1, 0);
        let body: [u8; 0] = [];

        h.set_checksum_body(&body);
        h.set_checksum();

        assert!(h.is_valid_checksum());
        assert!(h.is_valid_checksum_body(&body));
    }

    #[test]
    fn checksum_roundtrip_with_body() {
        let mut h = Header::new(Command::Request, 2, 1);
        let body = [1u8, 2, 3, 4, 5, 6, 7, 8];

        h.size = Header::SIZE_MIN + body.len() as u32;

        h.set_checksum_body(&body);
        h.set_checksum();

        assert!(h.body_len() == body.len() as u32);
        assert!(h.is_valid_checksum());
        assert!(h.is_valid_checksum_body(&body));
    }

    #[test]
    fn checksum_body_detects_corruption() {
        let mut h = Header::new(Command::Request, 2, 1);
        let mut body = vec![0xA5u8; 16];

        h.size = Header::SIZE_MIN + body.len() as u32;

        h.set_checksum_body(&body);
        h.set_checksum();

        assert!(h.is_valid_checksum_body(&body));

        body[0] ^= 0xFF;
        assert!(!h.is_valid_checksum_body(&body));
    }

    #[test]
    fn validate_basic_checks_protocol() {
        let mut h = Header::new(Command::Ping, 1, 0);

        assert!(h.validate_basic().is_ok());

        h.protocol = VSR_VERSION + 1;
        assert!(h.validate_basic().is_err());
    }

    #[test]
    fn validate_basic_checks_reserved() {
        let mut h = Header::new(Command::Ping, 1, 0);

        h.reserved_frame[0] = 1;
        assert!(h.validate_basic().is_err());
    }

    #[test]
    fn validate_basic_checks_reserved_command() {
        let mut h = Header::new(Command::Ping, 1, 0);

        h.reserved_command[0] = 1;
        assert!(h.validate_basic().is_err());
    }

    #[test]
    fn fuzz_from_bytes_never_panics() {
        use std::panic::catch_unwind;

        let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let next_rand = |state: &mut u64| -> u64 {
            // Simple xorshift64
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };

        for _ in 0..10_000 {
            let mut bytes = [0u8; HEADER_SIZE_USIZE];
            for chunk in bytes.chunks_mut(8) {
                let rand = next_rand(&mut rng_state);
                let rand_bytes = rand.to_le_bytes();
                let len = chunk.len().min(8);
                chunk[..len].copy_from_slice(&rand_bytes[..len]);
            }

            let result = catch_unwind(|| {
                let h = Header::from_bytes(&bytes);
                // Exercise methods that might panic on invalid state
                let _ = h.as_bytes();
                let _ = h.calculate_checksum();
                let _ = h.is_valid_checksum();
                let _ = h.validate_basic();
                // Note: body_len() and total_len() assert on size bounds,
                // which is intentional for invalid headers
            });

            assert!(
                result.is_ok(),
                "Header::from_bytes must handle arbitrary bytes without panicking"
            );
        }
    }

    #[test]
    fn checksum_body_with_various_sizes() {
        use crate::vsr::wire::constants::MESSAGE_BODY_SIZE_MAX;

        let body_sizes = [1, 100, 1024, 4096, MESSAGE_BODY_SIZE_MAX as usize];

        for &body_size in &body_sizes {
            let mut h = Header::new(Command::Request, 1, 0);
            h.size = Header::SIZE_MIN + body_size as u32;

            let body = vec![0x42u8; body_size];
            h.set_checksum_body(&body);
            h.set_checksum();

            assert!(
                h.is_valid_checksum_body(&body),
                "Checksum validation must work for body size {}",
                body_size
            );
            assert!(h.is_valid_checksum());
        }
    }

    #[test]
    fn all_commands_roundtrip() {
        for &cmd in Command::ALL.iter() {
            let original = Header::new(cmd, 42, 3);
            let bytes = original.as_bytes();
            let restored = Header::from_bytes(bytes);

            assert_eq!(
                restored.command, cmd,
                "Command {:?} must survive serialization round-trip",
                cmd
            );
            assert_eq!(restored.cluster, 42);
            assert_eq!(restored.replica, 3);
            assert_eq!(restored.as_bytes(), original.as_bytes());
        }
    }

    #[test]
    fn size_boundaries_comprehensive() {
        // Test cases: (size, should_error, label)
        let test_cases = [
            (0u32, true, "zero size"),
            (1, true, "1 byte"),
            (Header::SIZE_MIN - 1, true, "MIN - 1"),
            (Header::SIZE_MIN, false, "MIN (valid)"),
            (Header::SIZE_MIN + 1, false, "MIN + 1"),
            (Header::SIZE_MAX - 1, false, "MAX - 1"),
            (Header::SIZE_MAX, false, "MAX (valid)"),
            (Header::SIZE_MAX + 1, true, "MAX + 1"),
            (u32::MAX, true, "u32::MAX"),
        ];

        for (size, should_error, label) in test_cases {
            let mut h = Header::new(Command::Ping, 1, 0);
            h.size = size;

            let result = h.validate_basic();
            assert_eq!(
                result.is_err(),
                should_error,
                "Size {} ({}) validation mismatch: expected err={}, got {:?}",
                size,
                label,
                should_error,
                result
            );
        }
    }

    #[test]
    fn checksum_detects_all_field_corruption() {
        // Test that checksum protects all fields after the checksum itself
        let field_corruptors: [FieldCorruptor; 9] = [
            ("cluster", |h| h.cluster ^= 1),
            ("size", |h| h.size ^= 1),
            ("epoch", |h| h.epoch ^= 1),
            ("view", |h| h.view ^= 1),
            ("release", |h| h.release.0 ^= 1),
            ("protocol", |h| h.protocol ^= 1),
            ("replica", |h| h.replica ^= 1),
            ("reserved_frame", |h| h.reserved_frame[0] ^= 1),
            ("reserved_command", |h| h.reserved_command[0] ^= 1),
        ];

        for (field_name, corrupt_fn) in field_corruptors {
            let mut h = Header::new(Command::Commit, 1, 0);
            h.set_checksum();
            assert!(
                h.is_valid_checksum(),
                "Precondition: checksum must be valid"
            );

            corrupt_fn(&mut h);

            assert!(
                !h.is_valid_checksum(),
                "Checksum must detect corruption of field: {}",
                field_name
            );
        }
    }

    #[test]
    fn checksum_padding_fields_are_covered() {
        let mut h = Header::new(Command::Prepare, 1, 0);
        h.set_checksum();

        // Mutate checksum_padding - should invalidate checksum
        h.checksum_padding = h.checksum_padding.wrapping_add(1);
        assert!(
            !h.is_valid_checksum(),
            "checksum_padding must be protected by checksum"
        );

        // Reset and test checksum_body_padding
        let mut h = Header::new(Command::Prepare, 1, 0);
        h.set_checksum();
        h.checksum_body_padding = h.checksum_body_padding.wrapping_add(1);
        assert!(
            !h.is_valid_checksum(),
            "checksum_body_padding must be protected by checksum"
        );

        // Reset and test nonce_reserved
        let mut h = Header::new(Command::Prepare, 1, 0);
        h.set_checksum();
        h.nonce_reserved = h.nonce_reserved.wrapping_add(1);
        assert!(
            !h.is_valid_checksum(),
            "nonce_reserved must be protected by checksum"
        );
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

        fn arb_header() -> impl Strategy<Value = Header> {
            (
                arb_command(),
                any::<u128>(),                       // cluster
                any::<u8>(),                         // replica
                any::<u32>(),                        // epoch
                any::<u32>(),                        // view
                Header::SIZE_MIN..=Header::SIZE_MAX, // size (valid range only)
            )
                .prop_map(|(command, cluster, replica, epoch, view, size)| {
                    let mut h = Header::new(command, cluster, replica);
                    h.epoch = epoch;
                    h.view = view;
                    h.size = size;
                    h
                })
        }

        proptest! {
            #[test]
            fn prop_serialization_roundtrip(h in arb_header()) {
                let bytes = h.as_bytes();
                let restored = Header::from_bytes(bytes);

                prop_assert_eq!(restored.command, h.command);
                prop_assert_eq!(restored.cluster, h.cluster);
                prop_assert_eq!(restored.replica, h.replica);
                prop_assert_eq!(restored.epoch, h.epoch);
                prop_assert_eq!(restored.view, h.view);
                prop_assert_eq!(restored.size, h.size);
                prop_assert_eq!(restored.as_bytes(), h.as_bytes());
            }

            #[test]
            fn prop_body_len_invariant(h in arb_header()) {
                let body_len = h.body_len();
                let expected = h.size - Header::SIZE_MIN;
                prop_assert_eq!(body_len, expected);
                prop_assert!(body_len <= Header::SIZE_MAX - Header::SIZE_MIN);
            }

            #[test]
            fn prop_checksum_deterministic(h in arb_header()) {
                let c1 = h.calculate_checksum();
                let c2 = h.calculate_checksum();
                prop_assert_eq!(c1, c2, "Checksum must be deterministic");
            }

            #[test]
            fn prop_validate_basic_accepts_valid_headers(h in arb_header()) {
                // All headers from arb_header() have valid size range
                prop_assert!(h.validate_basic().is_ok());
            }

            #[test]
            fn prop_set_checksum_makes_valid(mut h in arb_header()) {
                h.set_checksum();
                prop_assert!(h.is_valid_checksum());
            }

            #[test]
            fn prop_total_len_equals_size(h in arb_header()) {
                prop_assert_eq!(h.total_len(), h.size);
            }
        }
    }

    // =========================================================================
    // Completeness Tests
    // =========================================================================

    #[test]
    fn header_memory_layout_stable() {
        use std::mem::{align_of, size_of};

        // Size must be exactly HEADER_SIZE
        assert_eq!(size_of::<Header>(), HEADER_SIZE_USIZE);

        // Alignment should be u128 for efficient network I/O
        assert_eq!(align_of::<Header>(), align_of::<u128>());

        // repr(transparent) for Release
        assert_eq!(size_of::<Release>(), size_of::<u32>());
        assert_eq!(align_of::<Release>(), align_of::<u32>());
    }

    #[test]
    fn header_field_offsets_stable() {
        // Verify that the checksum field is at offset 0
        // This is critical for the checksum calculation which skips the first 16 bytes
        let h = Header::new(Command::Ping, 0, 0);
        let h_ptr = &h as *const Header as usize;
        let checksum_ptr = &h.checksum as *const Checksum128 as usize;

        assert_eq!(
            checksum_ptr - h_ptr,
            0,
            "checksum must be at offset 0 for wire protocol"
        );
    }

    #[test]
    fn protocol_version_always_current() {
        for &cmd in Command::ALL.iter() {
            let h = Header::new(cmd, 1, 0);
            assert_eq!(
                h.protocol, VSR_VERSION,
                "Header::new must use VSR_VERSION for {:?}",
                cmd
            );
        }
    }

    #[test]
    fn cluster_id_edge_values() {
        let edge_clusters: [u128; 4] = [0, 1, u64::MAX as u128, u128::MAX];

        for cluster in edge_clusters {
            let h = Header::new(Command::Ping, cluster, 0);
            assert_eq!(h.cluster, cluster);

            // Round-trip
            let bytes = h.as_bytes();
            let restored = Header::from_bytes(bytes);
            assert_eq!(
                restored.cluster, cluster,
                "Cluster ID {} must survive round-trip",
                cluster
            );
        }
    }

    #[test]
    fn replica_id_all_values() {
        for replica in 0..=255u8 {
            let h = Header::new(Command::Pong, 1, replica);
            assert_eq!(h.replica, replica);

            let restored = Header::from_bytes(h.as_bytes());
            assert_eq!(restored.replica, replica);
        }
    }

    #[test]
    fn nonce_reserved_initialized_to_zero() {
        let h = Header::new(Command::Prepare, 1, 0);
        assert_eq!(
            h.nonce_reserved, 0,
            "nonce_reserved must be zero-initialized"
        );
    }

    #[test]
    fn epoch_view_initialization() {
        let h = Header::new(Command::StartView, 1, 0);
        assert_eq!(h.epoch, 0, "Initial epoch should be 0");
        assert_eq!(h.view, 0, "Initial view should be 0");
    }

    #[test]
    fn epoch_view_edge_values() {
        let values = [0u32, 1, u32::MAX];

        for &epoch in &values {
            for &view in &values {
                let mut h = Header::new(Command::Commit, 1, 0);
                h.epoch = epoch;
                h.view = view;

                let restored = Header::from_bytes(h.as_bytes());
                assert_eq!(restored.epoch, epoch);
                assert_eq!(restored.view, view);
            }
        }
    }

    #[test]
    fn release_zero_constant() {
        assert_eq!(Release::ZERO.value(), 0);
    }

    #[test]
    fn release_round_trip() {
        let releases = [Release(0), Release(1), Release(u32::MAX)];

        for release in releases {
            let mut h = Header::new(Command::Ping, 1, 0);
            h.release = release;

            let restored = Header::from_bytes(h.as_bytes());
            assert_eq!(restored.release, release);
        }
    }

    #[test]
    fn full_message_workflow() {
        // Create header
        let mut header = Header::new(Command::Request, 42, 3);

        // Prepare body
        let body = b"client_request_payload_data";
        header.size = Header::SIZE_MIN + body.len() as u32;

        // Set checksums
        header.set_checksum_body(body);
        header.set_checksum();

        // Validate
        assert!(header.validate_basic().is_ok());
        assert!(header.is_valid_checksum());
        assert!(header.is_valid_checksum_body(body));

        // Serialize complete message
        let mut message = Vec::new();
        message.extend_from_slice(header.as_bytes());
        message.extend_from_slice(body);

        // Deserialize
        let header_bytes: [u8; HEADER_SIZE_USIZE] =
            message[..HEADER_SIZE_USIZE].try_into().unwrap();
        let parsed_header = Header::from_bytes(&header_bytes);
        let parsed_body = &message[HEADER_SIZE_USIZE..];

        // Verify
        assert!(parsed_header.is_valid_checksum());
        assert!(parsed_header.is_valid_checksum_body(parsed_body));
        assert_eq!(parsed_header.command, Command::Request);
        assert_eq!(parsed_header.cluster, 42);
        assert_eq!(parsed_header.replica, 3);
        assert_eq!(parsed_body, body);
    }

    #[test]
    fn debug_format_includes_key_fields() {
        let h = Header::new(Command::Request, 42, 3);
        let debug_str = format!("{:?}", h);

        assert!(debug_str.contains("Request"), "Debug should show command");
        assert!(
            debug_str.contains("cluster"),
            "Debug should show cluster field"
        );
        assert!(debug_str.contains("42"), "Debug should show cluster value");
        assert!(
            debug_str.contains("replica"),
            "Debug should show replica field"
        );
        assert!(debug_str.contains("3"), "Debug should show replica value");
    }
}
