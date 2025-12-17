//! Wire protocol constants for VSR messages.
//!
//! # Design Decisions
//!
//! Size constants use `u32` instead of `usize` for portability and to prevent
//! truncation on 32-bit systems. Use `_USIZE` variants for array sizing.
//!
//! All invariants verified at compile time via `const` assertions.

// =============================================================================
// Platform verification
// =============================================================================

// Compile-time proof that u32 -> usize is safe on this platform.
const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

// =============================================================================
// Wire format constants
// =============================================================================

/// Message header size. Fixed at 256 bytes for command metadata, cluster ID,
/// checksums, and routing information. Must be 16-byte aligned.
pub const HEADER_SIZE: u32 = 256;

/// Maximum message size (header + body). Balances memory, network, and disk efficiency.
pub const MESSAGE_SIZE_MAX: u32 = 1 << 20; // 1 MiB

/// Maximum message body size. Derived as [`MESSAGE_SIZE_MAX`] - [`HEADER_SIZE`].
pub const MESSAGE_BODY_SIZE_MAX: u32 = MESSAGE_SIZE_MAX - HEADER_SIZE;

/// VSR protocol version. Increment for wire-incompatible changes.
pub const VSR_VERSION: u16 = 1;

/// Cluster identifier. 128 bits prevents collisions and enables random generation.
pub type ClusterId = u128;

// =============================================================================
// Compile-time design integrity assertions
// =============================================================================

// Header constraints
const _: () = assert!(HEADER_SIZE > 0);
const _: () = assert!(HEADER_SIZE == 256, "Header must be exactly 256 bytes");
const _: () = assert!(
    HEADER_SIZE.is_multiple_of(16),
    "Header must be 16-byte aligned for checksums"
);

// Message size constraints
const _: () = assert!(MESSAGE_SIZE_MAX > 0);
const _: () = assert!(
    MESSAGE_SIZE_MAX > HEADER_SIZE,
    "Message must have room for body"
);
const _: () = assert!(
    MESSAGE_SIZE_MAX.is_power_of_two(),
    "Message size should be power of two for alignment"
);

// Body size consistency
const _: () = assert!(MESSAGE_BODY_SIZE_MAX == MESSAGE_SIZE_MAX - HEADER_SIZE);
const _: () = assert!(MESSAGE_BODY_SIZE_MAX > 0);

// Relationship constraints
const _: () = assert!(HEADER_SIZE < MESSAGE_SIZE_MAX);

// Protocol version
const _: () = assert!(VSR_VERSION > 0);

// =============================================================================
// Pre-converted usize constants
// =============================================================================

/// [`HEADER_SIZE`] as `usize`.
pub const HEADER_SIZE_USIZE: usize = HEADER_SIZE as usize;

/// [`MESSAGE_SIZE_MAX`] as `usize`.
pub const MESSAGE_SIZE_MAX_USIZE: usize = MESSAGE_SIZE_MAX as usize;

/// [`MESSAGE_BODY_SIZE_MAX`] as `usize`.
pub const MESSAGE_BODY_SIZE_MAX_USIZE: usize = MESSAGE_BODY_SIZE_MAX as usize;

// Verify usize conversions match source constants.
const _: () = assert!(HEADER_SIZE_USIZE == HEADER_SIZE as usize);
const _: () = assert!(MESSAGE_SIZE_MAX_USIZE == MESSAGE_SIZE_MAX as usize);
const _: () = assert!(MESSAGE_BODY_SIZE_MAX_USIZE == MESSAGE_BODY_SIZE_MAX as usize);

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_size_relationships_hold() {
        assert_eq!(HEADER_SIZE + MESSAGE_BODY_SIZE_MAX, MESSAGE_SIZE_MAX);
    }

    #[test]
    fn constants_are_powers_of_two() {
        assert!(MESSAGE_SIZE_MAX.is_power_of_two());
    }

    #[test]
    fn usize_constants_match_originals() {
        assert_eq!(HEADER_SIZE_USIZE, HEADER_SIZE as usize);
        assert_eq!(MESSAGE_SIZE_MAX_USIZE, MESSAGE_SIZE_MAX as usize);
        assert_eq!(MESSAGE_BODY_SIZE_MAX_USIZE, MESSAGE_BODY_SIZE_MAX as usize);
    }
}
