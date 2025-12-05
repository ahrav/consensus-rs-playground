//! Protocol and system constants for VSR consensus.
//!
//! # Design Decisions
//!
//! Size constants use `u32` instead of `usize` for portability and to prevent
//! truncation on 32-bit systems. Use [`as_usize`] or `_USIZE` variants for arrays.
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

/// Sector size for disk I/O alignment. Must be power of two for bitwise alignment.
pub const SECTOR_SIZE: u32 = 4096;

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

// Sector constraints
const _: () = assert!(SECTOR_SIZE > 0);
const _: () = assert!(
    SECTOR_SIZE.is_power_of_two(),
    "Sector size must be power of two for alignment arithmetic"
);
const _: () = assert!(
    SECTOR_SIZE >= HEADER_SIZE,
    "Sector must fit at least one header"
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
const _: () = assert!(
    MESSAGE_SIZE_MAX >= SECTOR_SIZE,
    "Message must be at least one sector"
);

// Body size consistency
const _: () = assert!(MESSAGE_BODY_SIZE_MAX == MESSAGE_SIZE_MAX - HEADER_SIZE);
const _: () = assert!(MESSAGE_BODY_SIZE_MAX > 0);

// Relationship constraints
const _: () = assert!(HEADER_SIZE < MESSAGE_SIZE_MAX);
const _: () = assert!(HEADER_SIZE <= SECTOR_SIZE);

// Protocol version
const _: () = assert!(VSR_VERSION > 0);

// =============================================================================
// Helper functions
// =============================================================================

/// Rounds up to next [`SECTOR_SIZE`] multiple. Idempotent for aligned inputs.
///
/// # Panics
/// Panics on overflow.
///
/// # Examples
/// ```
/// # use consensus::constants::*;
/// assert_eq!(sector_ceil(0), 0);
/// assert_eq!(sector_ceil(1), SECTOR_SIZE);
/// assert_eq!(sector_ceil(SECTOR_SIZE), SECTOR_SIZE);
/// assert_eq!(sector_ceil(SECTOR_SIZE + 1), SECTOR_SIZE * 2);
/// ```
#[inline]
pub const fn sector_ceil(n: u32) -> u32 {
    const _: () = assert!(SECTOR_SIZE.is_power_of_two());

    let mask = SECTOR_SIZE - 1;

    assert!(n <= u32::MAX - mask, "sector_ceil overflow");

    let result = (n + mask) & !mask;
    assert!(
        result.is_multiple_of(SECTOR_SIZE),
        "sector_ceil produced unaligned result"
    );

    result
}

/// Convert protocol constant from `u32` to `usize` for array sizing.
///
/// # Examples
/// ```
/// # use consensus::constants::*;
/// let buffer = [0u8; as_usize(MESSAGE_SIZE_MAX)];
/// ```
#[inline(always)]
pub const fn as_usize(n: u32) -> usize {
    n as usize
}

// =============================================================================
// Pre-converted usize constants
// =============================================================================

/// [`HEADER_SIZE`] as `usize`.
pub const HEADER_SIZE_USIZE: usize = HEADER_SIZE as usize;

/// [`SECTOR_SIZE`] as `usize`.
pub const SECTOR_SIZE_USIZE: usize = SECTOR_SIZE as usize;

/// [`MESSAGE_SIZE_MAX`] as `usize`.
pub const MESSAGE_SIZE_MAX_USIZE: usize = MESSAGE_SIZE_MAX as usize;

/// [`MESSAGE_BODY_SIZE_MAX`] as `usize`.
pub const MESSAGE_BODY_SIZE_MAX_USIZE: usize = MESSAGE_BODY_SIZE_MAX as usize;

// Verify usize conversions match source constants.
const _: () = assert!(HEADER_SIZE_USIZE == HEADER_SIZE as usize);
const _: () = assert!(SECTOR_SIZE_USIZE == SECTOR_SIZE as usize);
const _: () = assert!(MESSAGE_SIZE_MAX_USIZE == MESSAGE_SIZE_MAX as usize);
const _: () = assert!(MESSAGE_BODY_SIZE_MAX_USIZE == MESSAGE_BODY_SIZE_MAX as usize);

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sector_ceil_rounds_zero_to_zero() {
        assert_eq!(sector_ceil(0), 0);
    }

    #[test]
    fn sector_ceil_rounds_non_aligned_up() {
        assert_eq!(sector_ceil(1), SECTOR_SIZE);
        assert_eq!(sector_ceil(SECTOR_SIZE - 1), SECTOR_SIZE);
        assert_eq!(sector_ceil(SECTOR_SIZE + 1), SECTOR_SIZE * 2);
    }

    #[test]
    fn sector_ceil_identity_for_aligned() {
        assert_eq!(sector_ceil(SECTOR_SIZE), SECTOR_SIZE);
        assert_eq!(sector_ceil(SECTOR_SIZE * 2), SECTOR_SIZE * 2);
        assert_eq!(sector_ceil(SECTOR_SIZE * 100), SECTOR_SIZE * 100);
    }

    #[test]
    fn header_fits_in_sector() {
        assert_eq!(sector_ceil(HEADER_SIZE), SECTOR_SIZE);
    }

    #[test]
    fn message_size_relationships_hold() {
        assert_eq!(HEADER_SIZE + MESSAGE_BODY_SIZE_MAX, MESSAGE_SIZE_MAX);
    }

    #[test]
    fn constants_are_powers_of_two() {
        assert!(SECTOR_SIZE.is_power_of_two());
        assert!(MESSAGE_SIZE_MAX.is_power_of_two());
    }

    #[test]
    fn usize_constants_match_originals() {
        assert_eq!(HEADER_SIZE_USIZE, HEADER_SIZE as usize);
        assert_eq!(SECTOR_SIZE_USIZE, SECTOR_SIZE as usize);
        assert_eq!(MESSAGE_SIZE_MAX_USIZE, MESSAGE_SIZE_MAX as usize);
        assert_eq!(MESSAGE_BODY_SIZE_MAX_USIZE, MESSAGE_BODY_SIZE_MAX as usize);
    }
}
