//! System constants (currently unused, retained for future use).
//!
//! # Design Decisions
//!
//! Size constants use `u32` instead of `usize` for portability and to prevent
//! truncation on 32-bit systems. Use [`as_usize`] or `_USIZE` variants for arrays.
//!
//! All invariants verified at compile time via `const` assertions.
//!
//! # Note
//!
//! VSR wire-format constants have been moved to [`crate::vsr::wire::constants`].

// =============================================================================
// Platform verification
// =============================================================================

// Compile-time proof that u32 -> usize is safe on this platform.
const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

// =============================================================================
// Disk I/O constants
// =============================================================================

/// Sector size for disk I/O alignment. Must be power of two for bitwise alignment.
pub const SECTOR_SIZE: u32 = 4096;

// =============================================================================
// Compile-time design integrity assertions
// =============================================================================

// Sector constraints
const _: () = assert!(SECTOR_SIZE > 0);
const _: () = assert!(
    SECTOR_SIZE.is_power_of_two(),
    "Sector size must be power of two for alignment arithmetic"
);

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
/// let buffer = [0u8; as_usize(SECTOR_SIZE)];
/// ```
#[inline(always)]
pub const fn as_usize(n: u32) -> usize {
    n as usize
}

// =============================================================================
// Pre-converted usize constants
// =============================================================================

/// [`SECTOR_SIZE`] as `usize`.
pub const SECTOR_SIZE_USIZE: usize = SECTOR_SIZE as usize;

// Verify usize conversions match source constants.
const _: () = assert!(SECTOR_SIZE_USIZE == SECTOR_SIZE as usize);

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
    fn constants_are_powers_of_two() {
        assert!(SECTOR_SIZE.is_power_of_two());
    }

    #[test]
    fn usize_constants_match_originals() {
        assert_eq!(SECTOR_SIZE_USIZE, SECTOR_SIZE as usize);
    }
}
