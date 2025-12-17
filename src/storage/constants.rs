//! Storage protocol constants and limits.

/// Default sector size for most modern storage devices.
pub const SECTOR_SIZE_DEFAULT: usize = 4096;

/// Minimum sector size (legacy 512-byte sectors).
pub const SECTOR_SIZE_MIN: usize = 512;

/// Maximum supported sector size (64 KiB for large-block devices).
pub const SECTOR_SIZE_MAX: usize = 65536;

/// Default number of concurrent I/O operations for the storage engine.
/// Must be a power of two. Provides good concurrency for async I/O without excessive memory overhead.
pub const IO_ENTRIES_DEFAULT: u32 = 256;

/// Bounds next-tick callback processing to prevent runaway loops.
pub const MAX_NEXT_TICK_ITERATIONS: usize = 10_000;

const _: () = {
    assert!(SECTOR_SIZE_DEFAULT.is_power_of_two());
    assert!(SECTOR_SIZE_MIN.is_power_of_two());
    assert!(SECTOR_SIZE_MAX.is_power_of_two());
    assert!(SECTOR_SIZE_MIN <= SECTOR_SIZE_DEFAULT);
    assert!(SECTOR_SIZE_DEFAULT <= SECTOR_SIZE_MAX);

    assert!(std::mem::size_of::<usize>() >= 8);

    assert!(IO_ENTRIES_DEFAULT > 0);
    assert!(IO_ENTRIES_DEFAULT.is_power_of_two());

    assert!(MAX_NEXT_TICK_ITERATIONS > 0);
    assert!(MAX_NEXT_TICK_ITERATIONS <= 100_000);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_pow2() {
        assert!(SECTOR_SIZE_MIN.is_power_of_two());
        assert!(SECTOR_SIZE_DEFAULT.is_power_of_two());
        assert!(SECTOR_SIZE_MAX.is_power_of_two());
    }

    #[test]
    fn const_values() {
        assert_eq!(SECTOR_SIZE_MIN, 512);
        assert_eq!(SECTOR_SIZE_DEFAULT, 4096);
        assert_eq!(SECTOR_SIZE_MAX, 65536);
        const { assert!(SECTOR_SIZE_MIN <= SECTOR_SIZE_DEFAULT) };
        const { assert!(SECTOR_SIZE_DEFAULT <= SECTOR_SIZE_MAX) };
    }
}
