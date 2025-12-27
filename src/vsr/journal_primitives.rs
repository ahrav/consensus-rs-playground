use core::mem::size_of;

use crate::constants;
use crate::vsr;

pub const fn div_exact_usize(n: usize, d: usize) -> usize {
    assert!(d > 0);
    assert!(n % d == 0);
    n / d
}

pub const fn div_exact_u64(n: u64, d: u64) -> u64 {
    assert!(d > 0);
    assert!(n % d == 0);
    n / d
}

/// Floor an offset down to a sector boundary.
#[inline]
pub const fn sector_floor(offset: u64) -> u64 {
    (offset / constants::SECTOR_SIZE as u64) * (constants::SECTOR_SIZE as u64)
}

// -----------------------------------------------------------------------------
// Core journal constants
// -----------------------------------------------------------------------------

pub const SLOT_COUNT: usize = constants::JOURNAL_SLOT_COUNT;

pub const HEADERS_SIZE: u64 = constants::JOURNAL_SIZE_HEADERS;

pub const PREPARES_SIZE: u64 = constants::JOURNAL_SIZE_PREPARES;

pub const WRITE_AHEAD_LOG_ZONE_SIZE: u64 = HEADERS_SIZE + PREPARES_SIZE;

pub const HEADERS_PER_SECTOR: usize =
    div_exact_usize(constants::SECTOR_SIZE, size_of::<vsr::Header>());

pub const HEADERS_PER_MESSAGE: usize = div_exact_usize(
    constants::MESSAGE_SIZE_MAX as usize,
    size_of::<vsr::Header>(),
);

const IS_PRODUCTION: bool = cfg!(feature = "production");

const _: () = {
    assert!(HEADERS_PER_SECTOR > 0);
    assert!(HEADERS_PER_MESSAGE > 0);

    assert!(SLOT_COUNT > 0);
    assert!(SLOT_COUNT.is_multiple_of(2));
    assert!(SLOT_COUNT.is_multiple_of(HEADERS_PER_SECTOR));
    assert!(SLOT_COUNT >= HEADERS_PER_SECTOR);

    assert!(SLOT_COUNT > constants::PIPELINE_PREPARE_QUEUE_MAX);

    assert!(HEADERS_SIZE > 0);
    assert!(HEADERS_SIZE % constants::SECTOR_SIZE as u64 == 0);

    {
        let headers_sector = div_exact_u64(HEADERS_SIZE, constants::SECTOR_SIZE as u64);
        assert!(headers_sector > constants::JOURNAL_IOPS_WRITE_MAX as u64 || !IS_PRODUCTION);

        // let _ =
    }

    assert!(PREPARES_SIZE > 0);
    assert!(PREPARES_SIZE % constants::SECTOR_SIZE as u64 == 0);
    assert!(PREPARES_SIZE % constants::MESSAGE_SIZE_MAX as u64 == 0);
};
