//! Journal write-ahead log primitives.
//!
//! Provides layout constants and offset calculations for the WAL's two circular
//! buffers: headers and prepares. Used by the journal to map operations to
//! on-disk locations.

use core::mem::size_of;

use crate::constants;
use crate::vsr;

/// Divides `n` by `d`, panicking if the division is not exact.
pub const fn div_exact_usize(n: usize, d: usize) -> usize {
    assert!(d > 0);
    assert!(n % d == 0);
    n / d
}

/// Divides `n` by `d`, panicking if the division is not exact.
pub const fn div_exact_u64(n: u64, d: u64) -> u64 {
    assert!(d > 0);
    assert!(n % d == 0);
    n / d
}

/// Rounds a byte offset down to the nearest sector boundary.
#[inline]
pub const fn sector_floor(offset: u64) -> u64 {
    (offset / constants::SECTOR_SIZE as u64) * (constants::SECTOR_SIZE as u64)
}

// -----------------------------------------------------------------------------
// Core journal constants
// -----------------------------------------------------------------------------

/// Total slots in the journal's circular buffers.
pub const SLOT_COUNT: usize = constants::JOURNAL_SLOT_COUNT;

/// Size in bytes of the headers ring.
pub const HEADERS_SIZE: u64 = constants::JOURNAL_SIZE_HEADERS;

/// Size in bytes of the prepares ring.
pub const PREPARES_SIZE: u64 = constants::JOURNAL_SIZE_PREPARES;

/// Combined size of headers and prepares rings.
pub const WRITE_AHEAD_LOG_ZONE_SIZE: u64 = HEADERS_SIZE + PREPARES_SIZE;

/// Number of [`vsr::Header`]s that fit in one sector.
pub const HEADERS_PER_SECTOR: usize =
    div_exact_usize(constants::SECTOR_SIZE, size_of::<vsr::Header>());

/// Number of [`vsr::Header`]s that fit in one message.
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
    }

    assert!(PREPARES_SIZE > 0);
    assert!(PREPARES_SIZE % constants::SECTOR_SIZE as u64 == 0);
    assert!(PREPARES_SIZE % constants::MESSAGE_SIZE_MAX as u64 == 0);
};

// -----------------------------------------------------------------------------
// Ring / Slot
// -----------------------------------------------------------------------------

/// Identifies which WAL circular buffer to operate on.
///
/// The WAL stores headers and prepares in separate rings. Reserved headers
/// have their `op` set to the slot index, enabling detection of misdirected
/// reads/writes during recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ring {
    /// Fixed-size header metadata ring (sector-aligned).
    Headers,
    /// Fixed-size prepare payload ring (each slot padded to `MESSAGE_SIZE_MAX`).
    Prepares,
}

impl Ring {
    /// Returns the byte offset within this ring for the given slot.
    ///
    /// For `Headers`, returns a sector-aligned offset for direct I/O.
    /// For `Prepares`, returns a message-aligned offset.
    #[inline]
    pub fn offset(&self, slot: Slot) -> u64 {
        assert!(slot.0 < SLOT_COUNT);

        match self {
            Ring::Headers => {
                let byte_offset = (slot.0 as u64)
                    .checked_mul(size_of::<vsr::Header>() as u64)
                    .expect("slot.index & size of header overflow");
                let ring_offset = sector_floor(byte_offset);

                assert!(ring_offset < HEADERS_SIZE);
                ring_offset
            }
            Ring::Prepares => {
                let ring_offset = (constants::MESSAGE_SIZE_MAX as u64)
                    .checked_mul(slot.0 as u64)
                    .expect("slot.index * MESSAGE_SIZE_MAX overflow");

                assert!(ring_offset < PREPARES_SIZE);
                ring_offset
            }
        }
    }
}

/// Index into the journal's circular buffers, computed as `op % SLOT_COUNT`.
///
/// Maps an operation to its position in both rings and in-memory tracking
/// arrays (`headers`, `dirty`, `faulty`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot(usize);
