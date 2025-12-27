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
    assert!(n.is_multiple_of(d));
    n / d
}

/// Divides `n` by `d`, panicking if the division is not exact.
pub const fn div_exact_u64(n: u64, d: u64) -> u64 {
    assert!(d > 0);
    assert!(n.is_multiple_of(d));
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
    assert!(HEADERS_SIZE.is_multiple_of(constants::SECTOR_SIZE as u64));

    {
        let headers_sector = div_exact_u64(HEADERS_SIZE, constants::SECTOR_SIZE as u64);
        assert!(headers_sector > constants::JOURNAL_IOPS_WRITE_MAX as u64 || !IS_PRODUCTION);
    }

    assert!(PREPARES_SIZE > 0);
    assert!(PREPARES_SIZE.is_multiple_of(constants::SECTOR_SIZE as u64));
    assert!(PREPARES_SIZE.is_multiple_of(constants::MESSAGE_SIZE_MAX as u64));

    // Ring/Slot invariants: verify offset calculations cannot overflow or exceed bounds.
    {
        let header_size = size_of::<vsr::Header>() as u64;

        // Headers ring: SLOT_COUNT headers fit exactly in HEADERS_SIZE.
        let total_header_bytes = (SLOT_COUNT as u64) * header_size;
        assert!(total_header_bytes == HEADERS_SIZE);

        // Last headers slot offset must be within bounds after sector_floor.
        // Slot 1023: byte_offset = 1023 * 256 = 261,888
        // sector_floor(261,888) = 63 * 4096 = 258,048 < 262,144
        let last_slot_byte_offset = ((SLOT_COUNT - 1) as u64) * header_size;
        let last_slot_sector_offset =
            (last_slot_byte_offset / constants::SECTOR_SIZE as u64) * constants::SECTOR_SIZE as u64;
        assert!(last_slot_sector_offset < HEADERS_SIZE);

        // Prepares ring: SLOT_COUNT messages fit exactly in PREPARES_SIZE.
        let total_prepare_bytes = (SLOT_COUNT as u64) * (constants::MESSAGE_SIZE_MAX as u64);
        assert!(total_prepare_bytes == PREPARES_SIZE);

        // Last prepares slot offset must be within bounds.
        // Slot 1023: offset = 1023 * 1,048,576 = 1,072,693,248 < 1,073,741,824
        let last_prepare_offset = ((SLOT_COUNT - 1) as u64) * (constants::MESSAGE_SIZE_MAX as u64);
        assert!(last_prepare_offset < PREPARES_SIZE);

        // Verify multiplication cannot overflow u64.
        // Max header offset: (SLOT_COUNT - 1) * header_size
        // Max prepare offset: (SLOT_COUNT - 1) * MESSAGE_SIZE_MAX
        assert!((SLOT_COUNT as u64) <= u64::MAX / header_size);
        assert!((SLOT_COUNT as u64) <= u64::MAX / (constants::MESSAGE_SIZE_MAX as u64));
    }
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

impl Slot {
    /// Creates a new slot from an index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= SLOT_COUNT`.
    #[inline]
    pub const fn new(index: usize) -> Self {
        assert!(index < SLOT_COUNT, "slot index out of bounds");
        Slot(index)
    }

    /// Creates a slot from an operation number.
    ///
    /// Computes `op % SLOT_COUNT` to map the operation to its ring position.
    #[inline]
    pub const fn from_op(op: u64) -> Self {
        Slot((op % SLOT_COUNT as u64) as usize)
    }

    /// Returns the raw slot index.
    #[inline]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Inclusive range of slots in the circular buffer.
///
/// Represents a contiguous segment when `head <= tail`, or a wrap-around
/// segment when `head > tail`. Excludes `head == tail` since it's ambiguous
/// in circular buffers—could mean "one slot" or "all slots after full wrap".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotRange {
    pub head: Slot,
    pub tail: Slot,
}

impl SlotRange {
    /// Checks if `slot` falls within this range.
    ///
    /// Visual (`·`=included, ` `=excluded):
    ///
    /// * `head < tail` → `  head··tail  `
    /// * `head > tail` → `··tail  head··` (wraps around)
    ///
    /// # Panics
    ///
    /// Panics if `head == tail`. For single-slot checks, compare `slot == head` directly.
    #[inline]
    pub fn contains(&self, slot: Slot) -> bool {
        assert!(self.head.0 != self.tail.0);

        if self.head.index() < self.tail.index() {
            return self.head.index() <= slot.index() && slot.index() <= self.tail.index();
        }
        if self.head.index() > self.tail.index() {
            return slot.index() <= self.tail.index() || self.head.index() <= slot.index();
        }

        unreachable!("head == tail is handled by assertion above");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // sector_floor tests
    // -------------------------------------------------------------------------

    #[test]
    fn sector_floor_zero_stays_zero() {
        assert_eq!(sector_floor(0), 0);
    }

    #[test]
    fn sector_floor_rounds_down_within_first_sector() {
        // Any offset < SECTOR_SIZE should floor to 0.
        assert_eq!(sector_floor(1), 0);
        assert_eq!(sector_floor(255), 0);
        assert_eq!(sector_floor(256), 0);
        assert_eq!(sector_floor(4095), 0);
    }

    #[test]
    fn sector_floor_aligned_value_unchanged() {
        // Sector-aligned values should remain unchanged.
        assert_eq!(sector_floor(4096), 4096);
        assert_eq!(sector_floor(8192), 8192);
        assert_eq!(sector_floor(262_144), 262_144);
    }

    // -------------------------------------------------------------------------
    // Slot API tests
    // -------------------------------------------------------------------------

    #[test]
    fn slot_new_creates_valid_slot() {
        let slot = Slot::new(0);
        assert_eq!(slot.index(), 0);

        let slot = Slot::new(SLOT_COUNT - 1);
        assert_eq!(slot.index(), SLOT_COUNT - 1);
    }

    // -------------------------------------------------------------------------
    // Ring::Headers offset tests
    // -------------------------------------------------------------------------

    #[test]
    fn headers_ring_last_valid_slot() {
        // Slot 1023 is the last valid slot (SLOT_COUNT - 1).
        // byte_offset = 1023 * 256 = 261,888
        // sector_floor(261,888) = (261,888 / 4096) * 4096 = 63 * 4096 = 258,048
        let offset = Ring::Headers.offset(Slot::new(SLOT_COUNT - 1));
        assert_eq!(offset, 258_048);
        assert!(offset < HEADERS_SIZE);
    }

    // -------------------------------------------------------------------------
    // Ring::Prepares offset tests
    // -------------------------------------------------------------------------

    #[test]
    fn prepares_ring_slot_0_is_zero() {
        assert_eq!(Ring::Prepares.offset(Slot::new(0)), 0);
    }

    #[test]
    fn prepares_ring_slot_1_is_one_mib() {
        // Each prepare slot gets MESSAGE_SIZE_MAX bytes (1 MiB).
        assert_eq!(
            Ring::Prepares.offset(Slot::new(1)),
            constants::MESSAGE_SIZE_MAX as u64
        );
    }

    #[test]
    fn prepares_ring_last_valid_slot() {
        // Slot 1023: offset = 1023 * 1,048,576 = 1,072,693,248.
        let offset = Ring::Prepares.offset(Slot::new(SLOT_COUNT - 1));
        assert_eq!(offset, 1_072_693_248);
        assert!(offset < PREPARES_SIZE);
    }

    // -------------------------------------------------------------------------
    // Panic tests
    // -------------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "slot index out of bounds")]
    fn slot_new_panics_on_slot_at_boundary() {
        Slot::new(SLOT_COUNT);
    }

    #[test]
    #[should_panic(expected = "slot index out of bounds")]
    fn slot_new_panics_on_slot_beyond_boundary() {
        Slot::new(SLOT_COUNT + 1);
    }

    // -------------------------------------------------------------------------
    // div_exact tests
    // -------------------------------------------------------------------------

    #[test]
    fn div_exact_usize_works_for_exact_division() {
        assert_eq!(div_exact_usize(100, 10), 10);
        assert_eq!(div_exact_usize(256, 256), 1);
        assert_eq!(div_exact_usize(4096, 256), 16);
    }

    #[test]
    #[should_panic]
    fn div_exact_usize_panics_on_non_exact() {
        div_exact_usize(100, 3);
    }

    #[test]
    #[should_panic]
    fn div_exact_usize_panics_on_zero_divisor() {
        div_exact_usize(100, 0);
    }

    #[test]
    fn div_exact_u64_works_for_exact_division() {
        assert_eq!(div_exact_u64(100, 10), 10);
        assert_eq!(div_exact_u64(262_144, 4096), 64);
    }

    #[test]
    #[should_panic]
    fn div_exact_u64_panics_on_non_exact() {
        div_exact_u64(100, 3);
    }

    // -------------------------------------------------------------------------
    // SlotRange unit tests
    // -------------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn slotrange_panics_when_head_equals_tail() {
        let range = SlotRange {
            head: Slot::new(42),
            tail: Slot::new(42),
        };
        range.contains(Slot::new(0));
    }

    #[test]
    fn slotrange_contains_wrapping_at_extreme_boundaries() {
        // Wrapping range: slots 1023 (head) through 0 (tail).
        let range = SlotRange {
            head: Slot::new(SLOT_COUNT - 1),
            tail: Slot::new(0),
        };

        // Included: head (1023) and tail (0).
        assert!(range.contains(Slot::new(SLOT_COUNT - 1)));
        assert!(range.contains(Slot::new(0)));

        // Excluded: middle slots.
        assert!(!range.contains(Slot::new(1)));
        assert!(!range.contains(Slot::new(SLOT_COUNT - 2)));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn headers_ring_offset_always_sector_aligned(slot_index in 0usize..SLOT_COUNT) {
            // Headers ring offsets must always be sector-aligned due to sector_floor.
            let offset = Ring::Headers.offset(Slot::new(slot_index));
            prop_assert_eq!(
                offset % (constants::SECTOR_SIZE as u64),
                0,
                "Headers offset {} not sector-aligned for slot {}",
                offset,
                slot_index
            );
        }

        #[test]
        fn headers_ring_offset_within_bounds(slot_index in 0usize..SLOT_COUNT) {
            // All valid slot indices must produce offsets within HEADERS_SIZE.
            let offset = Ring::Headers.offset(Slot::new(slot_index));
            prop_assert!(
                offset < HEADERS_SIZE,
                "Headers offset {} exceeds HEADERS_SIZE for slot {}",
                offset,
                slot_index
            );
        }

        #[test]
        fn prepares_ring_offset_within_bounds(slot_index in 0usize..SLOT_COUNT) {
            // All valid slot indices must produce offsets within PREPARES_SIZE.
            let offset = Ring::Prepares.offset(Slot::new(slot_index));
            prop_assert!(
                offset < PREPARES_SIZE,
                "Prepares offset {} exceeds PREPARES_SIZE for slot {}",
                offset,
                slot_index
            );
        }

        #[test]
        fn prepares_ring_offset_exact_multiple(slot_index in 0usize..SLOT_COUNT) {
            // Prepares offset = slot_index * MESSAGE_SIZE_MAX (no rounding).
            let offset = Ring::Prepares.offset(Slot::new(slot_index));
            let expected = (slot_index as u64) * (constants::MESSAGE_SIZE_MAX as u64);
            prop_assert_eq!(
                offset,
                expected,
                "Prepares offset {} doesn't match expected {} for slot {}",
                offset,
                expected,
                slot_index
            );
        }

        #[test]
        fn headers_ring_slots_in_same_sector_share_offset(sector_num in 0usize..64) {
            // All slots mapping to the same sector must have the same offset.
            // 64 sectors in headers ring (262,144 / 4096).
            let base_slot = sector_num * HEADERS_PER_SECTOR;
            if base_slot + HEADERS_PER_SECTOR <= SLOT_COUNT {
                let base_offset = Ring::Headers.offset(Slot::new(base_slot));

                for i in 1..HEADERS_PER_SECTOR {
                    let offset = Ring::Headers.offset(Slot::new(base_slot + i));
                    prop_assert_eq!(
                        offset,
                        base_offset,
                        "Slot {} in sector {} has different offset than base slot {}",
                        base_slot + i,
                        sector_num,
                        base_slot
                    );
                }
            }
        }

        #[test]
        fn headers_ring_offset_monotonic(
            slot_a in 0usize..SLOT_COUNT,
            slot_b in 0usize..SLOT_COUNT
        ) {
            // If slot_a < slot_b, then offset(slot_a) <= offset(slot_b).
            let offset_a = Ring::Headers.offset(Slot::new(slot_a));
            let offset_b = Ring::Headers.offset(Slot::new(slot_b));

            if slot_a < slot_b {
                prop_assert!(
                    offset_a <= offset_b,
                    "Offset regressed: slot {} offset {} > slot {} offset {}",
                    slot_a,
                    offset_a,
                    slot_b,
                    offset_b
                );
            }
        }

        #[test]
        fn prepares_ring_offset_strictly_monotonic(
            slot_a in 0usize..SLOT_COUNT,
            slot_b in 0usize..SLOT_COUNT
        ) {
            // If slot_a < slot_b, then offset(slot_a) < offset(slot_b).
            let offset_a = Ring::Prepares.offset(Slot::new(slot_a));
            let offset_b = Ring::Prepares.offset(Slot::new(slot_b));

            if slot_a < slot_b {
                prop_assert!(
                    offset_a < offset_b,
                    "Prepares offset not strictly increasing: slot {} offset {} >= slot {} offset {}",
                    slot_a,
                    offset_a,
                    slot_b,
                    offset_b
                );
            } else if slot_a > slot_b {
                prop_assert!(
                    offset_a > offset_b,
                    "Prepares offset ordering wrong: slot {} offset {} <= slot {} offset {}",
                    slot_a,
                    offset_a,
                    slot_b,
                    offset_b
                );
            }
        }

        #[test]
        fn slot_from_op_always_within_bounds(op: u64) {
            // Result of from_op must always be < SLOT_COUNT.
            let slot = Slot::from_op(op);
            prop_assert!(slot.index() < SLOT_COUNT);
        }

        #[test]
        fn slot_from_op_is_modulo(op: u64) {
            // from_op(op) == op % SLOT_COUNT.
            let slot = Slot::from_op(op);
            let expected = (op % SLOT_COUNT as u64) as usize;
            prop_assert_eq!(slot.index(), expected);
        }

        #[test]
        fn sector_floor_always_aligned(offset: u64) {
            // sector_floor result is always sector-aligned.
            let floored = sector_floor(offset);
            prop_assert_eq!(floored % constants::SECTOR_SIZE as u64, 0);
        }

        #[test]
        fn sector_floor_never_increases(offset: u64) {
            // sector_floor(x) <= x.
            let floored = sector_floor(offset);
            prop_assert!(floored <= offset);
        }

        #[test]
        fn sector_floor_idempotent(offset: u64) {
            // sector_floor(sector_floor(x)) == sector_floor(x).
            let once = sector_floor(offset);
            let twice = sector_floor(once);
            prop_assert_eq!(once, twice);
        }

        // ---------------------------------------------------------------------
        // SlotRange property tests
        // ---------------------------------------------------------------------

        #[test]
        fn slotrange_non_wrapping_is_inclusive(
            head in 0usize..SLOT_COUNT,
            tail in 0usize..SLOT_COUNT,
            slot in 0usize..SLOT_COUNT
        ) {
            prop_assume!(head < tail);

            let range = SlotRange {
                head: Slot::new(head),
                tail: Slot::new(tail),
            };
            let slot = Slot::new(slot);

            let expected = head <= slot.index() && slot.index() <= tail;
            prop_assert_eq!(range.contains(slot), expected);
        }

        #[test]
        fn slotrange_wrapping_splits_at_boundary(
            head in 0usize..SLOT_COUNT,
            tail in 0usize..SLOT_COUNT,
            slot in 0usize..SLOT_COUNT
        ) {
            prop_assume!(head > tail);

            let range = SlotRange {
                head: Slot::new(head),
                tail: Slot::new(tail),
            };
            let slot = Slot::new(slot);

            let expected = slot.index() <= tail || head <= slot.index();
            prop_assert_eq!(range.contains(slot), expected);
        }

        #[test]
        fn slotrange_endpoints_always_included(
            head in 0usize..SLOT_COUNT,
            tail in 0usize..SLOT_COUNT
        ) {
            prop_assume!(head != tail);

            let range = SlotRange {
                head: Slot::new(head),
                tail: Slot::new(tail),
            };

            prop_assert!(range.contains(range.head));
            prop_assert!(range.contains(range.tail));
        }

        #[test]
        fn slotrange_contains_at_least_two_slots(
            head in 0usize..SLOT_COUNT,
            tail in 0usize..SLOT_COUNT
        ) {
            prop_assume!(head != tail);

            let range = SlotRange {
                head: Slot::new(head),
                tail: Slot::new(tail),
            };

            let contained_count = (0..SLOT_COUNT)
                .map(Slot::new)
                .filter(|&slot| range.contains(slot))
                .count();

            prop_assert!(contained_count >= 2);
        }

        #[test]
        fn slotrange_containment_partitions_slot_space(
            head in 0usize..SLOT_COUNT,
            tail in 0usize..SLOT_COUNT,
            slot in 0usize..SLOT_COUNT
        ) {
            prop_assume!(head != tail);

            let range = SlotRange {
                head: Slot::new(head),
                tail: Slot::new(tail),
            };
            let slot = Slot::new(slot);

            let in_range = range.contains(slot);

            if head < tail {
                // Non-wrapping: gap is [0, head) ∪ (tail, SLOT_COUNT)
                let in_gap = slot.index() < head || slot.index() > tail;
                prop_assert_eq!(in_range, !in_gap);
            } else {
                // Wrapping: gap is (tail, head)
                let in_gap = slot.index() > tail && slot.index() < head;
                prop_assert_eq!(in_range, !in_gap);
            }
        }
    }
}
