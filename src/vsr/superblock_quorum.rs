//! Superblock quorum logic for fault-tolerant storage.
//!
//! The superblock is replicated across multiple copies to survive disk failures.
//! This module determines how many copies must agree (quorum) for safe operations.

use crate::vsr::superblock::SuperBlockHeader;

/// Maximum superblock copies. Bounded by practical storage constraints.
const MAX_COPIES: usize = 8;

/// Minimum superblock copies. Below this, fault tolerance is insufficient.
const MIN_COPIES: usize = 4;

const _: () = {
    assert!(MIN_COPIES > 0);
    assert!(MIN_COPIES <= MAX_COPIES);
    // Limit to 16 copies: beyond this, coordination overhead outweighs benefits.
    assert!(MAX_COPIES <= 16);
};

/// Errors from quorum validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// No valid superblock found across any copy.
    NotFound,
    /// Insufficient agreeing copies to establish consensus.
    QuorumLost,
    /// Divergent superblock histories detected (unrecoverable).
    Fork,
    /// Parent sequence gap: expected parent missing from chain.
    ParentNotConnected,
    /// Non-consecutive parent sequence numbers.
    ParentSkipped,
    /// VSR state regressed or failed monotonicity check.
    VsrStateNotMonotonic,
}

/// Quorum threshold context.
///
/// Different operations require different fault tolerance levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Threshold {
    /// Opening: requires lower quorum (tolerates 2 faults) because repairs
    /// update copies in place, so we need headroom for concurrent failures.
    Open,
    /// Post-write verification: higher quorum (tolerates 1 fault) since
    /// we just wrote and expect most copies to be consistent.
    Verify,
}

const fn validate_copies<const COPIES: usize>() {
    assert!(COPIES >= MIN_COPIES, "COPIES below minimum");
    assert!(COPIES <= MAX_COPIES, "COPIES exceeds maximum");
    assert!(COPIES.is_multiple_of(2), "COPIES must be even");

    assert!(
        COPIES == 4 || COPIES == 6 || COPIES == 8,
        "unsupported COPIES (expected 4, 6, or 8)"
    );
}

impl Threshold {
    /// Computes required agreeing copies for the given redundancy level.
    ///
    /// # Panics
    ///
    /// Panics if `COPIES` is outside `[MIN_COPIES, MAX_COPIES]` or odd.
    pub const fn count<const COPIES: usize>(self) -> u8 {
        const { validate_copies::<COPIES>() };

        // Quorum formula: ⌈(COPIES + fault_tolerance) / 2⌉
        // - Verify: fault_tolerance = 1
        // - Open: fault_tolerance = 2
        match (self, COPIES) {
            (Threshold::Verify, 4) => 3,
            (Threshold::Verify, 6) => 4,
            (Threshold::Verify, 8) => 5,
            (Threshold::Open, 4) => 2,
            (Threshold::Open, 6) => 3,
            (Threshold::Open, 8) => 4,
            _ => unreachable!(),
        }
    }
}

/// Tracks which superblock copies participate in a quorum decision.
///
/// Bit `i` indicates copy `i` is present. A bitset provides O(1) membership,
/// 2-byte storage, and hardware `popcnt` for counting.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuorumCount(u16);

impl QuorumCount {
    /// Maximum valid copy index (0-15 for 16 copies).
    const MAX_INDEX: u8 = 15;

    /// Creates an empty set with no copies marked.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Marks copy `idx` as present in this quorum group.
    ///
    /// Idempotent: setting an already-set bit is a no-op.
    ///
    /// # Panics
    ///
    /// Panics if `idx > 15`.
    #[inline]
    pub fn set(&mut self, idx: u8) {
        assert!(idx <= Self::MAX_INDEX);
        self.0 |= 1u16 << idx;
    }

    /// Returns `true` if copy `idx` is marked present.
    ///
    /// # Panics
    ///
    /// Panics if `idx > 15`.
    #[inline]
    pub fn is_set(&self, idx: u8) -> bool {
        assert!(idx <= Self::MAX_INDEX);
        self.0 & (1u16 << idx) != 0
    }

    /// Returns the number of copies marked present.
    #[inline]
    pub fn count(&self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Returns `true` if exactly copies 0..N are set, with no extras.
    ///
    /// This is stricter than `count() == N`: it verifies the set contains
    /// precisely the expected indices. Used to confirm all expected copies
    /// responded without spurious entries.
    ///
    /// # Compile-time constraint
    ///
    /// `N` must be ≤ 16.
    #[inline]
    pub fn full<const N: usize>(self) -> bool {
        const { assert!(N <= (Self::MAX_INDEX as usize + 1)) };

        let mask: u16 = if N == 16 { u16::MAX } else { (1u16 << N) - 1 };

        (self.0 & mask) == mask && (self.0 & !mask) == 0
    }
}

/// Result of quorum analysis across superblock copies.
///
/// After reading all `COPIES` superblock slots, this struct captures which copies
/// agreed on the same header content. The quorum is "valid" when enough copies
/// match to meet the [`Threshold`] requirement.
///
/// # Slot Mapping
///
/// Physical storage has `COPIES` slots numbered 0..COPIES. Each valid superblock
/// contains a "copy index" field indicating which slot it *should* occupy.
/// Misdirected writes (copy in wrong slot) and duplicates are tracked here
/// to enable repair decisions.
///
/// # Lifetime
///
/// The `header` pointer is borrowed from the caller's buffer. This struct must
/// not outlive the superblock data it references.
#[derive(Clone, Copy, Debug)]
pub struct Quorum<const COPIES: usize> {
    /// First-seen header for this quorum group. Used as the canonical reference
    /// when multiple copies agree. Null until a valid header is encountered.
    header: *const SuperBlockHeader,

    /// `true` if this quorum meets the threshold for the intended operation.
    pub valid: bool,

    /// Bitset tracking which copy indices are present in this quorum.
    pub copies: QuorumCount,

    /// Slot-to-copy mapping. `slots[i] = Some(c)` means physical slot `i`
    /// contains copy index `c`. `None` indicates invalid/missing/ignored.
    pub slots: [Option<u8>; COPIES],
}

impl<const COPIES: usize> Default for Quorum<COPIES> {
    fn default() -> Self {
        Self {
            header: std::ptr::null(),
            valid: false,
            copies: QuorumCount::default(),
            slots: [None; COPIES],
        }
    }
}

impl<const COPIES: usize> Quorum<COPIES> {
    #[inline]
    fn has_header(&self) -> bool {
        !self.header.is_null()
    }

    /// Returns the representative header for this quorum.
    ///
    /// # Safety
    ///
    /// Caller must ensure the original superblock buffer outlives this reference.
    /// The pointer was set during quorum construction from caller-provided data.
    ///
    /// # Panics
    ///
    /// Panics if no header was ever recorded (quorum is empty).
    #[inline]
    pub unsafe fn header(&self) -> &SuperBlockHeader {
        assert!(self.has_header());
        unsafe { &*self.header }
    }

    /// Validates internal consistency. Used defensively before operations.
    ///
    /// # Invariants checked
    ///
    /// - Valid quorums have a header
    /// - `copies.count()` equals the number of `Some` entries in `slots`
    /// - All copy indices in `slots` are within bounds
    fn assert_invariants(&self) {
        if self.valid {
            assert!(self.has_header());
        }

        let slot_count = self.slots.iter().filter(|slot| slot.is_some()).count();
        assert_eq!(self.copies.count() as usize, slot_count);

        assert!(
            self.slots
                .iter()
                .flatten()
                .all(|&copy| (copy as usize) < COPIES)
        );
    }

    /// Returns an iterator yielding slot indices that need repair.
    ///
    /// Repairs restore each slot to contain its "home" copy (slot `i` should
    /// hold copy `i`). See [`RepairIterator`] for priority ordering.
    ///
    /// # Panics
    ///
    /// Panics if the quorum is not valid.
    pub fn repairs(&self) -> RepairIterator<COPIES> {
        assert!(self.valid);
        self.assert_invariants();
        RepairIterator::new(self.slots)
    }
}

/// Iterates over slot indices requiring repair, in priority order.
///
/// The goal is to restore the invariant that slot `i` contains copy `i`.
/// After all repairs, the superblock array will have each copy in its home slot.
///
/// # Repair Priority (highest to lowest)
///
/// 1. **Empty slot, missing copy**: Slot `i` is empty AND copy `i` doesn't exist
///    anywhere. This is the most critical—we're missing data entirely.
///
/// 2. **Empty slot, misdirected copy**: Slot `i` is empty but copy `i` exists
///    in some other slot. We can recover by copying from the misdirected location.
///
/// 3. **Duplicate in wrong slot**: Slot `i` contains copy `j` (where `j ≠ i`)
///    and copy `j` also exists elsewhere. Safe to overwrite since we have a backup.
///
/// # Mutation
///
/// Each call to `next()` updates internal state as if the repair succeeded,
/// marking `slots[repair_slot] = Some(repair_slot)`. This prevents returning
/// the same slot twice and allows priority recalculation.
///
/// # Example
///
/// ```ignore
/// // slots: [Some(1), None, Some(1), Some(3)]
/// // - slot 0 has copy 1 (misdirected)
/// // - slot 1 is empty, copy 1 exists elsewhere (priority 2)
/// // - slot 2 has copy 1 (duplicate of slot 0)
/// // - slot 3 has copy 3 (correct)
/// //
/// // Iterator yields: 1 (empty, copy exists), 0 (has duplicate), 2 (has duplicate)
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RepairIterator<const COPIES: usize> {
    /// Current slot state, mutated as repairs are "applied" during iteration.
    slots: [Option<u8>; COPIES],
    /// Count of repairs yielded so far. Bounded by `COPIES`.
    repairs_returned: u8,
}

impl<const COPIES: usize> RepairIterator<COPIES> {
    /// Creates a new repair iterator from the given slot mapping.
    ///
    /// # Compile-time constraint
    ///
    /// `COPIES` must be 4, 6, or 8.
    pub fn new(slots: [Option<u8>; COPIES]) -> Self {
        const { validate_copies::<COPIES>() };
        Self {
            slots,
            repairs_returned: 0,
        }
    }

    /// Computes two sets: all copies present, and copies appearing more than once.
    ///
    /// Returns `(copies_any, copies_duplicate)` where:
    /// - `copies_any`: bitset of all copy indices found in any slot
    /// - `copies_duplicate`: subset of `copies_any` appearing in 2+ slots
    fn compute_copy_sets(&self) -> (QuorumCount, QuorumCount) {
        let mut copies_any = QuorumCount::empty();
        let mut copies_duplicate = QuorumCount::empty();

        for slot in self.slots.iter() {
            if let Some(copy) = *slot {
                assert!((copy as usize) < COPIES);

                if copies_any.is_set(copy) {
                    copies_duplicate.set(copy);
                }
                copies_any.set(copy);
            }
        }

        (copies_any, copies_duplicate)
    }
}

impl<const COPIES: usize> Iterator for RepairIterator<COPIES> {
    type Item = u8;

    /// Returns the next slot index to repair, or `None` when all slots are correct.
    ///
    /// Internally marks the returned slot as repaired (sets `slots[i] = Some(i)`)
    /// so subsequent calls see updated state.
    fn next(&mut self) -> Option<Self::Item> {
        const { validate_copies::<COPIES>() };

        // Precondition: all copy indices are in bounds.
        assert!(
            self.slots
                .iter()
                .all(|slot| slot.is_none() || (slot.unwrap() as usize) < COPIES)
        );

        let (copies_any, copies_duplicate) = self.compute_copy_sets();

        // Find candidates at each priority level.
        // We take the first match at the highest priority that has one.
        let mut priority_1: Option<u8> = None; // Empty slot, copy missing entirely
        let mut priority_2: Option<u8> = None; // Empty slot, copy exists elsewhere
        let mut priority_3: Option<u8> = None; // Wrong copy, but it's a duplicate

        for i in 0..COPIES {
            let i_u8 = i as u8;
            let slot = self.slots[i];

            if slot.is_none() && !copies_any.is_set(i_u8) {
                priority_1 = Some(i_u8);
            }
            if slot.is_none() && copies_any.is_set(i_u8) {
                priority_2 = Some(i_u8);
            }
            if let Some(slot_copy) = slot {
                // Slot contains wrong copy, and that copy is duplicated elsewhere.
                if slot_copy != i_u8 && copies_duplicate.is_set(slot_copy) {
                    priority_3 = Some(i_u8);
                }
            }
        }

        let repair = priority_1.or(priority_2).or(priority_3);

        let Some(repair_slot) = repair else {
            // No repairs needed: all slots should be populated.
            assert!(self.slots.iter().all(|s| s.is_some()));
            return None;
        };

        assert!((repair_slot as usize) < COPIES);

        // Mark as repaired: slot now contains its home copy.
        self.slots[repair_slot as usize] = Some(repair_slot);

        self.repairs_returned += 1;
        assert!((self.repairs_returned as usize) <= COPIES);

        Some(repair_slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // QuorumCount Tests
    // =========================================================================

    #[test]
    fn quorum_count_empty() {
        let q = QuorumCount::empty();
        assert_eq!(q.count(), 0);
        assert!(q.full::<0>());
        assert!(!q.full::<1>());
    }

    #[test]
    fn quorum_count_set_is_set_and_count() {
        let mut q = QuorumCount::empty();
        q.set(0);
        q.set(3);
        q.set(15);
        q.set(3);

        assert!(q.is_set(0));
        assert!(q.is_set(3));
        assert!(q.is_set(15));
        assert!(!q.is_set(1));
        assert_eq!(q.count(), 3);
    }

    #[test]
    fn quorum_count_full_prefix() {
        let mut q = QuorumCount::empty();
        q.set(0);
        q.set(1);
        q.set(2);

        assert!(q.full::<3>());
        assert!(!q.full::<4>());

        q.set(4);
        assert!(!q.full::<3>());
    }

    #[test]
    fn quorum_count_full_16() {
        let mut q = QuorumCount::empty();
        for i in 0..16u8 {
            q.set(i);
        }
        assert!(q.full::<16>());
    }

    #[test]
    #[should_panic]
    fn quorum_count_set_out_of_range_panics() {
        let mut q = QuorumCount::empty();
        q.set(16);
    }

    #[test]
    #[should_panic]
    fn quorum_count_is_set_out_of_range_panics() {
        let q = QuorumCount::empty();
        q.is_set(16);
    }

    // =========================================================================
    // Threshold::count() Tests
    // =========================================================================

    #[test]
    fn threshold_count_verify_4_copies() {
        assert_eq!(Threshold::Verify.count::<4>(), 3);
    }

    #[test]
    fn threshold_count_verify_6_copies() {
        assert_eq!(Threshold::Verify.count::<6>(), 4);
    }

    #[test]
    fn threshold_count_verify_8_copies() {
        assert_eq!(Threshold::Verify.count::<8>(), 5);
    }

    #[test]
    fn threshold_count_open_4_copies() {
        assert_eq!(Threshold::Open.count::<4>(), 2);
    }

    #[test]
    fn threshold_count_open_6_copies() {
        assert_eq!(Threshold::Open.count::<6>(), 3);
    }

    #[test]
    fn threshold_count_open_8_copies() {
        assert_eq!(Threshold::Open.count::<8>(), 4);
    }

    #[test]
    fn threshold_open_tolerates_more_faults_than_verify() {
        // Open quorum is lower (tolerates 2 faults vs 1)
        assert!(Threshold::Open.count::<4>() < Threshold::Verify.count::<4>());
        assert!(Threshold::Open.count::<6>() < Threshold::Verify.count::<6>());
        assert!(Threshold::Open.count::<8>() < Threshold::Verify.count::<8>());
    }
}
