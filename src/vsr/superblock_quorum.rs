//! Superblock quorum logic for fault-tolerant storage.
//!
//! The superblock is replicated across multiple copies to survive disk failures.
//! This module determines how many copies must agree (quorum) for safe operations.

use std::cmp::Ordering;

use crate::vsr::state::VsrState;
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
    /// - `copies` matches the set of unique copy indices found in `slots`
    /// - All copy indices in `slots` are within bounds
    fn assert_invariants(&self) {
        if self.valid {
            assert!(self.has_header());
        }

        let mut slot_copies = QuorumCount::empty();
        for &copy in self.slots.iter().flatten() {
            assert!((copy as usize) < COPIES);
            slot_copies.set(copy);
        }

        assert_eq!(self.copies, slot_copies);
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

/// Workspace for grouping superblock copies by content and selecting a valid quorum.
///
/// During recovery or verification, we read all `COPIES` superblock slots and need
/// to determine which (if any) form a valid quorum. Copies with identical checksums
/// are grouped together, then ranked by validity, sequence, and count.
///
/// Uses a fixed-size array because at most `COPIES` distinct versions can exist
/// (one per slot in worst-case divergence), and heap allocation is avoided in
/// the recovery path.
///
/// The contained [`Quorum`] entries hold raw pointers to superblock headers.
/// This struct must not outlive the superblock data buffer.
#[derive(Clone, Copy, Debug)]
pub struct Quorums<const COPIES: usize> {
    /// Quorum groups found during analysis. Only `array[..count]` are valid.
    array: [Quorum<COPIES>; COPIES],
    /// Number of distinct quorum groups discovered.
    count: u8,
}

impl<const COPIES: usize> Default for Quorums<COPIES> {
    fn default() -> Self {
        const { validate_copies::<COPIES>() };
        Self {
            array: [Quorum::default(); COPIES],
            count: 0,
        }
    }
}

impl<const COPIES: usize> Quorums<COPIES> {
    /// Returns the active quorum groups (only the first `count` entries are valid).
    #[inline]
    fn slice(&self) -> &[Quorum<COPIES>] {
        &self.array[..self.count as usize]
    }

    #[inline]
    fn slice_mut(&mut self) -> &mut [Quorum<COPIES>] {
        &mut self.array[..self.count as usize]
    }

    /// Identifies a valid, connected superblock quorum from the given copies.
    ///
    /// Groups copies by checksum, validates against threshold, sorts by priority,
    /// and verifies parent chain connectivity. Returns the best quorum on success.
    ///
    /// If a copy with `sequence = best.sequence - 1` exists from the same replica,
    /// its checksum must match `best.parent`—this detects interrupted writes.
    ///
    /// # Errors
    ///
    /// - [`Error::NotFound`]: No copies have valid checksums
    /// - [`Error::QuorumLost`]: Best group doesn't meet threshold
    /// - [`Error::Fork`]: Same sequence, different checksums
    /// - [`Error::ParentSkipped`]: Gap in sequence numbers (e.g., 2,2,2,4)
    /// - [`Error::ParentNotConnected`]: Immediate parent exists but doesn't chain
    /// - [`Error::VsrStateNotMonotonic`]: VSR state regressed between parent and child
    pub fn working(
        &mut self,
        copies: &[SuperBlockHeader; COPIES],
        threshold: Threshold,
    ) -> Result<Quorum<COPIES>, Error> {
        const { validate_copies::<COPIES>() };

        let threshold_count = threshold.count::<COPIES>();
        assert!((2..=5).contains(&threshold_count));

        self.count = 0;

        copies
            .iter()
            .enumerate()
            .for_each(|(i, copy)| self.count_copy(copy, i, threshold));

        self.slice_mut().sort_by(Self::sort_priority_descending);

        if self.slice().is_empty() {
            return Err(Error::NotFound);
        }

        let b = self.slice()[0];
        assert!(b.has_header());
        let b_h = unsafe { &*b.header };

        for a in &self.slice()[1..] {
            assert!(Self::sort_priority_descending(&b, a) == Ordering::Less);
            assert!(unsafe { &*a.header }.valid_checksum());
        }

        if !b.valid {
            return Err(Error::QuorumLost);
        }

        // Parent connectivity check: compare best quorum against all others.
        // We only check headers from the same cluster and replica—cross-cluster
        // or cross-replica headers are irrelevant to our chain.
        for a in &self.slice()[1..] {
            assert!(a.has_header());
            let a_h = unsafe { &*a.header };

            // Skip headers from different clusters or replicas.
            if a_h.cluster != b_h.cluster {
                continue;
            }

            if a_h.vsr_state.replica_id != b_h.vsr_state.replica_id {
                continue;
            }

            // Same sequence, different checksum = fork (unrecoverable divergence).
            if a_h.sequence == b_h.sequence {
                assert_ne!(a_h.checksum, b_h.checksum);
                return Err(Error::Fork);
            }

            // Sequence beyond b+1 means we're missing intermediate states.
            // Example: copies show sequences [2, 2, 2, 4]—where is 3?
            let b_plus_1 = b_h.sequence.checked_add(1).expect("sequence overflow");
            if a_h.sequence > b_plus_1 {
                return Err(Error::ParentSkipped);
            }

            // `a` is the immediate parent of `b`. Verify the chain is intact.
            if a_h.sequence.checked_add(1) == Some(b_h.sequence) {
                assert_ne!(a_h.checksum, b_h.checksum);
                assert_eq!(a_h.cluster, b_h.cluster);
                assert_eq!(a_h.vsr_state.replica_id, b_h.vsr_state.replica_id);

                // Parent's checksum must match what `b` recorded as its parent.
                if a_h.checksum != b_h.parent {
                    return Err(Error::ParentNotConnected);
                }

                // VSR state must never regress (view, commit op, etc.).
                if !VsrState::monotonic(&a_h.vsr_state, &b_h.vsr_state) {
                    return Err(Error::VsrStateNotMonotonic);
                }

                assert!(b_h.valid_checksum());
                b.assert_invariants();
                return Ok(b);
            }
        }

        // No immediate parent found among other quorums—this is fine.
        // The parent may have been fully overwritten by subsequent updates.
        assert!(b_h.valid_checksum());
        b.assert_invariants();
        Ok(b)
    }

    /// Comparator for sorting quorums by selection priority (best first).
    ///
    /// Ordering: valid > invalid, then higher sequence, then more copies,
    /// then higher checksum (deterministic tie-breaker for reproducible recovery).
    fn sort_priority_descending(a: &Quorum<COPIES>, b: &Quorum<COPIES>) -> Ordering {
        assert!(a.has_header());
        assert!(b.has_header());

        let a_h = unsafe { &*a.header };
        let b_h = unsafe { &*b.header };

        if a.valid != b.valid {
            return b.valid.cmp(&a.valid);
        }

        if a_h.sequence != b_h.sequence {
            return b_h.sequence.cmp(&a_h.sequence);
        }

        let a_count = a.copies.count();
        let b_count = b.copies.count();
        if a_count != b_count {
            return b_count.cmp(&a_count);
        }

        // Deterministic tie-breaker: higher checksum wins.
        b_h.checksum.cmp(&a_h.checksum)
    }

    /// Processes a single superblock copy, adding it to its quorum group.
    ///
    /// Copies are grouped by checksum. Within a group, we track which copy indices
    /// are present to determine if we have quorum.
    ///
    /// The header's `copy` field indicates which slot it *should* occupy; `slot`
    /// is where we actually *found* it. These may differ due to misdirected writes.
    ///
    /// Edge cases:
    /// - Corrupt copy index (`>= COPIES`): fall back to slot position
    /// - Duplicate copy index: ignore (already counted)
    /// - Misdirected write: trust the header's copy field over slot position
    fn count_copy(&mut self, copy: &SuperBlockHeader, slot: usize, threshold: Threshold) {
        assert!(slot < COPIES);

        let threshold_count = threshold.count::<COPIES>();
        assert!((2..=5).contains(&threshold_count));

        if !copy.valid_checksum() {
            return;
        }

        let quorum = self.find_or_insert_quorum_for_copy(copy);
        assert!(quorum.has_header());
        let q_h = unsafe { quorum.header() };
        assert_eq!(q_h.checksum, copy.checksum);
        assert!(q_h.equal(copy));

        // Handle the three cases: corrupt index, duplicate, or normal.
        if (copy.copy as usize) >= COPIES {
            // Corrupt copy index—fall back to slot position.
            quorum.slots[slot] = Some(slot as u8);
            quorum.copies.set(slot as u8);
        } else if quorum.copies.is_set(copy.copy as u8) {
            // Already have this copy index—ignore duplicate.
        } else {
            // Normal case: record by the header's copy field.
            quorum.slots[slot] = Some(copy.copy as u8);
            quorum.copies.set(copy.copy as u8);
        }

        quorum.valid = quorum.copies.count() >= threshold_count
    }

    /// Returns a mutable reference to the quorum for this copy, creating one if needed.
    ///
    /// Quorums are identified by checksum—all copies with the same checksum belong
    /// to the same quorum group. Uses two-phase lookup (find then insert) to
    /// satisfy the borrow checker.
    ///
    /// Panics if the copy has an invalid checksum or if we exceed `COPIES` quorums
    /// (impossible: can't have more distinct superblocks than slots).
    fn find_or_insert_quorum_for_copy(&mut self, copy: &SuperBlockHeader) -> &mut Quorum<COPIES> {
        assert!(copy.valid_checksum());

        // Phase 1: Search for existing quorum by checksum.
        let existing_idx = (0..self.count as usize).find(|&i| {
            let q = &self.array[i];
            assert!(q.has_header());

            let q_h = unsafe { &*q.header };
            q_h.checksum == copy.checksum
        });

        if let Some(idx) = existing_idx {
            return &mut self.array[idx];
        }

        // Phase 2: No match found—insert new quorum group.
        let idx = self.count as usize;
        assert!(idx < COPIES);

        self.array[idx] = Quorum {
            header: copy as *const _,
            valid: false,
            copies: QuorumCount::empty(),
            slots: [None; COPIES],
        };
        self.count += 1;

        &mut self.array[idx]
    }
}

/// Iterates over slot indices requiring repair, in priority order.
///
/// The goal is to ensure all copies 0..COPIES exist somewhere. Repairs only
/// occur when it's safe to do so without losing unique data. The iterator
/// does NOT guarantee `slot[i] = copy i` after completion—only that all
/// copies are present.
///
/// Slots with unique wrong copies (copy `j` in slot `i` where `j ≠ i` and
/// `j` is not duplicated) are NOT repaired, as overwriting would lose the
/// only instance of copy `j`.
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
        // We take a match at the highest priority that has one.
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
                // Safe to overwrite since we have a backup.
                if slot_copy != i_u8 && copies_duplicate.is_set(slot_copy) {
                    priority_3 = Some(i_u8);
                }
                // Note: We do NOT repair slots with unique wrong copies.
                // Overwriting would lose the only instance of that copy.
            }
        }

        let repair = priority_1.or(priority_2).or(priority_3);

        let Some(repair_slot) = repair else {
            // No more safe repairs available.
            // Remaining slots either:
            // - Are correct (slot[i] = Some(i))
            // - Have unique wrong copies (cannot overwrite without losing data)
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

    // =========================================================================
    // Quorum Tests
    // =========================================================================

    mod quorum_tests {
        use super::*;

        #[test]
        fn default_is_invalid() {
            let q = Quorum::<4>::default();
            assert!(!q.valid);
            assert!(!q.has_header());
            assert_eq!(q.copies.count(), 0);
            assert!(q.slots.iter().all(|s| s.is_none()));
        }

        #[test]
        fn header_returns_reference() {
            let header = SuperBlockHeader::zeroed();
            let q = Quorum::<4> {
                header: &header as *const SuperBlockHeader,
                ..Default::default()
            };

            let returned = unsafe { q.header() };
            assert_eq!(
                returned as *const SuperBlockHeader,
                &header as *const SuperBlockHeader
            );
        }

        #[test]
        fn invariants_hold_when_copies_match_slots() {
            let mut q = Quorum::<4>::default();
            q.copies.set(0);
            q.copies.set(2);
            q.slots[0] = Some(0);
            q.slots[2] = Some(2);

            q.assert_invariants(); // Should not panic
        }

        #[test]
        fn invariants_allow_duplicate_copies() {
            let mut q = Quorum::<4>::default();
            q.copies.set(1);
            q.slots[0] = Some(1);
            q.slots[1] = Some(1);

            q.assert_invariants();
        }

        #[test]
        #[should_panic(expected = "assertion")]
        fn invariants_panic_on_copy_set_mismatch() {
            let mut q = Quorum::<4>::default();
            q.copies.set(0);
            q.copies.set(1);
            q.slots[0] = Some(0);
            q.slots[1] = Some(2); // Count matches, but set differs

            q.assert_invariants();
        }

        #[test]
        #[should_panic(expected = "assertion")]
        fn invariants_panic_on_out_of_bounds_copy() {
            let mut q = Quorum::<4>::default();
            q.copies.set(0);
            q.slots[0] = Some(5); // Copy 5 is out of bounds for COPIES=4

            q.assert_invariants();
        }

        #[test]
        #[should_panic(expected = "assertion")]
        fn repairs_requires_valid_quorum() {
            let q = Quorum::<4>::default();
            let _ = q.repairs();
        }

        #[test]
        fn repairs_allow_duplicate_slots() {
            let header = SuperBlockHeader::zeroed();
            let mut q = Quorum::<4> {
                header: &header as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                slots: [None; 4],
            };
            q.copies.set(1);
            q.slots[0] = Some(1);
            q.slots[1] = Some(1);

            let mut repairs: Vec<u8> = q.repairs().collect();
            repairs.sort();
            assert_eq!(repairs, vec![0, 2, 3]);
        }

        // =====================================================================
        // Quorum.repairs() Integration Tests
        // =====================================================================

        #[test]
        fn quorum_repairs_all_correct_slots() {
            let header = SuperBlockHeader::zeroed();
            let mut q = Quorum::<4> {
                header: &header as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                slots: [Some(0), Some(1), Some(2), Some(3)],
            };
            for i in 0..4 {
                q.copies.set(i);
            }

            let repairs: Vec<u8> = q.repairs().collect();
            assert_eq!(repairs.len(), 0, "No repairs needed when all slots correct");
        }

        #[test]
        fn quorum_repairs_sparse_slots() {
            let header = SuperBlockHeader::zeroed();
            let mut q = Quorum::<4> {
                header: &header as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                slots: [Some(0), None, Some(2), None],
            };
            q.copies.set(0);
            q.copies.set(2);

            let mut repairs: Vec<u8> = q.repairs().collect();
            repairs.sort();
            assert_eq!(repairs, vec![1, 3], "Should repair empty slots");
        }

        #[test]
        fn quorum_repairs_with_misdirected_copies() {
            let header = SuperBlockHeader::zeroed();
            let mut q = Quorum::<4> {
                header: &header as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                // All copies present but in wrong slots (cycle)
                slots: [Some(1), Some(0), Some(3), Some(2)],
            };
            q.copies.set(0);
            q.copies.set(1);
            q.copies.set(2);
            q.copies.set(3);

            let repairs: Vec<u8> = q.repairs().collect();
            // All copies exist, just misdirected - no repairs possible
            assert_eq!(repairs.len(), 0);
        }

        #[test]
        fn quorum_repairs_with_duplicates_and_missing() {
            let header = SuperBlockHeader::zeroed();
            let mut q = Quorum::<4> {
                header: &header as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                // Slot 0 and 1 both have copy 0; copy 1 missing
                slots: [Some(0), Some(0), Some(2), Some(3)],
            };
            q.copies.set(0);
            q.copies.set(2);
            q.copies.set(3);

            let repairs: Vec<u8> = q.repairs().collect();
            assert_eq!(repairs, vec![1], "Should repair slot 1 (duplicate)");
        }

        #[test]
        fn quorum_repairs_6_copies() {
            let header = SuperBlockHeader::zeroed();
            let mut q = Quorum::<6> {
                header: &header as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                slots: [Some(0), Some(1), None, Some(3), None, Some(5)],
            };
            q.copies.set(0);
            q.copies.set(1);
            q.copies.set(3);
            q.copies.set(5);

            let mut repairs: Vec<u8> = q.repairs().collect();
            repairs.sort();
            assert_eq!(repairs, vec![2, 4], "Should repair empty slots 2 and 4");
        }

        #[test]
        fn quorum_repairs_8_copies() {
            let header = SuperBlockHeader::zeroed();
            let mut q = Quorum::<8> {
                header: &header as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                slots: [
                    Some(0),
                    Some(1),
                    Some(2),
                    Some(3),
                    Some(4),
                    Some(5),
                    Some(6),
                    Some(7),
                ],
            };
            for i in 0..8 {
                q.copies.set(i);
            }

            let repairs: Vec<u8> = q.repairs().collect();
            assert_eq!(repairs.len(), 0, "No repairs needed when all slots correct");
        }
    }

    // =========================================================================
    // Quorums Tests
    // =========================================================================

    mod quorums_tests {
        use super::*;
        use crate::vsr::members::Members;
        use crate::vsr::wire::checksum::checksum;

        // =====================================================================
        // Test Helpers
        // =====================================================================

        /// Creates a SuperBlockHeader with specified values and valid checksum.
        fn make_header(
            sequence: u64,
            copy: u16,
            cluster: u128,
            replica_id: u128,
            parent: u128,
        ) -> SuperBlockHeader {
            let mut h = SuperBlockHeader::zeroed();
            h.sequence = sequence;
            h.copy = copy;
            h.cluster = cluster;
            h.parent = parent;
            h.vsr_state.replica_id = replica_id;
            h.checksum = h.calculate_checksum();
            h
        }

        /// Creates a header with a VsrState that passes assert_internally_consistent().
        /// Use this for tests that involve VsrState::monotonic() checks.
        fn make_header_with_valid_vsr(
            sequence: u64,
            copy: u16,
            cluster: u128,
            replica_id: u128,
            parent: u128,
            view: u32,
        ) -> SuperBlockHeader {
            use crate::constants;

            let mut h = SuperBlockHeader::zeroed();
            h.sequence = sequence;
            h.copy = copy;
            h.cluster = cluster;
            h.parent = parent;

            // Set up a minimal valid VsrState.
            h.vsr_state.replica_id = replica_id;
            h.vsr_state.replica_count = 1;
            h.vsr_state.view = view;
            h.vsr_state.log_view = view; // view >= log_view

            // Members must contain replica_id.
            let mut members = Members::zeroed();
            members.0[0] = replica_id;
            h.vsr_state.members = members;

            // Checksum fields must be checksum(&[]) for empty block references.
            let empty_checksum = checksum(&[]);
            h.vsr_state.checkpoint.free_set_blocks_acquired_checksum = empty_checksum;
            h.vsr_state.checkpoint.free_set_blocks_released_checksum = empty_checksum;
            h.vsr_state.checkpoint.client_sessions_checksum = empty_checksum;

            // Storage size must be >= DATA_FILE_SIZE_MIN.
            h.vsr_state.checkpoint.storage_size = constants::DATA_FILE_SIZE_MIN;

            // Set non-zero checkpoint header checksum to avoid root baseline constraints
            // (which require view=0 and log_view=0 when checksum=0 and op=0).
            h.vsr_state.checkpoint.header.checksum = 0xDEADBEEF;

            h.checksum = h.calculate_checksum();
            h
        }

        /// Creates COPIES agreeing headers (different copy indices, same content).
        fn make_agreeing_copies<const COPIES: usize>(
            sequence: u64,
            cluster: u128,
            replica_id: u128,
            parent: u128,
        ) -> [SuperBlockHeader; COPIES] {
            std::array::from_fn(|i| make_header(sequence, i as u16, cluster, replica_id, parent))
        }

        /// Corrupts a header's checksum to make it invalid.
        fn corrupt_checksum(h: &mut SuperBlockHeader) {
            h.checksum ^= 1;
        }

        /// Creates a child header with valid VsrState for monotonicity checks.
        fn make_child_with_valid_vsr(parent: &SuperBlockHeader, copy: u16) -> SuperBlockHeader {
            let mut child = make_header_with_valid_vsr(
                parent.sequence + 1,
                copy,
                parent.cluster,
                parent.vsr_state.replica_id,
                parent.checksum,       // Link to parent
                parent.vsr_state.view, // Same view (monotonic)
            );
            // Copy parent's members and replica_count for matching.
            child.vsr_state.members = parent.vsr_state.members;
            child.vsr_state.replica_count = parent.vsr_state.replica_count;
            child.checksum = child.calculate_checksum();
            child
        }

        // =====================================================================
        // working() Error Cases
        // =====================================================================

        #[test]
        fn working_error_not_found() {
            // All copies have invalid checksums.
            let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(10, 1, 1, 0);
            for c in &mut copies {
                corrupt_checksum(c);
            }

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            assert_eq!(result.unwrap_err(), Error::NotFound);
        }

        #[test]
        fn working_error_quorum_lost() {
            // Best group has threshold-1 copies (1 for Open with COPIES=4).
            let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(10, 1, 1, 0);
            // Corrupt all but one copy.
            corrupt_checksum(&mut copies[1]);
            corrupt_checksum(&mut copies[2]);
            corrupt_checksum(&mut copies[3]);

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            // Open threshold for 4 copies is 2, we only have 1.
            assert_eq!(result.unwrap_err(), Error::QuorumLost);
        }

        #[test]
        fn working_error_fork() {
            // Two quorums with same sequence, different checksums, same cluster/replica.
            // Quorum A: copies 0, 1 with one checksum.
            // Quorum B: copies 2, 3 with different checksum (different parent).
            let mut copies: [SuperBlockHeader; 4] = [SuperBlockHeader::zeroed(); 4];

            // Quorum A: sequence 10, parent = 100.
            copies[0] = make_header(10, 0, 1, 1, 100);
            copies[1] = make_header(10, 1, 1, 1, 100);

            // Quorum B: sequence 10, parent = 200 (different parent -> different checksum).
            copies[2] = make_header(10, 2, 1, 1, 200);
            copies[3] = make_header(10, 3, 1, 1, 200);

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            assert_eq!(result.unwrap_err(), Error::Fork);
        }

        #[test]
        fn working_error_parent_skipped() {
            // Best quorum seq=10 (valid), other quorum seq=12 (invalid, 1 copy).
            // The ParentSkipped error is triggered when another quorum has
            // sequence > best.sequence + 1, indicating missing intermediate states.
            let mut copies: [SuperBlockHeader; 4] = [SuperBlockHeader::zeroed(); 4];

            // Best quorum: 3 copies at sequence 10 (valid).
            copies[0] = make_header(10, 0, 1, 1, 0);
            copies[1] = make_header(10, 1, 1, 1, 0);
            copies[2] = make_header(10, 2, 1, 1, 0);

            // Other quorum: 1 copy at sequence 12 (invalid, but has valid checksum).
            // Since it's invalid, seq=10 is best. But 12 > 10+1=11, so ParentSkipped.
            copies[3] = make_header(12, 3, 1, 1, 0);

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            assert_eq!(result.unwrap_err(), Error::ParentSkipped);
        }

        #[test]
        fn working_error_parent_not_connected() {
            // Parent seq=9, child seq=10 with child.parent != parent.checksum.
            let parent = make_header(9, 2, 1, 1, 0);

            // Child with WRONG parent checksum.
            let mut child0 = make_header(10, 0, 1, 1, 0xDEADBEEF);
            child0.vsr_state.view = parent.vsr_state.view;
            child0.checksum = child0.calculate_checksum();

            let mut child1 = make_header(10, 1, 1, 1, 0xDEADBEEF);
            child1.vsr_state.view = parent.vsr_state.view;
            child1.checksum = child1.calculate_checksum();

            let mut copies: [SuperBlockHeader; 4] = [SuperBlockHeader::zeroed(); 4];
            // Child quorum: 2 copies at sequence 10 with DIFFERENT copy indices.
            copies[0] = child0;
            copies[1] = child1;

            // Parent quorum: 2 copies at sequence 9.
            copies[2] = parent;
            let mut parent2 = make_header(9, 3, 1, 1, 0);
            parent2.checksum = parent2.calculate_checksum();
            copies[3] = parent2;

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            assert_eq!(result.unwrap_err(), Error::ParentNotConnected);
        }

        #[test]
        fn working_error_vsr_state_not_monotonic() {
            // Parent view=5, child view=4 (regression).
            // Use valid VsrState so monotonic() can run without panicking.
            let parent = make_header_with_valid_vsr(9, 2, 1, 1, 0, 5);

            // Child with correct parent link but regressed view.
            let mut child0 = make_header_with_valid_vsr(10, 0, 1, 1, parent.checksum, 4);
            child0.vsr_state.members = parent.vsr_state.members;
            child0.vsr_state.replica_count = parent.vsr_state.replica_count;
            child0.checksum = child0.calculate_checksum();

            let mut child1 = make_header_with_valid_vsr(10, 1, 1, 1, parent.checksum, 4);
            child1.vsr_state.members = parent.vsr_state.members;
            child1.vsr_state.replica_count = parent.vsr_state.replica_count;
            child1.checksum = child1.calculate_checksum();

            let mut copies: [SuperBlockHeader; 4] = [SuperBlockHeader::zeroed(); 4];
            // Child quorum: 2 copies at sequence 10 with DIFFERENT copy indices.
            copies[0] = child0;
            copies[1] = child1;

            // Parent quorum: 2 copies at sequence 9.
            copies[2] = parent;
            let mut parent2 = make_header_with_valid_vsr(9, 3, 1, 1, 0, 5);
            parent2.vsr_state.members = parent.vsr_state.members;
            parent2.vsr_state.replica_count = parent.vsr_state.replica_count;
            parent2.checksum = parent2.calculate_checksum();
            copies[3] = parent2;

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            assert_eq!(result.unwrap_err(), Error::VsrStateNotMonotonic);
        }

        // =====================================================================
        // working() Success Cases
        // =====================================================================

        #[test]
        fn working_exactly_threshold_copies_open() {
            // Only 2 copies valid (meets Open threshold for COPIES=4).
            let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(10, 1, 1, 0);
            corrupt_checksum(&mut copies[2]);
            corrupt_checksum(&mut copies[3]);

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            assert!(result.is_ok());
            let q = result.unwrap();
            assert!(q.valid);
            assert_eq!(q.copies.count(), 2);
        }

        #[test]
        fn working_exactly_threshold_copies_verify() {
            // Only 3 copies valid (meets Verify threshold for COPIES=4).
            let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(10, 1, 1, 0);
            corrupt_checksum(&mut copies[3]);

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Verify);

            assert!(result.is_ok());
            let q = result.unwrap();
            assert!(q.valid);
            assert_eq!(q.copies.count(), 3);
        }

        #[test]
        fn working_highest_sequence_wins() {
            // Multiple quorums with different sequences.
            let mut copies: [SuperBlockHeader; 4] = [SuperBlockHeader::zeroed(); 4];

            // Low sequence quorum (2 copies).
            copies[0] = make_header(5, 0, 1, 1, 0);
            copies[1] = make_header(5, 1, 1, 1, 0);

            // High sequence quorum (2 copies).
            copies[2] = make_header(10, 2, 1, 1, 0);
            copies[3] = make_header(10, 3, 1, 1, 0);

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            assert!(result.is_ok());
            let q = result.unwrap();
            let header = unsafe { q.header() };
            assert_eq!(header.sequence, 10);
        }

        #[test]
        fn working_parent_chains_correctly() {
            // Parent seq=9, child seq=10 with correct parent checksum.
            // Use valid VsrState for monotonicity check.
            let parent = make_header_with_valid_vsr(9, 2, 1, 1, 0, 0);

            // Child with correct parent link and valid VsrState.
            let child0 = make_child_with_valid_vsr(&parent, 0);
            let child1 = make_child_with_valid_vsr(&parent, 1);

            let mut copies: [SuperBlockHeader; 4] = [SuperBlockHeader::zeroed(); 4];
            // Child quorum: 2 copies with DIFFERENT copy indices.
            copies[0] = child0;
            copies[1] = child1;

            // Parent quorum: 2 copies.
            copies[2] = parent;
            let mut parent2 = make_header_with_valid_vsr(9, 3, 1, 1, 0, 0);
            parent2.vsr_state.members = parent.vsr_state.members;
            parent2.vsr_state.replica_count = parent.vsr_state.replica_count;
            parent2.checksum = parent2.calculate_checksum();
            copies[3] = parent2;

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            assert!(result.is_ok());
            let q = result.unwrap();
            let header = unsafe { q.header() };
            assert_eq!(header.sequence, 10); // Returns child, not parent.
        }

        #[test]
        fn working_no_parent_found_is_valid() {
            // Single quorum, no parent among others.
            let copies: [SuperBlockHeader; 4] = make_agreeing_copies(10, 1, 1, 0);

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            // Parent absence is acceptable (may have been overwritten).
            assert!(result.is_ok());
        }

        #[test]
        fn working_ignores_different_cluster() {
            // Quorums from different clusters should not trigger Fork error.
            let mut copies: [SuperBlockHeader; 4] = [SuperBlockHeader::zeroed(); 4];

            // Cluster 1 quorum: 2 copies at sequence 10.
            copies[0] = make_header(10, 0, 1, 1, 0);
            copies[1] = make_header(10, 1, 1, 1, 0);

            // Cluster 2 quorum: 2 copies at same sequence 10.
            copies[2] = make_header(10, 2, 2, 1, 0); // Different cluster!
            copies[3] = make_header(10, 3, 2, 1, 0);

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            // Should NOT be Fork—different clusters are ignored.
            assert!(result.is_ok());
        }

        #[test]
        fn working_ignores_different_replica() {
            // Quorums from different replicas should not trigger Fork error.
            let mut copies: [SuperBlockHeader; 4] = [SuperBlockHeader::zeroed(); 4];

            // Replica 1 quorum: 2 copies at sequence 10.
            copies[0] = make_header(10, 0, 1, 1, 0);
            copies[1] = make_header(10, 1, 1, 1, 0);

            // Replica 2 quorum: 2 copies at same sequence 10.
            copies[2] = make_header(10, 2, 1, 2, 0); // Different replica!
            copies[3] = make_header(10, 3, 1, 2, 0);

            let mut quorums = Quorums::<4>::default();
            let result = quorums.working(&copies, Threshold::Open);

            // Should NOT be Fork—different replicas are ignored.
            assert!(result.is_ok());
        }

        #[test]
        fn working_threshold_open_vs_verify() {
            // 2 copies: Valid for Open, QuorumLost for Verify.
            let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(10, 1, 1, 0);
            corrupt_checksum(&mut copies[2]);
            corrupt_checksum(&mut copies[3]);

            let mut quorums = Quorums::<4>::default();
            let result_open = quorums.working(&copies, Threshold::Open);
            assert!(result_open.is_ok());

            let mut quorums = Quorums::<4>::default();
            let result_verify = quorums.working(&copies, Threshold::Verify);
            assert_eq!(result_verify.unwrap_err(), Error::QuorumLost);
        }

        // =====================================================================
        // sort_priority_descending() Tests
        // =====================================================================

        #[test]
        fn sort_valid_before_invalid() {
            let h1 = make_header(10, 0, 1, 1, 0);
            let h2 = make_header(10, 0, 1, 1, 0);

            let valid_q = Quorum::<4> {
                header: &h1 as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                slots: [None; 4],
            };

            let invalid_q = Quorum::<4> {
                header: &h2 as *const SuperBlockHeader,
                valid: false,
                copies: QuorumCount::empty(),
                slots: [None; 4],
            };

            // Valid should sort before invalid (Less in descending order).
            assert_eq!(
                Quorums::sort_priority_descending(&valid_q, &invalid_q),
                Ordering::Less
            );
            assert_eq!(
                Quorums::sort_priority_descending(&invalid_q, &valid_q),
                Ordering::Greater
            );
        }

        #[test]
        fn sort_higher_sequence_first() {
            let h_low = make_header(5, 0, 1, 1, 0);
            let h_high = make_header(10, 0, 1, 1, 0);

            let low_q = Quorum::<4> {
                header: &h_low as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                slots: [None; 4],
            };

            let high_q = Quorum::<4> {
                header: &h_high as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                slots: [None; 4],
            };

            // Higher sequence should sort first.
            assert_eq!(
                Quorums::sort_priority_descending(&high_q, &low_q),
                Ordering::Less
            );
            assert_eq!(
                Quorums::sort_priority_descending(&low_q, &high_q),
                Ordering::Greater
            );
        }

        #[test]
        fn sort_more_copies_first() {
            let h = make_header(10, 0, 1, 1, 0);

            let mut few_copies = QuorumCount::empty();
            few_copies.set(0);
            few_copies.set(1);

            let mut many_copies = QuorumCount::empty();
            many_copies.set(0);
            many_copies.set(1);
            many_copies.set(2);
            many_copies.set(3);

            let few_q = Quorum::<4> {
                header: &h as *const SuperBlockHeader,
                valid: true,
                copies: few_copies,
                slots: [None; 4],
            };

            let many_q = Quorum::<4> {
                header: &h as *const SuperBlockHeader,
                valid: true,
                copies: many_copies,
                slots: [None; 4],
            };

            // More copies should sort first.
            assert_eq!(
                Quorums::sort_priority_descending(&many_q, &few_q),
                Ordering::Less
            );
            assert_eq!(
                Quorums::sort_priority_descending(&few_q, &many_q),
                Ordering::Greater
            );
        }

        #[test]
        fn sort_higher_checksum_tiebreaker() {
            // Same sequence, same validity, same copy count—checksum decides.
            let h1 = make_header(10, 0, 1, 1, 100);
            let h2 = make_header(10, 0, 1, 1, 200);

            // Determine which actually has higher checksum (AEGIS-128L is not predictable).
            let (h_high, h_low) = if h1.checksum > h2.checksum {
                (h1, h2)
            } else {
                (h2, h1)
            };

            let low_q = Quorum::<4> {
                header: &h_low as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                slots: [None; 4],
            };

            let high_q = Quorum::<4> {
                header: &h_high as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                slots: [None; 4],
            };

            // Different checksums should produce deterministic ordering.
            assert_ne!(h_high.checksum, h_low.checksum);

            // Higher checksum should sort first.
            assert_eq!(
                Quorums::sort_priority_descending(&high_q, &low_q),
                Ordering::Less
            );
            assert_eq!(
                Quorums::sort_priority_descending(&low_q, &high_q),
                Ordering::Greater
            );
        }

        #[test]
        fn sort_equal_quorums() {
            let h = make_header(10, 0, 1, 1, 0);

            let q = Quorum::<4> {
                header: &h as *const SuperBlockHeader,
                valid: true,
                copies: QuorumCount::empty(),
                slots: [None; 4],
            };

            // Equal quorums should compare as Equal.
            assert_eq!(Quorums::sort_priority_descending(&q, &q), Ordering::Equal);
        }

        #[test]
        fn sort_full_precedence_chain() {
            // Test that validity > sequence > count > checksum.
            let h_invalid_high_seq = make_header(100, 0, 1, 1, 0);
            let h_valid_low_seq = make_header(5, 0, 1, 1, 0);
            let h_valid_high_seq_few = make_header(10, 0, 1, 1, 0);
            let h_cksum_a = make_header(10, 0, 1, 1, 100);
            let h_cksum_b = make_header(10, 0, 1, 1, 200);

            // Determine which actually has higher checksum.
            let (h_valid_high_seq_many_high_cksum, h_valid_high_seq_many_low_cksum) =
                if h_cksum_a.checksum > h_cksum_b.checksum {
                    (h_cksum_a, h_cksum_b)
                } else {
                    (h_cksum_b, h_cksum_a)
                };

            let mut few_copies = QuorumCount::empty();
            few_copies.set(0);

            let mut many_copies = QuorumCount::empty();
            many_copies.set(0);
            many_copies.set(1);

            let invalid_high_seq = Quorum::<4> {
                header: &h_invalid_high_seq as *const SuperBlockHeader,
                valid: false,
                copies: many_copies,
                slots: [None; 4],
            };

            let valid_low_seq = Quorum::<4> {
                header: &h_valid_low_seq as *const SuperBlockHeader,
                valid: true,
                copies: many_copies,
                slots: [None; 4],
            };

            let valid_high_seq_few = Quorum::<4> {
                header: &h_valid_high_seq_few as *const SuperBlockHeader,
                valid: true,
                copies: few_copies,
                slots: [None; 4],
            };

            let valid_high_seq_many_low_cksum = Quorum::<4> {
                header: &h_valid_high_seq_many_low_cksum as *const SuperBlockHeader,
                valid: true,
                copies: many_copies,
                slots: [None; 4],
            };

            let valid_high_seq_many_high_cksum = Quorum::<4> {
                header: &h_valid_high_seq_many_high_cksum as *const SuperBlockHeader,
                valid: true,
                copies: many_copies,
                slots: [None; 4],
            };

            // Expected order (best first):
            // 1. valid_high_seq_many_high_cksum (valid, seq=10, 2 copies, high checksum)
            // 2. valid_high_seq_many_low_cksum (valid, seq=10, 2 copies, low checksum)
            // 3. valid_high_seq_few (valid, seq=10, 1 copy)
            // 4. valid_low_seq (valid, seq=5)
            // 5. invalid_high_seq (invalid)

            // Valid beats invalid (even with higher sequence).
            assert_eq!(
                Quorums::sort_priority_descending(&valid_low_seq, &invalid_high_seq),
                Ordering::Less
            );

            // Higher sequence beats lower sequence.
            assert_eq!(
                Quorums::sort_priority_descending(&valid_high_seq_few, &valid_low_seq),
                Ordering::Less
            );

            // More copies beats fewer copies.
            assert_eq!(
                Quorums::sort_priority_descending(
                    &valid_high_seq_many_low_cksum,
                    &valid_high_seq_few
                ),
                Ordering::Less
            );

            // Higher checksum is tie-breaker.
            assert_eq!(
                Quorums::sort_priority_descending(
                    &valid_high_seq_many_high_cksum,
                    &valid_high_seq_many_low_cksum
                ),
                Ordering::Less
            );
        }

        // =====================================================================
        // count_copy() Tests
        // =====================================================================

        #[test]
        fn count_copy_normal() {
            let header = make_header(10, 1, 1, 1, 0);

            let mut quorums = Quorums::<4>::default();
            quorums.count_copy(&header, 1, Threshold::Open);

            assert_eq!(quorums.count, 1);
            let q = &quorums.array[0];
            assert!(q.copies.is_set(1));
            assert_eq!(q.slots[1], Some(1));
        }

        #[test]
        fn count_copy_corrupt_index_fallback() {
            // Copy index >= COPIES should fall back to slot position.
            let mut header = SuperBlockHeader::zeroed();
            header.sequence = 10;
            header.copy = 100; // Invalid for COPIES=4.
            header.checksum = header.calculate_checksum();

            let mut quorums = Quorums::<4>::default();
            quorums.count_copy(&header, 2, Threshold::Open);

            assert_eq!(quorums.count, 1);
            let q = &quorums.array[0];
            // Falls back to slot position.
            assert!(q.copies.is_set(2));
            assert_eq!(q.slots[2], Some(2));
        }

        #[test]
        fn count_copy_duplicate_ignored() {
            // Two headers with same checksum, same copy index.
            let header = make_header(10, 1, 1, 1, 0);

            let mut quorums = Quorums::<4>::default();
            quorums.count_copy(&header, 0, Threshold::Open);
            quorums.count_copy(&header, 1, Threshold::Open); // Same copy index.

            // Should only count once.
            assert_eq!(quorums.count, 1);
            let q = &quorums.array[0];
            assert_eq!(q.copies.count(), 1);
        }

        #[test]
        fn count_copy_invalid_checksum_skipped() {
            let mut header = make_header(10, 0, 1, 1, 0);
            corrupt_checksum(&mut header);

            let mut quorums = Quorums::<4>::default();
            quorums.count_copy(&header, 0, Threshold::Open);

            // Should be skipped.
            assert_eq!(quorums.count, 0);
        }

        #[test]
        fn count_copy_threshold_exactly_met() {
            // Add exactly 2 copies (Open threshold for COPIES=4).
            let h1 = make_header(10, 0, 1, 1, 0);
            let h2 = make_header(10, 1, 1, 1, 0);

            let mut quorums = Quorums::<4>::default();
            quorums.count_copy(&h1, 0, Threshold::Open);
            assert!(!quorums.array[0].valid);

            quorums.count_copy(&h2, 1, Threshold::Open);
            assert!(quorums.array[0].valid);
        }

        #[test]
        fn count_copy_threshold_minus_one() {
            // Add only 1 copy (threshold-1 for Open with COPIES=4).
            let header = make_header(10, 0, 1, 1, 0);

            let mut quorums = Quorums::<4>::default();
            quorums.count_copy(&header, 0, Threshold::Open);

            assert!(!quorums.array[0].valid);
        }

        #[test]
        fn count_copy_different_checksums_separate_quorums() {
            // Two headers with different checksums create separate quorums.
            let h1 = make_header(10, 0, 1, 1, 100);
            let h2 = make_header(10, 1, 1, 1, 200); // Different parent -> different checksum.

            let mut quorums = Quorums::<4>::default();
            quorums.count_copy(&h1, 0, Threshold::Open);
            quorums.count_copy(&h2, 1, Threshold::Open);

            assert_eq!(quorums.count, 2);
        }

        // =====================================================================
        // Multi-COPIES Variants
        // =====================================================================

        #[test]
        fn working_error_fork_6() {
            let mut copies: [SuperBlockHeader; 6] = [SuperBlockHeader::zeroed(); 6];

            // Quorum A: 3 copies at sequence 10.
            copies[0] = make_header(10, 0, 1, 1, 100);
            copies[1] = make_header(10, 1, 1, 1, 100);
            copies[2] = make_header(10, 2, 1, 1, 100);

            // Quorum B: 3 copies at same sequence, different checksum.
            copies[3] = make_header(10, 3, 1, 1, 200);
            copies[4] = make_header(10, 4, 1, 1, 200);
            copies[5] = make_header(10, 5, 1, 1, 200);

            let mut quorums = Quorums::<6>::default();
            let result = quorums.working(&copies, Threshold::Open);

            assert_eq!(result.unwrap_err(), Error::Fork);
        }

        #[test]
        fn working_error_fork_8() {
            let mut copies: [SuperBlockHeader; 8] = [SuperBlockHeader::zeroed(); 8];

            // Quorum A: 4 copies at sequence 10.
            copies[0] = make_header(10, 0, 1, 1, 100);
            copies[1] = make_header(10, 1, 1, 1, 100);
            copies[2] = make_header(10, 2, 1, 1, 100);
            copies[3] = make_header(10, 3, 1, 1, 100);

            // Quorum B: 4 copies at same sequence, different checksum.
            copies[4] = make_header(10, 4, 1, 1, 200);
            copies[5] = make_header(10, 5, 1, 1, 200);
            copies[6] = make_header(10, 6, 1, 1, 200);
            copies[7] = make_header(10, 7, 1, 1, 200);

            let mut quorums = Quorums::<8>::default();
            let result = quorums.working(&copies, Threshold::Open);

            assert_eq!(result.unwrap_err(), Error::Fork);
        }

        // =====================================================================
        // Property-Based Tests for Quorums
        // =====================================================================

        mod quorums_properties {
            use super::*;
            use proptest::prelude::*;

            /// Generates a corruption mask (which copies to corrupt).
            fn corruption_mask<const COPIES: usize>() -> impl Strategy<Value = [bool; COPIES]> {
                proptest::collection::vec(prop::bool::ANY, COPIES).prop_map(|v| {
                    let mut arr = [false; COPIES];
                    for (i, &val) in v.iter().enumerate().take(COPIES) {
                        arr[i] = val;
                    }
                    arr
                })
            }

            /// Generates a corruption mask with at most `max_faults` errors.
            fn limited_corruption_mask<const COPIES: usize>(
                max_faults: usize,
            ) -> impl Strategy<Value = [bool; COPIES]> {
                // Select a random subset of indices to corrupt, size 0..=max_faults
                prop::collection::vec(0..COPIES, 0..=max_faults).prop_map(|indices| {
                    let mut mask = [false; COPIES];
                    for i in indices {
                        mask[i] = true;
                    }
                    mask
                })
            }

            /// Helper: apply corruption mask to copies.
            fn apply_corruption<const COPIES: usize>(
                copies: &mut [SuperBlockHeader; COPIES],
                mask: &[bool; COPIES],
            ) {
                for (i, should_corrupt) in mask.iter().enumerate() {
                    if *should_corrupt {
                        corrupt_checksum(&mut copies[i]);
                    }
                }
            }

            proptest! {
                /// If working(Open) succeeds, the returned quorum is valid/sufficient.
                #[test]
                fn working_open_success_implies_valid_quorum(
                    seq in 1u64..100u64,
                    mask in corruption_mask::<4>()
                ) {
                    let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(seq, 1, 1, 0);
                    apply_corruption(&mut copies, &mask);

                    let mut quorums = Quorums::<4>::default();
                    if let Ok(quorum) = quorums.working(&copies, Threshold::Open) {
                        prop_assert!(quorum.valid, "Successful quorum must be valid");
                        prop_assert!(
                            quorum.copies.count() >= 2,
                            "Open threshold requires at least 2 copies, got {}",
                            quorum.copies.count()
                        );
                    }
                }

                /// If working(Verify) succeeds, the returned quorum is valid/sufficient.
                #[test]
                fn working_verify_success_implies_valid_quorum(
                    seq in 1u64..100u64,
                    mask in corruption_mask::<4>()
                ) {
                    let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(seq, 1, 1, 0);
                    apply_corruption(&mut copies, &mask);

                    let mut quorums = Quorums::<4>::default();
                    if let Ok(quorum) = quorums.working(&copies, Threshold::Verify) {
                        prop_assert!(quorum.valid, "Successful quorum must be valid");
                        prop_assert!(
                            quorum.copies.count() >= 3,
                            "Verify threshold requires at least 3 copies, got {}",
                            quorum.copies.count()
                        );
                    }
                }

                /// Liveness: working(Open) MUST succeed if faults <= 2 (for COPIES=4).
                #[test]
                fn working_open_liveness(
                    seq in 1u64..100u64,
                    mask in limited_corruption_mask::<4>(2)
                ) {
                    let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(seq, 1, 1, 0);
                    apply_corruption(&mut copies, &mask);

                    let mut quorums = Quorums::<4>::default();
                    prop_assert!(
                        quorums.working(&copies, Threshold::Open).is_ok(),
                        "Open threshold must tolerate 2 faults"
                    );
                }

                /// Liveness: working(Verify) MUST succeed if faults <= 1 (for COPIES=4).
                #[test]
                fn working_verify_liveness(
                    seq in 1u64..100u64,
                    mask in limited_corruption_mask::<4>(1)
                ) {
                    let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(seq, 1, 1, 0);
                    apply_corruption(&mut copies, &mask);

                    let mut quorums = Quorums::<4>::default();
                    prop_assert!(
                        quorums.working(&copies, Threshold::Verify).is_ok(),
                        "Verify threshold must tolerate 1 fault"
                    );
                }

                /// All slot copy indices in a successful quorum are within bounds.
                #[test]
                fn working_slots_in_bounds(
                    seq in 1u64..100u64,
                    mask in corruption_mask::<4>()
                ) {
                    let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(seq, 1, 1, 0);
                    apply_corruption(&mut copies, &mask);

                    let mut quorums = Quorums::<4>::default();
                    if let Ok(quorum) = quorums.working(&copies, Threshold::Open) {
                        for copy_idx in quorum.slots.iter().flatten() {
                            prop_assert!(
                                (*copy_idx as usize) < 4,
                                "Copy index {} out of bounds for COPIES=4",
                                copy_idx
                            );
                        }
                    }
                }

                /// The returned quorum is the best according to the sort order.
                #[test]
                fn working_returns_best_quorum(
                    seq in 1u64..100u64,
                    mask in corruption_mask::<4>()
                ) {
                    let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(seq, 1, 1, 0);
                    apply_corruption(&mut copies, &mask);

                    let mut quorums = Quorums::<4>::default();
                    // Process all copies first.
                    for (slot, copy) in copies.iter().enumerate() {
                        quorums.count_copy(copy, slot, Threshold::Open);
                    }

                    if quorums.count > 0 {
                        quorums.slice_mut().sort_by(Quorums::<4>::sort_priority_descending);
                        let best = &quorums.array[0];

                        // The best quorum should be >= all others by sort order.
                        for i in 1..quorums.count as usize {
                            let other = &quorums.array[i];
                            let cmp = Quorums::<4>::sort_priority_descending(best, other);
                            prop_assert!(
                                cmp != std::cmp::Ordering::Greater,
                                "Best quorum is not actually best"
                            );
                        }
                    }
                }

                /// Calling working() twice with the same input yields the same result.
                #[test]
                fn working_is_deterministic(
                    seq in 1u64..100u64,
                    mask in corruption_mask::<4>()
                ) {
                    let mut copies: [SuperBlockHeader; 4] = make_agreeing_copies(seq, 1, 1, 0);
                    apply_corruption(&mut copies, &mask);

                    let mut quorums1 = Quorums::<4>::default();
                    let result1 = quorums1.working(&copies, Threshold::Open);

                    let mut quorums2 = Quorums::<4>::default();
                    let result2 = quorums2.working(&copies, Threshold::Open);

                    match (&result1, &result2) {
                        (Ok(q1), Ok(q2)) => {
                            prop_assert_eq!(q1.copies.count(), q2.copies.count());
                            prop_assert_eq!(q1.valid, q2.valid);
                        }
                        (Err(e1), Err(e2)) => {
                            prop_assert_eq!(e1, e2);
                        }
                        _ => {
                            prop_assert!(false, "Results differ");
                        }
                    }
                }

                /// For COPIES=6, if working() succeeds, quorum is valid.
                #[test]
                fn working_6_copies_valid(
                    seq in 1u64..100u64,
                    mask in corruption_mask::<6>()
                ) {
                    let mut copies: [SuperBlockHeader; 6] = make_agreeing_copies(seq, 1, 1, 0);
                    apply_corruption(&mut copies, &mask);

                    let mut quorums = Quorums::<6>::default();
                    if let Ok(quorum) = quorums.working(&copies, Threshold::Open) {
                        prop_assert!(quorum.valid);
                        prop_assert!(quorum.copies.count() >= 3); // Open threshold for COPIES=6.
                    }
                }

                /// For COPIES=8, if working() succeeds, quorum is valid.
                #[test]
                fn working_8_copies_valid(
                    seq in 1u64..100u64,
                    mask in corruption_mask::<8>()
                ) {
                    let mut copies: [SuperBlockHeader; 8] = make_agreeing_copies(seq, 1, 1, 0);
                    apply_corruption(&mut copies, &mask);

                    let mut quorums = Quorums::<8>::default();
                    if let Ok(quorum) = quorums.working(&copies, Threshold::Open) {
                        prop_assert!(quorum.valid);
                        prop_assert!(quorum.copies.count() >= 4); // Open threshold for COPIES=8.
                    }
                }
            }
        }
    }

    // =========================================================================
    // RepairIterator Unit Tests
    // =========================================================================

    mod repair_iterator_tests {
        use super::*;

        #[test]
        fn cycle_no_repairs_4() {
            // A pure permutation cycle: each slot has a unique wrong copy.
            // No repairs possible without losing data.
            let slots = [Some(1), Some(2), Some(3), Some(0)];
            let mut iter = RepairIterator::<4>::new(slots);

            // No repairs should occur - all copies exist, just misplaced
            let repairs: Vec<u8> = iter.by_ref().collect();
            assert_eq!(
                repairs.len(),
                0,
                "Permutation cycles should not be repaired"
            );

            // All copies still present
            let (copies_any, _) = iter.compute_copy_sets();
            for i in 0..4u8 {
                assert!(copies_any.is_set(i), "Copy {} should still exist", i);
            }
        }

        #[test]
        fn all_same_copy_catastrophic_corruption() {
            // All slots have copy 0 (catastrophic corruption)
            let slots = [Some(0), Some(0), Some(0), Some(0)];
            let iter = RepairIterator::<4>::new(slots);

            // Slots 1, 2, 3 need their copies (all priority 1 - missing)
            let repairs: Vec<u8> = iter.collect();
            assert_eq!(repairs.len(), 3);
            assert!(repairs.contains(&1));
            assert!(repairs.contains(&2));
            assert!(repairs.contains(&3));
            assert!(!repairs.contains(&0)); // Slot 0 is correct
        }

        #[test]
        fn state_mutation_marks_slot_as_repaired() {
            let slots = [Some(0), None, Some(2), None];
            let mut iter = RepairIterator::<4>::new(slots);

            assert_eq!(iter.repairs_returned, 0);

            let first = iter.next().unwrap();
            assert_eq!(iter.repairs_returned, 1);
            assert_eq!(iter.slots[first as usize], Some(first));

            let second = iter.next().unwrap();
            assert_eq!(iter.repairs_returned, 2);
            assert_eq!(iter.slots[second as usize], Some(second));

            assert_eq!(iter.next(), None);
        }

        #[test]
        fn copies_6_works() {
            let slots = [Some(0), None, None, Some(3), Some(4), Some(5)];
            let repairs: Vec<u8> = RepairIterator::<6>::new(slots).collect();

            assert!(repairs.contains(&1));
            assert!(repairs.contains(&2));
            assert_eq!(repairs.len(), 2);
        }

        #[test]
        fn copies_8_works() {
            let slots = [Some(0), Some(1), Some(2), Some(3), None, None, None, None];
            let repairs: Vec<u8> = RepairIterator::<8>::new(slots).collect();

            assert_eq!(repairs.len(), 4);
            assert!(repairs.contains(&4));
            assert!(repairs.contains(&5));
            assert!(repairs.contains(&6));
            assert!(repairs.contains(&7));
        }

        #[test]
        fn cycle_no_repairs_6() {
            // A pure permutation cycle: each slot has a unique wrong copy.
            let slots = [Some(1), Some(2), Some(3), Some(4), Some(5), Some(0)];
            let mut iter = RepairIterator::<6>::new(slots);

            // No repairs - all copies exist, just misplaced
            let repairs: Vec<u8> = iter.by_ref().collect();
            assert_eq!(
                repairs.len(),
                0,
                "Permutation cycles should not be repaired"
            );

            let (copies_any, _) = iter.compute_copy_sets();
            for i in 0..6u8 {
                assert!(copies_any.is_set(i), "Copy {} should still exist", i);
            }
        }

        #[test]
        fn cycle_no_repairs_8() {
            // A pure permutation cycle: each slot has a unique wrong copy.
            let slots = [
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
                Some(0),
            ];
            let mut iter = RepairIterator::<8>::new(slots);

            // No repairs - all copies exist, just misplaced
            let repairs: Vec<u8> = iter.by_ref().collect();
            assert_eq!(
                repairs.len(),
                0,
                "Permutation cycles should not be repaired"
            );

            let (copies_any, _) = iter.compute_copy_sets();
            for i in 0..8u8 {
                assert!(copies_any.is_set(i), "Copy {} should still exist", i);
            }
        }

        #[test]
        fn repair_cycle_missing_4() {
            // Regression case: Permutation cycle with missing element.
            // Slot 1 is empty (missing copy 1).
            // Cycle: 0->2, 2->3, 3->0, 1 missing.
            let slots = [Some(2), None, Some(3), Some(0)];
            let repairs: Vec<u8> = RepairIterator::<4>::new(slots).collect();

            // Should repair slot 1 (priority 1).
            // Result: [Some(2), Some(1), Some(3), Some(0)].
            // Now 1 is correct, others form cycle 0->2->3->0.
            assert_eq!(repairs, vec![1]);
        }

        #[test]
        fn repair_cycle_missing_6() {
            // Regression case.
            // Missing: 0, 3, 5.
            // Present: 1, 2, 4 (in slots 4, 1, 2).
            let slots = [None, Some(2), Some(4), None, Some(1), None];
            let mut repairs: Vec<u8> = RepairIterator::<6>::new(slots).collect();
            repairs.sort();

            // Should repair missing copies: 0, 3, 5.
            assert_eq!(repairs, vec![0, 3, 5]);
        }

        #[test]
        fn repair_cycle_missing_8() {
            // Regression case.
            // Missing: 0, 1, 2, 3, 7.
            // Present: 4, 5, 6 (in slots 6, 4, 5).
            let slots = [None, None, None, None, Some(5), Some(6), Some(4), None];
            let mut repairs: Vec<u8> = RepairIterator::<8>::new(slots).collect();
            repairs.sort();

            // Should repair missing copies.
            assert_eq!(repairs, vec![0, 1, 2, 3, 7]);
        }

        #[test]
        fn disjoint_swap_cycles_no_repairs() {
            // Two independent swap cycles: 0↔1 and 2↔3
            let slots = [Some(1), Some(0), Some(3), Some(2)];
            let repairs: Vec<u8> = RepairIterator::new(slots).collect();

            assert_eq!(repairs.len(), 0);

            let (copies_any, _) = RepairIterator::new(slots).compute_copy_sets();
            for i in 0..4u8 {
                assert!(copies_any.is_set(i));
            }
        }

        #[test]
        fn swapped_pair_no_repair() {
            // Slots 0 and 1 swapped, all copies present
            let slots = [Some(1), Some(0), Some(2), Some(3)];
            let repairs: Vec<u8> = RepairIterator::new(slots).collect();

            assert_eq!(repairs.len(), 0);
        }

        #[test]
        fn triple_duplicate() {
            let slots = [None, Some(1), Some(1), Some(1)];
            let repairs: Vec<u8> = RepairIterator::new(slots).collect();

            assert_eq!(repairs.len(), 3);
            assert!(repairs.contains(&0));
            assert!(repairs.contains(&2));
            assert!(repairs.contains(&3));
            assert!(!repairs.contains(&1));
        }

        #[test]
        fn quadruple_duplicate_copies_8() {
            let slots = [None, None, None, None, Some(7), Some(7), Some(7), Some(7)];
            let repairs: Vec<u8> = RepairIterator::new(slots).collect();

            assert_eq!(repairs.len(), 7);
            for i in 0..7u8 {
                assert!(repairs.contains(&i));
            }
            assert!(!repairs.contains(&7));
        }

        #[test]
        fn all_same_copy_non_zero() {
            let slots = [Some(2), Some(2), Some(2), Some(2)];
            let repairs: Vec<u8> = RepairIterator::new(slots).collect();

            assert_eq!(repairs.len(), 3);
            assert!(repairs.contains(&0));
            assert!(repairs.contains(&1));
            assert!(repairs.contains(&3));
            assert!(!repairs.contains(&2));
        }

        #[test]
        fn copies_6_mixed_duplicates_and_missing() {
            // Slot 0: correct, Slot 1: dup of 0, Slot 2: empty
            // Slot 3-4: dup of 5, Slot 5: correct
            let slots = [Some(0), Some(0), None, Some(5), Some(5), Some(5)];
            let repairs: Vec<u8> = RepairIterator::new(slots).collect();

            assert_eq!(repairs.len(), 4);
            assert!(!repairs.contains(&0));
            assert!(repairs.contains(&1));
            assert!(repairs.contains(&2));
            assert!(repairs.contains(&3));
            assert!(repairs.contains(&4));
            assert!(!repairs.contains(&5));
        }

        #[test]
        fn copies_8_mixed_scenario() {
            // Correct: 0, 1, 5. Duplicates of 1: slots 2, 3. Empty: 4, 6, 7.
            let slots = [
                Some(0),
                Some(1),
                Some(1),
                Some(1),
                None,
                Some(5),
                None,
                None,
            ];
            let repairs: Vec<u8> = RepairIterator::new(slots).collect();

            assert_eq!(repairs.len(), 5);
            assert!(repairs.contains(&4));
            assert!(repairs.contains(&6));
            assert!(repairs.contains(&7));
            assert!(repairs.contains(&2));
            assert!(repairs.contains(&3));
        }

        #[test]
        #[should_panic(expected = "assertion")]
        fn panics_on_out_of_bounds_copy() {
            let slots = [Some(0), Some(5), Some(2), Some(3)];
            let mut iter = RepairIterator::new(slots);
            let _ = iter.next();
        }

        mod compute_copy_sets {
            use super::*;

            #[test]
            fn identifies_duplicates() {
                let slots = [Some(1), Some(1), Some(2), Some(3)];
                let iter = RepairIterator::<4>::new(slots);

                let (copies_any, copies_duplicate) = iter.compute_copy_sets();

                assert!(copies_any.is_set(1));
                assert!(copies_any.is_set(2));
                assert!(copies_any.is_set(3));
                assert!(!copies_any.is_set(0)); // Copy 0 missing

                assert!(copies_duplicate.is_set(1)); // Copy 1 is duplicated
                assert!(!copies_duplicate.is_set(2));
                assert!(!copies_duplicate.is_set(3));
            }

            #[test]
            fn empty_slots_yields_empty_sets() {
                let slots: [Option<u8>; 4] = [None, None, None, None];
                let iter = RepairIterator::<4>::new(slots);

                let (copies_any, copies_duplicate) = iter.compute_copy_sets();

                assert_eq!(copies_any.count(), 0);
                assert_eq!(copies_duplicate.count(), 0);
            }

            #[test]
            fn all_unique_no_duplicates() {
                let slots = [Some(0), Some(1), Some(2), Some(3)];
                let iter = RepairIterator::<4>::new(slots);

                let (copies_any, copies_duplicate) = iter.compute_copy_sets();

                assert_eq!(copies_any.count(), 4);
                assert_eq!(copies_duplicate.count(), 0);
            }
        }
    }

    // =========================================================================
    // RepairIterator Property-Based Tests
    // =========================================================================

    mod repair_iterator_properties {
        use super::*;
        use proptest::prelude::*;
        use std::collections::HashSet;

        /// Strategy to generate a valid slot array where all copy indices are < COPIES.
        fn valid_slot_array<const COPIES: usize>() -> impl Strategy<Value = [Option<u8>; COPIES]> {
            proptest::collection::vec(
                prop_oneof![
                    1 => Just(None),
                    3 => (0u8..(COPIES as u8)).prop_map(Some),
                ],
                COPIES,
            )
            .prop_map(|v| {
                let mut arr = [None; COPIES];
                for (i, slot) in v.into_iter().enumerate() {
                    arr[i] = slot;
                }
                arr
            })
        }

        proptest! {
            /// The iterator must always terminate within COPIES iterations.
            #[test]
            fn always_terminates_within_copies_iterations(
                slots in valid_slot_array::<4>()
            ) {
                let mut iter = RepairIterator::new(slots);
                let mut count = 0;

                while iter.next().is_some() {
                    count += 1;
                    prop_assert!(count <= 4, "Iterator produced more than COPIES items");
                }
            }

            /// Each slot is yielded at most once (no duplicate repairs).
            #[test]
            fn no_duplicate_repairs(slots in valid_slot_array::<4>()) {
                let repairs: Vec<u8> = RepairIterator::new(slots).collect();
                let unique: HashSet<_> = repairs.iter().collect();

                prop_assert_eq!(
                    repairs.len(),
                    unique.len(),
                    "Duplicate repairs detected: {:?}",
                    repairs
                );
            }

            /// After full iteration, all copies 0..COPIES must exist somewhere.
            /// The iterator ensures no data is lost by only repairing safe slots.
            #[test]
            fn all_copies_present_after_repairs(slots in valid_slot_array::<4>()) {
                let mut iter = RepairIterator::new(slots);

                // Consume all repairs
                while iter.next().is_some() {}

                // Verify all copies exist somewhere
                let (copies_any, _) = iter.compute_copy_sets();
                for i in 0..4u8 {
                    prop_assert!(
                        copies_any.is_set(i),
                        "Copy {} is missing after repairs. Slots: {:?}",
                        i,
                        iter.slots
                    );
                }
            }

            /// All yielded repair indices must be within bounds.
            #[test]
            fn all_repairs_in_bounds(slots in valid_slot_array::<4>()) {
                for repair in RepairIterator::new(slots) {
                    prop_assert!(
                        (repair as usize) < 4,
                        "Repair index {} out of bounds",
                        repair
                    );
                }
            }

            /// repairs_returned counter matches actual count.
            #[test]
            fn repairs_returned_matches_count(slots in valid_slot_array::<4>()) {
                let mut iter = RepairIterator::new(slots);
                let mut expected_count = 0u8;

                while iter.next().is_some() {
                    expected_count += 1;
                    prop_assert_eq!(
                        iter.repairs_returned,
                        expected_count,
                        "repairs_returned mismatch"
                    );
                }
            }

            /// Priority ordering: repairs are yielded in priority order based on
            /// the current state at each iteration. Priority is recalculated
            /// after each repair since duplicates may become unique.
            ///
            /// This test verifies that within each iteration, the selected repair
            /// is the highest priority available at that moment.
            ///
            /// Note: Priority 4 (unique wrong copies) are NOT repaired to avoid data loss.
            #[test]
            fn priority_ordering_is_strict(slots in valid_slot_array::<4>()) {
                let mut current_slots = slots;

                loop {
                    let (copies_any, copies_dup) = {
                        let temp_iter = RepairIterator::new(current_slots);
                        temp_iter.compute_copy_sets()
                    };

                    // Find the highest priority REPAIRABLE slot.
                    // Priority 4 (unique wrong copy) is not repairable.
                    let mut best_priority = 4u8; // 4 means "no repairable slot"
                    let mut has_repairable = false;

                    for (i, &slot_value) in current_slots.iter().enumerate() {
                        let i_u8 = i as u8;

                        let priority = if slot_value.is_none() && !copies_any.is_set(i_u8) {
                            1 // Empty slot, copy missing entirely
                        } else if slot_value.is_none() && copies_any.is_set(i_u8) {
                            2 // Empty slot, copy exists elsewhere
                        } else if let Some(copy) = slot_value {
                            if copy != i_u8 && copies_dup.is_set(copy) {
                                3 // Wrong copy, duplicated - repairable
                            } else {
                                continue; // Correct or unique wrong copy (not repairable)
                            }
                        } else {
                            continue;
                        };

                        has_repairable = true;
                        if priority < best_priority {
                            best_priority = priority;
                        }
                    }

                    // Get the actual repair from the iterator
                    let mut iter = RepairIterator::new(current_slots);
                    let actual_repair = iter.next();

                    match (has_repairable, actual_repair) {
                        (false, None) => break, // Both agree: no repairs available
                        (true, None) => {
                            prop_assert!(false, "Iterator stopped but repairable slots remain");
                        }
                        (false, Some(r)) => {
                            prop_assert!(false, "Iterator returned {} but no repairable slots", r);
                        }
                        (true, Some(actual)) => {
                            // The iterator should return a slot at the best priority level
                            let actual_slot_value = current_slots[actual as usize];
                            let actual_priority = if actual_slot_value.is_none() && !copies_any.is_set(actual) {
                                1
                            } else if actual_slot_value.is_none() && copies_any.is_set(actual) {
                                2
                            } else if let Some(copy) = actual_slot_value {
                                if copy != actual && copies_dup.is_set(copy) {
                                    3
                                } else {
                                    prop_assert!(false, "Iterator repaired non-repairable slot {}", actual);
                                    unreachable!()
                                }
                            } else {
                                prop_assert!(false, "Unexpected state");
                                unreachable!()
                            };

                            prop_assert_eq!(
                                actual_priority,
                                best_priority,
                                "Iterator chose priority {} but best available was {}",
                                actual_priority,
                                best_priority
                            );

                            // Apply the repair
                            current_slots[actual as usize] = Some(actual);
                        }
                    }
                }
            }

            // Tests for COPIES=6
            #[test]
            fn all_copies_present_after_repairs_6_copies(slots in valid_slot_array::<6>()) {
                let mut iter = RepairIterator::new(slots);
                while iter.next().is_some() {}

                let (copies_any, _) = iter.compute_copy_sets();
                for i in 0..6u8 {
                    prop_assert!(copies_any.is_set(i), "Copy {} missing", i);
                }
            }

            #[test]
            fn no_duplicate_repairs_6_copies(slots in valid_slot_array::<6>()) {
                let repairs: Vec<u8> = RepairIterator::new(slots).collect();
                let unique: HashSet<_> = repairs.iter().collect();
                prop_assert_eq!(repairs.len(), unique.len());
            }

            // Tests for COPIES=8
            #[test]
            fn all_copies_present_after_repairs_8_copies(slots in valid_slot_array::<8>()) {
                let mut iter = RepairIterator::new(slots);
                while iter.next().is_some() {}

                let (copies_any, _) = iter.compute_copy_sets();
                for i in 0..8u8 {
                    prop_assert!(copies_any.is_set(i), "Copy {} missing", i);
                }
            }

            #[test]
            fn no_duplicate_repairs_8_copies(slots in valid_slot_array::<8>()) {
                let repairs: Vec<u8> = RepairIterator::new(slots).collect();
                let unique: HashSet<_> = repairs.iter().collect();
                prop_assert_eq!(repairs.len(), unique.len());
            }
        }
    }
}
