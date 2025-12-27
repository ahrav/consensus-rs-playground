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

#[derive(Clone, Copy, Debug)]
pub struct Quorum<const COPIES: usize> {
    /// Pointer to the representative header for this quorum (first encountered).
    header: *const SuperBlockHeader,

    pub valid: bool,
    pub copies: QuorumCount,

    /// For each physical slot `i`, `slots[i]` contains the *copy index* found in that slot,
    /// or `None` if the slot is invalid or ignored (e.g. duplicate).
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

    #[inline]
    pub unsafe fn header(&self) -> &SuperBlockHeader {
        assert!(self.has_header());
        unsafe { &*self.header }
    }

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

    // pub fn repairs(&self) -> RepairIterator<COPIES> {
    //     assert!(self.valid);
    //     self.assert_invariants();
    //     RepairIterator::new(self.slots);
    // }
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
