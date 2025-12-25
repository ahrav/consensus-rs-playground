//! Superblock quorum logic for fault-tolerant storage.
//!
//! The superblock is replicated across multiple copies to survive disk failures.
//! This module determines how many copies must agree (quorum) for safe operations.

use crate::vsr::superblock::SuperBlockHeader;

/// Maximum superblock copies. Bounded by practical storage constraints.
const MAX_COPIES: usize = 8;

/// Minimum superblock copies. Below this, fault tolerance is insufficient.
const MIN_COPIES: usize = 4;

/// Bounds repair iteration to prevent infinite loops on malformed state.
const MAX_REPAIR_ITERATIONS: usize = 16;

const _: () = {
    assert!(MIN_COPIES > 0);
    assert!(MIN_COPIES <= MAX_COPIES);
    // Limit to 16 copies: beyond this, coordination overhead outweighs benefits.
    assert!(MAX_COPIES <= 16);

    // Repairs must be able to iterate over all copies.
    assert!(MAX_REPAIR_ITERATIONS >= MAX_COPIES);
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

impl Threshold {
    /// Computes required agreeing copies for the given redundancy level.
    ///
    /// # Panics
    ///
    /// Panics if `COPIES` is outside `[MIN_COPIES, MAX_COPIES]` or odd.
    pub const fn count<const COPIES: usize>(self) -> u8 {
        assert!(COPIES >= MIN_COPIES);
        assert!(COPIES <= MAX_COPIES);
        assert!(COPIES.is_multiple_of(2));

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

/// Selects the highest-sequence superblock that satisfies the quorum threshold.
///
/// This function is stateless: it examines the provided headers and selects the best
/// candidate based solely on validity, consensus, and sequence number.
///
/// # Returns
///
/// * `Ok(&header)`: The selected canonical superblock header.
/// * `Err(Error::NotFound)`: No headers with valid checksums were found.
/// * `Err(Error::QuorumLost)`: Valid headers exist, but no group meets the threshold.
/// * `Err(Error::Fork)`: Two disjoint groups satisfy the threshold with the same maximum sequence number.
pub fn select_superblock<const N: usize>(
    headers: &[SuperBlockHeader; N],
    threshold: Threshold,
) -> Result<&SuperBlockHeader, Error> {
    // 1. Identify valid headers (checksum pass).
    // Using a bitmask since N <= MAX_COPIES (16).
    let mut valid_mask = 0u16;
    let mut valid_count = 0;

    for (i, header) in headers.iter().enumerate() {
        if header.valid_checksum() {
            valid_mask |= 1 << i;
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return Err(Error::NotFound);
    }

    let required = threshold.count::<N>();
    let mut best_clique_header: Option<&SuperBlockHeader> = None;

    // 2. Iterate over valid headers to find the best clique.
    // We only need to check each valid header as a potential "clique leader".
    for i in 0..N {
        if (valid_mask & (1 << i)) == 0 {
            continue;
        }
        let candidate = &headers[i];

        // Optimization: If we already found a better clique (higher sequence),
        // we can skip this candidate if its sequence is strictly lower.
        // However, we must process equal sequence numbers to detect forks.
        if let Some(best) = best_clique_header
            && candidate.sequence < best.sequence
        {
            continue;
        }

        // Count members of this candidate's clique.
        let mut clique_size = 0;
        for (j, header) in headers.iter().enumerate().take(N) {
            if (valid_mask & (1 << j)) != 0 {
                // Logic equality check (ignores copy index and checksum fields)
                if header.equal(candidate) {
                    clique_size += 1;
                }
            }
        }

        if clique_size < required {
            continue;
        }

        // We found a quorum. Compare with best_clique_header.
        match best_clique_header {
            None => {
                best_clique_header = Some(candidate);
            }
            Some(current_best) => {
                if candidate.sequence > current_best.sequence {
                    best_clique_header = Some(candidate);
                } else if candidate.sequence == current_best.sequence {
                    // Fork detection:
                    // If sequences match but headers are logically different,
                    // we have two disjoint quorums claiming the same sequence.
                    if !candidate.equal(current_best) {
                        return Err(Error::Fork);
                    }
                    // Else, they are equal, so it's the same clique we already found.
                }
            }
        }
    }

    best_clique_header.ok_or(Error::QuorumLost)
}

/// Iterator over superblock copies that require repair.
#[derive(Debug, Clone)]
pub struct RepairIterator<const N: usize> {
    mask: u16,
    index: usize,
}

impl<const N: usize> Iterator for RepairIterator<N> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < N {
            let i = self.index;
            self.index += 1;
            if (self.mask & (1 << i)) != 0 {
                return Some(i);
            }
        }
        None
    }
}

/// Identifies which copies deviate from the canonical state and need repair.
///
/// A copy needs repair if:
/// - It has an invalid checksum.
/// - It disagrees with the canonical header (sequence, cluster, data, etc).
///   This includes cases where the copy has a *higher* sequence number but
///   is not part of the quorum (minority write).
pub fn detect_repairs<const N: usize>(
    headers: &[SuperBlockHeader; N],
    canonical: &SuperBlockHeader,
) -> RepairIterator<N> {
    let mut mask = 0u16;
    for (i, header) in headers.iter().enumerate() {
        // If checksum is invalid, strictly needs repair.
        if !header.valid_checksum() {
            mask |= 1 << i;
            continue;
        }

        // If valid, check if it matches canonical state.
        if !header.equal(canonical) {
            mask |= 1 << i;
        }
    }

    RepairIterator { mask, index: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use crate::vsr::ViewChangeArray;
    use crate::vsr::superblock::SuperBlockHeader;
    use proptest::prelude::*;

    // =========================================================================
    // Test Helpers
    // =========================================================================

    /// Creates a valid header with given sequence and cluster.
    fn make_header(sequence: u64, cluster: u128) -> SuperBlockHeader {
        let mut header = SuperBlockHeader::zeroed();
        header.version = constants::SUPERBLOCK_VERSION;
        header.cluster = cluster;
        header.sequence = sequence;
        header.view_headers_all = ViewChangeArray::root(cluster);
        header.set_checksum();
        header
    }

    /// Assigns copy indices and recomputes checksums.
    fn finalize_headers<const N: usize>(headers: &mut [SuperBlockHeader; N]) {
        for (i, h) in headers.iter_mut().enumerate() {
            h.copy = i as u16;
            h.set_checksum();
        }
    }

    /// Corrupts a header's checksum to make it invalid.
    fn corrupt_checksum(header: &mut SuperBlockHeader) {
        header.checksum ^= 1;
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
    // select_superblock() - Basic Scenarios
    // =========================================================================

    #[test]
    fn select_superblock_basic_quorum() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // 3 copies at seq 10, 1 at seq 5
        for h in &mut headers[..3] {
            *h = make_header(10, 1);
        }
        headers[3] = make_header(5, 1);
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 10);
    }

    #[test]
    fn select_superblock_unanimous() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // All copies agree
        for h in &mut headers {
            *h = make_header(42, 1);
        }
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 42);

        // Also works for Verify
        let result = select_superblock(&headers, Threshold::Verify);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 42);
    }

    #[test]
    fn select_superblock_exactly_at_quorum_boundary() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // Exactly 2 copies at seq 10 (Open threshold = 2)
        headers[0] = make_header(10, 1);
        headers[1] = make_header(10, 1);
        headers[2] = make_header(5, 1);
        headers[3] = make_header(3, 1);
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 10);
    }

    #[test]
    fn select_superblock_one_below_quorum() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // Only 1 copy at highest seq (below Open threshold of 2)
        headers[0] = make_header(10, 1);
        headers[1] = make_header(5, 1);
        headers[2] = make_header(3, 1);
        headers[3] = make_header(1, 1);
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(matches!(result, Err(Error::QuorumLost)));
    }

    // =========================================================================
    // select_superblock() - Error Cases
    // =========================================================================

    #[test]
    fn select_superblock_quorum_lost() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // All different sequences
        for (i, h) in headers.iter_mut().enumerate() {
            *h = make_header(10 + i as u64, 1);
        }
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(matches!(result, Err(Error::QuorumLost)));
    }

    #[test]
    fn select_superblock_all_invalid() {
        // Zeroed headers have invalid checksums
        let headers = [SuperBlockHeader::zeroed(); 4];

        let result = select_superblock(&headers, Threshold::Open);
        assert!(matches!(result, Err(Error::NotFound)));
    }

    #[test]
    fn select_superblock_all_corrupt() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        for h in &mut headers {
            *h = make_header(10, 1);
        }
        finalize_headers(&mut headers);

        // Corrupt all checksums
        for h in &mut headers {
            corrupt_checksum(h);
        }

        let result = select_superblock(&headers, Threshold::Open);
        assert!(matches!(result, Err(Error::NotFound)));
    }

    #[test]
    fn select_superblock_fork_detection() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // Group A: Seq 10, Cluster 1 (indices 0, 1)
        headers[0] = make_header(10, 1);
        headers[1] = make_header(10, 1);

        // Group B: Seq 10, Cluster 2 (indices 2, 3)
        headers[2] = make_header(10, 2);
        headers[3] = make_header(10, 2);

        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(matches!(result, Err(Error::Fork)));
    }

    #[test]
    fn select_superblock_fork_different_parent() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // Same sequence, same cluster, but different parent
        for h in &mut headers {
            *h = make_header(10, 1);
        }
        headers[0].parent = 111;
        headers[1].parent = 111;
        headers[2].parent = 222;
        headers[3].parent = 222;
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(matches!(result, Err(Error::Fork)));
    }

    // =========================================================================
    // select_superblock() - Sequence Selection
    // =========================================================================

    #[test]
    fn select_superblock_higher_sequence_wins() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // Lower seq with quorum, higher seq without quorum
        headers[0] = make_header(10, 1);
        headers[1] = make_header(10, 1);
        headers[2] = make_header(20, 1); // Higher but alone
        headers[3] = make_header(5, 1);
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 10); // Highest with quorum
    }

    #[test]
    fn select_superblock_higher_sequence_has_quorum() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // Higher seq with quorum beats lower seq with quorum
        headers[0] = make_header(20, 1);
        headers[1] = make_header(20, 1);
        headers[2] = make_header(10, 1);
        headers[3] = make_header(10, 1);
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 20);
    }

    // =========================================================================
    // select_superblock() - Partial Corruption
    // =========================================================================

    #[test]
    fn select_superblock_survives_single_corruption() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        for h in &mut headers {
            *h = make_header(10, 1);
        }
        finalize_headers(&mut headers);

        // Corrupt one copy
        corrupt_checksum(&mut headers[0]);

        // Still have 3 valid (Verify threshold = 3)
        let result = select_superblock(&headers, Threshold::Verify);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 10);
    }

    #[test]
    fn select_superblock_survives_two_corruptions_open() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        for h in &mut headers {
            *h = make_header(10, 1);
        }
        finalize_headers(&mut headers);

        // Corrupt two copies
        corrupt_checksum(&mut headers[0]);
        corrupt_checksum(&mut headers[1]);

        // 2 valid copies: enough for Open (threshold=2), not Verify (threshold=3)
        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 10);

        let result = select_superblock(&headers, Threshold::Verify);
        assert!(matches!(result, Err(Error::QuorumLost)));
    }

    #[test]
    fn select_superblock_mixed_valid_invalid() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // 2 valid at seq 10, 1 valid at seq 5, 1 corrupted at seq 10
        headers[0] = make_header(10, 1);
        headers[1] = make_header(10, 1);
        headers[2] = make_header(5, 1);
        headers[3] = make_header(10, 1);
        finalize_headers(&mut headers);

        corrupt_checksum(&mut headers[3]);

        // Only 2 valid copies at seq 10
        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 10);
    }

    // =========================================================================
    // select_superblock() - Threshold::Verify
    // =========================================================================

    #[test]
    fn select_superblock_verify_threshold_strict() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // 2 copies at each sequence (neither meets Verify threshold of 3)
        headers[0] = make_header(10, 1);
        headers[1] = make_header(10, 1);
        headers[2] = make_header(5, 1);
        headers[3] = make_header(5, 1);
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Verify);
        assert!(matches!(result, Err(Error::QuorumLost)));
    }

    #[test]
    fn select_superblock_verify_with_minority_old() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // 3 copies at seq 10, 1 at seq 5
        for h in &mut headers[..3] {
            *h = make_header(10, 1);
        }
        headers[3] = make_header(5, 1);
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Verify);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 10);
    }

    // =========================================================================
    // select_superblock() - N=6 Configuration
    // =========================================================================

    #[test]
    fn select_superblock_6_copies_open_quorum() {
        let mut headers = [SuperBlockHeader::zeroed(); 6];

        // 3 copies at seq 10 (Open threshold = 3)
        for h in &mut headers[..3] {
            *h = make_header(10, 1);
        }
        for h in &mut headers[3..] {
            *h = make_header(5, 1);
        }
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 10);
    }

    // =========================================================================
    // select_superblock() - N=8 Configuration
    // =========================================================================

    #[test]
    fn select_superblock_8_copies_open_quorum() {
        let mut headers = [SuperBlockHeader::zeroed(); 8];

        // 4 copies at seq 10 (Open threshold = 4)
        for h in &mut headers[..4] {
            *h = make_header(10, 1);
        }
        for h in &mut headers[4..] {
            *h = make_header(5, 1);
        }
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 10);
    }

    #[test]
    fn select_superblock_8_copies_fork() {
        let mut headers = [SuperBlockHeader::zeroed(); 8];

        // Group A: 4 copies (meets Open quorum)
        for h in &mut headers[..4] {
            *h = make_header(10, 1);
        }
        // Group B: 4 copies at same seq, different cluster (also meets quorum)
        for h in &mut headers[4..] {
            *h = make_header(10, 2);
        }
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(matches!(result, Err(Error::Fork)));
    }

    // =========================================================================
    // select_superblock() - Edge Cases
    // =========================================================================

    #[test]
    fn select_superblock_single_valid_copy() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        headers[0] = make_header(10, 1);
        finalize_headers(&mut headers);

        // Invalidate all but one
        for h in &mut headers[1..] {
            *h = SuperBlockHeader::zeroed();
        }

        // 1 valid copy is below any threshold
        let result = select_superblock(&headers, Threshold::Open);
        assert!(matches!(result, Err(Error::QuorumLost)));
    }

    #[test]
    fn select_superblock_sequence_zero() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // Sequence 0 is valid (initial state)
        for h in &mut headers {
            *h = make_header(0, 1);
        }
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 0);
    }

    #[test]
    fn select_superblock_max_sequence() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        for h in &mut headers {
            *h = make_header(u64::MAX, 1);
        }
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, u64::MAX);
    }

    #[test]
    fn select_superblock_ptr_stability() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        for h in &mut headers {
            *h = make_header(10, 1);
        }
        finalize_headers(&mut headers);

        let result = select_superblock(&headers, Threshold::Open).unwrap();

        // Verify returned reference points into original array
        let result_ptr = result as *const _;
        let array_start = headers.as_ptr();
        let array_end = unsafe { array_start.add(4) };

        assert!(result_ptr >= array_start && result_ptr < array_end);
    }

    // =========================================================================
    // detect_repairs() Tests
    // =========================================================================

    #[test]
    fn detect_repairs_all_match() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];
        for h in &mut headers {
            *h = make_header(10, 1);
        }
        finalize_headers(&mut headers);

        let canonical = &headers[0];
        let repairs: Vec<usize> = detect_repairs(&headers, canonical).collect();
        assert!(repairs.is_empty());
    }

    #[test]
    fn detect_repairs_invalid_checksum() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];
        for h in &mut headers {
            *h = make_header(10, 1);
        }
        finalize_headers(&mut headers);

        // Corrupt index 2
        corrupt_checksum(&mut headers[2]);

        let canonical = &headers[0];
        let repairs: Vec<usize> = detect_repairs(&headers, canonical).collect();
        assert_eq!(repairs, vec![2]);
    }

    #[test]
    fn detect_repairs_old_sequence() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];
        for h in &mut headers[..3] {
            *h = make_header(10, 1);
        }
        // Old sequence at index 3
        headers[3] = make_header(5, 1);
        finalize_headers(&mut headers);

        let canonical = &headers[0];
        let repairs: Vec<usize> = detect_repairs(&headers, canonical).collect();
        assert_eq!(repairs, vec![3]);
    }

    #[test]
    fn detect_repairs_higher_sequence_without_quorum() {
        // Scenario: canonical is seq 10 (quorum), but index 3 is seq 11 (minority).
        // Index 3 should be repaired (rolled back).
        let mut headers = [SuperBlockHeader::zeroed(); 4];
        for h in &mut headers[..3] {
            *h = make_header(10, 1);
        }
        headers[3] = make_header(11, 1);
        finalize_headers(&mut headers);

        let canonical = &headers[0];
        let repairs: Vec<usize> = detect_repairs(&headers, canonical).collect();
        assert_eq!(repairs, vec![3]);
    }

    #[test]
    fn detect_repairs_divergent_cluster() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];
        for h in &mut headers[..3] {
            *h = make_header(10, 1);
        }
        // Different cluster at index 3
        headers[3] = make_header(10, 2);
        finalize_headers(&mut headers);

        let canonical = &headers[0];
        let repairs: Vec<usize> = detect_repairs(&headers, canonical).collect();
        assert_eq!(repairs, vec![3]);
    }

    #[test]
    fn detect_repairs_multiple_issues() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];
        headers[0] = make_header(10, 1);
        headers[1] = make_header(5, 1); // Old sequence
        headers[2] = make_header(10, 1);
        headers[3] = make_header(10, 2); // Different cluster
        finalize_headers(&mut headers);

        corrupt_checksum(&mut headers[2]); // Corrupt checksum

        let canonical = &headers[0];
        let repairs: Vec<usize> = detect_repairs(&headers, canonical).collect();
        assert_eq!(repairs, vec![1, 2, 3]);
    }

    #[test]
    fn detect_repairs_all_need_repair() {
        let mut headers = [SuperBlockHeader::zeroed(); 4];
        for (i, h) in headers.iter_mut().enumerate() {
            *h = make_header(i as u64, 1);
        }
        finalize_headers(&mut headers);

        // Canonical is seq 10, none match
        let canonical = make_header(10, 1);
        let repairs: Vec<usize> = detect_repairs(&headers, &canonical).collect();
        assert_eq!(repairs, vec![0, 1, 2, 3]);
    }

    // =========================================================================
    // RepairIterator Tests
    // =========================================================================

    #[test]
    fn repair_iterator_empty_mask() {
        let iter: RepairIterator<4> = RepairIterator { mask: 0, index: 0 };
        let repairs: Vec<usize> = iter.collect();
        assert!(repairs.is_empty());
    }

    #[test]
    fn repair_iterator_all_set() {
        let iter: RepairIterator<4> = RepairIterator {
            mask: 0b1111,
            index: 0,
        };
        let repairs: Vec<usize> = iter.collect();
        assert_eq!(repairs, vec![0, 1, 2, 3]);
    }

    #[test]
    fn repair_iterator_sparse() {
        let iter: RepairIterator<8> = RepairIterator {
            mask: 0b10100010,
            index: 0,
        };
        let repairs: Vec<usize> = iter.collect();
        assert_eq!(repairs, vec![1, 5, 7]);
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    proptest! {
        /// Unanimous copies always succeed.
        #[test]
        fn prop_unanimous_always_succeeds(
            sequence in any::<u64>(),
            cluster in 1u128..=u128::MAX,
        ) {
            let mut headers = [SuperBlockHeader::zeroed(); 4];
            for h in &mut headers {
                *h = make_header(sequence, cluster);
            }
            finalize_headers(&mut headers);

            prop_assert!(select_superblock(&headers, Threshold::Open).is_ok());
            prop_assert!(select_superblock(&headers, Threshold::Verify).is_ok());
        }

        /// Selected header has the highest sequence among quorum groups.
        #[test]
        fn prop_selected_has_highest_quorum_sequence(
            seq_a in 0u64..1000,
            seq_b in 0u64..1000,
        ) {
            let mut headers = [SuperBlockHeader::zeroed(); 4];

            // 2 copies each at different sequences
            headers[0] = make_header(seq_a, 1);
            headers[1] = make_header(seq_a, 1);
            headers[2] = make_header(seq_b, 1);
            headers[3] = make_header(seq_b, 1);
            finalize_headers(&mut headers);

            let result = select_superblock(&headers, Threshold::Open);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.unwrap().sequence, seq_a.max(seq_b));
        }

        /// Fork requires exactly two disjoint quorums at same sequence.
        #[test]
        fn prop_fork_requires_same_sequence(
            seq_a in 0u64..1000,
            seq_b in 0u64..1000,
        ) {
            let mut headers = [SuperBlockHeader::zeroed(); 4];

            // Group A at seq_a, Group B at seq_b, different clusters
            headers[0] = make_header(seq_a, 1);
            headers[1] = make_header(seq_a, 1);
            headers[2] = make_header(seq_b, 2);
            headers[3] = make_header(seq_b, 2);
            finalize_headers(&mut headers);

            let result = select_superblock(&headers, Threshold::Open);

            if seq_a == seq_b {
                prop_assert!(matches!(result, Err(Error::Fork)));
            } else {
                prop_assert!(result.is_ok());
            }
        }

        /// Corrupting copies below quorum still succeeds.
        #[test]
        fn prop_survives_single_corruption(
            corrupt_idx in 0usize..4,
            sequence in any::<u64>(),
        ) {
            let mut headers = [SuperBlockHeader::zeroed(); 4];
            for h in &mut headers {
                *h = make_header(sequence, 1);
            }
            finalize_headers(&mut headers);

            corrupt_checksum(&mut headers[corrupt_idx]);

            // 3 valid copies: passes Verify (threshold=3)
            prop_assert!(select_superblock(&headers, Threshold::Verify).is_ok());
        }

        /// All invalid checksums returns NotFound.
        #[test]
        fn prop_all_invalid_returns_not_found(
            sequences in proptest::collection::vec(any::<u64>(), 4),
        ) {
            let mut headers = [SuperBlockHeader::zeroed(); 4];
            for (i, h) in headers.iter_mut().enumerate() {
                *h = make_header(sequences[i], 1);
            }
            finalize_headers(&mut headers);

            for h in &mut headers {
                corrupt_checksum(h);
            }

            prop_assert!(matches!(
                select_superblock(&headers, Threshold::Open),
                Err(Error::NotFound)
            ));
        }

        /// detect_repairs returns empty when all match canonical.
        #[test]
        fn prop_detect_repairs_empty_when_unanimous(
            sequence in any::<u64>(),
            cluster in 1u128..=u128::MAX,
        ) {
            let mut headers = [SuperBlockHeader::zeroed(); 4];
            for h in &mut headers {
                *h = make_header(sequence, cluster);
            }
            finalize_headers(&mut headers);

            let canonical = &headers[0];
            let repairs: Vec<usize> = detect_repairs(&headers, canonical).collect();
            prop_assert!(repairs.is_empty());
        }

        /// detect_repairs finds all mismatches.
        #[test]
        fn prop_detect_repairs_finds_all_different(
            seq_canonical in any::<u64>(),
            seq_other in any::<u64>(),
        ) {
            if seq_canonical == seq_other {
                return Ok(());
            }

            let mut headers = [SuperBlockHeader::zeroed(); 4];
            for h in &mut headers {
                *h = make_header(seq_other, 1);
            }
            finalize_headers(&mut headers);

            let canonical = make_header(seq_canonical, 1);
            let repairs: Vec<usize> = detect_repairs(&headers, &canonical).collect();
            prop_assert_eq!(repairs, vec![0, 1, 2, 3]);
        }
    }
}
