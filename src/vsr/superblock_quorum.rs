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
        assert!(COPIES % 2 == 0);

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
        if let Some(best) = best_clique_header {
            if candidate.sequence < best.sequence {
                continue;
            }
        }

        // Count members of this candidate's clique.
        let mut clique_size = 0;
        for j in 0..N {
            if (valid_mask & (1 << j)) != 0 {
                // Logic equality check (ignores copy index and checksum fields)
                if headers[j].equal(candidate) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use crate::vsr::ViewChangeArray;
    use crate::vsr::superblock::SuperBlockHeader;

    fn make_header(sequence: u64, cluster: u128) -> SuperBlockHeader {
        let mut header = SuperBlockHeader::zeroed();
        header.version = constants::SUPERBLOCK_VERSION;
        header.cluster = cluster;
        header.sequence = sequence;
        header.view_headers_all = ViewChangeArray::root(cluster);
        header.set_checksum();
        header
    }

    #[test]
    fn test_select_superblock_basic() {
        // N=4, Threshold::Open (2)
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // 3 copies at seq 10, 1 at seq 5
        for i in 0..3 {
            headers[i] = make_header(10, 1);
            headers[i].copy = i as u16;
            headers[i].set_checksum();
        }
        headers[3] = make_header(5, 1);
        headers[3].copy = 3;
        headers[3].set_checksum();

        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        let h = result.unwrap();
        assert_eq!(h.sequence, 10);
    }

    #[test]
    fn test_select_superblock_quorum_lost() {
        // N=4, Threshold::Open (2)
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // All different sequences: 10, 11, 12, 13
        for i in 0..4 {
            headers[i] = make_header(10 + i as u64, 1);
            headers[i].copy = i as u16;
            headers[i].set_checksum();
        }

        // Each has size 1, need 2.
        let result = select_superblock(&headers, Threshold::Open);
        assert!(matches!(result, Err(Error::QuorumLost)));
    }

    #[test]
    fn test_select_superblock_not_found() {
        // N=4
        let headers = [SuperBlockHeader::zeroed(); 4];
        // All checksums invalid (zeroed headers with non-zeroed checksum field due to Aegis)

        let result = select_superblock(&headers, Threshold::Open);
        assert!(matches!(result, Err(Error::NotFound)));
    }

    #[test]
    fn test_select_superblock_fork() {
        // N=4, Threshold::Open (2)
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // Group A: Seq 10, Cluster 1 (indices 0, 1)
        headers[0] = make_header(10, 1);
        headers[1] = make_header(10, 1);

        // Group B: Seq 10, Cluster 2 (indices 2, 3)
        headers[2] = make_header(10, 2);
        headers[3] = make_header(10, 2);

        // Fix copies and checksums
        for i in 0..4 {
            headers[i].copy = i as u16;
            headers[i].set_checksum();
        }

        let result = select_superblock(&headers, Threshold::Open);
        assert!(matches!(result, Err(Error::Fork)));
    }

    #[test]
    fn test_select_superblock_higher_sequence_wins() {
        // N=4, Threshold::Open (2)
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // Group A: Seq 10 (indices 0, 1) - Quorum
        headers[0] = make_header(10, 1);
        headers[1] = make_header(10, 1);

        // Group B: Seq 20 (index 2) - No Quorum
        headers[2] = make_header(20, 1);

        // Group C: Seq 5 (index 3) - No Quorum
        headers[3] = make_header(5, 1);

        for i in 0..4 {
            headers[i].copy = i as u16;
            headers[i].set_checksum();
        }

        let result = select_superblock(&headers, Threshold::Open);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().sequence, 10);
    }

    #[test]
    fn test_verify_threshold_strict() {
        // N=4, Threshold::Verify (3)
        let mut headers = [SuperBlockHeader::zeroed(); 4];

        // 2 copies at Seq 10 (Not enough for Verify)
        headers[0] = make_header(10, 1);
        headers[1] = make_header(10, 1);
        headers[2] = make_header(5, 1);
        headers[3] = make_header(5, 1);

        for i in 0..4 {
            headers[i].copy = i as u16;
            headers[i].set_checksum();
        }

        let result = select_superblock(&headers, Threshold::Verify);
        assert!(matches!(result, Err(Error::QuorumLost)));
    }
}
