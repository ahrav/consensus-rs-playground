//! Cluster membership tracking for VSR.
//!
//! Members are stored as a contiguous prefix of non-zero replica IDs with no duplicates.
//! Zero marks the end of the active member list; trailing slots are always zero.

use std::mem;

use crate::{constants, vsr::wire::checksum};

const CLUSTER_CONFIG_FIELD_COUNT: usize = 17;

const CLUSTER_CONFIG_SEED_SIZE: usize = CLUSTER_CONFIG_FIELD_COUNT * 8;

const ID_SEED_SIZE: usize = 33;

const _: () = {
    assert!(CLUSTER_CONFIG_FIELD_COUNT == 17);
    assert!(CLUSTER_CONFIG_SEED_SIZE == CLUSTER_CONFIG_FIELD_COUNT * 8);

    assert!(ID_SEED_SIZE == 33);
};

/// Fixed-size array of replica IDs representing cluster membership.
///
/// # Invariants
/// - Non-zero IDs are packed at the front (contiguous prefix)
/// - No duplicate IDs
/// - Zero terminates the active list; all subsequent slots are zero
///
/// Use [`valid_members`] to validate these invariants.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Members(pub [u128; constants::MEMBERS_MAX]);

const _: () = {
    assert!(constants::MEMBERS_MAX > 0);
    assert!(constants::MEMBERS_MAX <= 128);
    assert!(mem::size_of::<Members>() == constants::MEMBERS_MAX * mem::size_of::<u128>());
};

impl Members {
    #[inline]
    pub const fn zeroed() -> Self {
        Self([0u128; constants::MEMBERS_MAX])
    }

    #[inline]
    pub fn count(&self) -> u8 {
        member_count(self)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0[0] == 0
    }

    /// Iterates over active (non-zero) member IDs.
    #[inline]
    pub fn iter_used(&self) -> impl Iterator<Item = u128> + '_ {
        let n = member_count(self) as usize;
        assert!(n <= constants::MEMBERS_MAX);

        self.0[..n].iter().copied()
    }

    #[inline]
    pub fn get(&self, index: u8) -> Option<u128> {
        let idx = index as usize;
        if idx >= constants::MEMBERS_MAX {
            return None;
        }
        let val = self.0[idx];
        if val == 0 { None } else { Some(val) }
    }
}

impl core::ops::Deref for Members {
    type Target = [u128; constants::MEMBERS_MAX];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for Members {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Checks that members form a valid contiguous prefix of unique non-zero IDs.
pub fn valid_members(members: &Members) -> bool {
    for (i, &replica_i) in members.0.iter().enumerate() {
        for &replica_j in &members.0[..i] {
            if replica_j == 0 && replica_i != 0 {
                return false;
            }

            if replica_j != 0 && replica_i == replica_j {
                return false;
            }
        }
    }
    true
}

/// Counts active members (non-zero prefix length).
///
/// # Panics
/// Panics if members violate invariants (see [`valid_members`]).
fn member_count(members: &Members) -> u8 {
    assert!(valid_members(members));

    members
        .0
        .iter()
        .position(|&id| id == 0)
        .unwrap_or(constants::MEMBERS_MAX) as u8
}

/// Finds the index of a replica ID within the member list.
///
/// # Panics
/// Panics if `replica_id` is zero or members violate invariants.
pub fn member_index(members: &Members, replica_id: u128) -> Option<u8> {
    assert!(valid_members(members));
    assert!(replica_id > 0, "zero replica_id");

    members
        .0
        .iter()
        .position(|&id| id == replica_id)
        .map(|idx| idx as u8)
}

/// Generates deterministic member IDs for a cluster's initial configuration.
///
/// Each member ID is derived from: `checksum(config_checksum || cluster_id || replica_index)`.
/// This ensures identical clusters with identical configs produce identical member sets.
pub fn root_members(cluster: u128) -> Members {
    let cluster_config_checksum = cluster_config_checksum();

    let mut result = Members::zeroed();
    for (replica_id, slot) in result.0.iter_mut().enumerate() {
        let replica_u8 = replica_id as u8;

        let mut seed = [0u8; ID_SEED_SIZE];
        seed[0..16].copy_from_slice(&cluster_config_checksum.to_le_bytes());
        seed[16..32].copy_from_slice(&cluster.to_le_bytes());
        seed[32] = replica_u8;

        *slot = checksum(&seed);
    }

    assert!(valid_members(&result));
    assert_eq!(member_count(&result), constants::MEMBERS_MAX as u8);

    result
}

/// Checksums cluster configuration constants to detect incompatible builds.
///
/// Nodes with different config checksums cannot join the same cluster.
fn cluster_config_checksum() -> u128 {
    let fields: [u64; CLUSTER_CONFIG_FIELD_COUNT] = [
        constants::CACHE_LINE_SIZE,
        constants::CLIENTS_MAX as u64,
        constants::PIPELINE_PREPARE_QUEUE_MAX as u64,
        constants::VIEW_CHANGE_HEADERS_SUFFIX_MAX as u64,
        constants::QUORUM_REPLICATION_MAX,
        constants::JOURNAL_SLOT_COUNT as u64,
        constants::MESSAGE_SIZE_MAX as u64,
        constants::SUPERBLOCK_COPIES as u64,
        constants::BLOCK_SIZE,
        constants::LSM_LEVELS,
        constants::LSM_GROWTH_FACTOR,
        constants::LSM_COMPACTION_OPS,
        constants::LSM_SNAPSHOTS_MAX,
        constants::LSM_MANIFEST_COMPACT_EXTRA_BLOCKS,
        constants::LSM_TABLE_COALESCING_THRESHOLD_PERCENT,
        constants::VSR_RELEASES_MAX,
        constants::LSM_SCANS_MAX,
    ];

    // Build byte buffer
    let mut bytes = [0u8; CLUSTER_CONFIG_SEED_SIZE];

    for (i, &v) in fields.iter().enumerate() {
        let start = i * 8;
        let end = start + 8;

        bytes[start..end].copy_from_slice(&v.to_le_bytes());
    }

    assert_eq!(CLUSTER_CONFIG_FIELD_COUNT * 8, bytes.len());

    checksum(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn members_zeroed() {
        let m = Members::zeroed();

        assert!(valid_members(&m));
        assert_eq!(member_count(&m), 0);
        assert!(m.is_empty());
    }

    #[test]
    fn members_valid_partial() {
        let mut m = Members::zeroed();
        m.0[0] = 100;
        m.0[1] = 200;
        m.0[2] = 300;
        // Rest are zeros

        assert!(valid_members(&m));
        assert_eq!(member_count(&m), 3);
        assert!(!m.is_empty());
    }

    #[test]
    fn members_invalid_hole() {
        let mut m = Members::zeroed();
        m.0[0] = 100;
        m.0[1] = 0; // hole
        m.0[2] = 300;

        assert!(!valid_members(&m));
    }

    #[test]
    fn members_invalid_duplicate() {
        let mut m = Members::zeroed();
        m.0[0] = 100;
        m.0[1] = 200;
        m.0[2] = 100; // duplicate

        assert!(!valid_members(&m));
    }

    #[test]
    fn member_index_found() {
        let mut m = Members::zeroed();
        m.0[0] = 100;
        m.0[1] = 200;
        m.0[2] = 300;

        assert_eq!(member_index(&m, 100), Some(0));
        assert_eq!(member_index(&m, 200), Some(1));
        assert_eq!(member_index(&m, 300), Some(2));
    }

    #[test]
    fn member_index_not_found() {
        let mut m = Members::zeroed();
        m.0[0] = 100;
        m.0[1] = 200;

        assert_eq!(member_index(&m, 999), None);
    }

    #[test]
    #[should_panic(expected = "zero replica_id")]
    fn member_index_zero_panics() {
        let m = Members::zeroed();
        member_index(&m, 0);
    }

    #[test]
    fn root_members_deterministic() {
        let cluster = 0x123456789ABCDEF0;

        let m1 = root_members(cluster);
        let m2 = root_members(cluster);

        // Same cluster should produce same members
        assert_eq!(m1, m2);
        assert!(valid_members(&m1));

        // All slots should be filled
        assert_eq!(member_count(&m1), constants::MEMBERS_MAX as u8);

        // All IDs should be unique and non-zero
        for i in 0..constants::MEMBERS_MAX {
            assert!(m1.0[i] != 0);
            for j in 0..i {
                assert!(m1.0[i] != m1.0[j]);
            }
        }
    }

    #[test]
    fn root_members_different_clusters() {
        let m1 = root_members(1);
        let m2 = root_members(2);

        // Different clusters should produce different members
        assert_ne!(m1, m2);
    }

    #[test]
    fn members_iter_used() {
        let mut m = Members::zeroed();
        m.0[0] = 100;
        m.0[1] = 200;
        m.0[2] = 300;

        let collected: Vec<u128> = m.iter_used().collect();
        assert_eq!(collected, vec![100, 200, 300]);
    }

    #[test]
    fn members_get() {
        let mut m = Members::zeroed();
        m.0[0] = 100;
        m.0[1] = 200;

        assert_eq!(m.get(0), Some(100));
        assert_eq!(m.get(1), Some(200));
        assert_eq!(m.get(2), None); // zero
        assert_eq!(m.get(255), None); // out of bounds
    }

    #[test]
    #[should_panic]
    fn members_count_invalid_panics() {
        let mut m = Members::zeroed();
        m.0[0] = 1;
        m.0[1] = 0; // hole
        m.0[2] = 2;

        let _ = m.count();
    }

    #[test]
    fn members_full_valid() {
        let mut m = Members::zeroed();
        for i in 0..constants::MEMBERS_MAX {
            m.0[i] = (i + 1) as u128;
        }

        assert!(valid_members(&m));
        assert_eq!(m.count() as usize, constants::MEMBERS_MAX);
        assert!(!m.is_empty());
    }

    #[test]
    fn members_full_get() {
        let mut m = Members::zeroed();
        for i in 0..constants::MEMBERS_MAX {
            m.0[i] = (i + 1) as u128;
        }

        for i in 0..constants::MEMBERS_MAX {
            assert_eq!(m.get(i as u8), Some((i + 1) as u128));
        }
        assert_eq!(m.get(constants::MEMBERS_MAX as u8), None);
    }
}
