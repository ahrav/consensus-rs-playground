//! VSR replica state types for checkpoint persistence and initialization.
//!
//! This module defines the core state structures that replicas use for:
//! - Initializing a new replica with [`RootOptions`]
//! - Persisting checkpoint state to durable storage with [`CheckpointState`]

#[allow(unused_imports)]
use core::{mem, slice};

#[allow(unused_imports)]
use crate::{
    constants,
    util::{as_bytes, equal_bytes},
    vsr::{
        HeaderPrepare, Members, member_index, valid_members,
        wire::{Checksum128, checksum, header::Release},
    },
};

// Compile-time layout and constant validation.
const _: () = {
    assert!(mem::size_of::<CheckpointState>() == 1024);
    assert!(mem::align_of::<CheckpointState>() >= 8);

    assert!(constants::REPLICAS_MAX > 0);
    assert!(constants::REPLICAS_MAX <= 128);
    assert!(constants::CLIENTS_MAX > 0);
    assert!(constants::BLOCK_SIZE > 0);
    assert!(constants::BLOCK_SIZE.is_power_of_two());
    assert!(constants::LSM_COMPACTION_OPS > 0);

    assert!(HeaderPrepare::SIZE == 256);
    assert!(HeaderPrepare::SIZE % 8 == 0);
};

/// Configuration for initializing a replica in the cluster.
///
/// Validated via [`validate`](Self::validate) to ensure the replica can
/// participate correctly in the consensus protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootOptions {
    pub cluster: u128,
    pub replica_index: u8,
    pub replica_count: u8,
    pub members: Members,
    pub release: Release,
    pub view: u32,
}

impl RootOptions {
    /// Asserts all invariants required for a valid replica configuration.
    ///
    /// # Panics
    /// - `replica_count` is zero or exceeds [`constants::REPLICAS_MAX`]
    /// - `replica_index` is out of bounds
    /// - `members` contains invalid entries
    /// - This replica's member ID is zero (unassigned)
    #[inline]
    pub fn validate(&self) {
        assert!(self.replica_count > 0);
        assert!((self.replica_count as usize) <= constants::REPLICAS_MAX);
        assert!((self.replica_index as usize) < (self.replica_count as usize));
        assert!(valid_members(&self.members));

        let replica_id = self.members.0[self.replica_index as usize];
        assert!(replica_id != 0);
    }

    /// Returns this replica's unique identifier from the members array.
    ///
    /// # Panics
    /// If `replica_index` is out of bounds or the member ID is zero.
    #[inline]
    pub fn replica_id(&self) -> u128 {
        assert!((self.replica_index as usize) < self.members.0.len());
        let id = self.members.0[self.replica_index as usize];
        assert!(id != 0);
        id
    }
}

/// Durable checkpoint state persisted to storage.
///
/// Fixed 1024-byte `#[repr(C)]` layout for direct memory-mapped I/O.
/// Checksum padding fields ensure 16-byte alignment for AEGIS-128L operations.
///
/// # Wire Layout
/// The structure groups related data for cache efficiency:
/// - Header (256 bytes): The prepare message that triggered this checkpoint
/// - Free set tracking: Acquired/released block checksums, addresses, and sizes
/// - Client sessions: Session state for exactly-once semantics
/// - Manifest: LSM tree manifest block chain (oldest → newest)
/// - Snapshots: Point-in-time snapshot block reference
/// - Reserved: Future protocol extensions (must be zero)
///
/// # Checkpoint Chain
/// `parent_checkpoint_id` and `grandparent_checkpoint_id` form a hash chain
/// enabling crash recovery by walking back to a consistent state.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CheckpointState {
    /// The prepare message header that triggered this checkpoint.
    pub header: HeaderPrepare,

    // --- Free set: acquired blocks ---
    pub free_set_blocks_acquired_last_block_checksum: Checksum128,
    pub free_set_blocks_acquired_last_block_checksum_padding: Checksum128,

    // --- Free set: released blocks ---
    pub free_set_blocks_released_last_block_checksum: Checksum128,
    pub free_set_blocks_released_last_block_checksum_padding: Checksum128,

    // --- Client sessions ---
    pub client_sessions_last_block_checksum: Checksum128,
    pub client_sessions_last_block_checksum_padding: Checksum128,

    pub free_set_blocks_acquired_last_block_address: u64,
    pub free_set_blocks_released_last_block_address: u64,
    pub client_sessions_last_block_address: u64,

    pub free_set_blocks_acquired_size: u64,
    pub free_set_blocks_released_size: u64,
    pub client_sessions_size: u64,

    /// Rolling checksum over all acquired block data.
    pub free_set_blocks_acquired_checksum: Checksum128,
    /// Rolling checksum over all released block data.
    pub free_set_blocks_released_checksum: Checksum128,
    /// Rolling checksum over all client session data.
    pub client_sessions_checksum: Checksum128,

    /// Forms a hash chain with `grandparent_checkpoint_id` for crash recovery.
    pub parent_checkpoint_id: u128,
    pub grandparent_checkpoint_id: u128,

    // --- Manifest block chain ---
    pub manifest_oldest_checksum: Checksum128,
    pub manifest_oldest_checksum_padding: Checksum128,
    pub manifest_oldest_address: u64,

    pub manifest_newest_checksum: Checksum128,
    pub manifest_newest_checksum_padding: Checksum128,
    pub manifest_newest_address: u64,

    pub manifest_block_count: u32,
    pub reserved_manifest: [u8; 4],

    // --- Snapshots ---
    pub snapshots_block_checksum: Checksum128,
    pub snapshots_block_checksum_padding: Checksum128,
    pub snapshots_block_address: u64,

    pub storage_size: u64,

    /// Detects unclean shutdown when reopening the data file.
    pub release: Release,

    /// Reserved for future protocol versions. Must be zero.
    pub reserved: [u8; 388],
}

impl CheckpointState {
    /// Creates a zero-initialized checkpoint state.
    ///
    /// # Safety
    /// Uses `mem::zeroed()` which is safe for this `#[repr(C)]` struct containing
    /// only primitive types and arrays. Asserts verify critical fields are zero.
    #[inline]
    pub fn zeroed() -> Self {
        // SAFETY: CheckpointState is repr(C) with only primitive/array fields.
        // All-zeros is a valid bit pattern for every field type.
        let state: Self = unsafe { mem::zeroed() };

        // Sanity checks: if these fail, memory initialization is broken.
        assert!(state.header.op == 0, "zeroed header.op not zero");
        assert!(state.storage_size == 0, "zeroed storage_size not zero");
        assert!(
            state.manifest_block_count == 0,
            "zeroed manifest_block_count not zero"
        );

        assert!(state.free_set_blocks_acquired_last_block_checksum_padding == 0);
        assert!(state.free_set_blocks_released_last_block_checksum_padding == 0);
        assert!(state.client_sessions_last_block_checksum_padding == 0);
        assert!(state.manifest_oldest_checksum_padding == 0);
        assert!(state.manifest_newest_checksum_padding == 0);
        assert!(state.snapshots_block_checksum_padding == 0);

        state
    }

    /// Asserts all padding fields are zero.
    ///
    /// Call after deserializing from storage or network to detect corruption
    /// or protocol violations. Non-zero padding indicates invalid state.
    #[inline]
    pub fn assert_padding_zeroed(&self) {
        assert!(
            self.free_set_blocks_acquired_last_block_checksum_padding == 0,
            "free_set_blocks_acquired padding non-zero"
        );
        assert!(
            self.free_set_blocks_released_last_block_checksum_padding == 0,
            "free_set_blocks_released padding non-zero"
        );
        assert!(
            self.client_sessions_last_block_checksum_padding == 0,
            "client_sessions padding non-zero"
        );
        assert!(
            self.manifest_oldest_checksum_padding == 0,
            "manifest_oldest padding non-zero"
        );
        assert!(
            self.manifest_newest_checksum_padding == 0,
            "manifest_newest padding non-zero"
        );
        assert!(
            self.snapshots_block_checksum_padding == 0,
            "snapshots_block padding non-zero"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsr::wire::header::Release;

    #[test]
    fn test_root_options_validate() {
        let mut members = Members([0; constants::MEMBERS_MAX]);
        members.0[0] = 1;
        members.0[1] = 2;
        members.0[2] = 3;

        let options = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 3,
            members,
            release: Release::ZERO,
            view: 0,
        };

        options.validate();
        assert_eq!(options.replica_id(), 1);
    }

    #[test]
    #[should_panic]
    fn test_root_options_invalid_replica_count() {
        let mut members = Members([0; constants::MEMBERS_MAX]);
        members.0[0] = 1;

        let options = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 0, // Invalid
            members,
            release: Release::ZERO,
            view: 0,
        };
        options.validate();
    }

    #[test]
    #[should_panic]
    fn test_root_options_invalid_replica_index() {
        let mut members = Members([0; constants::MEMBERS_MAX]);
        members.0[0] = 1;

        let options = RootOptions {
            cluster: 1,
            replica_index: 1, // Invalid, >= replica_count (1)
            replica_count: 1,
            members,
            release: Release::ZERO,
            view: 0,
        };
        options.validate();
    }

    #[test]
    fn test_checkpoint_state_layout() {
        assert_eq!(mem::size_of::<CheckpointState>(), 1024);
        assert!(mem::align_of::<CheckpointState>() >= 8);
    }

    #[test]
    fn test_checkpoint_state_zeroed() {
        let state = CheckpointState::zeroed();
        state.assert_padding_zeroed();
        assert_eq!(state.storage_size, 0);
        assert_eq!(state.header.op, 0);
    }

    #[test]
    #[should_panic(expected = "padding non-zero")]
    fn test_checkpoint_state_padding_check() {
        let mut state = CheckpointState::zeroed();
        state.free_set_blocks_acquired_last_block_checksum_padding = 1;
        state.assert_padding_zeroed();
    }
}
