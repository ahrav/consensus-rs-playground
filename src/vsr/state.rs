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
    util::{Pod, as_bytes, equal_bytes},
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
    assert!(HeaderPrepare::SIZE.is_multiple_of(8));
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

// SAFETY: CheckpointState is repr(C) with fixed 1024-byte layout.
// All fields are primitives (u8, u32, u64, u128), arrays thereof, or types
// that implement Pod (HeaderPrepare, Release, Checksum128 which is u128).
// The struct has no implicit padding bytes due to careful field ordering
// and explicit padding fields.
unsafe impl Pod for CheckpointState {}

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

/// Durable replica state persisted to the superblock.
///
/// Combines [`CheckpointState`] with replica identity and protocol progress fields.
/// Written atomically during checkpoints; all fields must advance monotonically
/// (see [`Self::monotonic`]).
///
/// # Invariants
///
/// - `replica_id` must exist in `members`
/// - `commit_max >= checkpoint.header.op`
/// - `view >= log_view`
/// - `sync_op_max >= sync_op_min`
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VsrState {
    pub checkpoint: CheckpointState,

    pub replica_id: u128,
    pub members: Members,

    /// Highest committed operation number known to this replica.
    pub commit_max: u64,
    /// State sync range: minimum op being synchronized.
    pub sync_op_min: u64,
    /// State sync range: maximum op being synchronized.
    pub sync_op_max: u64,

    /// View number during state sync.
    pub sync_view: u32,

    /// View in which this replica last participated in the log.
    /// Lags behind `view` during view changes until the replica
    /// receives the new log suffix.
    pub log_view: u32,
    /// Current view number.
    pub view: u32,

    pub replica_count: u8,
    pub reserved: [u8; 779],
}

// SAFETY: VsrState is repr(C) with all fields being Pod types:
// - CheckpointState (Pod), u128, Members (Pod), u64s, u32s, u8, and [u8; 779].
// The struct has no implicit padding bytes due to careful field ordering.
unsafe impl Pod for VsrState {}

impl VsrState {
    /// Returns a zero-initialized state for use as a comparison baseline.
    #[inline]
    pub fn zeroed() -> Self {
        let state: Self = unsafe { mem::zeroed() };

        assert!(state.replica_id == 0);
        assert!(state.commit_max == 0);
        assert!(state.view == 0);
        assert!(state.replica_count == 0);

        state
    }

    /// Creates the initial state for a new replica joining the cluster.
    ///
    /// Initializes the checkpoint with the root prepare header and empty
    /// checksums for all tracked data structures.
    ///
    /// # Panics
    ///
    /// Panics if `opts` fails validation (see [`RootOptions::validate`]).
    pub fn root(opts: &RootOptions) -> Self {
        opts.validate();

        let replica_id = opts.replica_id();
        assert!(replica_id != 0);

        let checksum_empty = checksum(&[]);
        assert!(checksum_empty != 0);

        let checkpoint = CheckpointState {
            header: HeaderPrepare::root(opts.cluster),

            free_set_blocks_acquired_last_block_checksum: 0,
            free_set_blocks_acquired_last_block_checksum_padding: 0,

            free_set_blocks_released_last_block_checksum: 0,
            free_set_blocks_released_last_block_checksum_padding: 0,

            client_sessions_last_block_checksum: 0,
            client_sessions_last_block_checksum_padding: 0,

            free_set_blocks_acquired_last_block_address: 0,
            free_set_blocks_released_last_block_address: 0,
            client_sessions_last_block_address: 0,

            free_set_blocks_acquired_size: 0,
            free_set_blocks_released_size: 0,
            client_sessions_size: 0,

            free_set_blocks_acquired_checksum: checksum_empty,
            free_set_blocks_released_checksum: checksum_empty,
            client_sessions_checksum: checksum_empty,

            parent_checkpoint_id: 0,
            grandparent_checkpoint_id: 0,

            manifest_oldest_checksum: 0,
            manifest_oldest_checksum_padding: 0,
            manifest_oldest_address: 0,

            manifest_newest_checksum: 0,
            manifest_newest_checksum_padding: 0,
            manifest_newest_address: 0,

            manifest_block_count: 0,
            reserved_manifest: [0u8; 4],

            snapshots_block_checksum: 0,
            snapshots_block_checksum_padding: 0,
            snapshots_block_address: 0,

            storage_size: constants::DATA_FILE_SIZE_MIN,
            release: opts.release,

            reserved: [0u8; 388],
        };

        assert!(checkpoint.storage_size >= constants::DATA_FILE_SIZE_MIN);

        let state = VsrState {
            checkpoint,
            replica_id,
            members: opts.members,

            commit_max: 0,
            sync_op_min: 0,
            sync_op_max: 0,

            sync_view: 0,
            log_view: 0,
            view: opts.view,

            replica_count: opts.replica_count,
            reserved: [0u8; 779],
        };

        // state.assert_internally_consistent();

        assert!(state.replica_id == replica_id);
        assert!(member_index(&state.members, state.replica_id).is_some());

        state
    }

    /// Panics if any internal invariants are violated.
    ///
    /// Validates replica membership, field ordering constraints, checkpoint
    /// padding, and consistency of free set, client sessions, and manifest state.
    pub fn assert_internally_consistent(&self) {
        assert!(member_index(&self.members, self.replica_id).is_some());
        assert!(self.commit_max >= self.checkpoint.header.op);
        assert!(self.sync_op_max >= self.sync_op_min);
        assert!(self.view >= self.log_view);
        assert!(self.replica_count > 0);
        assert!((self.replica_count as usize) <= constants::REPLICAS_MAX);

        self.checkpoint.assert_padding_zeroed();
        assert!(self.checkpoint.snapshots_block_checksum == 0);
        assert!(self.checkpoint.snapshots_block_address == 0);

        assert!(self.checkpoint.storage_size >= constants::DATA_FILE_SIZE_MIN);

        let checksum_empty = checksum(&[]);
        self.assert_free_set_acquired_consistent(checksum_empty);
        self.assert_free_set_released_consistent(checksum_empty);
        self.assert_client_sessions_consistent(checksum_empty);
        self.assert_manifest_consistent();
    }

    #[inline]
    fn assert_free_set_acquired_consistent(&self, checksum_empty: u128) {
        let addr = self.checkpoint.free_set_blocks_acquired_last_block_address;
        let size = self.checkpoint.free_set_blocks_acquired_size;
        let checksum = self.checkpoint.free_set_blocks_acquired_checksum;
        let block_checksum = self.checkpoint.free_set_blocks_acquired_last_block_checksum;

        if addr == 0 {
            assert!(size == 0, "free_set_acquired: addr=0 but size={}", size);
            assert!(
                checksum == checksum_empty,
                "free_set_acquired: addr=0 but checksum != empty"
            );
            assert!(
                block_checksum == 0,
                "free_set_acquired: addr=0 but block_checksum != 0"
            );
        } else {
            assert!(size > 0, "free_set_acquired: addr={} but size=0", addr);
        }
    }

    /// Asserts free_set_blocks_released consistency.
    #[inline]
    fn assert_free_set_released_consistent(&self, checksum_empty: u128) {
        let addr = self.checkpoint.free_set_blocks_released_last_block_address;
        let size = self.checkpoint.free_set_blocks_released_size;
        let checksum = self.checkpoint.free_set_blocks_released_checksum;
        let block_checksum = self.checkpoint.free_set_blocks_released_last_block_checksum;

        if addr == 0 {
            assert!(size == 0, "free_set_released: addr=0 but size={}", size);
            assert!(
                checksum == checksum_empty,
                "free_set_released: addr=0 but checksum != empty"
            );
            assert!(
                block_checksum == 0,
                "free_set_released: addr=0 but block_checksum != 0"
            );
        } else {
            assert!(size > 0, "free_set_released: addr={} but size=0", addr);
        }
    }

    /// Asserts client_sessions consistency.
    #[inline]
    fn assert_client_sessions_consistent(&self, checksum_empty: u128) {
        let addr = self.checkpoint.client_sessions_last_block_address;
        let block_checksum = self.checkpoint.client_sessions_last_block_checksum;
        let size = self.checkpoint.client_sessions_size;
        let checksum = self.checkpoint.client_sessions_checksum;

        if addr == 0 {
            assert!(
                block_checksum == 0,
                "client_sessions: addr=0 but block_checksum != 0"
            );
            assert!(size == 0, "client_sessions: addr=0 but size={}", size);
            assert!(
                checksum == checksum_empty,
                "client_sessions: addr=0 but checksum != empty"
            );
        } else {
            let expected_size = client_sessions_encode_size();
            assert!(
                size == expected_size,
                "client_sessions: size {} != expected {}",
                size,
                expected_size
            );
        }
    }

    /// Asserts manifest consistency.
    #[inline]
    fn assert_manifest_consistent(&self) {
        let count = self.checkpoint.manifest_block_count;
        let oldest_addr = self.checkpoint.manifest_oldest_address;
        let oldest_checksum = self.checkpoint.manifest_oldest_checksum;
        let newest_addr = self.checkpoint.manifest_newest_address;
        let newest_checksum = self.checkpoint.manifest_newest_checksum;

        if count == 0 {
            assert!(
                oldest_addr == 0,
                "manifest: count=0 but oldest_addr={}",
                oldest_addr
            );
            assert!(
                oldest_checksum == 0,
                "manifest: count=0 but oldest_checksum != 0"
            );
            assert!(
                newest_addr == 0,
                "manifest: count=0 but newest_addr={}",
                newest_addr
            );
            assert!(
                newest_checksum == 0,
                "manifest: count=0 but newest_checksum != 0"
            );
        } else {
            // Non-zero manifest
            assert!(
                oldest_addr != 0,
                "manifest: count={} but oldest_addr=0",
                count
            );
            assert!(
                oldest_checksum != 0,
                "manifest: count={} but oldest_checksum=0",
                count
            );
            assert!(
                newest_addr != 0,
                "manifest: count={} but newest_addr=0",
                count
            );
            assert!(
                newest_checksum != 0,
                "manifest: count={} but newest_checksum=0",
                count
            );

            // Single block case
            if count == 1 {
                assert!(
                    oldest_addr == newest_addr,
                    "manifest: count=1 but oldest_addr {} != newest_addr {}",
                    oldest_addr,
                    newest_addr
                );
                assert!(
                    oldest_checksum == newest_checksum,
                    "manifest: count=1 but checksums differ"
                );
            }

            // Address ordering
            assert!(
                oldest_addr <= newest_addr,
                "manifest: oldest_addr {} > newest_addr {}",
                oldest_addr,
                newest_addr
            );

            // Block alignment
            let block = constants::BLOCK_SIZE;
            assert!(block > 0, "BLOCK_SIZE is zero");
            assert!(
                oldest_addr.is_multiple_of(block),
                "manifest: oldest_addr {} not aligned to block size {}",
                oldest_addr,
                block
            );
            assert!(
                newest_addr.is_multiple_of(block),
                "manifest: newest_addr {} not aligned to block size {}",
                newest_addr,
                block
            );
        }
    }

    /// Returns `true` if `new` represents valid forward progress from `old`.
    ///
    /// Monotonicity ensures state never regresses: checkpoint op, commit_max,
    /// sync bounds, and view numbers must all be non-decreasing. States with
    /// the same checkpoint op must be byte-identical (no silent overwrites).
    ///
    /// # Panics
    ///
    /// Panics if either state is internally inconsistent, or if replica
    /// identity fields (`replica_id`, `replica_count`, `members`) differ.
    pub fn monotonic(old: &Self, new: &Self) -> bool {
        old.assert_internally_consistent();
        new.assert_internally_consistent();

        assert!(old.replica_id == new.replica_id);
        assert!(old.replica_count == new.replica_count);
        assert!(old.members == new.members);

        if old.checkpoint.header.op == new.checkpoint.header.op {
            if old.checkpoint.header.checksum == 0 && old.checkpoint.header.op == 0 {
                assert!(old.commit_max == 0);
                assert!(old.sync_op_min == 0);
                assert!(old.sync_op_max == 0);
                assert!(old.log_view == 0);
                assert!(old.view == 0);
            } else {
                // SAFETY: CheckpointState implements Pod, guaranteeing no padding bytes
                // and all bytes initialized. The references are distinct (old != new).
                assert!(unsafe { equal_bytes(&old.checkpoint, &new.checkpoint) });
            }
        } else {
            assert!(old.checkpoint.header.checksum != new.checkpoint.header.checksum);
            assert!(old.checkpoint.parent_checkpoint_id != new.checkpoint.parent_checkpoint_id);
        }

        if old.checkpoint.header.op > new.checkpoint.header.op {
            return false;
        }
        if old.commit_max > new.commit_max {
            return false;
        }
        if old.sync_op_min > new.sync_op_min {
            return false;
        }
        if old.sync_op_max > new.sync_op_max {
            return false;
        }
        if old.log_view > new.log_view {
            return false;
        }
        if old.view > new.view {
            return false;
        }

        true
    }

    /// Returns `true` if persisting `new` would change the on-disk state.
    ///
    /// Use this to avoid redundant writes when the state hasn't changed.
    ///
    /// # Panics
    ///
    /// Panics if `new` is not monotonic with respect to `old`.
    pub fn would_be_updated_by(old: &Self, new: &Self) -> bool {
        assert!(Self::monotonic(old, new));

        // SAFETY: VsrState implements Pod, guaranteeing no padding bytes
        // and all bytes initialized.
        unsafe { !equal_bytes(old, new) }
    }

    /// Returns `true` if `op` has been compacted (removed from the journal).
    ///
    /// Operations at or before the checkpoint trigger are no longer available
    /// in the journal and must be recovered via state sync if needed.
    #[inline]
    pub fn op_compacted(&self, op: u64) -> bool {
        let checkpoint_op = self.checkpoint.header.op;
        if checkpoint_op == 0 {
            return false;
        }

        let trigger = trigger_for_checkpoint(checkpoint_op)
            .expect("checkpoint_op > 0 but trigger_for_checkpoint return None");
        op <= trigger
    }
}

/// Computes the encoded size of client sessions data for checkpointing.
#[inline]
fn client_sessions_encode_size() -> u64 {
    const { assert!(HeaderPrepare::SIZE > 0) };
    const { assert!(constants::CLIENTS_MAX > 0) };

    let header_bytes = (HeaderPrepare::SIZE as u64)
        .checked_mul(constants::CLIENTS_MAX as u64)
        .expect("header_bytes overflow");

    let aligned = (header_bytes + 7) & !7;
    assert!(aligned >= header_bytes);
    assert!(aligned.is_multiple_of(8));

    let session_bytes = 8u64
        .checked_mul(constants::CLIENTS_MAX as u64)
        .expect("session_bytes overflow");

    let total_bytes = aligned
        .checked_add(session_bytes)
        .expect("total_bytes overflow");
    assert!(total_bytes > 0);
    assert!(total_bytes >= aligned);

    total_bytes
}

/// Returns the op number that triggered a given checkpoint.
///
/// Returns `None` for checkpoint 0 (the root checkpoint has no trigger).
#[inline]
fn trigger_for_checkpoint(checkpoint: u64) -> Option<u64> {
    if checkpoint == 0 {
        return None;
    }

    let trigger = checkpoint
        .checked_add(constants::LSM_COMPACTION_OPS)
        .expect("trigger_for_checkpoint overflow");
    assert!(trigger > checkpoint);
    Some(trigger)
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

    #[test]
    fn test_vsr_state_zeroed() {
        let state = VsrState::zeroed();
        assert_eq!(state.replica_id, 0);
        assert_eq!(state.commit_max, 0);
        assert_eq!(state.view, 0);
        assert_eq!(state.replica_count, 0);
    }

    #[test]
    fn test_vsr_state_root() {
        let mut members = Members([0; constants::MEMBERS_MAX]);
        members.0[0] = 1;
        members.0[1] = 2;

        let options = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 2,
            members,
            release: Release::ZERO,
            view: 10,
        };

        let state = VsrState::root(&options);

        assert_eq!(state.replica_id, 1);
        assert_eq!(state.replica_count, 2);
        assert_eq!(state.view, 10);
        assert_eq!(state.log_view, 0);
        assert_eq!(state.checkpoint.header.op, 0);

        state.assert_internally_consistent();
    }

    #[test]
    fn test_vsr_state_monotonic_updates() {
        let mut members = Members([0; constants::MEMBERS_MAX]);
        members.0[0] = 1;

        let options = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 1,
            members,
            release: Release::ZERO,
            view: 0,
        };

        let mut old = VsrState::root(&options);
        let mut new = old;

        // No change
        assert!(VsrState::monotonic(&old, &new));
        assert!(!VsrState::would_be_updated_by(&old, &new));

        // Valid update: increase view
        new.view += 1;
        assert!(VsrState::monotonic(&old, &new));
        assert!(VsrState::would_be_updated_by(&old, &new));

        // Invalid update: decrease view
        let invalid = old;
        old.view = 2; // Move old ahead
        assert!(!VsrState::monotonic(&old, &invalid));
    }

    #[test]
    fn test_vsr_state_op_compacted() {
        let mut members = Members([0; constants::MEMBERS_MAX]);
        members.0[0] = 1;

        let options = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 1,
            members,
            release: Release::ZERO,
            view: 0,
        };

        let mut state = VsrState::root(&options);

        // Initial state op=0, so no compaction active
        assert!(!state.op_compacted(0));
        assert!(!state.op_compacted(100));

        // Simulate a checkpoint
        state.checkpoint.header.op = 100;
        let trigger = 100 + constants::LSM_COMPACTION_OPS;

        assert!(state.op_compacted(100));
        assert!(state.op_compacted(trigger));
        assert!(!state.op_compacted(trigger + 1));
    }
}
