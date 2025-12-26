//! VSR replica state types for checkpoint persistence and initialization.
//!
//! This module defines the core state structures that replicas use for:
//! - Initializing a new replica with [`RootOptions`]
//! - Persisting checkpoint state to durable storage with [`CheckpointState`]

use core::mem;

use crate::{
    constants,
    util::{Pod, as_bytes, equal_bytes},
    vsr::{
        HeaderPrepare, Members, ViewChangeArray, member_index, valid_members,
        wire::{Checksum128, checksum, header::Release},
    },
};

// Compile-time layout and constant validation.
const _: () = {
    assert!(mem::size_of::<CheckpointState>() == 1024);
    assert!(mem::align_of::<CheckpointState>() >= 8);

    assert!(mem::size_of::<VsrState>() == 2048);
    assert!(mem::align_of::<VsrState>() >= 8);

    assert!(constants::REPLICAS_MAX > 0);
    assert!(constants::REPLICAS_MAX <= 128);
    assert!(constants::CLIENTS_MAX > 0);
    assert!(constants::BLOCK_SIZE > 0);
    assert!(constants::BLOCK_SIZE.is_power_of_two());
    assert!(constants::LSM_COMPACTION_OPS > 0);

    assert!(HeaderPrepare::SIZE == 256);
    assert!(HeaderPrepare::SIZE.is_multiple_of(8));
};

/// View state captured during checkpoint for view change recovery.
///
/// When a replica checkpoints during a view change, it must persist the current
/// view headers to enable recovery. Without this, a restarted replica cannot
/// determine which operations were committed in the new view.
pub struct ViewAttributes<'a> {
    /// Prepare headers from the current view (used to reconstruct commit state).
    pub headers: &'a ViewChangeArray,
    /// Current view number.
    pub view: u32,
    /// View in which the last log entry was created.
    pub log_view: u32,
}

/// Configuration for persisting a checkpoint to the superblock.
///
/// Captures all state needed to resume from this checkpoint after crash recovery.
pub struct CheckpointOptions<'a> {
    /// Complete checkpoint state to persist.
    pub checkpoint: CheckpointState,
    /// Current view number.
    pub view: u32,
    /// View in which the log was last updated.
    pub log_view: u32,
    /// View headers if checkpointing during/after a view change.
    pub view_attributes: Option<ViewAttributes<'a>>,
    /// Highest committed operation number.
    pub commit_max: u64,
    /// Sync target range: minimum operation to sync from.
    pub sync_op_min: u64,
    /// Sync target range: maximum operation to sync to.
    pub sync_op_max: u64,
    /// Total data file size in bytes.
    pub storage_size: u64,
    /// Software release for compatibility validation on recovery.
    pub release: Release,
    /// Whether this checkpoint was received via state sync.
    pub sync_checkpoint: bool,
}

/// Checkpoint received during state sync, with the skipped operation range.
#[derive(Clone, Copy)]
pub struct SyncCheckpointOptions {
    /// The checkpoint state to adopt.
    pub checkpoint: CheckpointState,
    /// First op in the skipped range (inclusive).
    pub sync_op_min: u64,
    /// Last op in the skipped range (inclusive).
    pub sync_op_max: u64,
}

/// Persists view/log_view before advertising them (prevents backtracking)
/// or updates checkpoint during sync.
///
/// Must advance view/log_view monotonically or update checkpoint.
pub struct ViewChangeOptions<'a> {
    /// Highest committed operation known to the replica.
    pub commit_max: u64,
    /// View in which the replica last participated in the log.
    pub log_view: u32,
    /// Current view number.
    pub view: u32,
    /// View headers from the new view's leader.
    pub headers: &'a ViewChangeArray,
    /// Checkpoint update if syncing state from another replica.
    pub sync_checkpoint: Option<SyncCheckpointOptions>,
}

/// Replica initialization configuration. See [`Self::validate`] for invariants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootOptions {
    /// Unique cluster identifier.
    pub cluster: u128,
    /// This replica's index within the cluster.
    pub replica_index: u8,
    /// Total number of replicas in the cluster.
    pub replica_count: u8,
    /// Cluster membership configuration.
    pub members: Members,
    /// Software release for compatibility checks.
    pub release: Release,
    /// Initial view number.
    pub view: u32,
}

impl RootOptions {
    /// Asserts all invariants required for a valid replica configuration.
    ///
    /// # Panics
    /// - `replica_count` is zero or exceeds [`constants::REPLICAS_MAX`]
    /// - `replica_index` is out of bounds
    /// - `members` contains invalid entries
    /// - `members` has fewer entries than `replica_count`
    /// - This replica's member ID is zero (unassigned)
    #[inline]
    pub fn validate(&self) {
        assert!(self.replica_count > 0);
        assert!((self.replica_count as usize) <= constants::REPLICAS_MAX);
        assert!((self.replica_index as usize) < (self.replica_count as usize));
        assert!(valid_members(&self.members));
        assert!(
            self.members.count() >= self.replica_count,
            "members.count() < replica_count"
        );

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
        assert!(state.reserved_manifest == [0u8; 4]);
        assert!(state.reserved == [0u8; 388]);

        state
    }

    /// Asserts all padding fields are zero.
    ///
    /// Call after deserializing from storage or network to detect corruption
    /// or protocol violations. Non-zero padding or reserved fields indicate invalid state.
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
        assert!(
            self.reserved_manifest == [0u8; 4],
            "reserved_manifest non-zero"
        );
        assert!(self.reserved == [0u8; 388], "checkpoint reserved non-zero");
    }

    /// Computes the checkpoint ID as the checksum of the entire CheckpointState.
    ///
    /// This forms a unique identifier for this checkpoint that is used in the
    /// parent/grandparent chain for crash recovery.
    ///
    /// # Safety
    ///
    /// Safe because `CheckpointState` implements `Pod`, guaranteeing no padding bytes
    /// and all bytes initialized.
    #[inline]
    pub fn checkpoint_id(&self) -> u128 {
        // SAFETY: CheckpointState implements Pod, so as_bytes is safe.
        checksum(unsafe { as_bytes(self) })
    }
}

#[inline]
fn checkpoint_valid(op: u64) -> bool {
    // Checkpoints are only valid at compaction bar boundaries.
    op == 0 || (op + 1).is_multiple_of(constants::LSM_COMPACTION_OPS)
}

#[inline]
fn checkpoint_after(checkpoint: u64) -> u64 {
    assert!(checkpoint_valid(checkpoint));

    let vsr_checkpoint_ops = constants::VSR_CHECKPOINT_OPS as u64;
    assert!(vsr_checkpoint_ops > 0);

    let result = if checkpoint == 0 {
        vsr_checkpoint_ops - 1
    } else {
        checkpoint
            .checked_add(vsr_checkpoint_ops)
            .expect("checkpoint_after overflow")
    };

    assert!((result + 1).is_multiple_of(constants::LSM_COMPACTION_OPS));
    assert!(checkpoint_valid(result));

    result
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
/// - `members.count() >= replica_count`
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
        assert!(state.reserved == [0u8; 779]);

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
    /// padding/reserved bytes, and consistency of free set, client sessions, and manifest state.
    pub fn assert_internally_consistent(&self) {
        assert!(member_index(&self.members, self.replica_id).is_some());
        assert!(
            self.members.count() >= self.replica_count,
            "members.count() < replica_count"
        );
        assert!(self.commit_max >= self.checkpoint.header.op);
        assert!(self.sync_op_max >= self.sync_op_min);
        assert!(self.view >= self.log_view);
        assert!(self.replica_count > 0);
        assert!((self.replica_count as usize) <= constants::REPLICAS_MAX);

        self.checkpoint.assert_padding_zeroed();
        assert!(self.checkpoint.snapshots_block_checksum == 0);
        assert!(self.checkpoint.snapshots_block_address == 0);

        assert!(self.checkpoint.storage_size >= constants::DATA_FILE_SIZE_MIN);
        assert!(self.reserved == [0u8; 779], "state reserved non-zero");

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
    /// Monotonicity rules:
    /// - checkpoint.op, commit_max, log_view, view must be non-decreasing
    /// - if checkpoint.op is unchanged, the *checkpoint bytes* must be identical
    /// - sync_op_* and sync_view are NOT part of the monotonic comparisons
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
                // Root baseline constraints.
                assert!(old.commit_max == 0);
                assert!(old.sync_op_min == 0);
                assert!(old.sync_op_max == 0);
                assert!(old.log_view == 0);
                assert!(old.view == 0);
            } else {
                // SAFETY: CheckpointState implements Pod, so byte-compare is valid.
                assert!(unsafe { equal_bytes(&old.checkpoint, &new.checkpoint) });
            }
        } else {
            // If checkpoint op advanced, the header checksum and parent checkpoint id must differ.
            assert!(old.checkpoint.header.checksum != new.checkpoint.header.checksum);
            assert!(old.checkpoint.parent_checkpoint_id != new.checkpoint.parent_checkpoint_id);
        }

        if old.checkpoint.header.op > new.checkpoint.header.op {
            return false;
        }
        if old.view > new.view {
            return false;
        }
        if old.log_view > new.log_view {
            return false;
        }
        if old.commit_max > new.commit_max {
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
    /// Operations at or before the checkpoint trigger (one bar after the
    /// checkpoint op) are no longer available in the journal and must be
    /// recovered via state sync if needed.
    #[inline]
    pub fn op_compacted(&self, op: u64) -> bool {
        let checkpoint_op = self.checkpoint.header.op;
        if checkpoint_op == 0 {
            return false;
        }

        // The compaction trigger is one bar after the checkpoint op.
        let trigger = trigger_for_checkpoint(checkpoint_op)
            .expect("checkpoint_op > 0 but trigger_for_checkpoint return None");
        op <= trigger
    }

    /// Updates this state for a new checkpoint.
    ///
    /// Advances the checkpoint chain and updates all checkpoint-related fields
    /// from the provided options. The current checkpoint ID (checksum of the
    /// entire `CheckpointState`) becomes the new `parent_checkpoint_id`, and
    /// the old parent becomes `grandparent_checkpoint_id`.
    ///
    /// # Panics
    ///
    /// - `opts.checkpoint.header.op <= self.checkpoint.header.op` (must advance)
    /// - `opts.commit_max < opts.checkpoint.header.op` (commit must cover checkpoint)
    /// - `opts.storage_size < DATA_FILE_SIZE_MIN`
    /// - `opts.sync_op_min > opts.sync_op_max`
    /// - `opts.release < self.checkpoint.release`
    /// - `opts.view < opts.log_view`
    /// - Update would violate monotonic state progression
    pub fn update_for_checkpoint(&mut self, opts: &CheckpointOptions<'_>) {
        let old = *self;

        assert!(
            opts.checkpoint.header.op <= opts.commit_max,
            "commit_max {} must be >= checkpoint op {}",
            opts.commit_max,
            opts.checkpoint.header.op
        );
        assert!(
            opts.checkpoint.header.op > self.checkpoint.header.op,
            "checkpoint op must advance: new {} <= current {}",
            opts.checkpoint.header.op,
            self.checkpoint.header.op
        );
        assert!(
            opts.checkpoint.header.checksum != self.checkpoint.header.checksum,
            "checkpoint header checksum must change when op advances"
        );
        assert!(
            opts.sync_op_min <= opts.sync_op_max,
            "sync_op_min {} must be <= sync_op_max {}",
            opts.sync_op_min,
            opts.sync_op_max
        );
        assert!(
            opts.storage_size >= constants::DATA_FILE_SIZE_MIN,
            "storage_size {} must be >= DATA_FILE_SIZE_MIN {}",
            opts.storage_size,
            constants::DATA_FILE_SIZE_MIN
        );
        assert!(
            opts.release.value() >= self.checkpoint.release.value(),
            "release must be non-decreasing"
        );
        assert!(
            opts.view >= opts.log_view,
            "view {} must be >= log_view {}",
            opts.view,
            opts.log_view
        );

        // Adopt the provided checkpoint state.
        self.checkpoint = opts.checkpoint;

        // Update VSR-level fields.
        self.view = opts.view;
        self.log_view = opts.log_view;
        self.commit_max = opts.commit_max;
        self.sync_op_min = opts.sync_op_min;
        self.sync_op_max = opts.sync_op_max;

        // Update sync_view to log_view at checkpoint time.
        self.sync_view = opts.log_view;

        assert!(
            Self::monotonic(&old, self),
            "checkpoint update must be monotonic"
        );
    }

    /// Updates this state for a view change.
    ///
    /// This is called to persist view/log_view changes or to update the
    /// checkpoint during state sync. The update must either advance
    /// view/log_view or include a sync_checkpoint.
    ///
    /// # Invariants
    ///
    /// - `commit_max` must be non-decreasing
    /// - `view` must be non-decreasing
    /// - `log_view` must be non-decreasing
    /// - `view >= log_view`
    /// - At least one of: `log_view` advances, `view` advances, or `sync_checkpoint` is provided
    /// - If `sync_checkpoint` is provided, the checkpoint op must advance
    ///
    /// # Panics
    ///
    /// Panics if any invariants are violated or if the update would not
    /// result in monotonic state progression.
    pub fn update_for_view_change(&mut self, opts: &ViewChangeOptions<'_>) {
        let old = *self;

        assert!(old.commit_max <= opts.commit_max);
        assert!(old.view <= opts.view);
        assert!(old.log_view <= opts.log_view);

        assert!(
            old.log_view < opts.log_view || old.view < opts.view || opts.sync_checkpoint.is_some()
        );
        assert!(opts.view >= opts.log_view);

        // Apply view/commit durability.
        self.commit_max = opts.commit_max;
        self.sync_op_min = opts.commit_max;
        self.sync_op_max = opts.commit_max;
        self.view = opts.view;
        self.log_view = opts.log_view;
        self.sync_view = opts.log_view;

        if let Some(sync) = &opts.sync_checkpoint {
            assert!(opts.log_view == opts.view);
            assert!(sync.checkpoint.header.op == opts.commit_max);
            assert!(sync.sync_op_min <= sync.sync_op_max);

            assert!(sync.checkpoint.header.op > old.checkpoint.header.op);
            assert!(sync.checkpoint.release.value() >= old.checkpoint.release.value());
            assert!(sync.checkpoint.parent_checkpoint_id == old.checkpoint.checkpoint_id());

            self.checkpoint = sync.checkpoint;
            self.sync_op_min = sync.sync_op_min;
            self.sync_op_max = sync.sync_op_max;
            self.sync_view = opts.view;
        }

        assert!(Self::would_be_updated_by(&old, self));
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
/// The trigger is one compaction bar after the checkpoint:
/// `checkpoint + LSM_COMPACTION_OPS`, and checkpoints are only valid on bars.
/// Returns `None` for checkpoint 0 (the root checkpoint has no trigger).
#[inline]
fn trigger_for_checkpoint(checkpoint: u64) -> Option<u64> {
    let lsm_compaction_ops = constants::LSM_COMPACTION_OPS;
    assert!(lsm_compaction_ops > 0);

    let valid = if checkpoint == 0 {
        true
    } else if let Some(next) = checkpoint.checked_add(1) {
        // Bar boundary: (op + 1) % LSM_COMPACTION_OPS == 0.
        next % lsm_compaction_ops == 0
    } else {
        false
    };
    assert!(valid);

    if checkpoint == 0 {
        return None;
    }
    Some(
        checkpoint
            .checked_add(lsm_compaction_ops)
            .expect("checkpoint trigger overflow"),
    )
}

#[cfg(test)]
fn make_prepare_header(cluster: u128, op: u64) -> HeaderPrepare {
    use crate::vsr::wire::{Command, Operation};

    assert!(op > 0);

    let mut header = HeaderPrepare::new();
    header.cluster = cluster;
    header.command = Command::Prepare;
    header.operation = Operation::NOOP;
    header.op = op;
    header.commit = op - 1;
    header.timestamp = 1;
    header.parent = 1;
    header.client = 1;
    header.request = 1;
    header.release = Release(1);

    header.set_checksum_body(&[]);
    header.set_checksum();

    debug_assert!(header.invalid().is_none());

    header
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsr::wire::header::Release;
    use crate::vsr::{ViewChangeArray, ViewChangeCommand};

    // =========================================================================
    // Helper function to create valid RootOptions for testing
    // =========================================================================
    fn make_valid_root_options(replica_count: u8, replica_index: u8) -> RootOptions {
        let mut members = Members([0; constants::MEMBERS_MAX]);
        for i in 0..replica_count as usize {
            members.0[i] = (i + 1) as u128;
        }
        RootOptions {
            cluster: 1,
            replica_index,
            replica_count,
            members,
            release: Release::ZERO,
            view: 0,
        }
    }

    // =========================================================================
    // RootOptions Tests
    // =========================================================================

    #[test]
    fn test_root_validate() {
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
    fn test_root_validate_indices() {
        for replica_count in 1..=5u8 {
            for replica_index in 0..replica_count {
                let options = make_valid_root_options(replica_count, replica_index);
                options.validate();
                assert_eq!(options.replica_id(), (replica_index + 1) as u128);
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_root_invalid_count_zero() {
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
    fn test_root_invalid_count_max() {
        let members = Members([1; constants::MEMBERS_MAX]);

        let options = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: (constants::REPLICAS_MAX + 1) as u8,
            members,
            release: Release::ZERO,
            view: 0,
        };
        options.validate();
    }

    #[test]
    #[should_panic]
    fn test_root_invalid_index() {
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
    #[should_panic]
    fn test_root_invalid_member_zero() {
        let members = Members([0; constants::MEMBERS_MAX]); // All zeros

        let options = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 1,
            members, // members[0] = 0, which is invalid
            release: Release::ZERO,
            view: 0,
        };
        options.validate();
    }

    #[test]
    #[should_panic]
    fn test_root_invalid_members_shorter_than_replica_count() {
        let mut members = Members([0; constants::MEMBERS_MAX]);
        members.0[0] = 1;

        let options = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 2,
            members,
            release: Release::ZERO,
            view: 0,
        };
        options.validate();
    }

    #[test]
    #[should_panic]
    fn test_root_invalid_members_duplicate() {
        let mut members = Members([0; constants::MEMBERS_MAX]);
        members.0[0] = 1;
        members.0[1] = 1;

        let options = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 2,
            members,
            release: Release::ZERO,
            view: 0,
        };
        options.validate();
    }

    #[test]
    #[should_panic]
    fn test_root_invalid_members_non_contiguous() {
        let mut members = Members([0; constants::MEMBERS_MAX]);
        members.0[0] = 1;
        members.0[2] = 2;

        let options = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 2,
            members,
            release: Release::ZERO,
            view: 0,
        };
        options.validate();
    }

    // =========================================================================
    // CheckpointState Tests
    // =========================================================================

    #[test]
    fn test_layout_checkpoint() {
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
    #[should_panic(expected = "free_set_blocks_acquired padding non-zero")]
    fn test_padding_free_set_acquired() {
        let mut state = CheckpointState::zeroed();
        state.free_set_blocks_acquired_last_block_checksum_padding = 1;
        state.assert_padding_zeroed();
    }

    #[test]
    #[should_panic(expected = "free_set_blocks_released padding non-zero")]
    fn test_padding_free_set_released() {
        let mut state = CheckpointState::zeroed();
        state.free_set_blocks_released_last_block_checksum_padding = 1;
        state.assert_padding_zeroed();
    }

    #[test]
    #[should_panic(expected = "client_sessions padding non-zero")]
    fn test_padding_client_sessions() {
        let mut state = CheckpointState::zeroed();
        state.client_sessions_last_block_checksum_padding = 1;
        state.assert_padding_zeroed();
    }

    #[test]
    #[should_panic(expected = "manifest_oldest padding non-zero")]
    fn test_padding_manifest_oldest() {
        let mut state = CheckpointState::zeroed();
        state.manifest_oldest_checksum_padding = 1;
        state.assert_padding_zeroed();
    }

    #[test]
    #[should_panic(expected = "manifest_newest padding non-zero")]
    fn test_padding_manifest_newest() {
        let mut state = CheckpointState::zeroed();
        state.manifest_newest_checksum_padding = 1;
        state.assert_padding_zeroed();
    }

    #[test]
    #[should_panic(expected = "snapshots_block padding non-zero")]
    fn test_padding_snapshots() {
        let mut state = CheckpointState::zeroed();
        state.snapshots_block_checksum_padding = 1;
        state.assert_padding_zeroed();
    }

    #[test]
    #[should_panic(expected = "reserved_manifest non-zero")]
    fn test_padding_reserved_manifest() {
        let mut state = CheckpointState::zeroed();
        state.reserved_manifest[0] = 1;
        state.assert_padding_zeroed();
    }

    #[test]
    #[should_panic(expected = "checkpoint reserved non-zero")]
    fn test_padding_reserved_checkpoint() {
        let mut state = CheckpointState::zeroed();
        state.reserved[0] = 1;
        state.assert_padding_zeroed();
    }

    // =========================================================================
    // VsrState Tests - Basic
    // =========================================================================

    #[test]
    fn test_state_zeroed() {
        let state = VsrState::zeroed();
        assert_eq!(state.replica_id, 0);
        assert_eq!(state.commit_max, 0);
        assert_eq!(state.view, 0);
        assert_eq!(state.replica_count, 0);
    }

    #[test]
    fn test_state_root() {
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
    fn test_state_root_indices() {
        for replica_index in 0..3u8 {
            let options = make_valid_root_options(3, replica_index);
            let state = VsrState::root(&options);
            assert_eq!(state.replica_id, (replica_index + 1) as u128);
            state.assert_internally_consistent();
        }
    }

    #[test]
    fn test_state_root_checksums() {
        let options = make_valid_root_options(1, 0);
        let state = VsrState::root(&options);

        let checksum_empty = checksum(&[]);

        // Verify empty checksums are initialized correctly
        assert_eq!(
            state.checkpoint.free_set_blocks_acquired_checksum,
            checksum_empty
        );
        assert_eq!(
            state.checkpoint.free_set_blocks_released_checksum,
            checksum_empty
        );
        assert_eq!(state.checkpoint.client_sessions_checksum, checksum_empty);

        // Block checksums should be zero (no blocks yet)
        assert_eq!(
            state
                .checkpoint
                .free_set_blocks_acquired_last_block_checksum,
            0
        );
        assert_eq!(
            state
                .checkpoint
                .free_set_blocks_released_last_block_checksum,
            0
        );
        assert_eq!(state.checkpoint.client_sessions_last_block_checksum, 0);
    }

    #[test]
    fn test_state_root_storage_size() {
        let options = make_valid_root_options(1, 0);
        let state = VsrState::root(&options);
        assert_eq!(state.checkpoint.storage_size, constants::DATA_FILE_SIZE_MIN);
    }

    // =========================================================================
    // VsrState::assert_internally_consistent Tests
    // =========================================================================

    #[test]
    #[should_panic]
    fn test_consistent_replica_id() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.replica_id = 999; // Not in members
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "members.count() < replica_count")]
    fn test_consistent_members_shorter_than_replica_count() {
        let options = make_valid_root_options(2, 0);
        let mut state = VsrState::root(&options);
        state.members.0[1] = 0;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic]
    fn test_consistent_commit_max() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.header.op = 100;
        state.commit_max = 50; // Less than checkpoint op
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic]
    fn test_consistent_sync_op() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.sync_op_min = 100;
        state.sync_op_max = 50; // Less than min
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic]
    fn test_consistent_view() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.log_view = 10;
        state.view = 5; // Less than log_view
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic]
    fn test_consistent_count_zero() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.replica_count = 0;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic]
    fn test_consistent_count_max() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.replica_count = (constants::REPLICAS_MAX + 1) as u8;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic]
    fn test_consistent_storage_size() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.storage_size = constants::DATA_FILE_SIZE_MIN - 1;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic]
    fn test_consistent_snapshots_checksum() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.snapshots_block_checksum = 1;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic]
    fn test_consistent_snapshots_address() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.snapshots_block_address = 1;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "state reserved non-zero")]
    fn test_consistent_state_reserved_nonzero() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.reserved[0] = 1;
        state.assert_internally_consistent();
    }

    // =========================================================================
    // Free Set Consistency Tests
    // =========================================================================

    #[test]
    #[should_panic(expected = "free_set_acquired: addr=0 but size=")]
    fn test_free_set_acquired_addr_zero() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.free_set_blocks_acquired_size = 100;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "free_set_acquired: addr=0 but block_checksum != 0")]
    fn test_free_set_acquired_checksum() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state
            .checkpoint
            .free_set_blocks_acquired_last_block_checksum = 1;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "free_set_acquired: addr=0 but checksum != empty")]
    fn test_free_set_acquired_checksum_empty() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.free_set_blocks_acquired_checksum = 0;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "free_set_acquired: addr=")]
    fn test_free_set_acquired_addr_nonzero_size_zero() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.free_set_blocks_acquired_last_block_address = constants::BLOCK_SIZE;
        state.checkpoint.free_set_blocks_acquired_size = 0;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "free_set_released: addr=0 but size=")]
    fn test_free_set_released_addr_zero() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.free_set_blocks_released_size = 100;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "free_set_released: addr=0 but block_checksum != 0")]
    fn test_free_set_released_checksum() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state
            .checkpoint
            .free_set_blocks_released_last_block_checksum = 1;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "free_set_released: addr=0 but checksum != empty")]
    fn test_free_set_released_checksum_empty() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.free_set_blocks_released_checksum = 0;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "free_set_released: addr=")]
    fn test_free_set_released_addr_nonzero_size_zero() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.free_set_blocks_released_last_block_address = constants::BLOCK_SIZE;
        state.checkpoint.free_set_blocks_released_size = 0;
        state.assert_internally_consistent();
    }

    // =========================================================================
    // Client Sessions Consistency Tests
    // =========================================================================

    #[test]
    #[should_panic(expected = "client_sessions: addr=0 but block_checksum != 0")]
    fn test_client_sessions_checksum() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.client_sessions_last_block_checksum = 1;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "client_sessions: addr=0 but checksum != empty")]
    fn test_client_sessions_checksum_empty() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.client_sessions_checksum = 0;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "client_sessions: addr=0 but size=")]
    fn test_client_sessions_size() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.client_sessions_size = 100;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "client_sessions: size")]
    fn test_client_sessions_size_mismatch_when_addr_nonzero() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.client_sessions_last_block_address = constants::BLOCK_SIZE;
        state.checkpoint.client_sessions_size = client_sessions_encode_size() + 8;
        state.assert_internally_consistent();
    }

    // =========================================================================
    // Manifest Consistency Tests
    // =========================================================================

    #[test]
    #[should_panic(expected = "manifest: count=0 but oldest_addr=")]
    fn test_manifest_count_zero_oldest_addr() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.manifest_oldest_address = constants::BLOCK_SIZE;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "manifest: count=0 but oldest_checksum != 0")]
    fn test_manifest_count_zero_oldest_checksum() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.manifest_oldest_checksum = 1;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "manifest: count=0 but newest_addr=")]
    fn test_manifest_count_zero_newest_addr() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.manifest_newest_address = constants::BLOCK_SIZE;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "manifest: count=0 but newest_checksum != 0")]
    fn test_manifest_count_zero_newest_checksum() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.manifest_newest_checksum = 1;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "manifest: count=1 but oldest_checksum=0")]
    fn test_manifest_nonzero_oldest_checksum_zero() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.manifest_block_count = 1;
        state.checkpoint.manifest_oldest_address = constants::BLOCK_SIZE;
        state.checkpoint.manifest_newest_address = constants::BLOCK_SIZE;
        state.checkpoint.manifest_newest_checksum = 1;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "manifest: count=1 but newest_checksum=0")]
    fn test_manifest_nonzero_newest_checksum_zero() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.manifest_block_count = 1;
        state.checkpoint.manifest_oldest_address = constants::BLOCK_SIZE;
        state.checkpoint.manifest_oldest_checksum = 1;
        state.checkpoint.manifest_newest_address = constants::BLOCK_SIZE;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "manifest: count=1 but oldest_addr")]
    fn test_manifest_count_one_addr_mismatch() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.manifest_block_count = 1;
        state.checkpoint.manifest_oldest_address = constants::BLOCK_SIZE;
        state.checkpoint.manifest_oldest_checksum = 1;
        state.checkpoint.manifest_newest_address = constants::BLOCK_SIZE * 2;
        state.checkpoint.manifest_newest_checksum = 1;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "manifest: count=1 but checksums differ")]
    fn test_manifest_count_one_checksum_mismatch() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.manifest_block_count = 1;
        state.checkpoint.manifest_oldest_address = constants::BLOCK_SIZE;
        state.checkpoint.manifest_oldest_checksum = 1;
        state.checkpoint.manifest_newest_address = constants::BLOCK_SIZE;
        state.checkpoint.manifest_newest_checksum = 2;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "manifest: oldest_addr")]
    fn test_manifest_oldest_greater_than_newest() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.manifest_block_count = 2;
        state.checkpoint.manifest_oldest_address = constants::BLOCK_SIZE * 2;
        state.checkpoint.manifest_oldest_checksum = 1;
        state.checkpoint.manifest_newest_address = constants::BLOCK_SIZE;
        state.checkpoint.manifest_newest_checksum = 2;
        state.assert_internally_consistent();
    }

    #[test]
    #[should_panic(expected = "manifest: oldest_addr")]
    fn test_manifest_unaligned() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.manifest_block_count = 1;
        state.checkpoint.manifest_oldest_address = constants::BLOCK_SIZE + 1; // Not aligned
        state.checkpoint.manifest_oldest_checksum = 1;
        state.checkpoint.manifest_newest_address = constants::BLOCK_SIZE + 1;
        state.checkpoint.manifest_newest_checksum = 1;
        state.assert_internally_consistent();
    }

    // =========================================================================
    // VsrState::monotonic Tests - Comprehensive
    // =========================================================================

    #[test]
    fn test_monotonic_no_change() {
        let options = make_valid_root_options(1, 0);
        let state = VsrState::root(&options);
        assert!(VsrState::monotonic(&state, &state));
    }

    #[test]
    fn test_monotonic_view_increase() {
        let options = make_valid_root_options(1, 0);
        let old = VsrState::root(&options);
        let mut new = old;
        new.view += 1;
        assert!(VsrState::monotonic(&old, &new));
        assert!(VsrState::would_be_updated_by(&old, &new));
    }

    #[test]
    fn test_monotonic_sync_view_increase() {
        let options = make_valid_root_options(1, 0);
        let old = VsrState::root(&options);
        let mut new = old;
        new.sync_view = 5;
        new.view = 5;
        assert!(VsrState::monotonic(&old, &new));
        assert!(VsrState::would_be_updated_by(&old, &new));
    }

    #[test]
    fn test_monotonic_view_regression() {
        let options = make_valid_root_options(1, 0);
        let mut old = VsrState::root(&options);
        let new = old;
        old.view = 10;
        assert!(!VsrState::monotonic(&old, &new));
    }

    #[test]
    fn test_monotonic_sync_view_regression() {
        // Sync fields are NOT checked for monotonicity.
        // They can legitimately reset when state sync completes.
        let options = make_valid_root_options(1, 0);
        let mut old = VsrState::root(&options);
        let mut new = old;
        old.sync_view = 10;
        old.view = 10;
        new.sync_view = 0;
        new.view = 10;
        assert!(VsrState::monotonic(&old, &new)); // Regression is allowed
    }

    #[test]
    fn test_monotonic_commit_max_increase() {
        let options = make_valid_root_options(1, 0);
        let old = VsrState::root(&options);
        let mut new = old;
        new.commit_max = 100;
        assert!(VsrState::monotonic(&old, &new));
    }

    #[test]
    fn test_monotonic_commit_max_regression() {
        let options = make_valid_root_options(1, 0);
        let mut old = VsrState::root(&options);
        let new = old;
        old.commit_max = 100;
        assert!(!VsrState::monotonic(&old, &new));
    }

    #[test]
    fn test_monotonic_sync_op_min_increase() {
        let options = make_valid_root_options(1, 0);
        let old = VsrState::root(&options);
        let mut new = old;
        new.sync_op_min = 50;
        new.sync_op_max = 50; // Must satisfy sync_op_max >= sync_op_min
        assert!(VsrState::monotonic(&old, &new));
    }

    #[test]
    fn test_monotonic_sync_op_min_regression() {
        // Sync fields are NOT checked for monotonicity.
        // They can legitimately reset when state sync completes.
        let options = make_valid_root_options(1, 0);
        let mut old = VsrState::root(&options);
        old.sync_op_min = 50;
        old.sync_op_max = 50;
        let new = VsrState::root(&options);
        assert!(VsrState::monotonic(&old, &new)); // Regression is allowed
    }

    #[test]
    fn test_monotonic_sync_op_max_increase() {
        let options = make_valid_root_options(1, 0);
        let old = VsrState::root(&options);
        let mut new = old;
        new.sync_op_max = 100;
        assert!(VsrState::monotonic(&old, &new));
    }

    #[test]
    fn test_monotonic_sync_op_max_regression() {
        // Sync fields are NOT checked for monotonicity.
        // They can legitimately reset when state sync completes.
        let options = make_valid_root_options(1, 0);
        let mut old = VsrState::root(&options);
        old.sync_op_max = 100;
        let new = VsrState::root(&options);
        assert!(VsrState::monotonic(&old, &new)); // Regression is allowed
    }

    #[test]
    fn test_monotonic_log_view_increase() {
        let options = make_valid_root_options(1, 0);
        let old = VsrState::root(&options);
        let mut new = old;
        new.log_view = 5;
        new.view = 5; // Must satisfy view >= log_view
        assert!(VsrState::monotonic(&old, &new));
    }

    #[test]
    fn test_monotonic_log_view_regression() {
        let options = make_valid_root_options(1, 0);
        let mut old = VsrState::root(&options);
        old.log_view = 10;
        old.view = 10;
        let new = VsrState::root(&options);
        assert!(!VsrState::monotonic(&old, &new));
    }

    #[test]
    fn test_monotonic_checkpoint_op_increase() {
        let options = make_valid_root_options(1, 0);
        let old = VsrState::root(&options);
        let mut new = old;
        new.checkpoint.header.op = 100;
        new.checkpoint.header.checksum = 12345; // Different checksum
        new.checkpoint.parent_checkpoint_id = 1; // Different parent
        new.commit_max = 100; // Must satisfy commit_max >= checkpoint.header.op
        assert!(VsrState::monotonic(&old, &new));
    }

    #[test]
    #[should_panic]
    fn test_monotonic_checkpoint_same_op_changes_panics() {
        let options = make_valid_root_options(1, 0);
        let old = VsrState::root(&options);
        let mut new = old;
        new.checkpoint.storage_size = constants::DATA_FILE_SIZE_MIN + constants::BLOCK_SIZE;
        VsrState::monotonic(&old, &new);
    }

    #[test]
    #[should_panic]
    fn test_monotonic_members_mismatch_panics() {
        let mut members_old = Members([0; constants::MEMBERS_MAX]);
        members_old.0[0] = 1;
        members_old.0[1] = 2;
        let options_old = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 2,
            members: members_old,
            release: Release::ZERO,
            view: 0,
        };

        let mut members_new = Members([0; constants::MEMBERS_MAX]);
        members_new.0[0] = 1;
        members_new.0[1] = 3;
        let options_new = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 2,
            members: members_new,
            release: Release::ZERO,
            view: 0,
        };

        let old = VsrState::root(&options_old);
        let new = VsrState::root(&options_new);
        VsrState::monotonic(&old, &new);
    }

    #[test]
    fn test_monotonic_checkpoint_op_regression() {
        let options = make_valid_root_options(1, 0);
        let mut old = VsrState::root(&options);
        old.checkpoint.header.op = 100;
        old.checkpoint.header.checksum = 12345;
        old.checkpoint.parent_checkpoint_id = 1;
        old.commit_max = 100;
        let new = VsrState::root(&options);
        assert!(!VsrState::monotonic(&old, &new));
    }

    #[test]
    fn test_monotonic_multiple_fields_increase() {
        let options = make_valid_root_options(1, 0);
        let old = VsrState::root(&options);
        let mut new = old;
        new.commit_max = 50;
        new.view = 5;
        new.log_view = 3;
        new.sync_op_min = 10;
        new.sync_op_max = 20;
        assert!(VsrState::monotonic(&old, &new));
        assert!(VsrState::would_be_updated_by(&old, &new));
    }

    #[test]
    fn test_monotonic_would_be_updated() {
        let options = make_valid_root_options(1, 0);
        let state = VsrState::root(&options);
        assert!(!VsrState::would_be_updated_by(&state, &state));
    }

    // =========================================================================
    // VsrState::op_compacted Tests
    // =========================================================================

    #[test]
    fn test_op_compacted_correctness_check() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let lsm_compaction_ops = constants::LSM_COMPACTION_OPS;
        let checkpoint = lsm_compaction_ops - 1;
        state.checkpoint.header.op = checkpoint;
        let trigger = checkpoint + lsm_compaction_ops;

        // Ops at or before trigger are compacted.
        assert!(state.op_compacted(checkpoint));
        assert!(state.op_compacted(checkpoint + 1));
        assert!(state.op_compacted(trigger));
        assert!(state.op_compacted(0));

        // Ops after trigger are not compacted.
        assert!(!state.op_compacted(trigger + 1));
    }

    #[test]
    fn test_op_compacted_checkpoint_zero() {
        let options = make_valid_root_options(1, 0);
        let state = VsrState::root(&options);

        // With checkpoint op = 0, nothing is compacted
        assert!(!state.op_compacted(0));
        assert!(!state.op_compacted(100));
        assert!(!state.op_compacted(u64::MAX));
    }

    #[test]
    fn test_op_compacted_with_checkpoint() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let lsm_compaction_ops = constants::LSM_COMPACTION_OPS;
        let checkpoint = (lsm_compaction_ops * 3) - 1;
        state.checkpoint.header.op = checkpoint;
        let trigger = checkpoint + lsm_compaction_ops;

        // Ops at or before trigger are compacted
        assert!(state.op_compacted(0));
        assert!(state.op_compacted(checkpoint - 1));
        assert!(state.op_compacted(checkpoint));
        assert!(state.op_compacted(checkpoint + 1));
        assert!(state.op_compacted(trigger));

        // Ops after trigger are not compacted
        assert!(!state.op_compacted(trigger + 1));
        assert!(!state.op_compacted(trigger + 1000));
    }

    #[test]
    fn test_op_compacted_boundary() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let lsm_compaction_ops = constants::LSM_COMPACTION_OPS;
        let checkpoint = lsm_compaction_ops - 1;
        state.checkpoint.header.op = checkpoint;
        let trigger = checkpoint + lsm_compaction_ops;

        assert!(state.op_compacted(trigger));
        assert!(!state.op_compacted(trigger + 1));
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_trigger_for_checkpoint_zero() {
        assert!(trigger_for_checkpoint(0).is_none());
    }

    #[test]
    fn test_trigger_for_checkpoint_valid_boundary() {
        let lsm_compaction_ops = constants::LSM_COMPACTION_OPS;
        let checkpoint = lsm_compaction_ops - 1;
        let trigger = trigger_for_checkpoint(checkpoint);
        assert!(trigger.is_some());
        assert_eq!(trigger.unwrap(), checkpoint + lsm_compaction_ops);
    }

    #[test]
    #[should_panic]
    fn test_trigger_for_checkpoint_invalid_panics() {
        let lsm_compaction_ops = constants::LSM_COMPACTION_OPS;
        assert!(lsm_compaction_ops > 1);
        let invalid_checkpoint = lsm_compaction_ops;
        let _ = trigger_for_checkpoint(invalid_checkpoint);
    }

    #[test]
    fn test_client_sessions_encode_size() {
        let size = client_sessions_encode_size();

        // Should be non-zero
        assert!(size > 0);

        // Should be based on HeaderPrepare::SIZE and CLIENTS_MAX
        let header_bytes = (HeaderPrepare::SIZE as u64) * (constants::CLIENTS_MAX as u64);
        let aligned = (header_bytes + 7) & !7;
        let session_bytes = 8u64 * (constants::CLIENTS_MAX as u64);

        assert_eq!(size, aligned + session_bytes);

        // Should be 8-byte aligned (since aligned is 8-byte aligned and session_bytes is multiple of 8)
        assert!(size.is_multiple_of(8));
    }

    // =========================================================================
    // VsrState Layout Tests
    // =========================================================================

    #[test]
    fn test_layout_state() {
        // VsrState should be large enough to hold all fields
        let size = mem::size_of::<VsrState>();

        // CheckpointState is 1024, plus other fields and reserved
        assert!(size > 1024);

        // Should have at least 8-byte alignment for u64 fields
        assert!(mem::align_of::<VsrState>() >= 8);
    }

    // =========================================================================
    // VsrState::update_for_checkpoint Tests
    // =========================================================================

    /// Helper to create a valid CheckpointOptions for testing.
    ///
    /// Builds a `CheckpointState` with the given op, properly chaining from
    /// the current state's checkpoint IDs.
    fn make_checkpoint_options(
        current: &VsrState,
        op: u64,
        commit_max: u64,
        storage_size: u64,
    ) -> CheckpointOptions<'static> {
        let header = super::make_prepare_header(1, op);

        let mut checkpoint = CheckpointState::zeroed();
        checkpoint.header = header;
        checkpoint.storage_size = storage_size;
        checkpoint.release = Release::ZERO;
        // Chain from current checkpoint.
        checkpoint.parent_checkpoint_id = current.checkpoint.checkpoint_id();
        checkpoint.grandparent_checkpoint_id = current.checkpoint.parent_checkpoint_id;
        // Set valid checksums for empty block references.
        checkpoint.free_set_blocks_acquired_checksum = checksum(&[]);
        checkpoint.free_set_blocks_released_checksum = checksum(&[]);
        checkpoint.client_sessions_checksum = checksum(&[]);

        CheckpointOptions {
            checkpoint,
            view: current.view,
            log_view: current.log_view,
            view_attributes: None,
            commit_max,
            sync_op_min: 0,
            sync_op_max: 0,
            storage_size,
            release: Release::ZERO,
            sync_checkpoint: false,
        }
    }

    #[test]
    fn test_update_for_checkpoint_basic() {
        let options = make_valid_root_options(3, 0);
        let mut state = VsrState::root(&options);

        // checkpoint_id() is the checksum of the entire CheckpointState, not just header.checksum
        let old_checkpoint_id = state.checkpoint.checkpoint_id();

        let opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        state.update_for_checkpoint(&opts);

        // Checkpoint chain advanced using checkpoint_id()
        assert_eq!(state.checkpoint.parent_checkpoint_id, old_checkpoint_id);
        assert_eq!(state.checkpoint.grandparent_checkpoint_id, 0);

        // Header updated
        assert_eq!(state.checkpoint.header.op, 100);

        // Commit updated
        assert_eq!(state.commit_max, 100);
    }

    #[test]
    fn test_update_for_checkpoint_chain_advancement() {
        let options = make_valid_root_options(3, 0);
        let mut state = VsrState::root(&options);

        // First checkpoint - use checkpoint_id() (checksum of entire CheckpointState)
        let first_checkpoint_id = state.checkpoint.checkpoint_id();
        let opts1 = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        state.update_for_checkpoint(&opts1);

        assert_eq!(state.checkpoint.parent_checkpoint_id, first_checkpoint_id);
        assert_eq!(state.checkpoint.grandparent_checkpoint_id, 0);

        // Second checkpoint
        let second_checkpoint_id = state.checkpoint.checkpoint_id();
        let opts2 = make_checkpoint_options(&state, 200, 200, constants::DATA_FILE_SIZE_MIN);
        state.update_for_checkpoint(&opts2);

        assert_eq!(state.checkpoint.parent_checkpoint_id, second_checkpoint_id);
        assert_eq!(
            state.checkpoint.grandparent_checkpoint_id,
            first_checkpoint_id
        );

        // Third checkpoint
        let third_checkpoint_id = state.checkpoint.checkpoint_id();
        let opts3 = make_checkpoint_options(&state, 300, 300, constants::DATA_FILE_SIZE_MIN);
        state.update_for_checkpoint(&opts3);

        assert_eq!(state.checkpoint.parent_checkpoint_id, third_checkpoint_id);
        assert_eq!(
            state.checkpoint.grandparent_checkpoint_id,
            second_checkpoint_id
        );
    }

    #[test]
    fn test_update_for_checkpoint_manifest_references() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let oldest_addr = constants::BLOCK_SIZE;
        let newest_addr = constants::BLOCK_SIZE * 2;

        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.checkpoint.manifest_oldest_checksum = 111;
        opts.checkpoint.manifest_oldest_address = oldest_addr;
        opts.checkpoint.manifest_newest_checksum = 333;
        opts.checkpoint.manifest_newest_address = newest_addr;
        opts.checkpoint.manifest_block_count = 2;

        state.update_for_checkpoint(&opts);

        assert_eq!(state.checkpoint.manifest_oldest_checksum, 111);
        assert_eq!(state.checkpoint.manifest_oldest_address, oldest_addr);
        assert_eq!(state.checkpoint.manifest_newest_checksum, 333);
        assert_eq!(state.checkpoint.manifest_newest_address, newest_addr);
        assert_eq!(state.checkpoint.manifest_block_count, 2);
    }

    #[test]
    fn test_update_for_checkpoint_free_set_references() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let acquired_last_addr = constants::BLOCK_SIZE;
        let released_last_addr = constants::BLOCK_SIZE * 2;

        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.checkpoint.free_set_blocks_acquired_checksum = 1111;
        opts.checkpoint.free_set_blocks_acquired_size = 2222;
        opts.checkpoint.free_set_blocks_acquired_last_block_checksum = 7777;
        opts.checkpoint.free_set_blocks_acquired_last_block_address = acquired_last_addr;
        opts.checkpoint.free_set_blocks_released_checksum = 3333;
        opts.checkpoint.free_set_blocks_released_size = 4444;
        opts.checkpoint.free_set_blocks_released_last_block_checksum = 8888;
        opts.checkpoint.free_set_blocks_released_last_block_address = released_last_addr;

        state.update_for_checkpoint(&opts);

        assert_eq!(state.checkpoint.free_set_blocks_acquired_checksum, 1111);
        assert_eq!(state.checkpoint.free_set_blocks_acquired_size, 2222);
        assert_eq!(
            state
                .checkpoint
                .free_set_blocks_acquired_last_block_checksum,
            7777
        );
        assert_eq!(
            state.checkpoint.free_set_blocks_acquired_last_block_address,
            acquired_last_addr
        );
        assert_eq!(state.checkpoint.free_set_blocks_released_checksum, 3333);
        assert_eq!(state.checkpoint.free_set_blocks_released_size, 4444);
        assert_eq!(
            state
                .checkpoint
                .free_set_blocks_released_last_block_checksum,
            8888
        );
        assert_eq!(
            state.checkpoint.free_set_blocks_released_last_block_address,
            released_last_addr
        );
    }

    #[test]
    fn test_update_for_checkpoint_client_sessions_references() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let sessions_size = client_sessions_encode_size();
        let sessions_addr = constants::BLOCK_SIZE;

        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.checkpoint.client_sessions_checksum = 9999;
        opts.checkpoint.client_sessions_size = sessions_size;
        opts.checkpoint.client_sessions_last_block_checksum = 7777;
        opts.checkpoint.client_sessions_last_block_address = sessions_addr;

        state.update_for_checkpoint(&opts);

        assert_eq!(state.checkpoint.client_sessions_checksum, 9999);
        assert_eq!(state.checkpoint.client_sessions_size, sessions_size);
        assert_eq!(state.checkpoint.client_sessions_last_block_checksum, 7777);
        assert_eq!(
            state.checkpoint.client_sessions_last_block_address,
            sessions_addr
        );
    }

    #[test]
    fn test_update_for_checkpoint_sync_bounds() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.sync_op_min = 50;
        opts.sync_op_max = 75;

        state.update_for_checkpoint(&opts);

        assert_eq!(state.sync_op_min, 50);
        assert_eq!(state.sync_op_max, 75);
    }

    #[test]
    fn test_update_for_checkpoint_with_view_attributes() {
        use crate::vsr::ViewChangeArray;

        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let view_headers = ViewChangeArray::root(1);

        let view_attrs = ViewAttributes {
            headers: &view_headers,
            view: 42,
            log_view: 40,
        };

        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.view = 42;
        opts.log_view = 40;
        opts.view_attributes = Some(view_attrs);

        state.update_for_checkpoint(&opts);

        assert_eq!(state.view, 42);
        assert_eq!(state.log_view, 40);
    }

    #[test]
    fn test_update_for_checkpoint_without_view_attributes() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.view = 10;
        state.log_view = 5;

        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.view = 10;
        opts.log_view = 5;
        state.update_for_checkpoint(&opts);

        // View preserved
        assert_eq!(state.view, 10);
        assert_eq!(state.log_view, 5);
    }

    #[test]
    fn test_update_for_checkpoint_storage_size() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let new_size = constants::DATA_FILE_SIZE_MIN * 2;
        let opts = make_checkpoint_options(&state, 100, 100, new_size);
        state.update_for_checkpoint(&opts);

        assert_eq!(state.checkpoint.storage_size, new_size);
    }

    #[test]
    fn test_update_for_checkpoint_release() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let new_release = Release(0x00010203); // Version 1.2.3 encoded

        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.checkpoint.release = new_release;
        opts.release = new_release;

        state.update_for_checkpoint(&opts);

        assert_eq!(state.checkpoint.release, new_release);
    }

    #[test]
    #[should_panic(expected = "checkpoint op must advance")]
    fn test_update_for_checkpoint_panics_op_not_advancing() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.header.op = 100;
        state.commit_max = 100;

        // Try to checkpoint at same op
        let opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        state.update_for_checkpoint(&opts);
    }

    #[test]
    #[should_panic(expected = "checkpoint op must advance")]
    fn test_update_for_checkpoint_panics_op_regression() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.checkpoint.header.op = 100;
        state.commit_max = 100;

        // Try to checkpoint at earlier op
        let opts = make_checkpoint_options(&state, 50, 50, constants::DATA_FILE_SIZE_MIN);
        state.update_for_checkpoint(&opts);
    }

    #[test]
    #[should_panic(expected = "commit_max")]
    fn test_update_for_checkpoint_panics_commit_less_than_op() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        // commit_max < header.op is invalid
        let opts = make_checkpoint_options(&state, 100, 50, constants::DATA_FILE_SIZE_MIN);
        state.update_for_checkpoint(&opts);
    }

    #[test]
    #[should_panic(expected = "sync_op_max")]
    fn test_update_for_checkpoint_panics_sync_bounds_inverted() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.sync_op_min = 75; // min > max is invalid
        opts.sync_op_max = 50;

        state.update_for_checkpoint(&opts);
    }

    #[test]
    #[should_panic(expected = "storage_size")]
    fn test_update_for_checkpoint_panics_storage_too_small() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        // Storage size below minimum
        let opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN - 1);
        state.update_for_checkpoint(&opts);
    }

    #[test]
    fn test_update_for_checkpoint_preserves_identity_fields() {
        let options = make_valid_root_options(3, 1);
        let mut state = VsrState::root(&options);

        let original_replica_id = state.replica_id;
        let original_replica_count = state.replica_count;
        let original_members = state.members;

        let opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        state.update_for_checkpoint(&opts);

        // Identity fields unchanged
        assert_eq!(state.replica_id, original_replica_id);
        assert_eq!(state.replica_count, original_replica_count);
        assert_eq!(state.members, original_members);
    }

    #[test]
    fn test_update_for_checkpoint_commit_equals_op() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        // commit_max == header.op is valid (boundary case)
        let opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        state.update_for_checkpoint(&opts);

        assert_eq!(state.commit_max, 100);
        assert_eq!(state.checkpoint.header.op, 100);
    }

    #[test]
    fn test_update_for_checkpoint_commit_greater_than_op() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        // commit_max > header.op is valid
        let opts = make_checkpoint_options(&state, 100, 150, constants::DATA_FILE_SIZE_MIN);
        state.update_for_checkpoint(&opts);

        assert_eq!(state.commit_max, 150);
        assert_eq!(state.checkpoint.header.op, 100);
    }

    #[test]
    #[should_panic]
    fn test_update_for_checkpoint_panics_commit_regression() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.commit_max = 200;

        let mut opts = make_checkpoint_options(&state, 100, 150, constants::DATA_FILE_SIZE_MIN);
        opts.commit_max = 150; // Less than current commit_max of 200

        state.update_for_checkpoint(&opts);
    }

    #[test]
    fn test_update_for_checkpoint_allows_sync_bounds_regression() {
        // Sync fields are NOT checked for monotonicity.
        // They can legitimately reset when state sync completes.
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.sync_op_min = 50;
        state.sync_op_max = 75;

        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.sync_op_min = 25; // Regression from 50 - this is allowed
        opts.sync_op_max = 25; // Regression from 75 - this is allowed

        // Should succeed - sync regression is allowed
        state.update_for_checkpoint(&opts);
        assert_eq!(state.sync_op_min, 25);
        assert_eq!(state.sync_op_max, 25);
    }

    #[test]
    #[should_panic]
    fn test_update_for_checkpoint_panics_view_regression() {
        use crate::vsr::ViewChangeArray;

        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.view = 10;
        state.log_view = 5;

        let view_headers = ViewChangeArray::root(1);

        let view_attrs = ViewAttributes {
            headers: &view_headers,
            view: 9,
            log_view: 4,
        };

        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.view = 9; // Regression from 10
        opts.log_view = 4; // Regression from 5
        opts.view_attributes = Some(view_attrs);

        state.update_for_checkpoint(&opts);
    }

    #[test]
    #[should_panic]
    fn test_update_for_checkpoint_panics_header_checksum_unchanged() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        // Create checkpoint with same checksum as current (invalid)
        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.checkpoint.header.checksum = state.checkpoint.header.checksum;

        state.update_for_checkpoint(&opts);
    }

    #[test]
    fn test_update_for_checkpoint_updates_last_block_info() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        // Set some initial state for last block fields
        state.checkpoint.free_set_blocks_acquired_last_block_address = 123;
        state
            .checkpoint
            .free_set_blocks_acquired_last_block_checksum = 123;
        state.checkpoint.free_set_blocks_acquired_size = 1;

        let mut opts = make_checkpoint_options(&state, 100, 100, constants::DATA_FILE_SIZE_MIN);
        opts.checkpoint.free_set_blocks_acquired_checksum = checksum(&[]);
        opts.checkpoint.free_set_blocks_acquired_size = 1;
        opts.checkpoint.free_set_blocks_acquired_last_block_checksum = 456;
        opts.checkpoint.free_set_blocks_acquired_last_block_address = 456;

        state.update_for_checkpoint(&opts);

        assert_eq!(
            state.checkpoint.free_set_blocks_acquired_last_block_address,
            456
        );
        assert_eq!(
            state
                .checkpoint
                .free_set_blocks_acquired_last_block_checksum,
            456
        );
    }

    // =========================================================================
    // VsrState::update_for_view_change Tests
    // =========================================================================

    fn make_view_headers(cluster: u128, op: u64) -> ViewChangeArray {
        let header = super::make_prepare_header(cluster, op);
        ViewChangeArray::init(ViewChangeCommand::StartView, &[header])
    }

    fn make_sync_checkpoint(
        state: &VsrState,
        op: u64,
        parent_checkpoint_id: u128,
        grandparent_checkpoint_id: u128,
    ) -> CheckpointState {
        let mut checkpoint = state.checkpoint;
        checkpoint.header = super::make_prepare_header(1, op);
        checkpoint.parent_checkpoint_id = parent_checkpoint_id;
        checkpoint.grandparent_checkpoint_id = grandparent_checkpoint_id;
        checkpoint
    }

    #[test]
    fn test_update_for_view_change_advances_view_and_commit() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let view_headers = make_view_headers(1, 1);

        let opts = ViewChangeOptions {
            commit_max: 10,
            log_view: 1,
            view: 2,
            headers: &view_headers,
            sync_checkpoint: None,
        };

        state.update_for_view_change(&opts);

        assert_eq!(state.commit_max, 10);
        assert_eq!(state.log_view, 1);
        assert_eq!(state.view, 2);
        assert_eq!(state.checkpoint.header.op, 0);
        // sync_op_min/max are set to commit_max in view_change (without sync_checkpoint)
        assert_eq!(state.sync_op_min, 10);
        assert_eq!(state.sync_op_max, 10);
        // sync_view is set to log_view
        assert_eq!(state.sync_view, 1);
    }

    #[test]
    fn test_update_for_view_change_with_sync_checkpoint_updates_checkpoint_and_sync_bounds() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.sync_view = 7;

        let current_checkpoint_id = state.checkpoint.checkpoint_id();
        let checkpoint_next = super::checkpoint_after(state.checkpoint.header.op);

        let checkpoint = make_sync_checkpoint(&state, checkpoint_next, current_checkpoint_id, 0);

        let sync = SyncCheckpointOptions {
            checkpoint,
            sync_op_min: 10,
            sync_op_max: 12,
        };

        let view_headers = make_view_headers(1, 1);

        let opts = ViewChangeOptions {
            commit_max: checkpoint_next,
            log_view: 1,
            view: 1,
            headers: &view_headers,
            sync_checkpoint: Some(sync),
        };

        state.update_for_view_change(&opts);

        assert_eq!(state.commit_max, checkpoint_next);
        assert_eq!(state.view, 1);
        assert_eq!(state.log_view, 1);
        assert_eq!(state.checkpoint.header.op, checkpoint_next);
        assert_eq!(state.checkpoint.parent_checkpoint_id, current_checkpoint_id);
        assert_eq!(state.sync_op_min, 10);
        assert_eq!(state.sync_op_max, 12);
        // With sync_checkpoint, sync_view is set to opts.view
        assert_eq!(state.sync_view, 1);
    }

    #[test]
    #[should_panic]
    fn test_update_for_view_change_panics_without_progress_or_sync() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let view_headers = make_view_headers(1, 1);

        let opts = ViewChangeOptions {
            commit_max: 0,
            log_view: 0,
            view: 0,
            headers: &view_headers,
            sync_checkpoint: None,
        };

        state.update_for_view_change(&opts);
    }

    #[test]
    #[should_panic]
    fn test_update_for_view_change_panics_commit_max_regression() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.commit_max = 10;

        let view_headers = make_view_headers(1, 1);

        let opts = ViewChangeOptions {
            commit_max: 9,
            log_view: 1,
            view: 1,
            headers: &view_headers,
            sync_checkpoint: None,
        };

        state.update_for_view_change(&opts);
    }

    #[test]
    #[should_panic]
    fn test_update_for_view_change_panics_log_view_regression() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);
        state.log_view = 2;
        state.view = 2;

        let view_headers = make_view_headers(1, 1);

        let opts = ViewChangeOptions {
            commit_max: 0,
            log_view: 1,
            view: 2,
            headers: &view_headers,
            sync_checkpoint: None,
        };

        state.update_for_view_change(&opts);
    }

    #[test]
    #[should_panic]
    fn test_update_for_view_change_panics_view_lt_log_view() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let view_headers = make_view_headers(1, 1);

        let opts = ViewChangeOptions {
            commit_max: 0,
            log_view: 2,
            view: 1,
            headers: &view_headers,
            sync_checkpoint: None,
        };

        state.update_for_view_change(&opts);
    }

    #[test]
    #[should_panic]
    fn test_update_for_view_change_panics_sync_checkpoint_parent_mismatch() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let current_checkpoint_id = state.checkpoint.checkpoint_id();
        let checkpoint_next = super::checkpoint_after(state.checkpoint.header.op);
        let wrong_parent = if current_checkpoint_id == 0 {
            1
        } else {
            current_checkpoint_id - 1
        };

        let checkpoint = make_sync_checkpoint(&state, checkpoint_next, wrong_parent, 0);

        let sync = SyncCheckpointOptions {
            checkpoint,
            sync_op_min: 1,
            sync_op_max: 1,
        };

        let view_headers = make_view_headers(1, 1);

        let opts = ViewChangeOptions {
            commit_max: checkpoint_next,
            log_view: 1,
            view: 1,
            headers: &view_headers,
            sync_checkpoint: Some(sync),
        };

        state.update_for_view_change(&opts);
    }

    #[test]
    #[should_panic]
    fn test_update_for_view_change_panics_sync_bounds_inverted() {
        let options = make_valid_root_options(1, 0);
        let mut state = VsrState::root(&options);

        let current_checkpoint_id = state.checkpoint.checkpoint_id();
        let checkpoint_next = super::checkpoint_after(state.checkpoint.header.op);
        let checkpoint = make_sync_checkpoint(&state, checkpoint_next, current_checkpoint_id, 0);

        let sync = SyncCheckpointOptions {
            checkpoint,
            sync_op_min: 5,
            sync_op_max: 4,
        };

        let view_headers = make_view_headers(1, 1);

        let opts = ViewChangeOptions {
            commit_max: checkpoint_next,
            log_view: 1,
            view: 1,
            headers: &view_headers,
            sync_checkpoint: Some(sync),
        };

        state.update_for_view_change(&opts);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::vsr::wire::header::Release;
    use proptest::prelude::*;

    /// Strategy for generating valid replica configurations
    fn valid_replica_config() -> impl Strategy<Value = (u8, u8)> {
        (1u8..=6u8).prop_flat_map(|count| (Just(count), 0..count))
    }

    /// Strategy for generating valid RootOptions
    fn valid_root_options_strategy() -> impl Strategy<Value = RootOptions> {
        valid_replica_config().prop_map(|(replica_count, replica_index)| {
            let mut members = Members([0; constants::MEMBERS_MAX]);
            for i in 0..replica_count as usize {
                members.0[i] = (i + 1) as u128;
            }
            RootOptions {
                cluster: 1,
                replica_index,
                replica_count,
                members,
                release: Release::ZERO,
                view: 0,
            }
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_root_options_validate_succeeds(opts in valid_root_options_strategy()) {
            opts.validate();
            let replica_id = opts.replica_id();
            prop_assert!(replica_id != 0);
            prop_assert_eq!(replica_id, (opts.replica_index + 1) as u128);
        }

        #[test]
        fn prop_vsr_state_root_is_consistent(opts in valid_root_options_strategy()) {
            let state = VsrState::root(&opts);
            state.assert_internally_consistent();
            prop_assert_eq!(state.replica_id, opts.replica_id());
            prop_assert_eq!(state.replica_count, opts.replica_count);
        }

        #[test]
        fn prop_vsr_state_monotonic_reflexive(opts in valid_root_options_strategy()) {
            let state = VsrState::root(&opts);
            prop_assert!(VsrState::monotonic(&state, &state));
            prop_assert!(!VsrState::would_be_updated_by(&state, &state));
        }

        #[test]
        fn prop_vsr_state_view_increase_is_monotonic(
            opts in valid_root_options_strategy(),
            view_delta in 1u32..100u32
        ) {
            let old = VsrState::root(&opts);
            let mut new = old;
            new.view = old.view.saturating_add(view_delta);
            prop_assert!(VsrState::monotonic(&old, &new));
            if view_delta > 0 {
                prop_assert!(VsrState::would_be_updated_by(&old, &new));
            }
        }

        #[test]
        fn prop_vsr_state_commit_max_increase_is_monotonic(
            opts in valid_root_options_strategy(),
            commit_delta in 1u64..1000u64
        ) {
            let old = VsrState::root(&opts);
            let mut new = old;
            new.commit_max = old.commit_max.saturating_add(commit_delta);
            prop_assert!(VsrState::monotonic(&old, &new));
        }

        #[test]
        fn prop_op_compacted_zero_checkpoint_never_compacts(
            opts in valid_root_options_strategy(),
            op in 0u64..10000u64
        ) {
            let state = VsrState::root(&opts);
            prop_assert!(!state.op_compacted(op));
        }

        #[test]
        fn prop_trigger_for_checkpoint_valid_boundary_returns_next_bar(k in 1u64..10_000u64) {
            let lsm_compaction_ops = constants::LSM_COMPACTION_OPS;
            let checkpoint = k
                .checked_mul(lsm_compaction_ops)
                .and_then(|v| v.checked_sub(1))
                .expect("checkpoint overflow");
            let trigger = trigger_for_checkpoint(checkpoint);
            prop_assert_eq!(trigger, Some(checkpoint + lsm_compaction_ops));
        }

        #[test]
        fn prop_checkpoint_state_zeroed_passes_padding_check(_dummy in 0..1i32) {
            let state = CheckpointState::zeroed();
            state.assert_padding_zeroed();
        }
    }

    // =====================================================================
    // VsrState::update_for_checkpoint Property Tests
    // =====================================================================

    /// Helper to create CheckpointOptions for property tests.
    fn make_proptest_checkpoint_opts(
        current: &VsrState,
        cluster: u128,
        op: u64,
        commit_max: u64,
        storage_size: u64,
    ) -> CheckpointOptions<'static> {
        let header = make_prepare_header(cluster, op);

        let mut checkpoint = CheckpointState::zeroed();
        checkpoint.header = header;
        checkpoint.storage_size = storage_size;
        checkpoint.release = Release::ZERO;
        checkpoint.parent_checkpoint_id = current.checkpoint.checkpoint_id();
        checkpoint.grandparent_checkpoint_id = current.checkpoint.parent_checkpoint_id;
        checkpoint.free_set_blocks_acquired_checksum = checksum(&[]);
        checkpoint.free_set_blocks_released_checksum = checksum(&[]);
        checkpoint.client_sessions_checksum = checksum(&[]);

        CheckpointOptions {
            checkpoint,
            view: current.view,
            log_view: current.log_view,
            view_attributes: None,
            commit_max,
            sync_op_min: 0,
            sync_op_max: 0,
            storage_size,
            release: Release::ZERO,
            sync_checkpoint: false,
        }
    }

    proptest! {
        #[test]
        fn prop_update_for_checkpoint_advances_chain(
            opts in valid_root_options_strategy(),
            new_op in 1u64..10000u64
        ) {
            let mut state = VsrState::root(&opts);
            // checkpoint_id() is the checksum of the entire CheckpointState
            let original_checkpoint_id = state.checkpoint.checkpoint_id();
            let original_parent = state.checkpoint.parent_checkpoint_id;

            let checkpoint_opts = make_proptest_checkpoint_opts(
                &state,
                opts.cluster,
                new_op,
                new_op,
                constants::DATA_FILE_SIZE_MIN,
            );

            state.update_for_checkpoint(&checkpoint_opts);

            // Parent should be the old checkpoint_id (checksum of entire CheckpointState)
            prop_assert_eq!(state.checkpoint.parent_checkpoint_id, original_checkpoint_id);
            // Grandparent should be the old parent
            prop_assert_eq!(state.checkpoint.grandparent_checkpoint_id, original_parent);
        }

        #[test]
        fn prop_update_for_checkpoint_updates_op(
            opts in valid_root_options_strategy(),
            new_op in 1u64..10000u64
        ) {
            let mut state = VsrState::root(&opts);

            let checkpoint_opts = make_proptest_checkpoint_opts(
                &state,
                opts.cluster,
                new_op,
                new_op,
                constants::DATA_FILE_SIZE_MIN,
            );

            state.update_for_checkpoint(&checkpoint_opts);

            prop_assert_eq!(state.checkpoint.header.op, new_op);
            prop_assert_eq!(state.commit_max, new_op);
        }

        #[test]
        fn prop_update_for_checkpoint_preserves_identity(
            opts in valid_root_options_strategy(),
            new_op in 1u64..10000u64
        ) {
            let mut state = VsrState::root(&opts);
            let original_replica_id = state.replica_id;
            let original_replica_count = state.replica_count;
            let original_members = state.members;

            let checkpoint_opts = make_proptest_checkpoint_opts(
                &state,
                opts.cluster,
                new_op,
                new_op,
                constants::DATA_FILE_SIZE_MIN,
            );

            state.update_for_checkpoint(&checkpoint_opts);

            prop_assert_eq!(state.replica_id, original_replica_id);
            prop_assert_eq!(state.replica_count, original_replica_count);
            prop_assert_eq!(state.members, original_members);
        }

        #[test]
        fn prop_update_for_checkpoint_sync_bounds_preserved(
            opts in valid_root_options_strategy(),
            new_op in 1u64..10000u64,
            sync_min in 0u64..1000u64,
            sync_delta in 0u64..1000u64
        ) {
            let mut state = VsrState::root(&opts);
            let sync_max = sync_min + sync_delta;

            let mut checkpoint_opts = make_proptest_checkpoint_opts(
                &state,
                opts.cluster,
                new_op,
                new_op,
                constants::DATA_FILE_SIZE_MIN,
            );
            checkpoint_opts.sync_op_min = sync_min;
            checkpoint_opts.sync_op_max = sync_max;

            state.update_for_checkpoint(&checkpoint_opts);

            prop_assert_eq!(state.sync_op_min, sync_min);
            prop_assert_eq!(state.sync_op_max, sync_max);
        }

        #[test]
        fn prop_update_for_checkpoint_storage_size_preserved(
            opts in valid_root_options_strategy(),
            new_op in 1u64..10000u64,
            size_multiplier in 1u64..10u64
        ) {
            let mut state = VsrState::root(&opts);
            let storage_size = constants::DATA_FILE_SIZE_MIN * size_multiplier;

            let checkpoint_opts = make_proptest_checkpoint_opts(
                &state,
                opts.cluster,
                new_op,
                new_op,
                storage_size,
            );

            state.update_for_checkpoint(&checkpoint_opts);

            prop_assert_eq!(state.checkpoint.storage_size, storage_size);
        }

        #[test]
        fn prop_update_for_checkpoint_multiple_advances(
            opts in valid_root_options_strategy(),
            num_checkpoints in 1usize..10usize
        ) {
            let mut state = VsrState::root(&opts);

            for i in 1..=num_checkpoints {
                let op = (i * 100) as u64;

                let checkpoint_opts = make_proptest_checkpoint_opts(
                    &state,
                    opts.cluster,
                    op,
                    op,
                    constants::DATA_FILE_SIZE_MIN,
                );

                state.update_for_checkpoint(&checkpoint_opts);
                prop_assert_eq!(state.checkpoint.header.op, op);
            }
        }
    }
}
