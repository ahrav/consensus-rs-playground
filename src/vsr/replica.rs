//! Minimal replica stub embedding a Journal for Zig parity.

use crate::{constants, message_pool::MessagePool};

use super::{HeaderPrepare, checkpoint_after, prepare_max_for_checkpoint};
use super::constants::ClusterId;
use super::journal::Journal;
use super::state::{CheckpointState, VsrState};
use super::storage::Storage;
use super::superblock::SuperBlockHeader;
use super::view_change::ViewChangeArray;

/// Minimal replica stub used to align Journal callbacks with Zig.
///
/// # Safety
///
/// The replica (and embedded journal) must not be moved while I/O is in flight,
/// because write/read completions store raw pointers to the journal.
#[repr(C)]
pub struct Replica<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> {
    pub journal: Journal<S, WRITE_OPS, WRITE_OPS_WORDS>,
    pub message_pool: MessagePool,
    pub cluster: ClusterId,
    pub op: u64,
    pub replica: u8,
    pub replica_count: u8,
    pub superblock: ReplicaSuperBlock,
}

/// Minimal superblock stub providing `working` state for recovery.
///
/// TODO: Replace with full `SuperBlock` integration when ported.
#[repr(C)]
pub struct ReplicaSuperBlock {
    pub working: SuperBlockHeader,
}

impl ReplicaSuperBlock {
    fn new_root(cluster: ClusterId, replica_count: u8) -> Self {
        let mut working = SuperBlockHeader::zeroed();

        working.cluster = cluster;
        working.version = constants::SUPERBLOCK_VERSION;

        let mut state = VsrState::zeroed();
        let mut checkpoint = CheckpointState::zeroed();
        checkpoint.header = HeaderPrepare::root(cluster);
        state.checkpoint = checkpoint;
        state.replica_count = replica_count;
        state.view = 0;
        state.log_view = 0;

        working.vsr_state = state;

        let view_headers = ViewChangeArray::root(cluster);
        working.view_headers_count = view_headers.len() as u32;
        working.view_headers_all = view_headers.into_array();

        working.set_checksum();

        Self { working }
    }
}

impl<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize>
    Replica<S, WRITE_OPS, WRITE_OPS_WORDS>
{
    pub fn new(
        storage: *mut S,
        replica: u8,
        cluster: ClusterId,
        message_pool: MessagePool,
    ) -> Self {
        let replica_count = 1;

        Self {
            journal: Journal::new(storage, replica),
            message_pool,
            cluster,
            op: 0,
            replica,
            replica_count,
            superblock: ReplicaSuperBlock::new_root(cluster, replica_count),
        }
    }

    #[inline]
    pub fn op_checkpoint(&self) -> u64 {
        self.superblock.working.vsr_state.checkpoint.header.op
    }

    #[inline]
    pub fn op_checkpoint_next(&self) -> u64 {
        checkpoint_after(self.op_checkpoint())
    }

    #[inline]
    pub fn op_prepare_max(&self) -> u64 {
        prepare_max_for_checkpoint(self.op_checkpoint_next())
            .expect("op_prepare_max requires a valid checkpoint")
    }

    #[inline]
    pub fn solo(&self) -> bool {
        // Stub does not model standbys; `replica_count` must be configured by callers.
        self.replica_count == 1
    }
}
