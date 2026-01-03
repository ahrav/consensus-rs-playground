//! Minimal replica stub embedding a Journal for Zig parity.

use crate::message_pool::MessagePool;

use super::constants::ClusterId;
use super::journal::Journal;
use super::storage::Storage;

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
        Self {
            journal: Journal::new(storage, replica),
            message_pool,
            cluster,
            op: 0,
            replica,
        }
    }
}
