//! Superblock: durable replica state for crash recovery.
//!
//! The superblock stores critical VSR state that must survive crashes and power failures.
//! Multiple copies are written to tolerate partial disk corruption. On recovery, replicas
//! read all copies and use quorum logic to select the authoritative state.
//!
//! # Layout
//!
//! The superblock zone contains [`SUPERBLOCK_COPIES`](constants::SUPERBLOCK_COPIES) copies,
//! each [`SUPERBLOCK_COPY_SIZE`] bytes, sector-aligned for direct I/O.

#![allow(dead_code)]
#[allow(unused_imports)]
use core::mem::size_of;
use core::ptr::NonNull;

use crate::vsr::wire::checksum;
#[allow(unused_imports)]
use crate::{
    constants::{self, SUPERBLOCK_VERSION},
    storage::Storage,
    util::{AlignedBox, align_up, as_bytes_unchecked},
    vsr::{
        Header, ViewChangeArray, ViewChangeCommand, ViewChangeSlice, VsrState, wire::Checksum128,
    },
};
// use crate::vsr::superblock_quorums::{Quorums, RepairIterator, Threshold};

/// Extra space reserved in each superblock copy for future client session tracking.
///
/// Sized to hold headers for the difference between max clients and pipeline depth,
/// allowing session state to grow without breaking the wire format.
pub const SUPERBLOCK_COPY_PADDING: usize =
    (constants::CLIENTS_MAX - constants::PIPELINE_PREPARE_QUEUE_MAX) * size_of::<Header>();

/// Total size of one superblock copy on disk, sector-aligned.
pub const SUPERBLOCK_COPY_SIZE: usize = align_up(
    size_of::<SuperBlockHeader>() + SUPERBLOCK_COPY_PADDING,
    constants::SECTOR_SIZE,
);

/// Max concurrent I/O operations to the superblock zone.
const MAX_QUEUE_DEPTH: usize = 16;

/// Upper bound on repair attempts before giving up.
///
/// Set to 2x copies to allow re-reading each copy after corrective writes.
const MAX_REPAIR_ITERATIONS: usize = constants::SUPERBLOCK_COPIES * 2;

// Compile-time invariant checks.
const _: () = {
    assert!(SUPERBLOCK_VERSION > 0);

    assert!(constants::SUPERBLOCK_ZONE_SIZE > 0);
    assert!(
        constants::SUPERBLOCK_ZONE_SIZE
            >= SUPERBLOCK_COPY_SIZE as u64 * constants::SUPERBLOCK_COPIES as u64
    );

    assert!(SUPERBLOCK_COPY_SIZE > 0);
    assert!(SUPERBLOCK_COPY_SIZE % constants::SECTOR_SIZE == 0);

    assert!(size_of::<SuperBlockHeader>() > 0);
    assert!(size_of::<SuperBlockHeader>() <= SUPERBLOCK_COPY_SIZE);

    // 4 copies minimum for fault tolerance (tolerates 1 corrupted copy).
    // 8 copies maximum to bound storage overhead.
    assert!(constants::SUPERBLOCK_COPIES >= 4);
    assert!(constants::SUPERBLOCK_COPIES <= 8);

    assert!(MAX_QUEUE_DEPTH > 0);
    assert!(MAX_QUEUE_DEPTH <= 256);

    assert!(MAX_REPAIR_ITERATIONS >= constants::SUPERBLOCK_COPIES);
};

/// Validates that an I/O region falls within the superblock zone and is sector-aligned.
///
/// # Panics
///
/// Panics if the region is empty, exceeds zone bounds, or is not sector-aligned.
pub fn assert_bounds(offset: u64, len: usize) {
    assert!(len > 0);
    assert!(len <= constants::SUPERBLOCK_ZONE_SIZE as usize);
    assert!(offset < constants::SUPERBLOCK_ZONE_SIZE);
    assert!(offset.checked_add(len as u64).is_some());
    assert!(offset + (len as u64) <= constants::SUPERBLOCK_ZONE_SIZE);
    assert!(offset % constants::SECTOR_SIZE as u64 == 0);
    assert!(len % constants::SECTOR_SIZE == 0);
}

/// Persistent header written to each superblock copy.
///
/// Uses `#[repr(C)]` for stable wire layout. Field order matches disk format exactly.
/// Padding fields ensure future extensibility and must always be zero.
///
/// # Checksum Coverage
///
/// The checksum covers bytes starting after `copy` (offset 34) through end of struct.
/// This excludes `checksum`, `checksum_padding`, and `copy` since:
/// - `checksum` obviously can't cover itself
/// - `copy` differs between copies but shouldn't affect logical equality
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SuperBlockHeader {
    /// AEGIS-128L MAC covering bytes after `copy` field.
    pub checksum: Checksum128,
    /// Reserved for checksum extension; must be zero.
    pub checksum_padding: Checksum128,
    /// Which of the N redundant copies this is (0..SUPERBLOCK_COPIES).
    pub copy: u16,

    /// Schema version for forward compatibility.
    pub version: u16,
    /// Cluster identifier; must match across all replicas.
    pub cluster: u128,
    /// Monotonically increasing write sequence number.
    pub sequence: u64,
    /// Checksum of the previous superblock (chain integrity).
    pub parent: u128,
    /// Reserved; must be zero.
    pub parent_padding: u128,

    /// Current VSR protocol state (view, commit index, etc).
    pub vsr_state: VsrState,

    /// Number of valid entries in `view_headers_all`.
    pub view_headers_count: u32,
    /// Headers from recent view changes for view-change recovery.
    pub view_headers_all: ViewChangeArray,
}

impl SuperBlockHeader {
    /// Byte offset where checksum coverage begins.
    ///
    /// Excludes: checksum (16) + checksum_padding (16) + copy (2) = 34 bytes.
    const CHECKSUM_EXCLUDE_SIZE: usize = size_of::<u128>() + size_of::<u128>() + size_of::<u16>();

    /// Creates a zero-initialized header.
    ///
    /// # Safety
    ///
    /// Safe because `SuperBlockHeader` contains only primitive types and arrays thereof;
    /// zero is a valid bit pattern for all fields.
    pub fn zeroed() -> Self {
        // SAFETY: All fields are primitives or arrays of primitives where zero is valid.
        let header: Self = unsafe { core::mem::zeroed() };

        assert_eq!(header.checksum, 0);
        assert_eq!(header.sequence, 0);
        assert_eq!(header.copy, 0);

        header
    }

    /// Computes the checksum over all fields except checksum, padding, and copy.
    ///
    /// # Safety
    ///
    /// Relies on the invariant that headers are created via [`zeroed()`](Self::zeroed)
    /// or read from storage with all bytes initialized.
    pub fn calculate_checksum(&self) -> Checksum128 {
        // SAFETY: SuperBlockHeader instances are created via `zeroed()` or read from
        // storage, ensuring all bytes are initialized.
        let bytes = unsafe { as_bytes_unchecked(self) };
        assert!(bytes.len() > Self::CHECKSUM_EXCLUDE_SIZE);

        checksum(&bytes[Self::CHECKSUM_EXCLUDE_SIZE..])
    }

    /// Returns true if the stored checksum matches the computed value.
    ///
    /// Also validates that `checksum_padding` is zero.
    pub fn valid_checksum(&self) -> bool {
        if self.checksum_padding != 0 {
            return false;
        }
        self.checksum == self.calculate_checksum()
    }

    /// Computes and stores the checksum.
    ///
    /// # Panics
    ///
    /// Panics if `checksum_padding` is non-zero (indicates corruption or misuse).
    pub fn set_checksum(&mut self) {
        assert_eq!(self.checksum_padding, 0);
        self.checksum = self.calculate_checksum();
        assert!(self.valid_checksum())
    }

    /// Logical equality ignoring `checksum` and `copy` fields.
    ///
    /// Two headers are equal if they represent the same VSR state, even if
    /// stored in different copy slots or with different checksums.
    pub fn equal(&self, other: &SuperBlockHeader) -> bool {
        self.version == other.version
            && self.cluster == other.cluster
            && self.sequence == other.sequence
            && self.parent == other.parent
            // SAFETY: VsrState and ViewChangeArray are repr(C) with initialized bytes.
            && unsafe {
                as_bytes_unchecked(&self.vsr_state) == as_bytes_unchecked(&other.vsr_state)
            }
            && self.view_headers_count == other.view_headers_count
            && unsafe {
                as_bytes_unchecked(&self.view_headers_all)
                    == as_bytes_unchecked(&other.view_headers_all)
            }
    }

    /// Validates structural invariants (Tiger Style defense-in-depth).
    ///
    /// Call after reading from storage or before critical operations.
    fn assert_invariants(&self) {
        assert_eq!(self.checksum_padding, 0);
        assert_eq!(self.parent_padding, 0);
        assert!(self.version == 0 || self.version == SUPERBLOCK_VERSION);
        assert!((self.copy as usize) < constants::SUPERBLOCK_COPIES || self.copy == 0);
    }
}
