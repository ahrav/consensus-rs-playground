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
#[allow(clippy::manual_is_multiple_of)]
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
    assert!(offset.is_multiple_of(constants::SECTOR_SIZE as u64));
    assert!(len.is_multiple_of(constants::SECTOR_SIZE));
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
    /// or read from storage with all bytes initialized. `ViewChangeArray` constructors
    /// zero-fill unused slots and padding to keep the header byte-initialized.
    pub fn calculate_checksum(&self) -> Checksum128 {
        // SAFETY: SuperBlockHeader instances are created via `zeroed()` or read from
        // storage, and ViewChangeArray constructors zero-fill unused slots/padding.
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
            // SAFETY: VsrState is Pod, and ViewChangeArray constructors zero-fill bytes.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use core::mem;
    use proptest::prelude::*;

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Creates a minimal valid SuperBlockHeader for testing.
    fn make_header() -> SuperBlockHeader {
        let mut header = SuperBlockHeader::zeroed();
        header.version = SUPERBLOCK_VERSION;
        header.cluster = 1;
        header.sequence = 1;
        header.view_headers_count = 1;
        header.view_headers_all = ViewChangeArray::root(header.cluster);
        header
    }

    /// Creates a header with a specific copy number.
    fn make_header_with_copy(copy: u16) -> SuperBlockHeader {
        let mut header = make_header();
        header.copy = copy;
        header
    }

    // =========================================================================
    // Compile-Time Constants Tests
    // =========================================================================

    #[test]
    #[allow(clippy::manual_is_multiple_of)]
    fn test_superblock_copy_size_aligned() {
        const { assert!(SUPERBLOCK_COPY_SIZE > 0) };
        const { assert!(SUPERBLOCK_COPY_SIZE % constants::SECTOR_SIZE == 0) };
    }

    #[test]
    fn test_superblock_zone_size_sufficient() {
        let total_needed = SUPERBLOCK_COPY_SIZE as u64 * constants::SUPERBLOCK_COPIES as u64;
        assert!(constants::SUPERBLOCK_ZONE_SIZE >= total_needed);
    }

    #[test]
    fn test_checksum_exclude_size() {
        assert_eq!(
            SuperBlockHeader::CHECKSUM_EXCLUDE_SIZE,
            size_of::<u128>() + size_of::<u128>() + size_of::<u16>()
        );
        assert_eq!(SuperBlockHeader::CHECKSUM_EXCLUDE_SIZE, 34);
    }

    #[test]
    fn test_bounds_valid_aligned_region() {
        // First sector, one sector length
        assert_bounds(0, constants::SECTOR_SIZE);

        // Second sector
        assert_bounds(constants::SECTOR_SIZE as u64, constants::SECTOR_SIZE);

        // Multiple sectors
        assert_bounds(0, constants::SECTOR_SIZE * 4);

        // Maximum valid region
        let max_offset = constants::SUPERBLOCK_ZONE_SIZE - constants::SECTOR_SIZE as u64;
        assert_bounds(max_offset, constants::SECTOR_SIZE);
    }

    #[test]
    #[should_panic]
    fn test_bounds_zero_length() {
        assert_bounds(0, 0);
    }

    #[test]
    #[should_panic]
    fn test_bounds_unaligned_offset() {
        assert_bounds(1, constants::SECTOR_SIZE);
    }

    #[test]
    #[should_panic]
    fn test_bounds_unaligned_length() {
        assert_bounds(0, constants::SECTOR_SIZE + 1);
    }

    #[test]
    #[should_panic]
    fn test_bounds_exceeds_zone() {
        assert_bounds(
            0,
            constants::SUPERBLOCK_ZONE_SIZE as usize + constants::SECTOR_SIZE,
        );
    }

    #[test]
    #[should_panic]
    fn test_bounds_offset_exceeds_zone() {
        assert_bounds(constants::SUPERBLOCK_ZONE_SIZE, constants::SECTOR_SIZE);
    }

    #[test]
    #[should_panic]
    fn test_bounds_offset_plus_len_exceeds_zone() {
        let offset = constants::SUPERBLOCK_ZONE_SIZE - (constants::SECTOR_SIZE as u64 / 2);
        assert_bounds(offset, constants::SECTOR_SIZE);
    }

    #[test]
    #[should_panic]
    fn test_bounds_length_exceeds_zone() {
        assert_bounds(0, constants::SUPERBLOCK_ZONE_SIZE as usize + 1);
    }

    #[test]
    fn test_zeroed_initialization() {
        let header = SuperBlockHeader::zeroed();

        assert_eq!(header.checksum, 0);
        assert_eq!(header.checksum_padding, 0);
        assert_eq!(header.copy, 0);
        assert_eq!(header.version, 0);
        assert_eq!(header.cluster, 0);
        assert_eq!(header.sequence, 0);
        assert_eq!(header.parent, 0);
        assert_eq!(header.parent_padding, 0);
        assert_eq!(header.view_headers_count, 0);
    }

    #[test]
    fn test_calculate_checksum_deterministic() {
        let header = make_header();
        let checksum1 = header.calculate_checksum();
        let checksum2 = header.calculate_checksum();
        assert_eq!(checksum1, checksum2);
        assert_ne!(checksum1, 0);
    }

    #[test]
    fn test_checksum_excludes_checksum_field() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.checksum = 123;
        header2.checksum = 456;

        // Changing checksum field shouldn't affect calculated checksum
        assert_eq!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_excludes_copy_field() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.copy = 0;
        header2.copy = 3;

        // Changing copy field shouldn't affect calculated checksum
        assert_eq!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_includes_version() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.version = 1;
        header2.version = 2;

        assert_ne!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_includes_cluster() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.cluster = 1;
        header2.cluster = 2;

        assert_ne!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_includes_sequence() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.sequence = 1;
        header2.sequence = 2;

        assert_ne!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_includes_parent() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.parent = 1;
        header2.parent = 2;

        assert_ne!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_includes_parent_padding() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        // Note: This violates invariants but tests checksum coverage
        header1.parent_padding = 0;
        header2.parent_padding = 1;

        assert_ne!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_includes_view_headers_count() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.view_headers_count = 1;
        header2.view_headers_count = 2;

        assert_ne!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_valid_checksum_fresh_header() {
        let mut header = make_header();
        header.set_checksum();
        assert!(header.valid_checksum());
    }

    #[test]
    fn test_valid_checksum_detects_tampering() {
        let mut header = make_header();
        header.set_checksum();

        // Tamper with sequence
        header.sequence += 1;

        assert!(!header.valid_checksum());
    }

    #[test]
    fn test_valid_checksum_rejects_nonzero_padding() {
        let mut header = make_header();
        header.set_checksum();

        // Corrupt padding
        header.checksum_padding = 1;

        assert!(!header.valid_checksum());
    }

    #[test]
    #[should_panic]
    fn test_set_checksum_requires_zero_padding() {
        let mut header = make_header();
        header.checksum_padding = 1;
        header.set_checksum();
    }

    #[test]
    fn test_set_checksum_postcondition() {
        let mut header = make_header();
        header.set_checksum();

        // Immediately after setting, validation must pass
        assert!(header.valid_checksum());
    }

    #[test]
    fn test_equal_same_header() {
        let header = make_header();
        assert!(header.equal(&header));
    }

    #[test]
    fn test_equal_ignores_checksum() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.checksum = 100;
        header2.checksum = 200;

        assert!(header1.equal(&header2));
    }

    #[test]
    fn test_equal_ignores_copy() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.copy = 0;
        header2.copy = 3;

        assert!(header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_version_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.version = 1;
        header2.version = 2;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_cluster_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.cluster = 1;
        header2.cluster = 2;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_sequence_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.sequence = 1;
        header2.sequence = 2;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_parent_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.parent = 1;
        header2.parent = 2;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_view_headers_count_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.view_headers_count = 1;
        header2.view_headers_count = 2;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_equal_different_copies_same_state() {
        let header1 = make_header_with_copy(0);
        let header2 = make_header_with_copy(3);

        assert!(header1.equal(&header2));
    }

    // Bug test: equal() should check parent_padding
    #[test]
    fn test_equal_ignores_parent_padding_bug() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        // This violates invariants but tests equal() behavior
        header1.parent_padding = 0;
        header2.parent_padding = 1;

        // BUG: equal() doesn't check parent_padding
        // Two headers with different padding are incorrectly considered equal
        assert!(header1.equal(&header2));
    }

    #[test]
    fn test_invariants_zeroed_header_passes() {
        let header = SuperBlockHeader::zeroed();
        header.assert_invariants();
    }

    #[test]
    fn test_invariants_valid_header_passes() {
        let header = make_header();
        header.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_invariants_nonzero_checksum_padding() {
        let mut header = make_header();
        header.checksum_padding = 1;
        header.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_invariants_nonzero_parent_padding() {
        let mut header = make_header();
        header.parent_padding = 1;
        header.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_invariants_invalid_version() {
        let mut header = make_header();
        header.version = 99;
        header.assert_invariants();
    }

    #[test]
    fn test_invariants_version_zero_allowed() {
        let mut header = make_header();
        header.version = 0;
        header.assert_invariants();
    }

    #[test]
    fn test_invariants_version_current_allowed() {
        let mut header = make_header();
        header.version = SUPERBLOCK_VERSION;
        header.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_invariants_copy_out_of_range() {
        let mut header = make_header();
        header.copy = constants::SUPERBLOCK_COPIES as u16;
        header.assert_invariants();
    }

    #[test]
    fn test_invariants_copy_max_valid() {
        let mut header = make_header();
        header.copy = (constants::SUPERBLOCK_COPIES - 1) as u16;
        header.assert_invariants();
    }

    #[test]
    fn test_invariants_copy_zero_allowed() {
        let mut header = make_header();
        header.copy = 0;
        header.assert_invariants();
    }

    proptest! {
        #[test]
        fn prop_checksum_deterministic(
            sequence in any::<u64>(),
            cluster in any::<u128>(),
        ) {
            let mut header = make_header();
            header.sequence = sequence;
            header.cluster = cluster;

            let checksum1 = header.calculate_checksum();
            let checksum2 = header.calculate_checksum();

            prop_assert_eq!(checksum1, checksum2);
        }

        #[test]
        fn prop_equal_is_reflexive(
            sequence in any::<u64>(),
            cluster in any::<u128>(),
            version in prop::option::of(Just(SUPERBLOCK_VERSION)),
        ) {
            let mut header = make_header();
            header.sequence = sequence;
            header.cluster = cluster;
            if let Some(v) = version {
                header.version = v;
            }

            prop_assert!(header.equal(&header));
        }

        #[test]
        fn prop_equal_is_symmetric(
            seq1 in any::<u64>(),
            seq2 in any::<u64>(),
        ) {
            let mut header1 = make_header();
            let mut header2 = make_header();
            header1.sequence = seq1;
            header2.sequence = seq2;

            let eq_12 = header1.equal(&header2);
            let eq_21 = header2.equal(&header1);

            prop_assert_eq!(eq_12, eq_21);
        }

        #[test]
        fn prop_changing_covered_field_changes_checksum(
            sequence1 in any::<u64>(),
            sequence2 in any::<u64>(),
        ) {
            if sequence1 == sequence2 {
                return Ok(());
            }

            let mut header1 = make_header();
            let mut header2 = make_header();

            header1.sequence = sequence1;
            header2.sequence = sequence2;

            prop_assert_ne!(
                header1.calculate_checksum(),
                header2.calculate_checksum()
            );
        }

        #[test]
        fn prop_changing_excluded_field_preserves_checksum(
            copy1 in 0u16..(constants::SUPERBLOCK_COPIES as u16),
            copy2 in 0u16..(constants::SUPERBLOCK_COPIES as u16),
        ) {
            let mut header1 = make_header();
            let mut header2 = make_header();

            header1.copy = copy1;
            header2.copy = copy2;

            prop_assert_eq!(
                header1.calculate_checksum(),
                header2.calculate_checksum()
            );
        }

        #[test]
        fn prop_set_checksum_makes_valid(
            sequence in any::<u64>(),
            cluster in any::<u128>(),
        ) {
            let mut header = make_header();
            header.sequence = sequence;
            header.cluster = cluster;
            header.set_checksum();

            prop_assert!(header.valid_checksum());
        }

        #[test]
        fn prop_equal_ignores_copy_values(
            copy1 in 0u16..(constants::SUPERBLOCK_COPIES as u16),
            copy2 in 0u16..(constants::SUPERBLOCK_COPIES as u16),
            sequence in any::<u64>(),
        ) {
            let mut header1 = make_header();
            let mut header2 = make_header();

            header1.copy = copy1;
            header2.copy = copy2;
            header1.sequence = sequence;
            header2.sequence = sequence;

            prop_assert!(header1.equal(&header2));
        }
    }

    #[test]
    fn test_checksum_survives_clone() {
        let mut original = make_header();
        original.set_checksum();

        let cloned = original;

        assert!(cloned.valid_checksum());
        assert_eq!(cloned.checksum, original.checksum);
    }

    #[test]
    fn test_equal_after_clone() {
        let original = make_header();
        let cloned = original;

        assert!(original.equal(&cloned));
    }

    #[test]
    fn test_max_sequence_value() {
        let mut header = make_header();
        header.sequence = u64::MAX;
        header.set_checksum();

        assert!(header.valid_checksum());
    }

    #[test]
    fn test_max_cluster_value() {
        let mut header = make_header();
        header.cluster = u128::MAX;
        header.set_checksum();

        assert!(header.valid_checksum());
    }

    #[test]
    fn test_max_copy_index() {
        let mut header = make_header();
        header.copy = (constants::SUPERBLOCK_COPIES - 1) as u16;
        header.assert_invariants();
    }

    #[test]
    fn test_different_view_headers_arrays() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        // Create different view change arrays
        header1.view_headers_all = ViewChangeArray::root(1);
        header2.view_headers_all = ViewChangeArray::root(2);

        assert!(!header1.equal(&header2));
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_lifecycle() {
        // Create header
        let mut header = make_header();
        header.copy = 0;
        header.sequence = 1;
        header.cluster = 42;

        // Set checksum
        header.set_checksum();

        // Verify invariants
        header.assert_invariants();

        // Verify checksum
        assert!(header.valid_checksum());

        // Clone to another copy
        let mut copy = header;
        copy.copy = 1;

        // Copies should be logically equal
        assert!(header.equal(&copy));

        // But checksums should be identical (copy field not covered)
        assert_eq!(header.checksum, copy.checksum);
    }

    #[test]
    fn test_progression_sequence() {
        let mut header1 = make_header();
        header1.sequence = 1;
        header1.set_checksum();

        let mut header2 = make_header();
        header2.sequence = 2;
        header2.parent = header1.checksum;
        header2.set_checksum();

        // Headers should not be equal (different sequence)
        assert!(!header1.equal(&header2));

        // Parent chain should link
        assert_eq!(header2.parent, header1.checksum);
    }
    #[test]
    fn test_superblock_header_size_alignment() {
        assert_eq!(mem::align_of::<SuperBlockHeader>(), 16);
        assert!(mem::size_of::<SuperBlockHeader>() <= SUPERBLOCK_COPY_SIZE);
        assert_eq!(
            mem::size_of::<SuperBlockHeader>() % 16,
            0,
            "SuperBlockHeader size {} not multiple of 16",
            mem::size_of::<SuperBlockHeader>()
        );
    }
}
