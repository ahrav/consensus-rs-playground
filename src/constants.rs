//! Crate-wide protocol and storage constants.
//!
//! Changing constants that affect wire framing or on-disk layout is a breaking change.

// =============================================================================
// Platform verification
// =============================================================================

// Compile-time proof that u32 -> usize is safe on this platform.
const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

// =============================================================================
// System / CPU constants
// =============================================================================

/// CPU cache line size in bytes.
pub const CACHE_LINE_SIZE: u64 = 64;

/// Enable extra verification checks in debug builds or with the `verify` feature.
pub const VERIFY: bool = cfg!(any(debug_assertions, feature = "verify"));

// =============================================================================
// Disk / storage constants
// =============================================================================

/// Sector size in bytes used for disk I/O alignment.
pub const SECTOR_SIZE: usize = 4096;

/// Default sector size for the storage engine when not explicitly configured.
pub const SECTOR_SIZE_DEFAULT: usize = SECTOR_SIZE;

/// Minimum supported sector size (legacy 512-byte sectors).
pub const SECTOR_SIZE_MIN: usize = 512;

/// Maximum supported sector size (64 KiB for large-block devices).
pub const SECTOR_SIZE_MAX: usize = 65_536;

/// Storage block size.
pub const BLOCK_SIZE: u64 = 512 * 1024;

/// Default number of concurrent I/O operations for the storage engine.
pub const IO_ENTRIES_DEFAULT: u32 = 256;

/// Bounds next-tick callback processing to prevent runaway loops.
pub const MAX_NEXT_TICK_ITERATIONS: usize = 10_000;

// =============================================================================
// VSR wire constants
// =============================================================================

/// Fixed wire header size in bytes.
pub const HEADER_SIZE: u32 = 256;

/// Maximum wire message size in bytes (header + body).
pub const MESSAGE_SIZE_MAX: u32 = 1 << 20; // 1 MiB

/// Maximum message body size in bytes (`MESSAGE_SIZE_MAX - HEADER_SIZE`).
pub const MESSAGE_BODY_SIZE_MAX: u32 = MESSAGE_SIZE_MAX - HEADER_SIZE;

/// VSR protocol version. Increment for wire-incompatible changes.
pub const VSR_VERSION: u16 = 1;

/// Cluster identifier type used on the wire.
pub type ClusterId = u128;

/// Operations reserved for future use.
pub const VSR_OPERATIONS_RESERVED: u8 = 128;

// =============================================================================
// VSR / superblock constants
// =============================================================================

/// Maximum number of replicas in a cluster.
pub const REPLICAS_MAX: usize = 6;

/// Maximum number of standbys in a cluster.
pub const STANDBYS_MAX: usize = 6;

/// Maximum total members in a cluster (`replicas + standbys`).
pub const MEMBERS_MAX: usize = REPLICAS_MAX + STANDBYS_MAX;

/// Maximum number of replicas required for a quorum.
pub const QUORUM_REPLICATION_MAX: u64 = 3;

/// Number of superblock copies stored on disk.
pub const SUPERBLOCK_COPIES: usize = 4;

/// Superblock format version. Increment for on-disk layout changes.
pub const SUPERBLOCK_VERSION: u16 = 2;

/// Total size of the superblock zone (all copies) in bytes.
pub const SUPERBLOCK_ZONE_SIZE: u64 = 1024 * 1024;

/// Maximum number of client ids tracked for reply deduplication.
pub const CLIENTS_MAX: usize = 64;

/// Max in-flight prepares in the pipeline.
pub const PIPELINE_PREPARE_QUEUE_MAX: usize = 8;

/// View-change suffix capacity used to derive view header limits.
pub const VIEW_CHANGE_HEADERS_SUFFIX_MAX: usize = PIPELINE_PREPARE_QUEUE_MAX + 1;

/// Maximum number of view headers retained during view changes.
pub const VIEW_HEADERS_MAX: usize = VIEW_CHANGE_HEADERS_SUFFIX_MAX + 2;

/// Number of slots in the journal ring.
pub const JOURNAL_SLOT_COUNT: usize = 1024;

/// Size of the journal headers section in bytes.
pub const JOURNAL_SIZE_HEADERS: u64 = (JOURNAL_SLOT_COUNT as u64) * (HEADER_SIZE as u64);

/// Size of the journal prepares section in bytes.
pub const JOURNAL_SIZE_PREPARES: u64 = (JOURNAL_SLOT_COUNT as u64) * (MESSAGE_SIZE_MAX as u64);

/// The maximum number of concurrent WAL read I/O operations to allow at once.
pub const JOURNAL_IOPS_READ_MAX: u32 = 8;

/// The maximum number of concurrent WAL write I/O operations to allow at once.
/// Ideally this is at least as high as PIPELINE_PREPARE_QUEUE_MAX, but it is safe to be lower.
pub const JOURNAL_IOPS_WRITE_MAX: u32 = 32;

/// Maximum operations between checkpoints. Derived from journal size minus:
/// - One full LSM compaction cycle (32 ops)
/// - Space for in-flight pipeline prepares (doubled for view-change safety, rounded to compaction boundary)
///
/// This ensures the journal never wraps around uncommitted entries and LSM compaction
/// completes before each checkpoint.
pub const VSR_CHECKPOINT_OPS: usize = JOURNAL_SLOT_COUNT
    - LSM_COMPACTION_OPS as usize
    - LSM_COMPACTION_OPS as usize
        * (PIPELINE_PREPARE_QUEUE_MAX * 2).div_ceil(LSM_COMPACTION_OPS as usize);

/// Total journal size in bytes.
pub const JOURNAL_SIZE: usize = JOURNAL_SLOT_COUNT * MESSAGE_SIZE_MAX_USIZE;

/// On-disk size of a single client reply entry in bytes.
pub const CLIENT_REPLY_SIZE: usize = 32;

/// Total bytes reserved in the superblock for client replies.
pub const CLIENT_REPLIES_SIZE: usize = CLIENTS_MAX * CLIENT_REPLY_SIZE;

/// Upper bound for a configured storage size limit.
pub const STORAGE_SIZE_LIMIT_MAX: u64 = 1 << 40; // 1 TiB

/// Minimum data file size in bytes, aligned to block boundaries.
pub const DATA_FILE_SIZE_MIN: u64 = align_forward_u64(
    SUPERBLOCK_ZONE_SIZE + JOURNAL_SIZE as u64 + CLIENT_REPLIES_SIZE as u64,
    BLOCK_SIZE,
);

// =============================================================================
// LSM constants
// =============================================================================

/// Number of levels in the LSM tree.
pub const LSM_LEVELS: u64 = 7;

/// Growth factor for LSM levels.
pub const LSM_GROWTH_FACTOR: u64 = 8;

/// Number of compaction operations.
pub const LSM_COMPACTION_OPS: u64 = 32;

/// Maximum number of snapshots.
pub const LSM_SNAPSHOTS_MAX: u64 = 32;

/// Extra blocks for manifest compaction.
pub const LSM_MANIFEST_COMPACT_EXTRA_BLOCKS: u64 = 1;

/// Threshold for table coalescing (percent).
pub const LSM_TABLE_COALESCING_THRESHOLD_PERCENT: u64 = 50;

/// Maximum number of VSR releases.
pub const VSR_RELEASES_MAX: u64 = 64;

/// Maximum number of scans.
pub const LSM_SCANS_MAX: u64 = 6;

// =============================================================================
// Compile-time design integrity assertions
// =============================================================================

const _: () = {
    // Storage sector constraints.
    assert!(SECTOR_SIZE > 0);
    assert!(SECTOR_SIZE.is_power_of_two());
    assert!(SECTOR_SIZE_MIN.is_power_of_two());
    assert!(SECTOR_SIZE_MAX.is_power_of_two());
    assert!(SECTOR_SIZE_MIN <= SECTOR_SIZE_DEFAULT);
    assert!(SECTOR_SIZE_DEFAULT <= SECTOR_SIZE_MAX);

    // This crate currently assumes 64-bit addressing for storage sizing.
    assert!(size_of::<usize>() >= 8);

    assert!(IO_ENTRIES_DEFAULT > 0);
    assert!(IO_ENTRIES_DEFAULT.is_power_of_two());

    assert!(MAX_NEXT_TICK_ITERATIONS > 0);
    assert!(MAX_NEXT_TICK_ITERATIONS <= 100_000);

    // Wire header constraints.
    assert!(HEADER_SIZE > 0);
    assert!(HEADER_SIZE == 256);
    assert!(HEADER_SIZE.is_multiple_of(16));

    // Wire message size constraints.
    assert!(MESSAGE_SIZE_MAX > 0);
    assert!(MESSAGE_SIZE_MAX > HEADER_SIZE);
    assert!(MESSAGE_SIZE_MAX.is_power_of_two());
    assert!(MESSAGE_BODY_SIZE_MAX == MESSAGE_SIZE_MAX - HEADER_SIZE);
    assert!(MESSAGE_BODY_SIZE_MAX > 0);

    // Protocol version.
    assert!(VSR_VERSION > 0);

    // Superblock / cluster constraints.
    assert!(REPLICAS_MAX > 0);
    assert!(STANDBYS_MAX > 0);
    assert!(MEMBERS_MAX == REPLICAS_MAX + STANDBYS_MAX);

    assert!(SUPERBLOCK_COPIES >= 4);
    assert!(SUPERBLOCK_COPIES <= 8);
    assert!(SUPERBLOCK_COPIES.is_multiple_of(2));

    assert!(SUPERBLOCK_VERSION > 0);
    assert!(SUPERBLOCK_ZONE_SIZE > 0);
    assert!(SUPERBLOCK_ZONE_SIZE.is_power_of_two());

    assert!(DATA_FILE_SIZE_MIN > 0);
    assert!(
        DATA_FILE_SIZE_MIN
            >= SUPERBLOCK_ZONE_SIZE + JOURNAL_SIZE as u64 + CLIENT_REPLIES_SIZE as u64
    );
    assert!(DATA_FILE_SIZE_MIN.is_multiple_of(BLOCK_SIZE));

    assert!(CLIENTS_MAX > 0);
    assert!(PIPELINE_PREPARE_QUEUE_MAX > 0);
    assert!(VIEW_CHANGE_HEADERS_SUFFIX_MAX == PIPELINE_PREPARE_QUEUE_MAX + 1);
    assert!(VIEW_HEADERS_MAX == VIEW_CHANGE_HEADERS_SUFFIX_MAX + 2);

    assert!(JOURNAL_SLOT_COUNT > 0);
    assert!(CLIENT_REPLY_SIZE > 0);
    assert!(STORAGE_SIZE_LIMIT_MAX > 0);
};

// =============================================================================
// Helper functions
// =============================================================================

/// Rounds up to next [`SECTOR_SIZE`] multiple. Idempotent for aligned inputs.
///
/// # Panics
/// Panics on overflow.
///
/// # Examples
/// ```
/// # use consensus::constants::*;
/// assert_eq!(sector_ceil(0), 0);
/// assert_eq!(sector_ceil(1), SECTOR_SIZE);
/// assert_eq!(sector_ceil(SECTOR_SIZE), SECTOR_SIZE);
/// assert_eq!(sector_ceil(SECTOR_SIZE + 1), SECTOR_SIZE * 2);
/// ```
#[inline]
pub const fn sector_ceil(n: usize) -> usize {
    const _: () = assert!(SECTOR_SIZE.is_power_of_two());

    let mask = SECTOR_SIZE - 1;

    assert!(n <= usize::MAX - mask, "sector_ceil overflow");

    let result = (n + mask) & !mask;
    assert!(
        result.is_multiple_of(SECTOR_SIZE),
        "sector_ceil produced unaligned result"
    );

    result
}

/// Convert a `u32` protocol length to `usize` for array sizing.
///
/// # Examples
/// ```
/// # use consensus::constants::*;
/// let buffer = [0u8; as_usize(HEADER_SIZE)];
/// ```
#[inline(always)]
pub const fn as_usize(n: u32) -> usize {
    n as usize
}

/// Align a `u64` value forward to the specified alignment.
///
/// # Examples
/// ```
/// # use consensus::constants::*;
/// let aligned = align_forward_u64(1025, 512);
/// assert_eq!(aligned, 1536);
/// ```
#[inline(always)]
pub const fn align_forward_u64(x: u64, align: u64) -> u64 {
    let rem = x % align;
    if rem == 0 { x } else { x + (align - rem) }
}

// =============================================================================
// Pre-converted usize constants
// =============================================================================

/// [`HEADER_SIZE`] as `usize`.
pub const HEADER_SIZE_USIZE: usize = HEADER_SIZE as usize;

/// [`MESSAGE_SIZE_MAX`] as `usize`.
pub const MESSAGE_SIZE_MAX_USIZE: usize = MESSAGE_SIZE_MAX as usize;

/// [`MESSAGE_BODY_SIZE_MAX`] as `usize`.
pub const MESSAGE_BODY_SIZE_MAX_USIZE: usize = MESSAGE_BODY_SIZE_MAX as usize;

// Verify usize conversions match source constants.
const _: () = assert!(HEADER_SIZE_USIZE == HEADER_SIZE as usize);
const _: () = assert!(MESSAGE_SIZE_MAX_USIZE == MESSAGE_SIZE_MAX as usize);
const _: () = assert!(MESSAGE_BODY_SIZE_MAX_USIZE == MESSAGE_BODY_SIZE_MAX as usize);

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sector_ceil_rounds_zero_to_zero() {
        assert_eq!(sector_ceil(0), 0);
    }

    #[test]
    fn sector_ceil_rounds_non_aligned_up() {
        assert_eq!(sector_ceil(1), SECTOR_SIZE);
        assert_eq!(sector_ceil(SECTOR_SIZE - 1), SECTOR_SIZE);
        assert_eq!(sector_ceil(SECTOR_SIZE + 1), SECTOR_SIZE * 2);
    }

    #[test]
    fn sector_ceil_identity_for_aligned() {
        assert_eq!(sector_ceil(SECTOR_SIZE), SECTOR_SIZE);
        assert_eq!(sector_ceil(SECTOR_SIZE * 2), SECTOR_SIZE * 2);
        assert_eq!(sector_ceil(SECTOR_SIZE * 100), SECTOR_SIZE * 100);
    }

    #[test]
    fn storage_constants_are_powers_of_two() {
        assert!(SECTOR_SIZE.is_power_of_two());
        assert!(SECTOR_SIZE_MIN.is_power_of_two());
        assert!(SECTOR_SIZE_MAX.is_power_of_two());
        assert!(IO_ENTRIES_DEFAULT.is_power_of_two());
    }

    #[test]
    fn storage_constant_values() {
        assert_eq!(SECTOR_SIZE, 4096);
        assert_eq!(SECTOR_SIZE_MIN, 512);
        assert_eq!(SECTOR_SIZE_DEFAULT, 4096);
        assert_eq!(SECTOR_SIZE_MAX, 65_536);
    }

    #[test]
    fn wire_constant_relationships_hold() {
        assert_eq!(HEADER_SIZE + MESSAGE_BODY_SIZE_MAX, MESSAGE_SIZE_MAX);
        assert!(HEADER_SIZE.is_multiple_of(16));
        assert!(MESSAGE_SIZE_MAX.is_power_of_two());
    }

    #[test]
    fn wire_usize_constants_match_originals() {
        assert_eq!(HEADER_SIZE_USIZE, HEADER_SIZE as usize);
        assert_eq!(MESSAGE_SIZE_MAX_USIZE, MESSAGE_SIZE_MAX as usize);
        assert_eq!(MESSAGE_BODY_SIZE_MAX_USIZE, MESSAGE_BODY_SIZE_MAX as usize);
    }

    #[test]
    fn test_align_forward_u64() {
        assert_eq!(align_forward_u64(0, 512), 0);
        assert_eq!(align_forward_u64(1, 512), 512);
        assert_eq!(align_forward_u64(511, 512), 512);
        assert_eq!(align_forward_u64(512, 512), 512);
        assert_eq!(align_forward_u64(513, 512), 1024);
        assert_eq!(align_forward_u64(1024, 512), 1024);
        assert_eq!(align_forward_u64(1025, 512), 1536);
    }
}
