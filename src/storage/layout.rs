//! Storage layout for VSR data file.
//!
//! This module provides a translation layer between **logical addressing**
//! (zone + offset) and **physical addressing** (absolute file byte). This
//! decouples application logic (which cares about "the 3rd WAL entry") from
//! physical storage (which cares about "byte 4096 on disk").
//!
//! # Physical Layout
//!
//! The [`Layout`] struct describes the immutable shape of the file on disk.
//! Zones are contiguous, with [`GridPadding`](Zone::GridPadding) inserted
//! automatically to ensure [`Grid`](Zone::Grid) is block-aligned.
//!
//! ```text
//! File Start (0)
//! │
//! ▼
//! ┌──────────────┐ ◄─ Zone 0: SuperBlock (base: 0)
//! │              │
//! ├──────────────┤ ◄─ Zone 1: WalHeaders
//! │              │
//! ├──────────────┤ ◄─ Zone 2: WalPrepares
//! │              │
//! ├──────────────┤ ◄─ Zone 3: ClientReplies
//! │              │
//! ├──────────────┤ ◄─ Zone 4: GridPadding (variable, forces alignment)
//! │  //////////  │
//! ├──────────────┤ ◄─ Zone 5: Grid (base % block_size == 0)
//! │              │
//! │              │
//! └──────────────┘
//! ```
//!
//! # Key Relationships
//!
//! - **[`Layout`] → [`ZoneSpec`]**: The layout acts as a factory. Compute the
//!   layout once at startup, then extract [`ZoneSpec`]s to configure I/O
//!   operations.
//!
//! - **[`ZoneSpec`] → I/O**: I/O structs (like `Read`) embed a [`ZoneSpec`]
//!   snapshot. This makes them self-contained—they can be passed to isolated
//!   I/O threads without referencing the global layout.
//!
//! # Design Rationale
//!
//! - **Safety**: Zone bounds are enforced; reads cannot cross zone boundaries.
//! - **Performance**: No pointer chasing—base offset is embedded in the I/O struct.
//! - **Alignment**: [`Layout`] guarantees sector-aligned bases, so aligned
//!   relative offsets yield aligned physical accesses.

/// Identifies a storage region within the data file.
///
/// Each zone has a dedicated purpose in the VSR protocol. Discriminants are
/// stable (`#[repr(u8)]`) for array indexing. Use [`Zone::ALL`] for iteration
/// in layout order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Zone {
    /// Cluster metadata and replica state for crash recovery.
    SuperBlock = 0,
    /// Fixed-size headers for WAL entries (checksums, sequence numbers).
    WalHeaders = 1,
    /// Variable-size prepare message bodies referenced by WAL headers.
    WalPrepares = 2,
    /// Cached responses for client request deduplication.
    ClientReplies = 3,
    /// Auto-inserted padding to block-align the Grid zone.
    GridPadding = 4,
    /// LSM tree block storage for persistent state machine data.
    Grid = 5,
}

impl Zone {
    /// Total number of zones in the layout.
    pub const COUNT: usize = 6;

    /// All zones in physical layout order (file start → end).
    pub const ALL: [Zone; Self::COUNT] = [
        Zone::SuperBlock,
        Zone::WalHeaders,
        Zone::WalPrepares,
        Zone::ClientReplies,
        Zone::GridPadding,
        Zone::Grid,
    ];

    /// Returns the zone's index for array lookup (matches discriminant value).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

const _: () = {
    assert!(Zone::COUNT == 6);
    assert!(Zone::ALL.len() == Zone::COUNT);

    // Verify each zone has the correct discriminant value (contiguous 0..COUNT-1)
    assert!(Zone::SuperBlock as usize == 0);
    assert!(Zone::WalHeaders as usize == 1);
    assert!(Zone::WalPrepares as usize == 2);
    assert!(Zone::ClientReplies as usize == 3);
    assert!(Zone::GridPadding as usize == 4);
    assert!(Zone::Grid as usize == 5);
    assert!(Zone::Grid as usize == Zone::COUNT - 1);

    // Verify ALL array contains zones in correct order
    assert!(Zone::ALL[0] as usize == 0);
    assert!(Zone::ALL[1] as usize == 1);
    assert!(Zone::ALL[2] as usize == 2);
    assert!(Zone::ALL[3] as usize == 3);
    assert!(Zone::ALL[4] as usize == 4);
    assert!(Zone::ALL[5] as usize == 5);
};

/// Position and extent of a single zone, extracted from [`Layout`].
///
/// `ZoneSpec` is a lightweight, copyable snapshot of a zone's location. It's
/// designed to be embedded in I/O structs, providing all context needed for
/// address translation without referencing the global [`Layout`].
///
/// # Address Translation
///
/// ```text
/// ZoneSpec { base: 8192, size: 4096 }
///              │
///              ▼
/// ┌────────────┬────────────────────────────┐
/// │   File     │  Zone (WalPrepares)        │
/// │   ...      │ [8192]──────────[12288)    │
/// └────────────┴────────────────────────────┘
///                  ▲
///                  │
///              offset(100) → 8292
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ZoneSpec {
    /// Absolute byte offset from file start.
    pub base: u64,
    /// Size in bytes (may be zero for empty zones).
    pub size: u64,
}

impl ZoneSpec {
    /// Exclusive end offset (`base + size`).
    ///
    /// # Panics
    /// On arithmetic overflow.
    #[inline]
    pub const fn end(self) -> u64 {
        match self.base.checked_add(self.size) {
            Some(end) => end,
            None => panic!("ZoneSpec::end overflow"),
        }
    }

    /// Returns `true` if `abs_offset` falls within `[base, end)`.
    #[inline]
    pub const fn contains(self, abs_offset: u64) -> bool {
        self.base <= abs_offset && abs_offset < self.end()
    }

    /// Converts a zone-relative offset to an absolute file offset.
    ///
    /// # Panics
    /// If `relative > size` or on arithmetic overflow.
    #[inline]
    pub const fn offset(self, relative: u64) -> u64 {
        assert!(relative <= self.size);
        match self.base.checked_add(relative) {
            Some(offset) => offset,
            None => panic!("ZoneSpec::offset overflow"),
        }
    }
}

/// Complete storage layout with computed zone positions.
///
/// `Layout` is the **factory** for [`ZoneSpec`]s. Compute it once at startup,
/// then extract zone specs for I/O operations. Zones are laid out contiguously
/// with [`GridPadding`](Zone::GridPadding) auto-computed to block-align the Grid.
///
/// # Invariants
///
/// - All zone bases are sector-aligned
/// - Zones are contiguous: `zone[i].end() == zone[i+1].base`
/// - [`Grid`](Zone::Grid) base is block-aligned
/// - [`SuperBlock`](Zone::SuperBlock) starts at offset 0
///
/// # Example
///
/// ```ignore
/// let layout = Layout::new(512, 4096, sb, wh, wp, cr, grid);
///
/// // Extract spec for I/O
/// let wal_spec = layout.zone(Zone::WalPrepares);
/// let abs_offset = wal_spec.offset(relative_offset);
/// ```
#[derive(Debug, Clone)]
pub struct Layout {
    /// Minimum I/O alignment (typically 512 or 4096).
    pub sector_size: u64,
    /// Grid block size; must be a multiple of `sector_size`.
    pub block_size: u64,
    zones: [ZoneSpec; Zone::COUNT],
}

impl Layout {
    /// Constructs a layout from zone sizes.
    ///
    /// [`GridPadding`](Zone::GridPadding) size is computed automatically.
    ///
    /// # Panics
    /// - `sector_size` or `block_size` is zero or not a power of two
    /// - `block_size < sector_size`
    /// - Zone sizes are not sector-aligned (or block-aligned for grid)
    pub fn new(
        sector_size: u64,
        block_size: u64,
        superblock_size: u64,
        wal_headers_size: u64,
        wal_prepares_size: u64,
        client_replies_size: u64,
        grid_size: u64,
    ) -> Self {
        assert!(sector_size > 0);
        assert!(
            sector_size.is_power_of_two(),
            "sector_size must be power of two"
        );
        assert!(block_size > 0);
        assert!(
            block_size.is_power_of_two(),
            "block_size must be power of two"
        );
        assert!(block_size >= sector_size);
        assert!(block_size.is_multiple_of(sector_size));

        assert!(
            superblock_size.is_multiple_of(sector_size),
            "superblock_size not sector-aligned"
        );
        assert!(
            wal_headers_size.is_multiple_of(sector_size),
            "wal_headers_size not sector-aligned"
        );
        assert!(
            wal_prepares_size.is_multiple_of(sector_size),
            "wal_prepares_size not sector-aligned"
        );
        assert!(
            client_replies_size.is_multiple_of(sector_size),
            "client_replies_size not sector-aligned"
        );
        assert!(
            grid_size.is_multiple_of(block_size),
            "grid_size not block-aligned"
        );

        let z0 = ZoneSpec {
            base: 0,
            size: superblock_size,
        };
        let z1 = ZoneSpec {
            base: z0.end(),
            size: wal_headers_size,
        };
        let z2 = ZoneSpec {
            base: z1.end(),
            size: wal_prepares_size,
        };
        let z3 = ZoneSpec {
            base: z2.end(),
            size: client_replies_size,
        };

        let before_grid = z3.end();
        let grid_base = align_up(before_grid, block_size);
        let grid_padding_size = grid_base - before_grid;

        assert!(grid_padding_size.is_multiple_of(sector_size));

        let z4 = ZoneSpec {
            base: z3.end(),
            size: grid_padding_size,
        };
        let z5 = ZoneSpec {
            base: z4.end(),
            size: grid_size,
        };

        assert!(z5.base.is_multiple_of(block_size));

        let layout = Self {
            sector_size,
            block_size,
            zones: [z0, z1, z2, z3, z4, z5],
        };

        for i in 1..Zone::COUNT {
            assert!(layout.zones[i].base == layout.zones[i - 1].end());
        }

        for zone in &layout.zones {
            assert!(zone.base.is_multiple_of(sector_size));
        }

        layout
    }

    /// Total file size (sum of all zone sizes).
    #[inline]
    pub fn total_size(&self) -> u64 {
        self.zones
            .iter()
            .try_fold(0u64, |acc, z| acc.checked_add(z.size))
            .expect("total size overflow")
    }

    /// Returns the [`ZoneSpec`] for a zone.
    #[inline]
    pub fn zone(&self, z: Zone) -> ZoneSpec {
        self.zones[z.index()]
    }

    /// Absolute start offset of a zone.
    #[inline]
    pub fn start(&self, z: Zone) -> u64 {
        self.zone(z).base
    }

    /// Size of a zone in bytes.
    #[inline]
    pub fn size(&self, z: Zone) -> u64 {
        self.zone(z).size
    }

    /// Exclusive end offset of a zone.
    #[inline]
    pub fn end(&self, z: Zone) -> u64 {
        self.zone(z).end()
    }

    /// Converts a zone-relative offset to absolute.
    ///
    /// # Panics
    /// If `relative > zone.size`.
    #[inline]
    pub fn offset(&self, z: Zone, relative: u64) -> u64 {
        let spec = self.zone(z);
        assert!(relative <= spec.size, "offset out of bounds");

        spec.base
            .checked_add(relative)
            .expect("Layout::offset overflow")
    }

    /// Finds which zone contains `abs_offset`, or `None` if out of bounds.
    pub fn zone_for_absolute(&self, abs_offset: u64) -> Option<Zone> {
        Zone::ALL
            .into_iter()
            .find(|&z| self.zone(z).contains(abs_offset))
    }
}

/// Rounds `value` up to the next multiple of `align`.
///
/// # Panics
/// - `align` is zero or not a power of two
/// - Result would overflow `u64`
#[inline]
pub const fn align_up(value: u64, align: u64) -> u64 {
    assert!(align > 0);
    assert!(align.is_power_of_two());
    match value.checked_add(align - 1) {
        Some(v) => v & !(align - 1),
        None => panic!("align_up overflow"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_invariants() {
        assert_eq!(Zone::COUNT, Zone::ALL.len());
        for (i, z) in Zone::ALL.iter().enumerate() {
            assert_eq!(z.index(), i);
        }
    }

    #[test]
    fn test_align_up_basic() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(3, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(100, 64), 128);
    }

    #[test]
    fn test_align_up_identity() {
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(1, 1), 1);
        assert_eq!(align_up(12345, 1), 12345);
    }

    #[test]
    fn test_layout_new() {
        let sector = 512;
        let block = 4096;
        let l = Layout::new(
            sector, block, sector, // 512
            sector, // 512
            sector, // 512
            sector, // 512
            block,  // 4096
        );

        // Check zones
        assert_eq!(l.zone(Zone::SuperBlock).base, 0);
        assert_eq!(l.zone(Zone::SuperBlock).size, sector);

        assert_eq!(l.zone(Zone::WalHeaders).base, sector);
        assert_eq!(l.zone(Zone::WalHeaders).size, sector);

        assert_eq!(l.zone(Zone::WalPrepares).base, 2 * sector);
        assert_eq!(l.zone(Zone::WalPrepares).size, sector);

        assert_eq!(l.zone(Zone::ClientReplies).base, 3 * sector);
        assert_eq!(l.zone(Zone::ClientReplies).size, sector);

        // End of ClientReplies is 4 * 512 = 2048.
        // Grid must start at block alignment (4096).
        // Padding needed = 4096 - 2048 = 2048.
        assert_eq!(l.zone(Zone::GridPadding).base, 4 * sector);
        assert_eq!(l.zone(Zone::GridPadding).size, 2048);

        assert_eq!(l.zone(Zone::Grid).base, block);
        assert_eq!(l.zone(Zone::Grid).size, block);

        assert_eq!(l.total_size(), block + block);
    }

    #[test]
    fn test_layout_exact() {
        let sector = 512;
        let block = 4096;
        // 8 sectors = 4096 bytes
        let l = Layout::new(
            sector, block, 4096, // SuperBlock
            4096, // WalHeaders
            4096, // WalPrepares
            4096, // ClientReplies
            4096, // Grid
        );

        // Before Grid: 4 * 4096 = 16384. 16384 is multiple of 4096.
        // Padding should be 0.
        assert_eq!(l.zone(Zone::GridPadding).size, 0);
        assert_eq!(l.zone(Zone::Grid).base, 16384);
    }

    #[test]
    fn test_offset() {
        let l = Layout::new(512, 4096, 512, 512, 512, 512, 4096);

        let z = Zone::WalHeaders; // Starts at 512, size 512.
        assert_eq!(l.start(z), 512);
        assert_eq!(l.size(z), 512);
        assert_eq!(l.end(z), 1024);

        // Relative offset
        assert_eq!(l.offset(z, 0), 512);
        assert_eq!(l.offset(z, 100), 612);
        assert_eq!(l.offset(z, 512), 1024); // Valid now
    }

    #[test]
    #[should_panic(expected = "offset out of bounds")]
    fn test_offset_oob() {
        let l = Layout::new(512, 4096, 512, 512, 512, 512, 4096);
        l.offset(Zone::WalHeaders, 513);
    }

    #[test]
    fn test_lookup() {
        let l = Layout::new(512, 4096, 512, 512, 512, 512, 4096);
        // SB: 0..512
        // WH: 512..1024
        // WP: 1024..1536
        // CR: 1536..2048
        // Pad: 2048..4096
        // Grid: 4096..8192

        assert_eq!(l.zone_for_absolute(0), Some(Zone::SuperBlock));
        assert_eq!(l.zone_for_absolute(511), Some(Zone::SuperBlock));
        assert_eq!(l.zone_for_absolute(512), Some(Zone::WalHeaders));
        assert_eq!(l.zone_for_absolute(2047), Some(Zone::ClientReplies));
        assert_eq!(l.zone_for_absolute(2048), Some(Zone::GridPadding));
        assert_eq!(l.zone_for_absolute(4095), Some(Zone::GridPadding));
        assert_eq!(l.zone_for_absolute(4096), Some(Zone::Grid));
        assert_eq!(l.zone_for_absolute(8192), None); // Out of bounds
    }

    #[test]
    #[should_panic(expected = "sector_size must be power of two")]
    fn test_new_invalid_sector() {
        Layout::new(100, 4096, 4096, 4096, 4096, 4096, 4096);
    }

    #[test]
    #[should_panic(expected = "not sector-aligned")]
    fn test_new_unaligned() {
        Layout::new(4096, 4096, 1000, 4096, 4096, 4096, 4096);
    }

    // ==================== ZoneSpec Overflow and Boundary Tests ====================

    #[test]
    #[should_panic(expected = "ZoneSpec::end overflow")]
    fn test_zonespec_end_overflow() {
        let spec = ZoneSpec {
            base: u64::MAX,
            size: 1,
        };
        let _ = spec.end();
    }

    #[test]
    fn test_zonespec_contains() {
        let spec = ZoneSpec {
            base: 1000,
            size: 500,
        };

        // Before zone
        assert!(!spec.contains(999));
        // At start boundary (inclusive)
        assert!(spec.contains(1000));
        // Middle of zone
        assert!(spec.contains(1250));
        // Just before end
        assert!(spec.contains(1499));
        // At end boundary (exclusive)
        assert!(!spec.contains(1500));
        // After zone
        assert!(!spec.contains(1501));
    }

    #[test]
    fn test_zonespec_contains_empty() {
        let spec = ZoneSpec {
            base: 1000,
            size: 0,
        };

        assert!(!spec.contains(999));
        assert!(!spec.contains(1000)); // Zero-size zone contains nothing
        assert!(!spec.contains(1001));
    }

    #[test]
    #[should_panic(expected = "ZoneSpec::offset overflow")]
    fn test_zonespec_offset_overflow() {
        let spec = ZoneSpec {
            base: u64::MAX - 100,
            size: 1000,
        };
        let _ = spec.offset(200);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_zonespec_offset_oob() {
        let spec = ZoneSpec {
            base: 1000,
            size: 10,
        };
        let _ = spec.offset(11);
    }

    // ==================== align_up Edge Case Tests ====================

    #[test]
    fn test_align_up_aligned() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(8192, 4096), 8192);
        assert_eq!(align_up(1024 * 1024, 4096), 1024 * 1024);
    }

    #[test]
    fn test_align_up_variable() {
        assert_eq!(align_up(100, 512), 512);
        assert_eq!(align_up(512, 512), 512);
        assert_eq!(align_up(513, 512), 1024);
        assert_eq!(align_up(1, 1024 * 1024), 1024 * 1024);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_align_up_zero() {
        let _ = align_up(100, 0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_align_up_non_pot() {
        let _ = align_up(100, 100);
    }

    #[test]
    #[should_panic(expected = "align_up overflow")]
    fn test_align_up_overflow() {
        let _ = align_up(u64::MAX - 100, 4096);
    }

    // ==================== Layout Invalid Config Tests ====================

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_new_zero_sector() {
        Layout::new(0, 4096, 4096, 4096, 4096, 4096, 4096);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_new_zero_block() {
        Layout::new(4096, 0, 4096, 4096, 4096, 4096, 4096);
    }

    #[test]
    #[should_panic(expected = "block_size must be power of two")]
    fn test_new_block_non_pot() {
        Layout::new(4096, 5000, 4096, 4096, 4096, 4096, 4096);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_new_block_lt_sector() {
        Layout::new(4096, 2048, 4096, 4096, 4096, 4096, 4096);
    }

    #[test]
    #[should_panic(expected = "not sector-aligned")]
    fn test_new_unaligned_wal_header() {
        Layout::new(4096, 4096, 4096, 4097, 4096, 4096, 4096);
    }

    #[test]
    #[should_panic(expected = "not sector-aligned")]
    fn test_new_unaligned_wal_prepare() {
        Layout::new(4096, 4096, 4096, 4096, 4097, 4096, 4096);
    }

    #[test]
    #[should_panic(expected = "not sector-aligned")]
    fn test_new_unaligned_client_reply() {
        Layout::new(4096, 4096, 4096, 4096, 4096, 4097, 4096);
    }

    #[test]
    #[should_panic(expected = "not block-aligned")]
    fn test_new_unaligned_grid() {
        Layout::new(512, 4096, 4096, 4096, 4096, 4096, 4097);
    }

    // ==================== Layout Overflow Tests ====================

    #[test]
    #[should_panic(expected = "ZoneSpec::end overflow")]
    fn test_new_overflow_zone_end() {
        let _ = Layout::new(1, 1, u64::MAX, 1, 0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "align_up overflow")]
    fn test_new_overflow_align() {
        let _ = Layout::new(1, 2, u64::MAX, 0, 0, 0, 0);
    }

    // ==================== Layout Invariant Tests ====================

    #[test]
    fn test_sb_zero() {
        let layout = Layout::new(4096, 4096, 4096, 4096, 4096, 4096, 4096);
        assert_eq!(layout.start(Zone::SuperBlock), 0);
    }

    #[test]
    fn test_zones_overlap() {
        let layout = Layout::new(512, 4096, 4096, 8192, 16384, 8192, 65536);

        for i in 0..Zone::COUNT {
            for j in (i + 1)..Zone::COUNT {
                let zone_i = layout.zone(Zone::ALL[i]);
                let zone_j = layout.zone(Zone::ALL[j]);

                assert!(
                    zone_i.end() <= zone_j.base,
                    "Zone {:?} ({}..{}) overlaps Zone {:?} ({}..{})",
                    Zone::ALL[i],
                    zone_i.base,
                    zone_i.end(),
                    Zone::ALL[j],
                    zone_j.base,
                    zone_j.end()
                );
            }
        }
    }

    #[test]
    fn test_total_size_end() {
        let layout = Layout::new(4096, 4096, 4096, 4096, 4096, 4096, 4096);
        assert_eq!(layout.total_size(), layout.end(Zone::Grid));
    }

    #[test]
    fn test_empty_zones() {
        let layout = Layout::new(4096, 4096, 4096, 0, 0, 0, 4096);

        assert_eq!(layout.size(Zone::WalHeaders), 0);
        assert_eq!(layout.size(Zone::WalPrepares), 0);
        assert_eq!(layout.size(Zone::ClientReplies), 0);

        // Zones should still be contiguous
        assert_eq!(
            layout.start(Zone::WalPrepares),
            layout.end(Zone::WalHeaders)
        );
    }

    #[test]
    fn test_large_zones() {
        let gb = 1024 * 1024 * 1024;
        let layout = Layout::new(4096, 1024 * 1024, 4096, gb, 10 * gb, gb, 100 * gb);
        assert!(layout.total_size() > 100 * gb);
    }

    #[test]
    fn test_discriminants() {
        // Wire protocol stability - discriminants must never change
        assert_eq!(Zone::SuperBlock as u8, 0);
        assert_eq!(Zone::WalHeaders as u8, 1);
        assert_eq!(Zone::WalPrepares as u8, 2);
        assert_eq!(Zone::ClientReplies as u8, 3);
        assert_eq!(Zone::GridPadding as u8, 4);
        assert_eq!(Zone::Grid as u8, 5);
    }

    #[test]
    fn test_lookup_empty_zones() {
        let layout = Layout::new(4096, 4096, 4096, 0, 4096, 0, 4096);

        // Zero-sized zones should not contain any offset
        let wal_headers_base = layout.start(Zone::WalHeaders);
        assert_ne!(
            layout.zone_for_absolute(wal_headers_base),
            Some(Zone::WalHeaders)
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    const PROPTEST_CASES: u32 = 16;

    fn sector_size_strategy() -> impl Strategy<Value = u64> {
        prop::sample::select(vec![512u64, 1024, 2048, 4096, 8192, 16384])
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(
            crate::test_utils::proptest_cases(PROPTEST_CASES)
        ))]

        #[test]
        fn prop_zones_contiguous(
            sector_size in sector_size_strategy(),
            block_exp in 0u32..5u32,
        ) {
            let block_size = sector_size << block_exp;
            let layout = Layout::new(
                sector_size,
                block_size,
                sector_size * 3,
                sector_size * 5,
                sector_size * 7,
                sector_size * 2,
                block_size * 16,
            );

            for i in 1..Zone::COUNT {
                let prev = layout.zone(Zone::ALL[i - 1]);
                let curr = layout.zone(Zone::ALL[i]);
                prop_assert_eq!(curr.base, prev.end());
            }
        }

        #[test]
        fn prop_zones_aligned(
            sector_size in sector_size_strategy(),
            block_exp in 0u32..5u32,
        ) {
            let block_size = sector_size << block_exp;
            let layout = Layout::new(
                sector_size,
                block_size,
                sector_size * 3,
                sector_size * 5,
                sector_size * 7,
                sector_size * 2,
                block_size * 16,
            );

            for zone in Zone::ALL {
                prop_assert_eq!(layout.start(zone) % layout.sector_size, 0);
            }
        }

        #[test]
        fn prop_grid_aligned(
            sector_size in sector_size_strategy(),
            block_exp in 0u32..5u32,
        ) {
            let block_size = sector_size << block_exp;
            let layout = Layout::new(
                sector_size,
                block_size,
                sector_size * 3,
                sector_size * 5,
                sector_size * 7,
                sector_size * 2,
                block_size * 8,
            );
            prop_assert_eq!(layout.start(Zone::Grid) % layout.block_size, 0);
            prop_assert!(layout.size(Zone::GridPadding) < layout.block_size);
        }

        #[test]
        fn prop_total_size(
            sector_size in sector_size_strategy(),
            block_exp in 0u32..5u32,
        ) {
            let block_size = sector_size << block_exp;
            let layout = Layout::new(
                sector_size,
                block_size,
                sector_size * 3,
                sector_size * 5,
                sector_size * 7,
                sector_size * 2,
                block_size * 16,
            );

            let sum: u64 = Zone::ALL.iter().map(|&z| layout.size(z)).sum();
            prop_assert_eq!(layout.total_size(), sum);
        }

        #[test]
        fn prop_lookup_roundtrip(
            sector_size in sector_size_strategy(),
            block_exp in 0u32..5u32,
        ) {
            let block_size = sector_size << block_exp;
            let layout = Layout::new(
                sector_size,
                block_size,
                sector_size * 3,
                sector_size * 5,
                sector_size * 7,
                sector_size * 2,
                block_size * 16,
            );

            for zone in Zone::ALL {
                let zone_spec = layout.zone(zone);
                if zone_spec.size > 0 {
                    let offset = zone_spec.base;
                    let found = layout.zone_for_absolute(offset);
                    prop_assert_eq!(found, Some(zone));
                }
            }
        }

        #[test]
        fn prop_offset_bounds(
            sector_size in sector_size_strategy(),
            block_exp in 0u32..5u32,
            zone_idx in 0usize..Zone::COUNT,
        ) {
            let block_size = sector_size << block_exp;
            let layout = Layout::new(
                sector_size,
                block_size,
                sector_size * 3,
                sector_size * 5,
                sector_size * 7,
                sector_size * 2,
                block_size * 16,
            );
            let zone = Zone::ALL[zone_idx];
            let zone_spec = layout.zone(zone);

            if zone_spec.size > 0 {
                let relative = zone_spec.size / 2;
                let abs_offset = layout.offset(zone, relative);

                prop_assert_eq!(layout.zone_for_absolute(abs_offset), Some(zone));
                prop_assert!(abs_offset >= zone_spec.base);
                prop_assert!(abs_offset < zone_spec.end());
            }
        }

        #[test]
        fn prop_align_up(
            value in 0u64..1_000_000,
            align_exp in 9u32..20u32,
        ) {
            let align = 1u64 << align_exp;
            let result = align_up(value, align);

            prop_assert_eq!(result % align, 0);
            prop_assert!(result >= value);
            if value % align == 0 {
                prop_assert_eq!(result, value);
            }
        }

        #[test]
        fn prop_zonespec_contains(
            base in 0u64..1_000_000,
            size in 1u64..1_000_000,
        ) {
            let spec = ZoneSpec { base, size };

            prop_assert!(spec.contains(base));
            prop_assert!(!spec.contains(spec.end()));

            if base > 0 {
                prop_assert!(!spec.contains(base - 1));
            }
        }

        #[test]
        fn prop_zonespec_offset(
            base in 0u64..1_000_000,
            size in 1u64..1_000_000,
            relative_factor in 0.0f64..1.0,
        ) {
            let spec = ZoneSpec { base, size };
            let relative = ((size - 1) as f64 * relative_factor) as u64;

            let abs_offset = spec.offset(relative);
            prop_assert!(abs_offset >= base);
            prop_assert!(abs_offset < spec.end());
        }
    }
}
