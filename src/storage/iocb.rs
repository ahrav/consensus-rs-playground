//! I/O Control Blocks (IOCBs) and deferred callback structures.

use crate::io::Completion as IoCompletion;
use crate::stdx::queue::{QueueLink, QueueNode};
use super::layout::{Zone, ZoneSpec};

/// Controls whether I/O operations complete synchronously or asynchronously.
///
/// Used to configure storage behavior for testing (synchronous) vs production
/// (asynchronous with io_uring/epoll).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Synchronicity {
    /// Operations block until completion. Useful for deterministic testing.
    AlwaysSynchronous,
    /// Operations return immediately; completion notified via callback.
    AlwaysAsynchronous,
}

/// Identifies which subsystem a deferred callback belongs to.
///
/// Callbacks are partitioned to allow priority-based processing: VSR protocol
/// operations typically take precedence over LSM tree maintenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextTickQueue {
    /// Viewstamped Replication protocol callbacks (high priority).
    Vsr,
    /// Log-Structured Merge tree callbacks (background maintenance).
    Lsm,
}

// ─────────────────────────────────────────────────────────────────────────────
// I/O Control Blocks
// ─────────────────────────────────────────────────────────────────────────────
//
// `Read` and `Write` are **I/O Control Blocks (IOCBs)**—each represents a slot
// for an in-flight async operation. The embedded [`IoCompletion`] is bound to
// a fixed memory address where the kernel/io_uring writes results.
//
// # Address Translation
//
// IOCBs translate logical (zone-relative) offsets to physical (absolute) offsets:
//
// ```text
// Logical:   Read { zone: WalPrepares, offset: 100 }
//                                        │
//            ┌───────────────────────────┘
//            ▼
// Physical:  zone_spec.base + offset = 8192 + 100 = 8292
// ```
//
// The embedded `zone_spec` is copied from [`Layout`], making IOCBs self-contained.
// Once prepared, an IOCB doesn't need to access the global Layout.
//
// The storage engine only sees `absolute_offset()`—no knowledge of Zones. The
// `zone` field exists for **completion context**: callbacks inspect it to
// determine how to process results.
//
// # Memory Layout
//
// `#[repr(C)]` ensures predictable field ordering. The `io` field is first so
// the IOCB address equals the completion address.
// ─────────────────────────────────────────────────────────────────────────────

/// I/O Control Block for a pending read. See module comment for lifecycle.
#[repr(C)]
pub struct Read {
    /// I/O completion state (result, status flags).
    pub io: IoCompletion,
    /// Invoked when I/O completes; `None` if not pending.
    pub callback: Option<fn(&mut Read)>,
    /// Expected bytes for validation (short reads are errors in direct I/O).
    pub expected_len: usize,

    /// Which zone this read targets.
    pub zone: Zone,
    /// Snapshot of zone position/size from [`Layout`] at submission time.
    pub zone_spec: ZoneSpec,
    /// Logical offset within the zone (0 = zone start).
    pub offset: u64,
}

impl Read {
    /// Creates a zeroed `Read` targeting the SuperBlock at offset 0.
    ///
    /// Call site must populate `zone`, `zone_spec`, and `offset` before use.
    pub const fn new() -> Self {
        Self {
            io: IoCompletion::new(),
            callback: None,
            expected_len: 0,
            zone: Zone::SuperBlock,
            zone_spec: ZoneSpec { base: 0, size: 0 },
            offset: 0,
        }
    }

    /// Returns `true` if this read is awaiting I/O completion.
    #[inline]
    pub fn is_pending(&self) -> bool {
        self.callback.is_some()
    }

    /// Expected read length; short reads indicate I/O errors with O_DIRECT.
    #[inline]
    pub fn expected_len(&self) -> usize {
        self.expected_len
    }

    /// Computes the absolute file offset from zone-relative addressing.
    ///
    /// # Panics
    ///
    /// - `offset > zone_spec.size` (would read past zone boundary)
    #[inline]
    pub fn absolute_offset(&self) -> u64 {
        assert!(self.offset <= self.zone_spec.size);

        let abs = self.zone_spec.offset(self.offset);
        assert!(abs >= self.zone_spec.base);

        abs
    }
}

impl Default for Read {
    fn default() -> Self {
        Self::new()
    }
}

/// I/O Control Block for a pending write. See module comment for lifecycle.
#[repr(C)]
pub struct Write {
    /// I/O completion state (result, status flags).
    pub io: IoCompletion,
    /// Invoked when I/O completes; `None` if not pending.
    pub callback: Option<fn(&mut Write)>,
    /// Expected bytes for validation (short writes are errors in direct I/O).
    pub expected_len: usize,

    /// Which zone this write targets.
    pub zone: Zone,
    /// Snapshot of zone position/size from [`Layout`] at submission time.
    pub zone_spec: ZoneSpec,
    /// Logical offset within the zone (0 = zone start).
    pub offset: u64,
}

impl Write {
    /// Creates a zeroed `Write` targeting the SuperBlock at offset 0.
    ///
    /// Call site must populate `zone`, `zone_spec`, and `offset` before use.
    pub const fn new() -> Self {
        Self {
            io: IoCompletion::new(),
            callback: None,
            expected_len: 0,
            zone: Zone::SuperBlock,
            zone_spec: ZoneSpec { base: 0, size: 0 },
            offset: 0,
        }
    }

    /// Returns `true` if this write is awaiting I/O completion.
    #[inline]
    pub fn is_pending(&self) -> bool {
        self.callback.is_some()
    }

    /// Expected write length; short writes indicate I/O errors with O_DIRECT.
    #[inline]
    pub fn expected_len(&self) -> usize {
        self.expected_len
    }

    /// Computes the absolute file offset from zone-relative addressing.
    ///
    /// # Panics
    ///
    /// - `offset > zone_spec.size` (would write past zone boundary)
    #[inline]
    pub fn absolute_offset(&self) -> u64 {
        assert!(self.offset <= self.zone_spec.size);

        let abs = self.zone_spec.offset(self.offset);
        assert!(abs >= self.zone_spec.base);

        abs
    }
}

impl Default for Write {
    fn default() -> Self {
        Self::new()
    }
}

/// Phantom type tag for [`NextTick`] intrusive lists.
pub enum NextTickTag {}

/// Intrusive linked list node for deferred callbacks ("next tick" semantics).
///
/// Uses an intrusive design to avoid heap allocation during scheduling—nodes
/// live in caller-owned memory. The `callback` is `None` when idle, `Some`
/// when queued. This approach guarantees deterministic memory usage and
/// eliminates allocation failure paths during I/O completion.
#[repr(C)]
pub struct NextTick {
    link: QueueLink<NextTick, NextTickTag>,
    pub callback: Option<fn(&mut NextTick)>,
}

impl NextTick {
    pub const fn new() -> Self {
        Self {
            link: QueueLink::new(),
            callback: None,
        }
    }

    /// Returns `true` if this callback is queued and awaiting execution.
    #[inline]
    pub fn is_pending(&self) -> bool {
        self.callback.is_some()
    }
}

impl Default for NextTick {
    fn default() -> Self {
        Self::new()
    }
}

impl QueueNode<NextTickTag> for NextTick {
    fn queue_link(&mut self) -> &mut QueueLink<Self, NextTickTag> {
        &mut self.link
    }

    fn queue_link_ref(&self) -> &QueueLink<Self, NextTickTag> {
        &self.link
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::layout::Layout;

    // ==================== Read ====================

    #[test]
    fn read_init() {
        let read = Read::new();
        assert!(!read.is_pending());
        assert_eq!(read.expected_len(), 0);
        assert_eq!(read.zone, Zone::SuperBlock);
        assert_eq!(read.zone_spec.base, 0);
        assert_eq!(read.offset, 0);
    }

    #[test]
    fn read_default() {
        let r1 = Read::new();
        let r2 = Read::default();
        assert_eq!(r1.zone, r2.zone);
        assert_eq!(r1.offset, r2.offset);
    }

    #[test]
    fn read_pending() {
        let read = Read::new();
        assert!(!read.is_pending());
    }

    #[test]
    fn read_pending_true_when_callback_set() {
        fn cb(_: &mut Read) {}
        let mut read = Read::new();
        assert!(!read.is_pending());
        read.callback = Some(cb);
        assert!(read.is_pending());
        read.callback = None;
        assert!(!read.is_pending());
    }

    #[test]
    fn read_expected_len_accessors() {
        let mut read = Read::new();
        assert_eq!(read.expected_len(), 0);
        read.expected_len = 123;
        assert_eq!(read.expected_len(), 123);
    }

    #[test]
    fn read_io_starts_idle() {
        let read = Read::new();
        assert!(read.io.is_idle());
    }

    #[test]
    fn read_offset_logic() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec {
            base: 1000,
            size: 500,
        };

        // Start
        read.offset = 0;
        assert_eq!(read.absolute_offset(), 1000);

        // Middle
        read.offset = 250;
        assert_eq!(read.absolute_offset(), 1250);

        // Limit (inclusive of size for calculation, EOF check is separate)
        read.offset = 500;
        assert_eq!(read.absolute_offset(), 1500);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_offset_panic() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec {
            base: 1000,
            size: 500,
        };
        read.offset = 501;
        let _ = read.absolute_offset();
    }

    #[test]
    fn read_zero_size_zone_allows_only_zero_offset() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec { base: 42, size: 0 };
        read.offset = 0;
        assert_eq!(read.absolute_offset(), 42);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn read_zero_size_zone_panics_on_nonzero_offset() {
        let mut read = Read::new();
        read.zone_spec = ZoneSpec { base: 42, size: 0 };
        read.offset = 1;
        let _ = read.absolute_offset();
    }

    #[test]
    fn read_layout() {
        let layout = Layout::new(512, 4096, 512, 512, 512, 512, 4096);
        let mut read = Read::new();
        read.zone = Zone::WalPrepares;
        read.zone_spec = layout.zone(Zone::WalPrepares);
        read.offset = 100;
        assert_eq!(
            read.absolute_offset(),
            layout.start(Zone::WalPrepares) + 100
        );
    }

    // ==================== Write ====================

    #[test]
    fn write_init() {
        let write = Write::new();
        assert!(!write.is_pending());
        assert_eq!(write.expected_len(), 0);
        assert_eq!(write.zone, Zone::SuperBlock);
        assert_eq!(write.zone_spec.base, 0);
        assert_eq!(write.offset, 0);
    }

    #[test]
    fn write_default() {
        let w1 = Write::new();
        let w2 = Write::default();
        assert_eq!(w1.zone, w2.zone);
        assert_eq!(w1.offset, w2.offset);
    }

    #[test]
    fn write_pending() {
        let write = Write::new();
        assert!(!write.is_pending());
    }

    #[test]
    fn write_pending_true_when_callback_set() {
        fn cb(_: &mut Write) {}
        let mut write = Write::new();
        assert!(!write.is_pending());
        write.callback = Some(cb);
        assert!(write.is_pending());
        write.callback = None;
        assert!(!write.is_pending());
    }

    #[test]
    fn write_expected_len_accessors() {
        let mut write = Write::new();
        assert_eq!(write.expected_len(), 0);
        write.expected_len = 456;
        assert_eq!(write.expected_len(), 456);
    }

    #[test]
    fn write_io_starts_idle() {
        let write = Write::new();
        assert!(write.io.is_idle());
    }

    #[test]
    fn write_offset_logic() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec {
            base: 2000,
            size: 1000,
        };

        write.offset = 0;
        assert_eq!(write.absolute_offset(), 2000);

        write.offset = 500;
        assert_eq!(write.absolute_offset(), 2500);

        write.offset = 1000;
        assert_eq!(write.absolute_offset(), 3000);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_offset_panic() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec {
            base: 2000,
            size: 1000,
        };
        write.offset = 1001;
        let _ = write.absolute_offset();
    }

    #[test]
    fn write_zero_size_zone_allows_only_zero_offset() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec { base: 99, size: 0 };
        write.offset = 0;
        assert_eq!(write.absolute_offset(), 99);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn write_zero_size_zone_panics_on_nonzero_offset() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec { base: 99, size: 0 };
        write.offset = 1;
        let _ = write.absolute_offset();
    }

    #[test]
    fn write_layout() {
        let layout = Layout::new(512, 4096, 512, 512, 512, 512, 4096);
        let mut write = Write::new();
        write.zone = Zone::ClientReplies;
        write.zone_spec = layout.zone(Zone::ClientReplies);
        write.offset = 200;
        assert_eq!(
            write.absolute_offset(),
            layout.start(Zone::ClientReplies) + 200
        );
    }

    // ==================== Consistency ====================

    #[test]
    fn rw_consistency() {
        let zone_spec = ZoneSpec {
            base: 8192,
            size: 4096,
        };
        let mut read = Read::new();
        read.zone_spec = zone_spec;
        read.offset = 100;
        let mut write = Write::new();
        write.zone_spec = zone_spec;
        write.offset = 100;

        assert_eq!(read.absolute_offset(), write.absolute_offset());
    }

    #[test]
    fn rw_all_zones() {
        let layout = Layout::new(512, 4096, 4096, 8192, 16384, 8192, 65536);
        for zone in Zone::ALL {
            let zone_spec = layout.zone(zone);
            if zone_spec.size > 0 {
                let offset = zone_spec.size / 2;

                let mut read = Read::new();
                read.zone = zone;
                read.zone_spec = zone_spec;
                read.offset = offset;

                let mut write = Write::new();
                write.zone = zone;
                write.zone_spec = zone_spec;
                write.offset = offset;

                assert_eq!(read.absolute_offset(), write.absolute_offset());
                assert_eq!(read.absolute_offset(), zone_spec.base + offset);
            }
        }
    }

    #[test]
    fn read_io_field_is_first() {
        let offset = unsafe {
            let uninit = core::mem::MaybeUninit::<Read>::uninit();
            let base = uninit.as_ptr();
            let io_ptr = core::ptr::addr_of!((*base).io);
            (io_ptr as usize) - (base as usize)
        };
        assert_eq!(offset, 0);
    }

    #[test]
    fn write_io_field_is_first() {
        let offset = unsafe {
            let uninit = core::mem::MaybeUninit::<Write>::uninit();
            let base = uninit.as_ptr();
            let io_ptr = core::ptr::addr_of!((*base).io);
            (io_ptr as usize) - (base as usize)
        };
        assert_eq!(offset, 0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_read_absolute_offset(
            base in 0u64..1_000_000,
            size in 1u64..100_000,
            offset_factor in 0.0f64..1.0,
        ) {
            let mut read = Read::new();
            read.zone_spec = ZoneSpec { base, size };
            read.offset = (size as f64 * offset_factor) as u64;

            let abs = read.absolute_offset();
            prop_assert_eq!(abs, base + read.offset);
            prop_assert!(abs >= base);
        }

        #[test]
        fn prop_write_absolute_offset(
            base in 0u64..1_000_000,
            size in 1u64..100_000,
            offset_factor in 0.0f64..1.0,
        ) {
            let mut write = Write::new();
            write.zone_spec = ZoneSpec { base, size };
            write.offset = (size as f64 * offset_factor) as u64;

            let abs = write.absolute_offset();
            prop_assert_eq!(abs, base + write.offset);
            prop_assert!(abs >= base);
        }

        #[test]
        fn prop_read_write_consistency(
            base in 0u64..1_000_000,
            size in 1u64..100_000,
            offset_factor in 0.0f64..1.0,
        ) {
            let zone_spec = ZoneSpec { base, size };
            let offset = (size as f64 * offset_factor) as u64;

            let mut read = Read::new();
            read.zone_spec = zone_spec;
            read.offset = offset;

            let mut write = Write::new();
            write.zone_spec = zone_spec;
            write.offset = offset;

            prop_assert_eq!(read.absolute_offset(), write.absolute_offset());
        }

        #[test]
        fn prop_read_offset_within_zone(
            base in 0u64..1_000_000,
            size in 1u64..100_000,
            offset_factor in 0.0f64..1.0,
        ) {
            let zone_spec = ZoneSpec { base, size };
            let offset = (size as f64 * offset_factor) as u64;

            let mut read = Read::new();
            read.zone_spec = zone_spec;
            read.offset = offset;

            let abs = read.absolute_offset();
            prop_assert!(abs >= zone_spec.base);
            prop_assert!(abs <= zone_spec.end());
        }

        #[test]
        fn prop_layout_integration(
            sector_exp in 9u32..13u32,
            block_exp in 0u32..4u32,
            zone_idx in 0usize..Zone::COUNT,
        ) {
            let sector_size = 1u64 << sector_exp;
            let block_size = sector_size << block_exp;

            let layout = crate::storage::layout::Layout::new(
                sector_size,
                block_size,
                sector_size * 2,
                sector_size * 4,
                sector_size * 8,
                sector_size * 2,
                block_size * 16,
            );

            let zone = Zone::ALL[zone_idx];
            let zone_spec = layout.zone(zone);

            if zone_spec.size > 0 {
                let offset = zone_spec.size / 2;

                let mut read = Read::new();
                read.zone = zone;
                read.zone_spec = zone_spec;
                read.offset = offset;

                let abs = read.absolute_offset();
                prop_assert_eq!(abs, layout.offset(zone, offset));
            }
        }
    }
}

#[cfg(test)]
mod security_tests {
    use super::*;

    #[test]
    #[should_panic(expected = "ZoneSpec::offset overflow")]
    fn test_read_absolute_offset_overflow() {
        // Scenario: ZoneSpec is corrupted or maliciously crafted such that
        // base + size overflows u64, but we try to read within "size".
        let mut read = Read::new();
        read.zone_spec = ZoneSpec {
            base: u64::MAX - 10,
            size: 20, // base + size overflows
        };
        // offset is within "size" (15 <= 20), so the first check passes.
        // But base + offset (MAX - 10 + 15) = MAX + 5 -> Overflows.
        read.offset = 15;

        let _ = read.absolute_offset();
    }

    #[test]
    #[should_panic(expected = "ZoneSpec::offset overflow")]
    fn test_write_absolute_offset_overflow() {
        let mut write = Write::new();
        write.zone_spec = ZoneSpec {
            base: u64::MAX - 100,
            size: 200,
        };
        write.offset = 150;
        let _ = write.absolute_offset();
    }

    #[test]
    fn test_struct_sizes() {
        // Ensure structs don't accidentally grow or shrink in ways that might affect FFI or cache lines.
        // These assertions are platform-dependent (likely 64-bit), but valuable for tracking changes.
        // Read/Write:
        // IoCompletion (approx 88 bytes, includes Operation(32), ListLink(16), etc.)
        // + Option<fn> (8)
        // + usize (8)
        // + Zone (1+7pad = 8)
        // + ZoneSpec (16)
        // + u64 (8)
        // = 88 + 8 + 8 + 8 + 16 + 8 = 136 bytes.
        assert_eq!(std::mem::size_of::<Read>(), 136);
        assert_eq!(std::mem::size_of::<Write>(), 136);

        assert_eq!(std::mem::align_of::<Read>(), 8);
        assert_eq!(std::mem::align_of::<Write>(), 8);
    }
}
