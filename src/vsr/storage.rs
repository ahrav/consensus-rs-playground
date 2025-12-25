//! Storage trait for VSR protocol I/O.
//!
//! VSR requires deterministic, sector-aligned I/O with callback-based completion.
//! This trait captures those requirements, allowing mock backends for testing
//! and fault injection without modifying protocol logic.

use crate::container_of;
use crate::storage;
use crate::storage::iocb;
use crate::vsr::superblock;

/// Backend-agnostic storage interface for sector-aligned async I/O.
///
/// Implementations must guarantee sector-aligned operations and provide
/// mechanisms to recover parent contexts from embedded I/O control blocks.
pub trait Storage: Sized {
    /// I/O control block for pending read operations.
    type Read;
    /// I/O control block for pending write operations.
    type Write;

    /// Logical region within the storage file (superblock, WAL, LSM zones).
    type Zone: Copy;

    /// Zone identifier for superblock I/O operations.
    const SUPERBLOCK_ZONE: Self::Zone;

    /// Submits an async read at `offset` within `zone`.
    ///
    /// `callback` is invoked when the read completes. `read` must remain valid
    /// until the callback fires. `buffer` must be sector-aligned for O_DIRECT.
    fn read_sectors(
        &mut self,
        callback: fn(&mut Self::Read),
        read: &mut Self::Read,
        buffer: &mut [u8],
        zone: Self::Zone,
        offset: u64,
    );

    /// Submits an async write at `offset` within `zone`.
    ///
    /// `callback` is invoked when the write completes. `write` must remain valid
    /// until the callback fires. `buffer` must be sector-aligned for O_DIRECT.
    fn write_sectors(
        &mut self,
        callback: fn(&mut Self::Write),
        write: &mut Self::Write,
        buffer: &[u8],
        zone: Self::Zone,
        offset: u64,
    );

    /// Recovers the owning [`superblock::Context`] from an embedded `Read` iocb.
    ///
    /// Used in I/O callbacks to navigate from the completed iocb back to the
    /// parent context, enabling state machine progression without heap allocation.
    ///
    /// # Safety
    ///
    /// `read` must be the address of the `read` field inside a live
    /// `superblock::Context<Self>`. The context must outlive this call.
    unsafe fn context_from_read(read: &mut Self::Read) -> &mut superblock::Context<Self>;

    /// Recovers the owning [`superblock::Context`] from an embedded `Write` iocb.
    ///
    /// # Safety
    ///
    /// `write` must be the address of the `write` field inside a live
    /// `superblock::Context<Self>`. The context must outlive this call.
    unsafe fn context_from_write(write: &mut Self::Write) -> &mut superblock::Context<Self>;
}

impl Storage for storage::Storage {
    type Read = iocb::Read;
    type Write = iocb::Write;

    type Zone = storage::Zone;

    const SUPERBLOCK_ZONE: Self::Zone = storage::Zone::SuperBlock;

    #[inline]
    fn read_sectors(
        &mut self,
        callback: fn(&mut Self::Read),
        read: &mut Self::Read,
        buffer: &mut [u8],
        zone: Self::Zone,
        offset: u64,
    ) {
        storage::Storage::read_sectors(self, callback, read, buffer, zone, offset);
    }

    #[inline]
    fn write_sectors(
        &mut self,
        callback: fn(&mut Self::Write),
        write: &mut Self::Write,
        buffer: &[u8],
        zone: Self::Zone,
        offset: u64,
    ) {
        storage::Storage::write_sectors(self, callback, write, buffer, zone, offset);
    }

    #[inline]
    unsafe fn context_from_read(read: &mut Self::Read) -> &mut superblock::Context<Self> {
        let ptr = read as *mut Self::Read;
        unsafe { &mut *container_of!(ptr, superblock::Context<Self>, read) }
    }

    #[inline]
    unsafe fn context_from_write(write: &mut Self::Write) -> &mut superblock::Context<Self> {
        let ptr = write as *mut Self::Write;
        unsafe { &mut *container_of!(ptr, superblock::Context<Self>, write) }
    }
}
