//! Linux io_uring backend for high-performance async I/O.
//!
//! Uses io_uring for direct kernel I/O submission with minimal syscalls.
//! Chosen over epoll/aio for batch submission, lower latency, and unified
//! interface for file/socket operations.

use std::io;

use io_uring::{IoUring, opcode, types};

use crate::io::{IoBackend, Operation};

/// Maximum completions drained per `drain()` call.
///
/// Prevents unbounded iteration if completions arrive faster than we process.
/// Allows draining to yield control back to caller for fairness.
const MAX_DRAIN_PER_CALL: u32 = 16_384;

const _: () = {
    assert!(MAX_DRAIN_PER_CALL >= 1024);
    assert!(MAX_DRAIN_PER_CALL <= u32::MAX / 2);
};

/// io_uring-backed I/O backend for Linux.
///
/// Maintains a single io_uring instance with submission and completion queues.
/// The ring size must be a power of two for efficient modulo operations.
pub struct UringBackend {
    ring: IoUring,
    /// Cached ring size for runtime assertions. io_uring itself knows this value,
    /// but we store it to avoid repeated syscalls during validation.
    entries: u32,
}

impl IoBackend for UringBackend {
    const ENTRIES_MIN: u32 = 1;
    const ENTRIES_MAX: u32 = 1 << 15;

    fn new(entries: u32) -> Result<Self> {
        assert!(entries > 0);
        assert!(entries.is_power_of_two());
        assert!(entries <= Self::ENTRIES_MAX);

        let mut ring = IoUring::new(entries)?;
        assert!(ring.submission().capacity() >= entries as usize);

        Ok(Self { ring, entries })
    }

    /// # Safety
    ///
    /// Caller must ensure:
    /// - File descriptors are valid and open
    /// - Buffer pointers are valid for the specified length
    /// - Buffers remain valid until completion (no deallocation/reuse)
    /// - Offsets don't overflow `i64::MAX`
    unsafe fn try_push(&mut self, op: &Operation, user_data: u64) -> Result<(), ()> {
        match *op {
            Operation::Nop => {}
            Operation::Read {
                fd,
                buf,
                len,
                offset,
            } => {
                assert!(fd >= 0);
                assert!(!buf.as_ptr().is_null());
                assert!(len > 0);
                assert!(offset <= i64::MAX as u64);
            }
            Operation::Write {
                fd,
                buf,
                len,
                offset,
            } => {
                assert!(fd >= 0);
                assert!(!buf.as_ptr().is_null());
                assert!(len > 0);
                assert!(offset <= i64::MAX as u64);
            }
            Operation::Fsync { fd } => {
                assert!(fd >= 0);
            }
        }

        let mut sq = self.ring.submission();
        let was_full = sq.is_full();

        if was_full {
            return Err(());
        }

        let len_before = sq.len();

        let entry = match *op {
            Operation::Nop => opcode::Nop::new().build().user_data(user_data),
            Operation::Read {
                fd,
                buf,
                len,
                offset,
            } => opcode::Read::new(types::Fd(fd), buf.as_ptr(), len)
                .offset(offset as i64)
                .build()
                .user_data(user_data),
            Operation::Write {
                fd,
                buf,
                len,
                offset,
            } => opcode::Write::new(types::Fd(fd), buf.as_ptr(), len)
                .offset(offset as i64)
                .build()
                .user_data(user_data),
            Operation::Fsync { fd } => opcode::Fsync::new(types::Fd(fd))
                .build()
                .user_data(user_data),
        };

        let push_result = unsafe { sq.push(&entry) };

        assert!(push_result.is_ok());

        let len_after = sq.len();
        assert!(len_after == len_before + 1);

        Ok(())
    }

    /// Submits pending operations to the kernel.
    ///
    /// If `wait_for_one` is true, blocks until at least one completion is ready.
    /// This avoids busy-waiting when the caller needs results. If false, submits
    /// without blocking (fire-and-forget).
    fn flush(&mut self, wait_for_one: bool) -> io::Result<()> {
        let pending_before = self.ring.submission().len();

        if wait_for_one {
            assert!(pending_before > 0 || !self.ring.completion().is_empty());

            let submitted = self.ring.submit_and_wait(1)?;

            assert!(submitted <= pending_before);
        } else if pending_before > 0 {
            let submitted = self.ring.submit()?;
            assert!(submitted == pending_before);
        }

        Ok(())
    }

    /// Processes completed operations, invoking `f(user_data, result)` for each.
    ///
    /// Drains up to `MAX_DRAIN_PER_CALL` completions to prevent starvation of other tasks.
    /// The `result` passed to `f` is bytes transferred (positive) or `-errno` (negative).
    fn drain<F: FnMut(u64, i32)>(&mut self, mut f: F) {
        let mut cq = self.ring.completion();

        let mut drained_count: u32 = 0;
        for cqe in &mut cq {
            assert!(drained_count < MAX_DRAIN_PER_CALL);

            f(cqe.user_data(), cqe.result());
            drained_count += 1;
        }

        assert!(drained_count <= MAX_DRAIN_PER_CALL);

        if drained_count < MAX_DRAIN_PER_CALL {
            assert!(self.ring.completion().is_empty());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "ring must have at least one entry")]
    fn test_new_zero_entries_panics() {
        let _ = UringBackend::new(0);
    }

    #[test]
    #[should_panic(expected = "power-of-two")]
    fn test_new_non_power_of_two_panics() {
        let _ = UringBackend::new(100);
    }
}
