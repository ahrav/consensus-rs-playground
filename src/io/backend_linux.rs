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

    fn new(entries: u32) -> io::Result<Self> {
        assert!(entries > 0, "ring must have at least one entry");
        assert!(entries.is_power_of_two(), "power-of-two");
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
                len,
                offset,
                ..
            } => {
                assert!(fd >= 0);
                assert!(len > 0);
                assert!(offset <= i64::MAX as u64);
            }
            Operation::Write {
                fd,
                len,
                offset,
                ..
            } => {
                assert!(fd >= 0);
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
                .offset(offset)
                .build()
                .user_data(user_data),
            Operation::Write {
                fd,
                buf,
                len,
                offset,
            } => opcode::Write::new(types::Fd(fd), buf.as_ptr(), len)
                .offset(offset)
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
            // Note: We rely on the caller (IoCore) to ensure there are in-flight operations
            // before asking to wait.
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
        for cqe in (&mut cq).take(MAX_DRAIN_PER_CALL as usize) {
            f(cqe.user_data(), cqe.result());
            drained_count += 1;
        }

        assert!(drained_count <= MAX_DRAIN_PER_CALL);
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

    #[test]
    fn drain_respects_max_per_call() {
        let mut backend = UringBackend::new(UringBackend::ENTRIES_MAX).unwrap();

        let count = MAX_DRAIN_PER_CALL as usize + 1;
        assert!(count <= UringBackend::ENTRIES_MAX as usize);

        for i in 0..count {
            unsafe {
                backend.try_push(&Operation::Nop, i as u64).unwrap();
            }
        }

        backend.flush(false).unwrap();
        backend.ring.submit_and_wait(count).unwrap();

        let mut first = 0usize;
        backend.drain(|_, _| first += 1);
        assert_eq!(first, MAX_DRAIN_PER_CALL as usize);

        let mut second = 0usize;
        backend.drain(|_, _| second += 1);
        assert_eq!(second, 1);
    }

    #[test]
    fn test_flush_wait_on_submitted() {
        let mut backend = UringBackend::new(8).unwrap();

        // Submit one operation (NOP).
        unsafe {
            backend.try_push(&Operation::Nop, 1).unwrap();
        }

        // Flush without waiting (submit to kernel).
        backend.flush(false).unwrap();

        // Now ask to wait for completion.
        // This catches the bug where flush asserted `pending_before > 0` even if waiting on in-flight ops.
        backend.flush(true).unwrap();

        let mut count = 0;
        backend.drain(|_, _| count += 1);
        assert_eq!(count, 1);
    }

    #[test]
    fn new_min_size() {
        let backend = UringBackend::new(UringBackend::ENTRIES_MIN).unwrap();
        assert_eq!(backend.entries, UringBackend::ENTRIES_MIN);
    }

    #[test]
    fn new_common_sizes() {
        for &size in &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let backend = UringBackend::new(size).unwrap();
            assert_eq!(backend.entries, size);
        }
    }

    #[test]
    fn new_max_size() {
        let backend = UringBackend::new(UringBackend::ENTRIES_MAX).unwrap();
        assert_eq!(backend.entries, UringBackend::ENTRIES_MAX);
    }

    #[test]
    #[should_panic]
    fn new_exceeds_max() {
        let _ = UringBackend::new(1 << 16);
    }

    #[test]
    fn push_to_empty_ring() {
        let mut backend = UringBackend::new(8).unwrap();
        let result = unsafe { backend.try_push(&Operation::Nop, 0x1234) };
        assert!(result.is_ok());
    }

    #[test]
    fn push_fills_to_capacity() {
        let mut backend = UringBackend::new(4).unwrap();

        for i in 0..4 {
            let result = unsafe { backend.try_push(&Operation::Nop, i) };
            assert!(result.is_ok());
        }

        let result = unsafe { backend.try_push(&Operation::Nop, 99) };
        assert!(result.is_err());
    }

    #[test]
    fn push_err_when_full() {
        let mut backend = UringBackend::new(2).unwrap();

        unsafe {
            backend.try_push(&Operation::Nop, 1).unwrap();
            backend.try_push(&Operation::Nop, 2).unwrap();
        }

        let result = unsafe { backend.try_push(&Operation::Nop, 3) };
        assert!(result.is_err());
    }

    #[test]
    fn push_after_drain() {
        let mut backend = UringBackend::new(2).unwrap();

        unsafe {
            backend.try_push(&Operation::Nop, 1).unwrap();
            backend.try_push(&Operation::Nop, 2).unwrap();
        }

        backend.flush(false).unwrap();
        backend.flush(true).unwrap();
        backend.drain(|_, _| {});

        let result = unsafe { backend.try_push(&Operation::Nop, 3) };
        assert!(result.is_ok());
    }

    #[test]
    fn flush_no_wait() {
        let mut backend = UringBackend::new(8).unwrap();
        unsafe { backend.try_push(&Operation::Nop, 1).unwrap() };
        backend.flush(false).unwrap();
    }

    #[test]
    fn flush_empty_queue() {
        let mut backend = UringBackend::new(8).unwrap();
        backend.flush(false).unwrap();
    }

    #[test]
    fn drain_empty_queue() {
        let mut backend = UringBackend::new(8).unwrap();
        let mut count = 0;
        backend.drain(|_, _| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn nop_completes() {
        let mut backend = UringBackend::new(8).unwrap();
        unsafe { backend.try_push(&Operation::Nop, 0x42).unwrap() };
        backend.flush(true).unwrap();

        let mut completed = false;
        backend.drain(|user_data, result| {
            assert_eq!(user_data, 0x42);
            assert_eq!(result, 0);
            completed = true;
        });
        assert!(completed);
    }

    #[test]
    #[should_panic(expected = "fd >= 0")]
    fn read_negative_fd() {
        let mut backend = UringBackend::new(8).unwrap();
        let op = Operation::Read {
            fd: -1,
            buf: core::ptr::NonNull::dangling(),
            len: 1,
            offset: 0,
        };
        unsafe {
            let _ = backend.try_push(&op, 0);
        }
    }

    #[test]
    #[should_panic(expected = "fd >= 0")]
    fn write_negative_fd() {
        let mut backend = UringBackend::new(8).unwrap();
        let op = Operation::Write {
            fd: -1,
            buf: core::ptr::NonNull::dangling(),
            len: 1,
            offset: 0,
        };
        unsafe {
            let _ = backend.try_push(&op, 0);
        }
    }

    #[test]
    #[should_panic(expected = "fd >= 0")]
    fn fsync_negative_fd() {
        let mut backend = UringBackend::new(8).unwrap();
        unsafe {
            let _ = backend.try_push(&Operation::Fsync { fd: -1 }, 0);
        }
    }

    #[test]
    #[should_panic(expected = "len > 0")]
    fn read_zero_len() {
        let mut backend = UringBackend::new(8).unwrap();
        let op = Operation::Read {
            fd: 0,
            buf: core::ptr::NonNull::dangling(),
            len: 0,
            offset: 0,
        };
        unsafe {
            let _ = backend.try_push(&op, 0);
        }
    }

    #[test]
    #[should_panic(expected = "len > 0")]
    fn write_zero_len() {
        let mut backend = UringBackend::new(8).unwrap();
        let op = Operation::Write {
            fd: 0,
            buf: core::ptr::NonNull::dangling(),
            len: 0,
            offset: 0,
        };
        unsafe {
            let _ = backend.try_push(&op, 0);
        }
    }

    #[test]
    #[should_panic(expected = "offset <= i64::MAX")]
    fn read_offset_overflow() {
        let mut backend = UringBackend::new(8).unwrap();
        let op = Operation::Read {
            fd: 0,
            buf: core::ptr::NonNull::dangling(),
            len: 1,
            offset: u64::MAX,
        };
        unsafe {
            let _ = backend.try_push(&op, 0);
        }
    }

    #[test]
    #[should_panic(expected = "offset <= i64::MAX")]
    fn write_offset_overflow() {
        let mut backend = UringBackend::new(8).unwrap();
        let op = Operation::Write {
            fd: 0,
            buf: core::ptr::NonNull::dangling(),
            len: 1,
            offset: u64::MAX,
        };
        unsafe {
            let _ = backend.try_push(&op, 0);
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use core::ptr::NonNull;
    use std::fs::{File, OpenOptions};
    use std::io::{Read, Write};
    use std::os::unix::io::AsRawFd;
    use tempfile::tempdir;

    #[test]
    fn read_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("read_test.txt");
        let test_data = b"Hello, io_uring!";

        File::create(&path).unwrap().write_all(test_data).unwrap();

        let file = File::open(&path).unwrap();
        let fd = file.as_raw_fd();
        let mut backend = UringBackend::new(8).unwrap();

        let mut buf = vec![0u8; test_data.len()];
        let op = Operation::Read {
            fd,
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: buf.len() as u32,
            offset: 0,
        };

        unsafe { backend.try_push(&op, 0xDEADBEEF).unwrap() };
        backend.flush(true).unwrap();

        let mut completed = false;
        backend.drain(|data, result| {
            assert_eq!(data, 0xDEADBEEF);
            assert_eq!(result, test_data.len() as i32);
            completed = true;
        });

        assert!(completed);
        assert_eq!(&buf, test_data);
    }

    #[test]
    fn read_at_offset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("offset_read.txt");

        File::create(&path)
            .unwrap()
            .write_all(b"0123456789ABCDEF")
            .unwrap();

        let file = File::open(&path).unwrap();
        let fd = file.as_raw_fd();
        let mut backend = UringBackend::new(8).unwrap();

        let mut buf = vec![0u8; 4];
        let op = Operation::Read {
            fd,
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: 4,
            offset: 10,
        };

        unsafe { backend.try_push(&op, 1).unwrap() };
        backend.flush(true).unwrap();

        let mut bytes_read = 0;
        backend.drain(|_, result| bytes_read = result);

        assert_eq!(bytes_read, 4);
        assert_eq!(&buf, b"ABCD");
    }

    #[test]
    fn write_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("write_test.txt");

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        let fd = file.as_raw_fd();
        let mut backend = UringBackend::new(8).unwrap();

        let data = b"io_uring write test";
        let op = Operation::Write {
            fd,
            buf: NonNull::new(data.as_ptr() as *mut u8).unwrap(),
            len: data.len() as u32,
            offset: 0,
        };

        unsafe { backend.try_push(&op, 0x1).unwrap() };
        backend.flush(true).unwrap();

        let mut bytes_written = 0;
        backend.drain(|_, result| bytes_written = result);

        assert_eq!(bytes_written, data.len() as i32);

        drop(file);
        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents, "io_uring write test");
    }

    #[test]
    fn write_fsync() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("fsync_test.txt");

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        let fd = file.as_raw_fd();
        let mut backend = UringBackend::new(8).unwrap();

        let data = b"durable data";
        let write_op = Operation::Write {
            fd,
            buf: NonNull::new(data.as_ptr() as *mut u8).unwrap(),
            len: data.len() as u32,
            offset: 0,
        };
        unsafe { backend.try_push(&write_op, 0x1).unwrap() };
        unsafe { backend.try_push(&Operation::Fsync { fd }, 0x2).unwrap() };

        backend.flush(true).unwrap();

        let mut write_result = None;
        let mut fsync_result = None;

        for _ in 0..3 {
            backend.drain(|user_data, result| match user_data {
                0x1 => write_result = Some(result),
                0x2 => fsync_result = Some(result),
                _ => panic!("unexpected user_data"),
            });
            if write_result.is_some() && fsync_result.is_some() {
                break;
            }
            backend.flush(true).ok();
        }

        assert_eq!(write_result, Some(data.len() as i32));
        assert_eq!(fsync_result, Some(0));

        drop(file);
        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents, "durable data");
    }

    #[test]
    fn batch_operations() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("batch_test.txt");

        File::create(&path)
            .unwrap()
            .write_all(b"initial content here")
            .unwrap();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let fd = file.as_raw_fd();
        let mut backend = UringBackend::new(16).unwrap();

        let mut read_buf = vec![0u8; 7];
        let write_data = b"REPLACED";

        let read_op = Operation::Read {
            fd,
            buf: NonNull::new(read_buf.as_mut_ptr()).unwrap(),
            len: 7,
            offset: 0,
        };
        let write_op = Operation::Write {
            fd,
            buf: NonNull::new(write_data.as_ptr() as *mut u8).unwrap(),
            len: write_data.len() as u32,
            offset: 0,
        };

        unsafe {
            backend.try_push(&read_op, 1).unwrap();
            backend.try_push(&write_op, 2).unwrap();
            backend.try_push(&Operation::Nop, 3).unwrap();
        }

        backend.flush(true).unwrap();

        let mut completions = vec![];
        for _ in 0..5 {
            backend.drain(|user_data, result| completions.push((user_data, result)));
            if completions.len() >= 3 {
                break;
            }
            backend.flush(true).ok();
        }

        assert_eq!(completions.len(), 3);
        let user_datas: Vec<u64> = completions.iter().map(|(ud, _)| *ud).collect();
        assert!(user_datas.contains(&1));
        assert!(user_datas.contains(&2));
        assert!(user_datas.contains(&3));
    }

    #[test]
    fn read_closed_fd() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("closed_fd_test.txt");

        File::create(&path).unwrap();
        let file = File::open(&path).unwrap();
        let fd = file.as_raw_fd();

        let mut backend = UringBackend::new(8).unwrap();
        
        drop(file);

        let mut buf = vec![0u8; 64];
        let op = Operation::Read {
            fd,
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: 64,
            offset: 0,
        };

        unsafe { backend.try_push(&op, 0x99).unwrap() };
        backend.flush(true).unwrap();

        let mut got_error = false;
        backend.drain(|user_data, result| {
            assert_eq!(user_data, 0x99);
            assert!(result < 0);
            got_error = true;
        });
        assert!(got_error);
    }

    #[test]
    fn write_readonly_fd() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("readonly_test.txt");

        File::create(&path).unwrap();
        let file = File::open(&path).unwrap();
        let fd = file.as_raw_fd();
        let mut backend = UringBackend::new(8).unwrap();

        let data = b"should fail";
        let op = Operation::Write {
            fd,
            buf: NonNull::new(data.as_ptr() as *mut u8).unwrap(),
            len: data.len() as u32,
            offset: 0,
        };

        unsafe { backend.try_push(&op, 0xAA).unwrap() };
        backend.flush(true).unwrap();

        let mut got_error = false;
        backend.drain(|user_data, result| {
            assert_eq!(user_data, 0xAA);
            assert!(result < 0);
            got_error = true;
        });
        assert!(got_error);
    }

    #[test]
    fn short_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("short_read.txt");

        File::create(&path).unwrap().write_all(b"hello").unwrap();

        let file = File::open(&path).unwrap();
        let fd = file.as_raw_fd();
        let mut backend = UringBackend::new(8).unwrap();

        let mut buf = vec![0u8; 100];
        let op = Operation::Read {
            fd,
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: 100,
            offset: 0,
        };

        unsafe { backend.try_push(&op, 1).unwrap() };
        backend.flush(true).unwrap();

        let mut bytes_read = 0;
        backend.drain(|_, result| bytes_read = result);

        assert_eq!(bytes_read, 5);
        assert_eq!(&buf[..5], b"hello");
    }

    #[test]
    fn read_eof() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("eof_test.txt");

        File::create(&path)
            .unwrap()
            .write_all(b"0123456789")
            .unwrap();

        let file = File::open(&path).unwrap();
        let fd = file.as_raw_fd();
        let mut backend = UringBackend::new(8).unwrap();

        let mut buf = vec![0u8; 10];
        let op = Operation::Read {
            fd,
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: 10,
            offset: 10,
        };

        unsafe { backend.try_push(&op, 1).unwrap() };
        backend.flush(true).unwrap();

        let mut bytes_read = -1;
        backend.drain(|_, result| bytes_read = result);

        assert_eq!(bytes_read, 0);
    }
}
