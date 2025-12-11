use core::ffi::c_void;
use core::ptr::NonNull;
use std::os::unix::io::RawFd;
use std::time::{Duration, Instant};

use io_uring::{IoUring, opcode, types};

use crate::stdx::{DoublyLinkedList, ListLink, ListNode};

pub mod iops;

// Compile-time validations
const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);
