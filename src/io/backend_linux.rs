#![cfg(target_os = "linux")]

use std::io;

use io_uring::{IoUring, opcode, types};

use crate::io::{IoBackend, Operation};

pub struct UringBackend {
    ring: IoUring,
}
