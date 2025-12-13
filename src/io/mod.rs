pub mod iops;

use core::ffi::c_void;
use core::ptr::NonNull;
use std::os::unix::io::RawFd;
use std::time::{Duration, Instant};

use io_uring::{IoUring, opcode, types};

use crate::stdx::{DoublyLinkedList, ListLink, ListNode};

// Compile-time validations
const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

enum IoTag {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionState {
    Idle,
    Queued,
    Submitted,
    Completed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Nop,
    Read {
        fd: RawFd,
        buf: NonNull<u8>,
        len: u32,
        offset: u64,
    },
    Write {
        fd: RawFd,
        buf: NonNull<u8>,
        len: u32,
        offset: u64,
    },
    Fsync {
        fd: RawFd,
    },
}

impl Operation {
    #[inline]
    pub fn is_active(&self) -> bool {
        !matches!(self, Operation::Nop)
    }

    pub fn validate(&self) {
        match *self {
            Operation::Read {
                fd,
                buf,
                len,
                offset,
            } => {
                assert!(fd >= 0, "File descriptor must be non-negative");
                assert!(len > 0, "Length must be positive");
            }
            Operation::Write {
                fd,
                buf,
                len,
                offset,
            } => {
                assert!(fd >= 0, "File descriptor must be non-negative");
                assert!(len > 0, "Length must be positive");
            }
            Operation::Fsync { fd } => {
                assert!(fd >= 0, "File descriptor must be non-negative");
            }
            Operation::Nop => {}
        }
    }
}

pub type CompletionCallback = unsafe fn(*mut c_void, &mut Completion);

pub struct Completion {
    link: ListLink<Completion, IoTag>,
    state: CompletionState,

    pub result: i32,
    pub op: Operation,

    pub context: *mut c_void,
    pub callback: Option<CompletionCallback>,
}

impl Completion {
    pub const fn new() -> Self {
        Self {
            link: ListLink::new(),
            state: CompletionState::Idle,
            result: 0,
            op: Operation::Nop,
            context: core::ptr::null_mut(),
            callback: None,
        }
    }

    #[inline]
    pub fn state(&self) -> CompletionState {
        self.state
    }

    #[inline]
    pub fn is_idle(&self) -> bool {
        self.state == CompletionState::Idle
    }

    pub fn reset(&mut self) {
        assert!(!self.link.is_linked());
        assert!(self.state == CompletionState::Idle || self.state == CompletionState::Completed);

        self.link.reset();
        self.state = CompletionState::Idle;
        self.result = 0;
        self.op = Operation::Nop;
        self.context = core::ptr::null_mut();
        self.callback = None;

        assert!(self.is_idle());
        assert!(!self.link.is_linked());
    }

    #[inline]
    pub fn complete(&mut self) {
        assert!(self.state == CompletionState::Completed);
        assert!(!self.link.is_linked());

        let cb = self.callback.take();
        let ctx = self.context;

        self.state = CompletionState::Idle;
        assert!(self.is_idle());

        if let Some(cb) = cb {
            unsafe { cb(ctx, self) };
        }
    }

    fn set_queued(&mut self) {
        assert!(self.state == CompletionState::Idle);
        self.state = CompletionState::Queued;
    }

    fn set_submitted(&mut self) {
        assert!(self.state == CompletionState::Queued || self.state == CompletionState::Idle);
        self.state = CompletionState::Submitted;
    }

    fn set_completed(&mut self) {
        assert!(self.state == CompletionState::Submitted);
        self.state = CompletionState::Completed;
    }
}

impl Default for Completion {
    fn default() -> Self {
        Self::new()
    }
}
