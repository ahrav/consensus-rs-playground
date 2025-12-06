use super::{Command, Header};
use crate::constants::{
    HEADER_SIZE, HEADER_SIZE_USIZE, MESSAGE_BODY_SIZE_MAX, MESSAGE_SIZE_MAX, MESSAGE_SIZE_MAX_USIZE,
};
use core::cell::Cell;

// Compile-time: verify alignment requirements
const _: () = assert!(align_of::<Header>() <= 16);
const _: () = assert!(align_of::<u128>() <= 16);

// Compile-time: verify size relationships
const _: () = assert!(HEADER_SIZE_USIZE <= MESSAGE_SIZE_MAX_USIZE);
const _: () = assert!(MESSAGE_BODY_SIZE_MAX == MESSAGE_SIZE_MAX - HEADER_SIZE);

#[repr(C, align(16))]
struct AlignedBuffer {
    bytes: [u8; MESSAGE_SIZE_MAX_USIZE],
}

const _: () = assert!(align_of::<AlignedBuffer>() >= align_of::<Header>());
const _: () = assert!(size_of::<AlignedBuffer>() == MESSAGE_SIZE_MAX_USIZE);

impl AlignedBuffer {
    fn new_zeroed() -> Box<Self> {
        let buffer = Box::new(AlignedBuffer {
            bytes: [0; MESSAGE_SIZE_MAX_USIZE],
        });

        assert!(buffer.bytes.as_ptr() as usize % align_of::<Header>() == 0);
        buffer
    }
}

pub struct Message {
    pub reference: Cell<u32>,
    buffer: Box<AlignedBuffer>,
    // cached length of currently valid bytes in `buffer`.
    len: u32,
}

impl Message {
    pub const LEN_MIN: u32 = HEADER_SIZE;
    pub const LEN_MAX: u32 = MESSAGE_SIZE_MAX;

    pub fn new_zeroed() -> Self {
        let buffer = AlignedBuffer::new_zeroed();
        assert!(buffer.bytes.as_ptr() as usize % align_of::<Header>() == 0);

        let msg = Message {
            reference: Cell::new(0),
            buffer,
            len: HEADER_SIZE,
        };

        assert!(msg.len >= Self::LEN_MIN);
        assert!(msg.len <= Self::LEN_MAX);
        assert!(msg.reference.get() == 0);

        msg
    }

    #[inline]
    pub fn len(&self) -> u32 {
        assert!(self.len >= Self::LEN_MIN);
        assert!(self.len <= Self::LEN_MAX);
        self.len
    }

    #[inline]
    pub fn body_len(&self) -> u32 {
        assert!(self.len >= Self::LEN_MIN);
        assert!(self.len <= Self::LEN_MAX);
        self.len - Self::LEN_MIN
    }

    #[inline]
    pub fn header(&self) -> &Header {
        assert!(self.len >= Self::LEN_MIN);
        assert!(self.buffer.bytes.as_ptr() as usize % align_of::<Header>() == 0);

        // SAFETY:
        // - Buffer is aligned to 16 bytes (AlignedBuffer)
        // - Buffer is at least HEADER_SIZE bytes
        // - Header is repr(C) with known layout
        // - Lifetime tied to &self
        unsafe { &*(self.buffer.bytes.as_ptr() as *const Header) }
    }

    #[inline]
    pub fn header_mut(&mut self) -> &mut Header {
        assert!(self.len >= Self::LEN_MIN);
        assert!(self.buffer.bytes.as_ptr() as usize % align_of::<Header>() == 0);

        // SAFETY:
        // - Buffer is aligned to 16 bytes (AlignedBuffer)
        // - Buffer is at least HEADER_SIZE bytes
        // - Header is repr(C) with known layout
        // - Lifetime tied to &mut self
        unsafe { &mut *(self.buffer.bytes.as_mut_ptr() as *mut Header) }
    }
}
