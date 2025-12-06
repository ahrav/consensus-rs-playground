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

    #[inline]
    pub fn body(&self) -> &[u8] {
        let header = self.header();
        let total_len = self.len();

        assert!(total_len >= Self::LEN_MIN);
        assert!(total_len <= Self::LEN_MAX);
        assert!(total_len == self.len);

        let body = &self.buffer.bytes[HEADER_SIZE_USIZE..total_len as usize];
        assert!(body.len() as u32 == header.body_len());

        body
    }

    #[inline]
    pub fn body_mut(&mut self) -> &mut [u8] {
        let total_len = self.header().total_len();

        assert!(total_len >= Self::LEN_MIN);
        assert!(total_len <= Self::LEN_MAX);
        assert!(total_len == self.len);

        let body_len = total_len - Self::LEN_MIN;
        let body = &mut self.buffer.bytes[HEADER_SIZE_USIZE..total_len as usize];
        assert!(body.len() as u32 == body_len);

        body
    }

    pub fn reset(&mut self, command: Command, cluster: u128, replica: u8) {
        let header = Header::new(command, cluster, replica);
        let header_bytes = header.as_bytes();

        assert!(header_bytes.len() == HEADER_SIZE_USIZE);

        self.buffer.bytes[..HEADER_SIZE_USIZE].copy_from_slice(header_bytes);
        self.len = Self::LEN_MIN;

        assert!(self.header().command == command);
        assert!(self.header().cluster == cluster);
        assert!(self.header().replica == replica);
        assert!(self.header().size == Self::LEN_MIN);
    }

    pub fn set_body(&mut self, body: &[u8]) -> Result<(), &'static str> {
        // Preconditions
        assert!(body.len() <= u32::MAX as usize);
        let body_len = body.len() as u32;

        if body_len > MESSAGE_BODY_SIZE_MAX {
            return Err("body too large");
        }

        let total_len = Self::LEN_MIN + body_len;

        // Verify bounds
        assert!(total_len >= Self::LEN_MIN);
        assert!(total_len <= Self::LEN_MAX);

        // Copy body into buffer
        self.buffer.bytes[HEADER_SIZE_USIZE..total_len as usize].copy_from_slice(body);
        self.len = total_len;

        // Update header
        {
            let h = self.header_mut();
            h.size = total_len;

            // TB ordering: set_checksum_body then set_checksum
            h.set_checksum_body(body);
            h.set_checksum();
        }

        // Postconditions
        assert!(self.len == total_len);
        assert!(self.header().size == total_len);
        assert!(self.header().body_len() == body_len);
        assert!(self.header().is_valid_checksum());
        assert!(self.header().is_valid_checksum_body(body));

        assert!(self.body() == body);

        Ok(())
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        assert!(self.len() >= Self::LEN_MIN);
        assert!(self.len() <= Self::LEN_MAX);

        let bytes = &self.buffer.bytes[..self.len as usize];
        assert!(bytes.len() as u32 == self.len);

        bytes
    }

    #[inline]
    pub fn ref_acquire(&self) {
        let old = self.reference.get();

        assert!(old < u32::MAX);

        let new = old + 1;
        self.reference.set(new);

        assert!(self.reference.get() == new);
        assert!(self.reference.get() == old + 1);
    }

    #[inline]
    pub fn ref_release(&self) -> bool {
        let old = self.reference.get();

        assert!(old > 0, "reference count underflow");

        let new = old - 1;
        self.reference.set(new);

        assert!(self.reference.get() == new);
        assert!(self.reference.get() == old - 1);

        new == 0
    }

    #[inline]
    pub fn ref_count(&self) -> u32 {
        self.reference.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_is_correct() {
        let msg = Message::new_zeroed();
        let ptr = msg.buffer.bytes.as_ptr() as usize;

        assert!(ptr % align_of::<Header>() == 0);
        assert!(ptr % 16 == 0);
    }

    #[test]
    fn new_message_is_valid() {
        let msg = Message::new_zeroed();

        assert!(msg.len() == Message::LEN_MIN);
        assert!(msg.body_len() == 0);
        assert!(msg.ref_count() == 0);
    }

    #[test]
    fn reset_initializes_header() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Ping, 42, 3);

        assert!(msg.header().command == Command::Ping);
        assert!(msg.header().cluster == 42);
        assert!(msg.header().replica == 3);
        assert!(msg.len() == Message::LEN_MIN);
    }

    #[test]
    fn set_body_updates_message() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let body = b"hello world";
        assert!(msg.set_body(body).is_ok());

        assert!(msg.len() == Message::LEN_MIN + body.len() as u32);
        assert!(msg.body() == body);
        assert!(msg.header().is_valid_checksum());
        assert!(msg.header().is_valid_checksum_body(body));
    }

    #[test]
    fn set_body_rejects_oversized() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let big_body = vec![0u8; MESSAGE_BODY_SIZE_MAX as usize + 1];
        assert!(msg.set_body(&big_body).is_err());
    }

    #[test]
    fn reference_counting() {
        let msg = Message::new_zeroed();

        assert!(msg.ref_count() == 0);

        msg.ref_acquire();
        assert!(msg.ref_count() == 1);

        msg.ref_acquire();
        assert!(msg.ref_count() == 2);

        assert!(!msg.ref_release());
        assert!(msg.ref_count() == 1);

        assert!(msg.ref_release());
        assert!(msg.ref_count() == 0);
    }

    #[test]
    fn as_bytes_length() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Ping, 1, 0);

        // Header only
        assert!(msg.as_bytes().len() == HEADER_SIZE_USIZE);

        // With body
        let body = b"test body";
        msg.set_body(body).unwrap();
        assert!(msg.as_bytes().len() == HEADER_SIZE_USIZE + body.len());
    }

    #[test]
    fn body_roundtrip() {
        let mut msg = Message::new_zeroed();
        msg.reset(Command::Request, 1, 0);

        let original = b"some important data";
        msg.set_body(original).unwrap();

        assert!(msg.body() == original);
    }
}
