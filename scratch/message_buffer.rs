//! Port of TigerBeetle's message_buffer.zig.
//!
//! MessageBuffer is a producer-consumer ring buffer for bytes with support for:
//! - skipping ("suspending") messages to revisit later
//! - sticky checksum validation so headers/bodies are verified once

use core::{mem, slice};

use crate::{
    constants::{HEADER_SIZE, HEADER_SIZE_USIZE, MESSAGE_SIZE_MAX, MESSAGE_SIZE_MAX_USIZE},
    message_pool::{Message, MessagePool},
    vsr::{Command, Header, HeaderPrepareRaw},
    vsr::command::CommandMarker,
    vsr::header::ProtoHeader,
};

const COMMAND_OFFSET: usize = mem::offset_of!(Header, command);
const _: () = assert!(COMMAND_OFFSET == 114);
const _: () = assert!(mem::size_of::<Command>() == mem::size_of::<u8>());

/// Reason the buffer was invalidated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidReason {
    HeaderChecksum,
    HeaderSize,
    HeaderCluster,
    BodyChecksum,
    Misdirected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IteratorState {
    Idle,
    AfterPeek,
    AfterConsumeSuspend,
}

/// Marker type for "raw" messages in the pool.
pub struct RawCommand;

/// Message type used by the buffer for untyped, raw wire frames.
pub type RawMessage = Message<RawCommand>;

/// Treat HeaderPrepareRaw as a generic wire header for pool initialization.
/// Parsing for validation uses Header::from_bytes instead.
unsafe impl ProtoHeader for HeaderPrepareRaw {
    #[inline]
    fn command(&self) -> Command {
        Command::try_from_u8(self.command).unwrap_or(Command::Reserved)
    }

    #[inline]
    fn set_command(&mut self, command: Command) {
        self.command = command.as_u8();
    }

    #[inline]
    fn size(&self) -> u32 {
        self.size
    }

    #[inline]
    fn set_size(&mut self, size: u32) {
        self.size = size;
    }

    #[inline]
    fn protocol(&self) -> u16 {
        self.protocol
    }

    #[inline]
    fn set_protocol(&mut self, protocol: u16) {
        self.protocol = protocol;
    }
}

impl CommandMarker for RawCommand {
    const COMMAND: Command = Command::Reserved;
    type Header = HeaderPrepareRaw;
}

/// Interface between a MessageBus and a Replica for passing batches of messages.
///
/// Invariant: suspend_size <= process_size <= advance_size <= receive_size
pub struct MessageBuffer {
    message: RawMessage,
    suspend_size: u32,
    process_size: u32,
    advance_size: u32,
    receive_size: u32,
    pub invalid: Option<InvalidReason>,
    iterator_state: IteratorState,
}

impl MessageBuffer {
    pub fn init(pool: &MessagePool) -> Self {
        Self {
            message: pool.get::<RawCommand>(),
            suspend_size: 0,
            process_size: 0,
            advance_size: 0,
            receive_size: 0,
            invalid: None,
            iterator_state: IteratorState::Idle,
        }
    }

    pub fn deinit(self, _pool: &MessagePool) {
        drop(self);
    }

    /// Pass this to the kernel to read into.
    pub fn recv_slice(&mut self) -> &mut [u8] {
        assert!(self.receive_size < MESSAGE_SIZE_MAX);
        assert!(self.iterator_state == IteratorState::Idle);
        assert!(self.invalid.is_none());

        let start = self.receive_size as usize;
        &mut self.message_bytes_mut()[start..]
    }

    /// Inform the buffer about a successful receive.
    pub fn recv_advance(&mut self, size: u32) {
        assert!(self.iterator_state == IteratorState::Idle);
        assert!(self.process_size == 0);
        assert!(size > 0);
        assert!(size <= MESSAGE_SIZE_MAX);

        self.receive_size += size;
        assert!(self.receive_size <= MESSAGE_SIZE_MAX);
        self.advance();
    }

    pub fn invalidate(&mut self, reason: InvalidReason) {
        assert!(self.invalid.is_none());
        self.suspend_size = 0;
        self.process_size = 0;
        self.advance_size = 0;
        self.receive_size = 0;
        self.iterator_state = IteratorState::Idle;
        self.invalid = Some(reason);
        self.invariants();
    }

    /// Returns true if a fully validated message is available.
    pub fn has_message(&self) -> bool {
        let valid_unprocessed = self.advance_size - self.process_size;
        if valid_unprocessed >= HEADER_SIZE {
            let header = self.copy_header();
            if valid_unprocessed >= header.size {
                return true;
            }
        }
        false
    }

    /// Peek the next header. Must be followed by consume_message or suspend_message.
    pub fn next_header(&mut self) -> Option<Header> {
        match self.iterator_state {
            IteratorState::Idle | IteratorState::AfterConsumeSuspend => {}
            IteratorState::AfterPeek => unreachable!("next_header called without consume/suspend"),
        }

        let valid_unprocessed = self.advance_size - self.process_size;
        if valid_unprocessed >= HEADER_SIZE {
            let header = self.copy_header();
            if valid_unprocessed >= header.size {
                self.iterator_state = IteratorState::AfterPeek;
                return Some(header);
            }
        }

        assert!(self.suspend_size <= self.process_size);
        assert!(self.process_size <= self.receive_size);

        if self.suspend_size < self.process_size {
            let start = self.process_size as usize;
            let end = self.receive_size as usize;
            let dst = self.suspend_size as usize;
            self.message_bytes_mut().copy_within(start..end, dst);
        }

        let delta = self.process_size - self.suspend_size;
        self.receive_size -= delta;
        self.advance_size -= delta;
        self.suspend_size = 0;
        self.process_size = 0;
        self.iterator_state = IteratorState::Idle;

        let advance_size_idempotent = self.advance_size;
        self.advance();
        assert!(self.advance_size == advance_size_idempotent);

        None
    }

    pub fn consume_message(&mut self, pool: &MessagePool, header: &Header) -> RawMessage {
        assert!(self.iterator_state == IteratorState::AfterPeek);
        assert!(self.advance_size - self.process_size >= header.size);
        assert!(self.invalid.is_none());

        self.iterator_state = IteratorState::AfterConsumeSuspend;

        if self.process_size == 0 && self.receive_size == header.size {
            let header_in_buffer = self.copy_header();
            assert!(header_in_buffer.checksum == header.checksum);

            assert!(self.suspend_size == 0);
            self.process_size = 0;
            self.receive_size = 0;
            self.advance_size = 0;
            self.advance();
            assert!(self.advance_size == 0);

            return mem::replace(&mut self.message, pool.get::<RawCommand>());
        }

        let mut message = pool.get::<RawCommand>();
        {
            let src_start = self.process_size as usize;
            let src_end = src_start + header.size as usize;
            raw_message_bytes_mut(&mut message)[..header.size as usize]
                .copy_from_slice(&self.message_bytes()[src_start..src_end]);
        }

        self.process_size += header.size;
        assert!(self.process_size <= self.receive_size);
        self.advance();

        let header_bytes: &[u8; HEADER_SIZE_USIZE] =
            raw_message_bytes(&message)[..HEADER_SIZE_USIZE]
                .try_into()
                .expect("header slice length mismatch");
        let copied_header = Header::from_bytes(header_bytes);
        assert!(copied_header.checksum == header.checksum);

        message
    }

    pub fn suspend_message(&mut self, header: &Header) {
        assert!(self.iterator_state == IteratorState::AfterPeek);
        assert!(self.advance_size - self.process_size >= header.size);
        assert!(self.invalid.is_none());
        assert!(header.size <= MESSAGE_SIZE_MAX);

        let header_start = self.process_size as usize;
        let header_end = header_start + HEADER_SIZE_USIZE;
        let header_bytes = &self.message_bytes()[header_start..header_end];
        assert!(header_bytes == header.as_bytes());
        assert!(self.suspend_size <= self.process_size);

        self.iterator_state = IteratorState::AfterConsumeSuspend;

        if self.suspend_size < self.process_size {
            let src = self.process_size as usize;
            let dst = self.suspend_size as usize;
            let len = header.size as usize;
            self.message_bytes_mut().copy_within(src..src + len, dst);
        }

        self.suspend_size += header.size;
        self.process_size += header.size;
        self.advance();
    }

    fn advance(&mut self) {
        if self.invalid.is_none() {
            self.advance_header();
        }
        if self.invalid.is_none() {
            self.advance_body();
        }
        self.invariants();
    }

    fn advance_header(&mut self) {
        assert!(self.invalid.is_none());
        assert!(self.advance_size <= self.receive_size);
        if self.advance_size >= self.process_size + HEADER_SIZE {
            return;
        }
        assert!(self.advance_size == self.process_size);
        if self.receive_size - self.process_size < HEADER_SIZE {
            return;
        }

        let start = self.process_size as usize;
        let end = start + HEADER_SIZE_USIZE;
        let header_bytes: &[u8; HEADER_SIZE_USIZE] =
            self.message_bytes()[start..end]
                .try_into()
                .expect("header slice length mismatch");
        let header = Header::from_bytes(header_bytes);

        if !header.is_valid_checksum() {
            self.invalidate(InvalidReason::HeaderChecksum);
            return;
        }

        let command_raw = header_bytes[COMMAND_OFFSET];
        if Command::try_from_u8(command_raw).is_none() {
            panic!(
                "unknown VSR command (command={} protocol={} replica={} release={})",
                command_raw,
                header.protocol,
                header.replica,
                header.release.value()
            );
        }

        if header.size < HEADER_SIZE || header.size > MESSAGE_SIZE_MAX {
            self.invalidate(InvalidReason::HeaderSize);
            return;
        }

        self.advance_size += HEADER_SIZE;
    }

    fn advance_body(&mut self) {
        assert!(self.invalid.is_none());
        if self.advance_size < self.process_size + HEADER_SIZE {
            return;
        }

        let header = self.copy_header();

        if self.receive_size - self.process_size < header.size {
            return;
        }

        if self.advance_size >= self.process_size + header.size {
            return;
        }

        assert!(self.advance_size - self.process_size == HEADER_SIZE);

        let body_start = (self.process_size + HEADER_SIZE) as usize;
        let body_end = (self.process_size + header.size) as usize;
        let body = &self.message_bytes()[body_start..body_end];

        if !header.is_valid_checksum_body(body) {
            self.invalidate(InvalidReason::BodyChecksum);
            return;
        }

        self.advance_size += header.size - HEADER_SIZE;
    }

    fn copy_header(&self) -> Header {
        assert!(self.receive_size - self.process_size >= HEADER_SIZE);
        let start = self.process_size as usize;
        let end = start + HEADER_SIZE_USIZE;
        let header_bytes: &[u8; HEADER_SIZE_USIZE] =
            self.message_bytes()[start..end]
                .try_into()
                .expect("header slice length mismatch");
        Header::from_bytes(header_bytes)
    }

    #[inline]
    fn message_bytes(&self) -> &[u8] {
        raw_message_bytes(&self.message)
    }

    #[inline]
    fn message_bytes_mut(&mut self) -> &mut [u8] {
        raw_message_bytes_mut(&mut self.message)
    }

    fn invariants(&self) {
        assert!(self.suspend_size <= self.process_size);
        assert!(self.process_size <= self.advance_size);
        assert!(self.advance_size <= self.receive_size);
        if self.invalid.is_some() {
            assert!(self.suspend_size == 0);
            assert!(self.process_size == 0);
            assert!(self.advance_size == 0);
            assert!(self.receive_size == 0);
            assert!(self.iterator_state == IteratorState::Idle);
        }
    }
}

#[inline]
fn raw_message_bytes(message: &RawMessage) -> &[u8] {
    let ptr = message.buffer_ptr();
    unsafe { slice::from_raw_parts(ptr, MESSAGE_SIZE_MAX_USIZE) }
}

#[inline]
fn raw_message_bytes_mut(message: &mut RawMessage) -> &mut [u8] {
    assert!(message.is_unique(), "message must be unique for mutation");
    let ptr = message.buffer_ptr();
    unsafe { slice::from_raw_parts_mut(ptr, MESSAGE_SIZE_MAX_USIZE) }
}
