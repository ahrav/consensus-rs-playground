#![allow(dead_code)]

//! Message buffer for zero-copy I/O between MessageBus and Replica.
//!
//! `MessageBuffer` is the interface between a MessageBus and a Replica for passing batches of
//! messages while minimizing copies. It handles message framing but doesn't perform I/O directly.
//!
//! # Design
//!
//! MessageBuffer is a producer-consumer ring buffer of bytes with two key features:
//! - **Suspension**: The consumer can skip over or "suspend" certain slices to return to them later.
//! - **Sticky validation**: Messages are validated against checksums once, and this validation
//!   persists even if the message is skipped multiple times.
//!
//! The buffer uses a single fixed-size allocation ([`MESSAGE_SIZE_MAX`] bytes) from a message pool.
//! This enables a zero-copy fast path: if a `recv` syscall reads exactly one message, no copying
//! occurs.
//!
//! # Invariant
//!
//! The buffer maintains four cursors that satisfy the invariant:
//!
//! ```text
//! suspend_size ≤ process_size ≤ advance_size ≤ receive_size
//! ```
//!
//! - `suspend_size`: Bytes skipped (suspended) to be processed later
//! - `process_size`: Bytes consumed or suspended (always full messages)
//! - `advance_size`: Bytes validated against checksums (full messages + maybe a partial header)
//! - `receive_size`: Total bytes received from the kernel
//!
//! # Usage
//!
//! A typical usage pattern with kernel I/O:
//!
//! 1. Call [`recv_slice()`](MessageBuffer::recv_slice) to get a mutable buffer slice
//! 2. Pass this slice to a kernel `recv()` or `read()` syscall
//! 3. Call [`recv_advance()`](MessageBuffer::recv_advance) with the number of bytes read
//! 4. Iterate over messages, calling `peek()`, `consume()`, or `suspend()` as appropriate
//! 5. Reset the buffer when all messages are processed
//!
//! # Examples
//!
//! ```ignore
//! let mut buffer = MessageBuffer::init(&pool);
//!
//! // Prepare buffer for kernel read
//! let slice = buffer.recv_slice();
//!
//! // After kernel returns with bytes_read
//! buffer.recv_advance(bytes_read);
//!
//! // Process messages (iteration API not shown)
//! ```
//!
//! [`MESSAGE_SIZE_MAX`]: crate::vsr::MESSAGE_SIZE_MAX

use core::slice;
use std::mem;

use crate::{
    message_pool::{Message, MessagePool},
    vsr::{
        Command, HEADER_SIZE, HEADER_SIZE_USIZE, Header, HeaderPrepareRaw, MESSAGE_SIZE_MAX,
        MESSAGE_SIZE_MAX_USIZE, command::CommandMarker, header::ProtoHeader,
    },
};

const COMMAND_OFFSET: usize = mem::offset_of!(Header, command);
const _: () = assert!(COMMAND_OFFSET == 114);
const _: () = assert!(mem::size_of::<Command>() == mem::size_of::<u8>());

/// Reasons why a MessageBuffer may be marked invalid.
///
/// When a MessageBuffer encounters one of these errors, it sets the `invalid` field
/// and the MessageBus should terminate the connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidReason {
    /// The message header checksum failed verification.
    HeaderChecksum,

    /// The message header specifies an invalid size.
    ///
    /// This occurs when the size field is less than the header size, greater than
    /// [`MESSAGE_SIZE_MAX`], or not aligned to the required message boundary.
    HeaderSize,

    /// The message header specifies a cluster ID that doesn't match the local cluster.
    ///
    /// Messages from different clusters are rejected to prevent cross-cluster message corruption.
    HeaderCluster,

    /// The message body checksum failed verification.
    BodyChecksum,

    /// The message was sent to the wrong replica.
    ///
    /// This can occur when a message is addressed to a different replica ID than the local one,
    /// indicating a routing error in the network layer.
    Misdirected,
}

// Iterator state machine for message processing.
//
// The MessageBuffer uses this state to track the current position in the iteration protocol
// and ensure correct sequencing of peek/consume/suspend operations.
#[derive(Debug, Clone, PartialEq, Eq)]
enum IteratorState {
    // No iteration in progress. Ready for recv_slice/recv_advance or to start iteration.
    Idle,

    // After peek() has returned a message. Must call consume() or suspend() next.
    AfterPeek,

    // After consume() or suspend() has been called. Can call peek() again or finish iteration.
    AfterConsumeSuspend,
}

/// Marker type for raw, untyped message headers.
///
/// This is used with the message pool to allocate buffers that don't yet have a specific
/// command type. The MessageBuffer uses this to work with raw bytes before header validation.
pub struct RawCommand;

/// Type alias for a message with a raw, unvalidated header.
///
/// This allows MessageBuffer to work with untyped message memory before determining
/// the actual command type through header inspection.
pub type RawMessage = Message<RawCommand>;

// SAFETY: HeaderPrepareRaw is a valid protocol header that can represent any command type.
// It uses the raw `command` byte field which can hold all valid Command discriminants.
// This implementation allows MessageBuffer to inspect and validate headers before
// determining their specific command type.
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

// RawCommand acts as a marker for untyped message buffers.
// The Reserved command indicates that this message hasn't been validated yet.
impl CommandMarker for RawCommand {
    const COMMAND: Command = Command::Reserved;
    type Header = HeaderPrepareRaw;
}

/// A zero-copy buffer for batched message I/O between MessageBus and Replica.
///
/// MessageBuffer manages a single fixed-size allocation that serves as a producer-consumer
/// ring buffer. It tracks four cursor positions that advance as messages are received,
/// validated, and consumed.
///
/// # Fields
///
/// The buffer maintains the following state:
///
/// - `message`: Raw memory slab from the message pool. Using `Message<RawCommand>` enables
///   zero-copy fast path when exactly one message is received.
/// - `suspend_size`: Bytes of validated messages that were suspended (skipped) for later processing.
///   Always represents complete messages.
/// - `process_size`: Bytes that have been consumed or suspended. Always represents complete messages.
/// - `advance_size`: Bytes that have been validated against checksums. May include a partial
///   header at the boundary.
/// - `receive_size`: Total bytes received from the kernel via `recv()` or `read()` syscalls.
/// - `invalid`: If set, indicates a fatal error and the connection should be terminated.
/// - `iterator_state`: Internal state machine for tracking peek/consume/suspend sequencing.
///
/// # Invariant
///
/// The cursors must satisfy: `suspend_size ≤ process_size ≤ advance_size ≤ receive_size ≤ MESSAGE_SIZE_MAX`
///
/// # Examples
///
/// ```ignore
/// use message_buffer::MessageBuffer;
/// use message_pool::MessagePool;
///
/// let pool = MessagePool::init(...);
/// let mut buffer = MessageBuffer::init(&pool);
///
/// // Get slice for kernel to write into
/// let recv_buf = buffer.recv_slice();
///
/// // After kernel returns
/// buffer.recv_advance(bytes_read);
/// ```
pub struct MessageBuffer {
    // Raw memory slab from pool. Using Message<RawCommand> enables zero-copy optimization
    // when a recv() reads exactly one complete message.
    message: RawMessage,

    // Cursor positions satisfying: suspend_size ≤ process_size ≤ advance_size ≤ receive_size

    // Bytes of validated messages that were suspended for later processing (always full messages).
    suspend_size: u32,

    // Bytes consumed or suspended (always full messages).
    process_size: u32,

    // Bytes validated against checksums (full messages plus possibly a partial header).
    advance_size: u32,

    // Total bytes received from kernel.
    receive_size: u32,

    /// If set, indicates a fatal error. The MessageBus should terminate the connection.
    ///
    /// This field is public to allow the Replica to set semantic errors like wrong cluster ID.
    pub invalid: Option<InvalidReason>,

    // State machine for tracking iterator operations.
    iterator_state: IteratorState,
}

impl MessageBuffer {
    /// Creates a new MessageBuffer with memory allocated from the given pool.
    ///
    /// The buffer is initialized with all cursors at zero and ready to receive data.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let pool = MessagePool::init(...);
    /// let buffer = MessageBuffer::init(&pool);
    /// ```
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

    /// Returns a mutable slice to pass to the kernel for reading.
    ///
    /// This slice starts at `receive_size` and extends to the end of the buffer,
    /// providing space for new data from a `recv()` or `read()` syscall.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `receive_size >= MESSAGE_SIZE_MAX` (buffer is full)
    /// - `iterator_state != Idle` (cannot receive during iteration)
    /// - `invalid.is_some()` (buffer is in error state)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let slice = buffer.recv_slice();
    /// let bytes_read = socket.recv(slice)?;
    /// buffer.recv_advance(bytes_read);
    /// ```
    pub fn recv_slice(&mut self) -> &mut [u8] {
        assert!(self.receive_size < MESSAGE_SIZE_MAX);
        assert!(self.iterator_state == IteratorState::Idle);
        assert!(self.invalid.is_none());

        let start = self.receive_size as usize;
        &mut self.message_bytes_mut()[start..]
    }

    /// Updates the buffer after the kernel has written `size` bytes.
    ///
    /// Call this method after a `recv()` or `read()` syscall returns to advance
    /// the `receive_size` cursor by the number of bytes read.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `iterator_state != Idle` (cannot receive during iteration)
    /// - `process_size != 0` (messages must be fully processed before next receive)
    /// - `size == 0` (kernel should return positive size)
    /// - `size > MESSAGE_SIZE_MAX` (invalid size from kernel)
    /// - `receive_size + size > MESSAGE_SIZE_MAX` (would overflow buffer)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let slice = buffer.recv_slice();
    /// let bytes_read = socket.recv(slice)?;
    /// buffer.recv_advance(bytes_read as u32);
    /// ```
    pub fn recv_advance(&mut self, size: u32) {
        assert!(self.iterator_state == IteratorState::Idle);
        assert_eq!(self.process_size, 0);
        assert!(size > 0);
        assert!(size <= MESSAGE_SIZE_MAX);

        self.receive_size += size;
        assert!(self.receive_size <= MESSAGE_SIZE_MAX);
        self.advance()
    }

    /// Marks this buffer as invalid and resets all cursor positions to zero.
    ///
    /// The MessageBus should terminate the associated connection when it
    /// observes an invalid buffer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is already invalid.
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

    /// Returns `true` if a complete, validated message is available for processing.
    ///
    /// This checks whether enough validated bytes exist (from `process_size` to `advance_size`)
    /// to contain both a header and the full message body indicated by that header's size field.
    ///
    /// Note: This does not advance the iterator state. Use [`next_header`] to actually
    /// retrieve the message header for processing.
    pub fn has_message(&mut self) -> bool {
        let valid_unprocessed = self.advance_size - self.process_size;
        if valid_unprocessed >= HEADER_SIZE {
            let header = self.copy_header();
            if valid_unprocessed >= header.size {
                return true;
            }
        }
        false
    }

    /// Returns the header of the next complete message, or `None` when iteration ends.
    ///
    /// `MessageBuffer` is an iterator that must be driven to completion. When this method
    /// returns `Some(header)`, the caller **must** immediately call either [`consume_message`]
    /// or [`suspend_message`] before calling `next_header` again.
    ///
    /// # Iterator Protocol
    ///
    /// ```text
    /// loop {
    ///     match buffer.next_header() {
    ///         Some(header) => {
    ///             // MUST call one of these before next iteration:
    ///             buffer.consume_message(...) // or
    ///             buffer.suspend_message(...)
    ///         }
    ///         None => break, // Iteration complete, ready for next recv cycle
    ///     }
    /// }
    /// ```
    ///
    /// # Buffer Compaction
    ///
    /// When returning `None`, this method compacts the buffer by eliminating the hole
    /// between suspended and unprocessed bytes:
    ///
    /// ```text
    /// Before compaction:
    /// |  bytes  |     hole     |     bytes     |    hole    |
    ///           ^suspend_size  ^process_size   ^receive_size
    ///
    /// After compaction:
    /// |           bytes             |         hole          |
    /// ^ suspend_size,process_size   ^ receive_size
    /// ```
    ///
    /// This compaction preserves suspended messages at the front while making room
    /// for new data from the next `recv()` syscall.
    ///
    /// # Sticky Validation
    ///
    /// The method includes an idempotency check: after compaction, `advance()` is called
    /// to verify that checksum validation results are preserved. This ensures messages
    /// are validated exactly once, even across multiple iteration cycles.
    ///
    /// # Panics
    ///
    /// Panics if called while in `AfterPeek` state (i.e., after `next_header` returned
    /// `Some` but before `consume_message` or `suspend_message` was called).
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
        assert_eq!(self.advance_size, advance_size_idempotent);

        None
    }

    /// Consumes a message that was peeked via [`next_header`] and returns it as an owned [`RawMessage`].
    ///
    /// This method must be called after [`next_header`] returns `Some(header)`. It extracts the
    /// message from the buffer and advances `process_size` past the consumed bytes.
    ///
    /// # Zero-Copy Fast Path
    ///
    /// When the message is the only one in the buffer (i.e., `process_size == 0` and
    /// `receive_size == header.size`), the internal buffer is returned directly without copying.
    /// A fresh buffer is acquired from the pool to replace it. This optimization avoids memory
    /// copies when processing single messages.
    ///
    /// # Copy Path
    ///
    /// When multiple messages exist in the buffer, the message bytes are copied to a new buffer
    /// acquired from the pool. The original buffer is retained for continued iteration.
    ///
    /// # Arguments
    ///
    /// * `pool` - The message pool to acquire a new buffer from (for replacement or copying)
    /// * `header` - The header returned by the preceding [`next_header`] call
    ///
    /// # Returns
    ///
    /// An owned [`RawMessage`] containing the consumed message data. The caller is responsible
    /// for returning this message to the pool when done.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `iterator_state != AfterPeek` (must call [`next_header`] first)
    /// - `advance_size - process_size < header.size` (message not fully validated)
    /// - `invalid.is_some()` (buffer is in error state)
    /// - Header checksum mismatch (internal consistency check)
    pub fn consume_message(&mut self, pool: &MessagePool, header: &Header) -> RawMessage {
        assert!(self.iterator_state == IteratorState::AfterPeek);
        assert!(self.advance_size - self.process_size >= header.size);
        assert!(self.invalid.is_none());

        self.iterator_state = IteratorState::AfterConsumeSuspend;

        if self.process_size == 0 && self.receive_size == header.size {
            let header_in_buffer = self.copy_header();
            assert_eq!(header_in_buffer.checksum, header.checksum);

            assert_eq!(self.suspend_size, 0);
            self.process_size = 0;
            self.receive_size = 0;
            self.advance_size = 0;
            self.advance();
            assert_eq!(self.advance_size, 0);

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

        let header_bytes = raw_message_bytes(&message)[..HEADER_SIZE_USIZE]
            .try_into()
            .expect("header slice length mismatch");
        let copied_header = Header::from_bytes(header_bytes);
        assert_eq!(copied_header.checksum, header.checksum);

        message
    }

    /// Suspends a message to be processed later, keeping it in the buffer.
    ///
    /// This method must be called after [`next_header`] returns `Some(header)`. Unlike
    /// [`consume_message`], the message data stays in the buffer and will be available
    /// in subsequent iteration cycles.
    ///
    /// # Buffer Compaction
    ///
    /// When there's a hole between suspended bytes and the current message (i.e.,
    /// `suspend_size < process_size`), the message is moved to close the gap:
    ///
    /// ```text
    /// Before:
    /// |  bytes  |    hole     |    message    |     bytes    |
    ///           ^suspend_size ^process_size                  ^receive_size
    ///
    /// After:
    /// | bytes |    message    |     hole      |     bytes    |
    ///                         ^suspend_size   ^process_size  ^receive_size
    /// ```
    ///
    /// This compaction ensures suspended messages are contiguous at the front of the buffer.
    ///
    /// # Arguments
    ///
    /// * `header` - The header returned by the preceding [`next_header`] call
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `iterator_state != AfterPeek` (must call [`next_header`] first)
    /// - `advance_size - process_size < header.size` (message not fully validated)
    /// - `invalid.is_some()` (buffer is in error state)
    /// - `header.size > MESSAGE_SIZE_MAX` (invalid message size)
    /// - Header bytes in buffer don't match provided header (consistency check)
    pub fn suspend_message(&mut self, header: &Header) {
        assert!(self.iterator_state == IteratorState::AfterPeek);
        assert!(self.advance_size - self.process_size >= header.size);
        assert!(self.invalid.is_none());
        assert!(header.size <= MESSAGE_SIZE_MAX);

        let header_start = self.process_size as usize;
        let header_end = header_start + HEADER_SIZE_USIZE;
        let header_bytes = &self.message_bytes()[header_start..header_end];
        assert_eq!(header_bytes, header.as_bytes());

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

    /// Advances the parsing state machine by validating header and body.
    ///
    /// Calls [`advance_header`] then [`advance_body`], stopping early if
    /// the buffer becomes invalid. Idempotent.
    fn advance(&mut self) {
        if self.invalid.is_none() {
            self.advance_header();
        }

        if self.invalid.is_none() {
            self.advance_body();
        }
        self.invariants();
    }

    /// Validates the next message header if enough bytes are available.
    ///
    /// Idempotent: returns immediately if header already validated or if
    /// insufficient bytes are available. On success, advances `advance_size`
    /// by [`HEADER_SIZE`]. On validation failure, calls [`invalidate`].
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `invalid.is_some()` (buffer already invalid)
    /// - `advance_size > receive_size` (invariant violation)
    /// - Unknown VSR command discriminant (protocol mismatch)
    fn advance_header(&mut self) {
        assert!(self.invalid.is_none());
        assert!(self.advance_size <= self.receive_size);

        if self.advance_size >= self.process_size + HEADER_SIZE {
            return;
        }
        assert_eq!(self.advance_size, self.process_size);
        if self.receive_size - self.process_size < HEADER_SIZE {
            return;
        }

        let start = self.process_size as usize;
        let end = start + HEADER_SIZE as usize;
        let header_bytes = self.message_bytes()[start..end]
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
                "unknown VSR command (command={} protocol={} replica={} release={}",
                command_raw, header.protocol, header.replica, header.release.0,
            )
        }

        if header.size < HEADER_SIZE || header.size > MESSAGE_SIZE_MAX {
            self.invalidate(InvalidReason::HeaderSize);
            return;
        }

        self.advance_size += HEADER_SIZE;
    }

    /// Validates the message body checksum if enough bytes are available.
    ///
    /// Idempotent: returns immediately if body already validated, header not yet
    /// validated, or insufficient bytes. On validation failure, calls [`invalidate`].
    ///
    /// # Panics
    ///
    /// Panics if `invalid.is_some()` (buffer already invalid).
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

        assert_eq!(self.advance_size - self.process_size, HEADER_SIZE);

        let body_start = (self.process_size + HEADER_SIZE) as usize;
        let body_end = (self.process_size + header.size) as usize;
        let body = &self.message_bytes()[body_start..body_end];

        if !header.is_valid_checksum_body(body) {
            self.invalidate(InvalidReason::BodyChecksum);
            return;
        }

        self.advance_size += header.size - HEADER_SIZE;
    }

    /// Copies and returns the header at the current process position.
    ///
    /// # Panics
    ///
    /// Panics if fewer than [`HEADER_SIZE`] bytes are available.
    fn copy_header(&mut self) -> Header {
        assert!(self.receive_size - self.process_size >= HEADER_SIZE);
        let start = self.process_size as usize;
        let end = start + HEADER_SIZE as usize;
        let header_bytes = self.message_bytes()[start..end]
            .try_into()
            .expect("header slice length mismatch");
        Header::from_bytes(header_bytes)
    }

    /// Returns an immutable view of the message buffer bytes.
    #[inline]
    fn message_bytes(&self) -> &[u8] {
        raw_message_bytes(&self.message)
    }

    /// Returns a mutable view of the message buffer bytes.
    ///
    /// # Panics
    ///
    /// Panics if the message is not uniquely owned (internal invariant violation).
    #[inline]
    fn message_bytes_mut(&mut self) -> &mut [u8] {
        raw_message_bytes_mut(&mut self.message)
    }

    /// Asserts all internal invariants hold.
    ///
    /// Checks cursor ordering and invalid-state consistency.
    fn invariants(&self) {
        assert!(self.suspend_size <= self.process_size);
        assert!(self.process_size <= self.advance_size);
        assert!(self.advance_size <= self.receive_size);
        if self.invalid.is_some() {
            assert_eq!(self.suspend_size, 0);
            assert_eq!(self.process_size, 0);
            assert_eq!(self.advance_size, 0);
            assert_eq!(self.receive_size, 0);
            assert!(self.iterator_state == IteratorState::Idle);
        }
    }
}

/// Converts a RawMessage to an immutable byte slice.
///
/// This creates a view of the entire MESSAGE_SIZE_MAX buffer without copying.
#[inline]
fn raw_message_bytes(message: &RawMessage) -> &[u8] {
    unsafe { slice::from_raw_parts(message.buffer_ptr(), MESSAGE_SIZE_MAX_USIZE) }
}

/// Converts a RawMessage to a mutable byte slice.
///
/// # Safety
///
/// Requires that the message is uniquely owned (not shared). This is enforced
/// by the assertion, which will panic if violated.
///
/// # Panics
///
/// Panics if `message.is_unique()` returns false, indicating the message is shared.
#[inline]
fn raw_message_bytes_mut(message: &mut RawMessage) -> &mut [u8] {
    assert!(message.is_unique());
    unsafe { slice::from_raw_parts_mut(message.buffer_ptr(), MESSAGE_SIZE_MAX_USIZE) }
}
