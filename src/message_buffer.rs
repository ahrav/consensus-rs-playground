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
        Command, Header, HeaderPrepareRaw, MESSAGE_SIZE_MAX, MESSAGE_SIZE_MAX_USIZE,
        command::CommandMarker, header::ProtoHeader,
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
        // self.advance()
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
