//! Message framing for streaming byte input.
//!
//! Handles incremental decoding of wire messages from a byte stream. The [`MessageBuffer`]
//! accumulates incoming bytes and extracts complete, validated messages on demand.
//!
//! # Buffer Model
//!
//! ```text
//! [consumed bytes][available bytes]
//!                 ^cursor          ^len
//! ```
//!
//! The buffer tracks a `cursor` position separating consumed data from available data.
//! [`MessageBuffer::decode`] advances the cursor; [`MessageBuffer::compact`] reclaims consumed space.
//!
//! # Design Rationale: Double Copy
//!
//! This implementation uses a double-copy strategy to prioritize safety and determinism over raw throughput:
//!
//! 1.  **Kernel $\to$ Buffer:** Raw bytes are read into `MessageBuffer`.
//! 2.  **Buffer $\to$ Pool:** Validated messages are copied into a `Message` from the `MessagePool`.
//!
//! ## Reasoning
//!
//! *   **Validation:** Checksums are verified *before* acquiring a message handle or writing to the pool.
//! *   **Pool Integrity:** The `MessagePool` contains only valid messages. Reading directly into the pool (Zero-Copy) would leave "dirty" objects on validation failure, complicating cleanup.
//! *   **Resource Safety:** Invalid packets are rejected without allocating pool resources.

use super::constants::{HEADER_SIZE, HEADER_SIZE_USIZE, MESSAGE_SIZE_MAX, MESSAGE_SIZE_MAX_USIZE};
use super::{Header, MessageHandle, MessagePool};

// Compile-time design integrity
const _: () = assert!(HEADER_SIZE > 0);
const _: () = assert!(HEADER_SIZE < MESSAGE_SIZE_MAX);
const _: () = assert!(MESSAGE_SIZE_MAX_USIZE <= u32::MAX as usize);

const COMPACT_THRESHOLD_DIVISOR: u32 = 2;
const _: () = assert!(COMPACT_THRESHOLD_DIVISOR > 0);

/// Errors from [`MessageBuffer::decode`].
///
/// `NeedMoreBytes` is recoverable—feed more data and retry. Other variants indicate
/// malformed or corrupted messages that should be handled as protocol errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Insufficient data; call [`MessageBuffer::feed`] and retry.
    NeedMoreBytes,
    /// Header failed [`Header::validate_basic`]; contains the reason.
    ///
    /// This includes size bound violations (`size < HEADER_SIZE` or `size > MESSAGE_SIZE_MAX`),
    /// protocol version mismatch, and non-zero reserved fields.
    HeaderInvalid(&'static str),
    /// Header checksum mismatch—header bytes corrupted.
    BadHeaderChecksum,
    /// Body checksum mismatch—body bytes corrupted.
    BadBodyChecksum,
}

/// Accumulates bytes and decodes framed messages.
///
/// Use [`feed`](Self::feed) to append incoming bytes, then [`decode`](Self::decode) to
/// extract validated messages. The buffer auto-compacts after decodes to bound memory.
pub struct MessageBuffer {
    buf: Vec<u8>,
    cursor: u32,
}

impl Default for MessageBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageBuffer {
    const MAX_BUF_SIZE: u32 = MESSAGE_SIZE_MAX * 4;
    const _MIN: () = assert!(Self::MAX_BUF_SIZE >= MESSAGE_SIZE_MAX);
    // Compile-time overflow check: MESSAGE_SIZE_MAX * 4 must not wrap.
    // If it did, MAX_BUF_SIZE would be less than MESSAGE_SIZE_MAX.
    const _NO_OVERFLOW: () = assert!(Self::MAX_BUF_SIZE > MESSAGE_SIZE_MAX);

    pub fn new() -> Self {
        let buffer = Self {
            buf: Vec::new(),
            cursor: 0,
        };

        assert!(buffer.available() == 0);

        buffer
    }

    pub fn with_capacity(capacity: u32) -> Self {
        assert!(capacity <= Self::MAX_BUF_SIZE);

        let buffer = Self {
            buf: Vec::with_capacity(capacity as usize),
            cursor: 0,
        };

        assert!(buffer.buf.capacity() >= capacity as usize);
        assert!(buffer.available() == 0);

        buffer
    }

    /// Bytes available for decoding (buffered minus consumed).
    #[inline]
    pub fn available(&self) -> u32 {
        assert!(self.buf.len() <= Self::MAX_BUF_SIZE as usize);
        let buf_len = self.buf.len() as u32;

        assert!(self.cursor <= buf_len);

        buf_len - self.cursor
    }

    #[inline]
    pub fn cursor(&self) -> u32 {
        self.cursor
    }

    /// Total bytes in buffer (consumed + available).
    #[inline]
    pub fn buffered(&self) -> u32 {
        assert!(self.buf.len() <= Self::MAX_BUF_SIZE as usize);
        self.buf.len() as u32
    }

    #[inline]
    pub fn clear(&mut self) {
        self.buf.clear();
        self.cursor = 0;

        assert!(self.available() == 0);
        assert!(self.buf.is_empty());
    }

    /// Appends bytes to the buffer.
    ///
    /// # Panics
    ///
    /// Panics if total buffered bytes would exceed `MAX_BUF_SIZE` (4 MiB).
    pub fn feed(&mut self, bytes: &[u8]) {
        assert!(bytes.len() <= Self::MAX_BUF_SIZE as usize);
        let bytes_len = bytes.len() as u32;

        let old_buffered = self.buffered();
        let old_available = self.available();

        assert!(old_buffered + bytes_len <= Self::MAX_BUF_SIZE);

        self.buf.extend_from_slice(bytes);

        assert!(self.buffered() == old_buffered + bytes_len);
        assert!(self.available() == old_available + bytes_len);
        assert!(self.cursor == self.buffered() - self.available());
    }

    #[inline]
    pub fn peek(&self, len: u32) -> Option<&[u8]> {
        if len > self.available() {
            return None;
        }

        let start = self.cursor as usize;
        let end = start + len as usize;

        assert!(start <= self.buf.len());
        assert!(end <= self.buf.len());

        Some(&self.buf[start..end])
    }

    pub fn peek_header(&self) -> Option<Result<Header, DecodeError>> {
        if self.available() < HEADER_SIZE {
            return None;
        }

        let start = self.cursor() as usize;
        let end = start + HEADER_SIZE_USIZE;

        assert!(start <= end);
        assert!(end <= self.buf.len());

        let header_bytes: &[u8; HEADER_SIZE_USIZE] = self.buf[start..end]
            .try_into()
            .expect("Slice length mismatch");

        let header = Header::from_bytes(header_bytes);
        if let Err(e) = header.validate_basic() {
            return Some(Err(DecodeError::HeaderInvalid(e)));
        }

        if !header.is_valid_checksum() {
            return Some(Err(DecodeError::BadHeaderChecksum));
        }

        assert!(header.total_len() >= HEADER_SIZE);
        assert!(header.total_len() <= MESSAGE_SIZE_MAX);

        Some(Ok(header))
    }

    /// Decodes the next message if complete and valid.
    ///
    /// On success, copies the message into a buffer from `pool`, advances the cursor,
    /// and auto-compacts. On [`DecodeError::NeedMoreBytes`], buffer state is unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `pool` is exhausted (use [`MessagePool::try_acquire`] for fallible allocation).
    pub fn decode<const N: usize>(
        &mut self,
        pool: &mut MessagePool<N>,
    ) -> Result<MessageHandle, DecodeError> {
        let header = match self.peek_header() {
            None => return Err(DecodeError::NeedMoreBytes),
            Some(Err(e)) => return Err(e),
            Some(Ok(h)) => h,
        };

        let total_len = header.total_len();

        // Defense-in-depth: validate_basic() already enforces these bounds, but we
        // assert here to catch any regression in the validation logic.
        assert!(
            total_len >= HEADER_SIZE,
            "validate_basic must reject size < HEADER_SIZE"
        );
        assert!(
            total_len <= MESSAGE_SIZE_MAX,
            "validate_basic must reject size > MESSAGE_SIZE_MAX"
        );

        if total_len > self.available() {
            return Err(DecodeError::NeedMoreBytes);
        }

        let start = self.cursor() as usize;
        let end = start + total_len as usize;

        assert!(start <= end);
        assert!(end <= self.buf.len());

        let message_bytes: &[u8] = &self.buf[start..end];
        let body_bytes = &message_bytes[HEADER_SIZE_USIZE..];

        assert!(body_bytes.len() as u32 == header.body_len());

        if !header.is_valid_checksum_body(body_bytes) {
            return Err(DecodeError::BadBodyChecksum);
        }

        let mut handle = pool.acquire();

        unsafe {
            let msg = handle.as_mut();

            assert!(message_bytes.len() <= MESSAGE_SIZE_MAX_USIZE);

            msg.buffer.bytes[..message_bytes.len()].copy_from_slice(message_bytes);
            msg.len = total_len;

            assert!(msg.len() == total_len);
            assert!(msg.header().is_valid_checksum());
            assert!(msg.header().is_valid_checksum_body(body_bytes));
        }

        let old_cursor = self.cursor();
        self.cursor += total_len;

        assert!(self.cursor == old_cursor + total_len);

        self.maybe_compact();

        Ok(handle)
    }

    /// Compacts if consumed bytes exceed half the buffer, or clears if fully consumed.
    pub fn maybe_compact(&mut self) {
        let buf_len = self.buffered();

        if self.cursor == buf_len {
            self.clear();
            return;
        }

        if self.cursor > 0 && self.cursor > buf_len / COMPACT_THRESHOLD_DIVISOR {
            let old_available = self.available();

            self.buf.drain(0..self.cursor as usize);
            self.cursor = 0;

            assert!(self.available() == old_available);
            assert!(self.buffered() == old_available);
        }
    }

    /// Discards consumed bytes, shifting available data to the front.
    pub fn compact(&mut self) {
        if self.cursor == 0 {
            return;
        }

        let old_available = self.available();

        self.buf.drain(0..self.cursor as usize);
        self.cursor = 0;

        assert!(self.cursor == 0);
        assert!(self.available() == old_available);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsr::wire::Command;

    fn make_valid_message(command: Command, body: &[u8]) -> Vec<u8> {
        let mut header = Header::new(command, 1, 0);
        header.size = HEADER_SIZE + body.len() as u32;
        header.set_checksum_body(body);
        header.set_checksum();

        let mut bytes = Vec::with_capacity(header.size as usize);
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(body);

        // Verify construction
        assert!(bytes.len() == header.size as usize);

        bytes
    }

    #[test]
    fn new_buffer_is_empty() {
        let buf = MessageBuffer::new();

        assert!(buf.available() == 0);
        assert!(buf.cursor() == 0);
        assert!(buf.buffered() == 0);
    }

    #[test]
    fn peek_header_needs_enough_bytes() {
        let mut buf = MessageBuffer::new();

        // Not enough bytes
        buf.feed(&[0u8; HEADER_SIZE_USIZE - 1]);
        assert!(buf.peek_header().is_none());

        // Exactly enough (but will fail validation)
        buf.feed(&[0u8; 1]);
        assert!(buf.peek_header().is_some());
    }

    #[test]
    fn decode_needs_more_bytes_for_header() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        buf.feed(&[0u8; HEADER_SIZE_USIZE - 1]);

        let result = buf.decode(&mut pool);
        assert!(matches!(result, Err(DecodeError::NeedMoreBytes)));
    }

    #[test]
    fn decode_needs_more_bytes_for_body() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        // Create message with body
        let body = b"test body";
        let message_bytes = make_valid_message(Command::Ping, body);

        // Feed only header + partial body
        buf.feed(&message_bytes[..HEADER_SIZE_USIZE + 2]);

        let result = buf.decode(&mut pool);
        assert!(matches!(result, Err(DecodeError::NeedMoreBytes)));
    }

    #[test]
    fn decode_detects_bad_header_checksum() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        let mut message_bytes = make_valid_message(Command::Ping, b"");

        // Corrupt header (not the checksum field itself)
        message_bytes[20] ^= 0xFF;

        buf.feed(&message_bytes);

        let result = buf.decode(&mut pool);
        assert!(matches!(result, Err(DecodeError::BadHeaderChecksum)));
    }

    #[test]
    fn decode_detects_bad_body_checksum() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        let mut message_bytes = make_valid_message(Command::Ping, b"original body");

        // Corrupt body
        message_bytes[HEADER_SIZE_USIZE] ^= 0xFF;

        buf.feed(&message_bytes);

        let result = buf.decode(&mut pool);
        assert!(matches!(result, Err(DecodeError::BadBodyChecksum)));
    }

    #[test]
    fn peek_does_not_consume() {
        let mut buf = MessageBuffer::new();

        buf.feed(b"test data");

        let first = buf.peek(4);
        let second = buf.peek(4);

        assert!(first == second);
        assert!(first == Some(b"test".as_slice()));
        assert!(buf.available() == 9); // unchanged
    }

    #[test]
    fn maybe_compact_threshold_logic() {
        let mut buf = MessageBuffer::new();
        // COMPACT_THRESHOLD_DIVISOR is 2.

        let data = [1u8; 100];
        buf.feed(&data);
        assert!(buf.buffered() == 100);

        // Consume 40 bytes ( <= 100/2 = 50)
        buf.cursor = 40;

        buf.maybe_compact();
        // Should NOT compact
        assert!(buf.cursor == 40);
        assert!(buf.buffered() == 100);

        // Consume 11 more (total 51 > 50)
        buf.cursor = 51;
        buf.maybe_compact();
        // Should compact
        assert!(buf.cursor == 0);
        assert!(buf.buffered() == 49); // 100 - 51
        assert!(buf.available() == 49);
    }

    #[test]
    fn decode_header_invalid_protocol() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        let mut msg = make_valid_message(Command::Ping, b"");
        // Corrupt protocol version (offset 112)
        // Header::protocol is u16 at bytes 112-113.
        msg[112] = !msg[112]; // Flip bits to ensure it's different

        buf.feed(&msg);
        let result = buf.decode(&mut pool);
        match result {
            Err(DecodeError::HeaderInvalid(s)) => {
                assert!(s.contains("protocol mismatch"));
            }
            Err(e) => panic!("Expected HeaderInvalid, got {:?}", e),
            Ok(_) => panic!("Expected HeaderInvalid, got Ok"),
        }
    }

    #[test]
    fn decode_header_size_too_small_via_validate() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        let mut msg = make_valid_message(Command::Ping, b"");
        // Corrupt size to be < HEADER_SIZE. Size is u32 at 96.
        msg[96] = 0;
        msg[97] = 0;
        msg[98] = 0;
        msg[99] = 0;

        buf.feed(&msg);
        let result = buf.decode(&mut pool);
        match result {
            Err(DecodeError::HeaderInvalid(s)) => {
                assert!(s.contains("size < SIZE_MIN"));
            }
            Err(e) => panic!("Expected HeaderInvalid, got {:?}", e),
            Ok(_) => panic!("Expected HeaderInvalid, got Ok"),
        }
    }

    // =========================================================================
    // Constructor Tests
    // =========================================================================

    #[test]
    fn with_capacity_sizes() {
        let capacities = [0, 1, HEADER_SIZE, MESSAGE_SIZE_MAX, MESSAGE_SIZE_MAX * 2];

        for &cap in &capacities {
            let buf = MessageBuffer::with_capacity(cap);
            assert!(buf.buf.capacity() >= cap as usize);
            assert!(buf.available() == 0);
            assert!(buf.cursor() == 0);
            assert!(buf.buffered() == 0);
        }
    }

    #[test]
    fn with_capacity_at_max_boundary() {
        let buf = MessageBuffer::with_capacity(MessageBuffer::MAX_BUF_SIZE);
        assert!(buf.buf.capacity() >= MessageBuffer::MAX_BUF_SIZE as usize);
        assert!(buf.available() == 0);
    }

    #[test]
    #[should_panic]
    fn with_capacity_exceeds_max_panics() {
        let _ = MessageBuffer::with_capacity(MessageBuffer::MAX_BUF_SIZE + 1);
    }

    // =========================================================================
    // DecodeError Variant Tests
    // =========================================================================

    #[test]
    fn decode_header_size_too_large_via_validate() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        let mut msg = make_valid_message(Command::Ping, b"");
        // Set size to > MESSAGE_SIZE_MAX. Size is u32 at bytes 96-99 (little-endian).
        let invalid_size = MESSAGE_SIZE_MAX + 1;
        msg[96..100].copy_from_slice(&invalid_size.to_le_bytes());

        buf.feed(&msg);
        let result = buf.decode(&mut pool);
        match result {
            Err(DecodeError::HeaderInvalid(s)) => {
                assert!(s.contains("size > SIZE_MAX"));
            }
            Err(e) => panic!("Expected HeaderInvalid, got {:?}", e),
            Ok(_) => panic!("Expected HeaderInvalid, got Ok"),
        }
    }

    #[test]
    fn decode_header_reserved_frame_not_zero() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        let mut msg = make_valid_message(Command::Ping, b"");
        // reserved_frame is at bytes 116-127. Corrupt one byte.
        msg[116] = 0xFF;

        buf.feed(&msg);
        let result = buf.decode(&mut pool);
        match result {
            Err(DecodeError::HeaderInvalid(s)) => {
                assert!(s.contains("reserved_frame not zero"));
            }
            Err(e) => panic!("Expected HeaderInvalid, got {:?}", e),
            Ok(_) => panic!("Expected HeaderInvalid, got Ok"),
        }
    }

    #[test]
    fn decode_header_reserved_command_not_zero() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        let mut msg = make_valid_message(Command::Ping, b"");
        // reserved_command is at bytes 128-255. Corrupt one byte.
        msg[200] = 0xFF;

        buf.feed(&msg);
        let result = buf.decode(&mut pool);
        match result {
            Err(DecodeError::HeaderInvalid(s)) => {
                assert!(s.contains("reserved_command not zero"));
            }
            Err(e) => panic!("Expected HeaderInvalid, got {:?}", e),
            Ok(_) => panic!("Expected HeaderInvalid, got Ok"),
        }
    }

    // =========================================================================
    // Zero/Max Body Size Tests
    // =========================================================================

    #[test]
    fn decode_message_with_zero_body() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        let message_bytes = make_valid_message(Command::Ping, &[]);
        buf.feed(&message_bytes);

        let handle = buf.decode(&mut pool).expect("decode header-only message");

        unsafe {
            let msg = handle.as_ref();
            assert!(msg.header().body_len() == 0);
            assert!(msg.body().is_empty());
            assert!(msg.header().command == Command::Ping);
        }

        pool.release(handle);
    }

    #[test]
    fn decode_message_with_max_body_size() {
        use crate::vsr::wire::constants::MESSAGE_BODY_SIZE_MAX;

        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        let max_body = vec![0xAA; MESSAGE_BODY_SIZE_MAX as usize];
        let message_bytes = make_valid_message(Command::Request, &max_body);

        buf.feed(&message_bytes);

        let handle = buf.decode(&mut pool).expect("decode max-size message");

        unsafe {
            let msg = handle.as_ref();
            assert!(msg.header().total_len() == MESSAGE_SIZE_MAX);
            assert!(msg.body().len() == MESSAGE_BODY_SIZE_MAX as usize);
            // Verify body content
            assert!(msg.body().iter().all(|&b| b == 0xAA));
        }

        pool.release(handle);
    }

    // =========================================================================
    // feed() Boundary Tests
    // =========================================================================

    #[test]
    fn feed_up_to_max_buf_size() {
        let mut buf = MessageBuffer::new();

        // Feed in chunks to avoid huge single allocation
        let chunk_size = MESSAGE_SIZE_MAX as usize;
        for _ in 0..4 {
            let data = vec![0xBB; chunk_size];
            buf.feed(&data);
        }

        assert!(buf.buffered() == MessageBuffer::MAX_BUF_SIZE);
        assert!(buf.available() == MessageBuffer::MAX_BUF_SIZE);
    }

    #[test]
    #[should_panic]
    fn feed_exceeding_max_buf_size_panics() {
        let mut buf = MessageBuffer::new();

        // Fill to max
        let data = vec![0xCC; MessageBuffer::MAX_BUF_SIZE as usize];
        buf.feed(&data);

        // This should panic
        buf.feed(&[0x01]);
    }

    // =========================================================================
    // peek() Edge Case Tests
    // =========================================================================

    #[test]
    fn peek_zero_length() {
        let mut buf = MessageBuffer::new();
        buf.feed(b"data");

        let result = buf.peek(0);
        assert!(result == Some(&[][..]));
    }

    #[test]
    fn peek_exact_available() {
        let mut buf = MessageBuffer::new();
        buf.feed(b"exactly");

        let result = buf.peek(7);
        assert!(result == Some(b"exactly".as_slice()));
    }

    #[test]
    fn peek_more_than_available() {
        let mut buf = MessageBuffer::new();
        buf.feed(b"short");

        let result = buf.peek(100);
        assert!(result.is_none());
    }

    #[test]
    fn peek_at_non_zero_cursor() {
        let mut buf = MessageBuffer::new();
        buf.feed(b"hello world");
        buf.cursor = 6; // Simulate consumption of "hello "

        let result = buf.peek(5);
        assert!(result == Some(b"world".as_slice()));

        // Verify cursor unchanged
        assert!(buf.cursor() == 6);
    }

    #[test]
    fn peek_empty_buffer() {
        let buf = MessageBuffer::new();

        assert!(buf.peek(0) == Some(&[][..]));
        assert!(buf.peek(1).is_none());
    }

    // =========================================================================
    // State Transition Tests
    // =========================================================================

    #[test]
    fn feed_decode_feed_decode_cycle() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<16> = MessagePool::new();

        for i in 0..10 {
            let body = format!("message_{}", i);
            let msg = make_valid_message(Command::Ping, body.as_bytes());

            buf.feed(&msg);

            let handle = buf.decode(&mut pool).expect("decode in cycle");

            unsafe {
                assert!(handle.as_ref().body() == body.as_bytes());
            }

            pool.release(handle);
        }

        assert!(buf.available() == 0);
    }

    #[test]
    fn decode_error_leaves_buffer_unchanged() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        // Feed corrupt message (bad body checksum)
        let mut message = make_valid_message(Command::Ping, b"body content");
        message[HEADER_SIZE_USIZE] ^= 0xFF; // Corrupt body

        buf.feed(&message);

        let old_cursor = buf.cursor();
        let old_available = buf.available();
        let old_buffered = buf.buffered();

        let result = buf.decode(&mut pool);
        assert!(matches!(result, Err(DecodeError::BadBodyChecksum)));

        // Buffer state should be unchanged
        assert!(buf.cursor() == old_cursor);
        assert!(buf.available() == old_available);
        assert!(buf.buffered() == old_buffered);
    }

    // =========================================================================
    // compact() Tests
    // =========================================================================

    // =========================================================================
    // maybe_compact() Additional Tests
    // =========================================================================

    #[test]
    fn maybe_compact_clears_when_fully_consumed() {
        let mut buf = MessageBuffer::new();

        buf.feed(b"some data here");
        buf.cursor = buf.buffered(); // Simulate full consumption

        buf.maybe_compact();

        assert!(buf.cursor() == 0);
        assert!(buf.buffered() == 0);
        assert!(buf.available() == 0);
    }

    #[test]
    fn maybe_compact_does_nothing_at_zero_cursor() {
        let mut buf = MessageBuffer::new();

        buf.feed(&[0u8; 100]);

        let old_cursor = buf.cursor();
        let old_buffered = buf.buffered();

        buf.maybe_compact();

        // cursor is 0, so nothing should happen
        assert!(buf.cursor() == old_cursor);
        assert!(buf.buffered() == old_buffered);
    }

    // =========================================================================
    // buffered() vs available() Distinction Test
    // =========================================================================

    #[test]
    fn buffered_vs_available_distinction() {
        let mut buf = MessageBuffer::new();

        assert!(buf.buffered() == 0);
        assert!(buf.available() == 0);

        buf.feed(b"hello");
        assert!(buf.buffered() == 5);
        assert!(buf.available() == 5);

        buf.feed(b"world");
        assert!(buf.buffered() == 10);
        assert!(buf.available() == 10);

        buf.cursor = 3; // Consume 3 bytes
        assert!(buf.buffered() == 10); // Still 10 total buffered
        assert!(buf.available() == 7); // But only 7 available
    }

    #[test]
    fn decode_chunked_stream() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<8> = MessagePool::new();

        let messages: Vec<_> = (0..20)
            .map(|i| make_valid_message(Command::Request, format!("req_{}", i).as_bytes()))
            .collect();

        let mut decoded_count = 0;

        for msg_bytes in &messages {
            // Feed in random-ish chunks (simulate network)
            let chunk_size = (msg_bytes.len() / 3).max(1);

            for chunk in msg_bytes.chunks(chunk_size) {
                buf.feed(chunk);

                // Try to decode after each feed
                while let Ok(handle) = buf.decode(&mut pool) {
                    decoded_count += 1;
                    pool.release(handle);
                }
            }
        }

        assert_eq!(decoded_count, 20);
        assert!(buf.available() == 0);
    }

    #[test]
    fn decode_incremental_chunks() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        // Use a larger body so we have room to chunk
        let body = b"test body content here that is long enough for chunking";
        let message_bytes = make_valid_message(Command::Request, body);
        let total_len = message_bytes.len();

        // Feed message in small chunks
        // Chunk 1: first half of header
        buf.feed(&message_bytes[..HEADER_SIZE_USIZE / 2]);
        assert!(matches!(
            buf.decode(&mut pool),
            Err(DecodeError::NeedMoreBytes)
        ));

        // Chunk 2: second half of header
        buf.feed(&message_bytes[HEADER_SIZE_USIZE / 2..HEADER_SIZE_USIZE]);
        assert!(matches!(
            buf.decode(&mut pool),
            Err(DecodeError::NeedMoreBytes)
        ));

        // Chunk 3: first half of body
        let body_midpoint = HEADER_SIZE_USIZE + body.len() / 2;
        buf.feed(&message_bytes[HEADER_SIZE_USIZE..body_midpoint]);
        assert!(matches!(
            buf.decode(&mut pool),
            Err(DecodeError::NeedMoreBytes)
        ));

        // Chunk 4: rest of body
        buf.feed(&message_bytes[body_midpoint..total_len]);

        let handle = buf.decode(&mut pool).expect("decode after all chunks");
        unsafe {
            assert!(handle.as_ref().body() == body);
        }
        pool.release(handle);
    }

    // =========================================================================
    // Compaction Data Integrity Tests
    // =========================================================================

    /// Verifies that maybe_compact preserves trailing bytes when compaction
    /// is triggered by a large first message followed by partial second message data.
    #[test]
    fn maybe_compact_preserves_trailing_bytes_after_decode() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        // Create a large first message (larger body to ensure compaction triggers)
        let large_body = vec![0xAA; 512];
        let msg1 = make_valid_message(Command::Request, &large_body);

        // Create a second message
        let msg2_body = b"second message body";
        let msg2 = make_valid_message(Command::Pong, msg2_body);

        // Feed first message completely
        buf.feed(&msg1);

        // Feed only partial second message (header + partial body)
        let partial_len = HEADER_SIZE_USIZE + 5;
        buf.feed(&msg2[..partial_len]);

        // Decode first message - this should trigger maybe_compact since
        // cursor will exceed half the buffer after consuming msg1
        let handle = buf.decode(&mut pool).expect("first decode should succeed");
        unsafe {
            assert!(handle.as_ref().body() == large_body.as_slice());
        }
        pool.release(handle);

        // Verify the trailing bytes (partial second message) survived compaction
        assert!(buf.cursor() == 0, "cursor should be reset after compaction");
        assert!(
            buf.available() == partial_len as u32,
            "trailing bytes should be preserved"
        );

        // Verify the actual content of trailing bytes
        let trailing = buf
            .peek(partial_len as u32)
            .expect("should have trailing data");
        assert!(
            trailing == &msg2[..partial_len],
            "trailing byte content must match original partial message"
        );

        // Feed rest of second message and decode to prove data integrity
        buf.feed(&msg2[partial_len..]);
        let handle2 = buf.decode(&mut pool).expect("second decode should succeed");
        unsafe {
            assert!(handle2.as_ref().header().command == Command::Pong);
            assert!(handle2.as_ref().body() == msg2_body);
        }
        pool.release(handle2);
    }

    // =========================================================================
    // State Preservation Tests for All Error Paths
    // =========================================================================

    /// Verifies decode leaves state unchanged when returning NeedMoreBytes for partial header.
    #[test]
    fn decode_need_more_bytes_partial_header_preserves_state() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        // Feed partial header
        buf.feed(&[0u8; HEADER_SIZE_USIZE - 10]);

        let old_cursor = buf.cursor();
        let old_available = buf.available();
        let old_buffered = buf.buffered();

        let result = buf.decode(&mut pool);
        assert!(matches!(result, Err(DecodeError::NeedMoreBytes)));

        // State must be unchanged
        assert!(buf.cursor() == old_cursor);
        assert!(buf.available() == old_available);
        assert!(buf.buffered() == old_buffered);
    }

    /// Verifies decode leaves state unchanged when returning NeedMoreBytes for partial body.
    #[test]
    fn decode_need_more_bytes_partial_body_preserves_state() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        // Create valid message with body
        let body = b"some body content";
        let message = make_valid_message(Command::Ping, body);

        // Feed header + partial body only
        buf.feed(&message[..HEADER_SIZE_USIZE + 5]);

        let old_cursor = buf.cursor();
        let old_available = buf.available();
        let old_buffered = buf.buffered();

        let result = buf.decode(&mut pool);
        assert!(matches!(result, Err(DecodeError::NeedMoreBytes)));

        // State must be unchanged
        assert!(buf.cursor() == old_cursor);
        assert!(buf.available() == old_available);
        assert!(buf.buffered() == old_buffered);
    }

    /// Verifies decode leaves state unchanged on BadHeaderChecksum error.
    #[test]
    fn decode_bad_header_checksum_preserves_state() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        let mut message = make_valid_message(Command::Ping, b"body");
        // Corrupt header (not the checksum field, bytes 16+ are covered by checksum)
        message[20] ^= 0xFF;

        buf.feed(&message);

        let old_cursor = buf.cursor();
        let old_available = buf.available();
        let old_buffered = buf.buffered();

        let result = buf.decode(&mut pool);
        assert!(matches!(result, Err(DecodeError::BadHeaderChecksum)));

        // State must be unchanged
        assert!(buf.cursor() == old_cursor);
        assert!(buf.available() == old_available);
        assert!(buf.buffered() == old_buffered);
    }

    /// Verifies decode leaves state unchanged on HeaderInvalid error.
    #[test]
    fn decode_header_invalid_preserves_state() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<4> = MessagePool::new();

        let mut message = make_valid_message(Command::Ping, b"");
        // Corrupt protocol version (offset 112-113)
        message[112] ^= 0xFF;

        buf.feed(&message);

        let old_cursor = buf.cursor();
        let old_available = buf.available();
        let old_buffered = buf.buffered();

        let result = buf.decode(&mut pool);
        assert!(matches!(result, Err(DecodeError::HeaderInvalid(_))));

        // State must be unchanged
        assert!(buf.cursor() == old_cursor);
        assert!(buf.available() == old_available);
        assert!(buf.buffered() == old_buffered);
    }

    // =========================================================================
    // Pool Exhaustion Test
    // =========================================================================

    /// Verifies decode panics when the MessagePool is exhausted.
    #[test]
    #[should_panic(expected = "message pool exhausted")]
    fn decode_panics_on_exhausted_pool() {
        let mut buf = MessageBuffer::new();
        let mut pool: MessagePool<2> = MessagePool::new();

        let msg1 = make_valid_message(Command::Ping, b"first");
        let msg2 = make_valid_message(Command::Pong, b"second");
        let msg3 = make_valid_message(Command::Request, b"third");

        buf.feed(&msg1);
        buf.feed(&msg2);
        buf.feed(&msg3);

        // Acquire all pool slots
        let _h1 = buf.decode(&mut pool).expect("first decode");
        let _h2 = buf.decode(&mut pool).expect("second decode");

        // This should panic - pool exhausted
        let _h3 = buf.decode(&mut pool);
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        fn arb_command() -> impl Strategy<Value = Command> {
            (0u8..=Command::MAX).prop_map(|b| Command::try_from_u8(b).unwrap())
        }

        proptest! {
            /// Property: Feed/decode roundtrip preserves message content
            #[test]
            fn prop_feed_decode_roundtrip(
                commands in prop::collection::vec(arb_command(), 1..5),
                body_sizes in prop::collection::vec(0usize..256, 1..5),
            ) {
                let mut buf = MessageBuffer::new();
                let mut pool: MessagePool<16> = MessagePool::new();

                // Create messages with corresponding commands and body sizes
                let count = commands.len().min(body_sizes.len());
                let mut expected: Vec<(Command, Vec<u8>)> = Vec::new();

                for i in 0..count {
                    let cmd = commands[i];
                    let body: Vec<u8> = (0..body_sizes[i]).map(|j| (j & 0xFF) as u8).collect();
                    let msg = make_valid_message(cmd, &body);
                    buf.feed(&msg);
                    expected.push((cmd, body));
                }

                // Decode and verify all messages
                for (exp_cmd, exp_body) in &expected {
                    let handle = buf.decode(&mut pool).expect("decode should succeed");
                    unsafe {
                        let msg = handle.as_ref();
                        prop_assert_eq!(msg.header().command, *exp_cmd);
                        prop_assert_eq!(msg.body(), exp_body.as_slice());
                    }
                    pool.release(handle);
                }

                // Buffer should be empty
                prop_assert_eq!(buf.available(), 0);
            }

            /// Property: Compaction preserves available data
            #[test]
            fn prop_compact_preserves_available_data(
                data in prop::collection::vec(any::<u8>(), 1..500),
                cursor_frac in 0.0f64..1.0,
            ) {
                let mut buf = MessageBuffer::new();
                buf.feed(&data);

                let cursor = ((data.len() as f64) * cursor_frac) as u32;
                buf.cursor = cursor.min(buf.buffered());

                let old_available = buf.available();
                let old_data: Vec<u8> = buf.peek(old_available)
                    .map(|s| s.to_vec())
                    .unwrap_or_default();

                buf.compact();

                prop_assert_eq!(buf.cursor(), 0);
                prop_assert_eq!(buf.available(), old_available);

                if old_available > 0 {
                    let new_data = buf.peek(old_available).unwrap();
                    prop_assert_eq!(new_data, old_data.as_slice());
                }
            }

            /// Property: Peek never modifies buffer state
            #[test]
            fn prop_peek_never_modifies_state(
                data in prop::collection::vec(any::<u8>(), 0..500),
                peek_len in 0u32..600,
            ) {
                let mut buf = MessageBuffer::new();
                if !data.is_empty() {
                    buf.feed(&data);
                }

                let old_cursor = buf.cursor();
                let old_available = buf.available();
                let old_buffered = buf.buffered();

                let _ = buf.peek(peek_len);

                prop_assert_eq!(buf.cursor(), old_cursor);
                prop_assert_eq!(buf.available(), old_available);
                prop_assert_eq!(buf.buffered(), old_buffered);
            }

            /// Property: Feed increases available by exactly the input length
            #[test]
            fn prop_feed_increases_available_exactly(
                initial_data in prop::collection::vec(any::<u8>(), 0..1000),
                additional_data in prop::collection::vec(any::<u8>(), 0..1000),
            ) {
                // Ensure we don't exceed MAX_BUF_SIZE
                if initial_data.len() + additional_data.len() > MessageBuffer::MAX_BUF_SIZE as usize {
                    return Ok(());
                }

                let mut buf = MessageBuffer::new();

                if !initial_data.is_empty() {
                    buf.feed(&initial_data);
                }

                let old_available = buf.available();
                let old_buffered = buf.buffered();

                if !additional_data.is_empty() {
                    buf.feed(&additional_data);
                }

                prop_assert_eq!(buf.available(), old_available + additional_data.len() as u32);
                prop_assert_eq!(buf.buffered(), old_buffered + additional_data.len() as u32);
            }

            /// Property: Clear always results in empty buffer
            #[test]
            fn prop_clear_empties_buffer(
                data in prop::collection::vec(any::<u8>(), 0..1000),
                cursor_frac in 0.0f64..1.0,
            ) {
                let mut buf = MessageBuffer::new();

                if !data.is_empty() {
                    buf.feed(&data);
                    let cursor = ((data.len() as f64) * cursor_frac) as u32;
                    buf.cursor = cursor.min(buf.buffered());
                }

                buf.clear();

                prop_assert_eq!(buf.available(), 0);
                prop_assert_eq!(buf.buffered(), 0);
                prop_assert_eq!(buf.cursor(), 0);
            }

            /// Property: maybe_compact preserves available data
            #[test]
            fn prop_maybe_compact_preserves_available(
                data in prop::collection::vec(any::<u8>(), 1..500),
                cursor_frac in 0.0f64..1.0,
            ) {
                let mut buf = MessageBuffer::new();
                buf.feed(&data);

                let cursor = ((data.len() as f64) * cursor_frac) as u32;
                buf.cursor = cursor.min(buf.buffered());

                let old_available = buf.available();

                buf.maybe_compact();

                // After maybe_compact, available should be the same or buffer cleared
                // (if cursor == buffered, clear() is called)
                if cursor == buf.buffered() {
                    prop_assert_eq!(buf.available(), 0);
                } else {
                    prop_assert_eq!(buf.available(), old_available);
                }
            }
        }
    }
}
