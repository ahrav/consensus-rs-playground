//! Specialized header layout for `Prepare` messages in the VSR protocol.
//!
//! While the generic `Header` defines the wire format structure, `HeaderPrepare`
//! strictly maps the context-specific bytes (offsets 128-255) to the fields
//! used by log entries: operation numbers, client IDs, timestamps, and hash chains.

use core::{mem, slice};

use crate::{
    constants::{JOURNAL_SLOT_COUNT, VSR_VERSION},
    util::{Pod, Zeroable},
    vsr::{Checksum128, Command, Operation, Release, checksum, constants, header::ProtoHeader},
};

#[cfg(not(target_endian = "little"))]
compile_error!("Wire/layout code assumes a little-endian target");

const _: () = {
    assert!(mem::size_of::<HeaderPrepare>() == 256);
    assert!(mem::align_of::<HeaderPrepare>() == 16);

    assert!(mem::size_of::<Release>() == 4);
    assert!(mem::size_of::<Command>() == 1);
    assert!(mem::size_of::<Operation>() == 1);
    assert!(mem::size_of::<u128>() == 16);

    assert!(mem::offset_of!(HeaderPrepare, checksum) == 0);

    assert!(HeaderPrepare::SIZE == 256);
    assert!(HeaderPrepare::SIZE == mem::size_of::<HeaderPrepare>());
};

unsafe impl ProtoHeader for HeaderPrepare {
    fn command(&self) -> Command {
        self.command
    }

    fn set_command(&mut self, command: Command) {
        self.command = command
    }

    fn size(&self) -> u32 {
        self.size
    }

    fn set_size(&mut self, size: u32) {
        self.size = size
    }
}

/// Specialized header for `Command::Prepare` messages.
///
/// This struct guarantees the standard 256-byte header layout but interprets
/// the payload area as specific VSR log fields (op, commit, client, etc.).
///
/// # Layout
/// - **0-63**:   Checksums (Header + Body).
/// - **64-95**:  Routing (Cluster ID).
/// - **96-127**: Frame (Size, Epoch, View, Protocol).
/// - **128-255**: Prepare-specific log data.
#[repr(C, align(16))]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct HeaderPrepare {
    // --- Checksum region (offset 0-63) ---
    /// 128-bit checksum of header bytes 16-255.
    pub checksum: Checksum128,
    /// Reserved; must be zero.
    pub checksum_padding: Checksum128,
    /// 128-bit checksum of the message body.
    pub checksum_body: Checksum128,
    /// Reserved; must be zero.
    pub checksum_body_padding: Checksum128,

    // --- Nonce/cluster region (offset 64-95) ---
    /// Reserved for nonce; must be zero.
    pub nonce_reserved: Checksum128,
    /// Cluster identifier for routing.
    pub cluster: constants::ClusterId,

    // --- Frame header (offset 96-127) ---
    /// Total message size in bytes (header + body).
    pub size: u32,
    /// Epoch number for reconfiguration.
    pub epoch: u32,
    /// View number for leader election.
    pub view: u32,
    /// Software release version.
    pub release: Release,
    /// Protocol version; must equal [`VSR_VERSION`].
    pub protocol: u16,
    /// Message type discriminator.
    pub command: Command,
    /// Sender's replica index.
    pub replica: u8,
    /// Reserved for frame-level extensions; must be zero.
    pub reserved_frame: [u8; 12],

    // --- Prepare-specific fields (offset 128-255) ---
    /// Checksum of the parent `Prepare` message (the previous operation in the log).
    /// Used to enforce the hash chain and log integrity; must be non-zero for
    /// non-`ROOT`/non-`RESERVED` prepares.
    pub parent: u128,
    /// Reserved; must be zero.
    pub parent_padding: u128,
    /// Checksum of the client's request (if applicable).
    pub request_checksum: Checksum128,
    /// Reserved; must be zero.
    pub request_checksum_padding: Checksum128,
    /// Identifier for the checkpoint this operation belongs to (if any).
    pub checkpoint_id: u128,
    /// The ID of the client that issued the request.
    /// Zero for system-initiated operations (e.g., `Pulse`, `Upgrade`, `Root`).
    pub client: u128,
    /// The unique operation number (sequence number) assigned to this request.
    /// Must be strictly greater than `commit` for normal operations.
    pub op: u64,
    /// The highest operation number known to be committed (canonical) by the leader.
    pub commit: u64,
    /// Cluster-synchronized timestamp assigned by the leader.
    pub timestamp: u64,
    /// The client's request sequence number.
    /// Used for client session tracking and exactly-once semantics.
    pub request: u32,
    /// The semantic operation type (e.g., `Transfer`, `CreateAccount`, `Pulse`).
    pub operation: Operation,
    /// Reserved; must be zero.
    pub reserved: [u8; 3],
}

// SAFETY: HeaderPrepare is repr(C, align(16)) with fixed 256-byte layout.
// All fields are primitives (u8, u16, u32, u64, u128) or arrays thereof,
// which are Pod. The struct has no padding bytes due to careful field ordering
// and explicit padding fields (checksum_padding, etc.).
unsafe impl Pod for HeaderPrepare {}

// SAFETY: HeaderPrepare is #[repr(C)] with only primitive types and arrays thereof.
// All-zeros is a valid bit pattern for every field, as validated by `HeaderPrepare::zeroed()`.
unsafe impl Zeroable for HeaderPrepare {}

impl HeaderPrepare {
    pub const SIZE: usize = 256;

    pub const SIZE_MIN: u32 = Self::SIZE as u32;

    /// Returns a strictly zero-initialized header.
    ///
    /// # Safety
    /// Validates that `mem::zeroed` produces a valid initial state (e.g. 0 checksums).
    #[inline]
    pub fn zeroed() -> Self {
        let header = unsafe { mem::zeroed::<Self>() };
        assert_eq!(header.checksum, 0);
        assert_eq!(header.size, 0);
        assert_eq!(header.protocol, 0);

        header
    }

    /// Creates a new header with default values (Size, Protocol).
    #[inline]
    pub fn new() -> Self {
        let mut h = Self::zeroed();
        h.size = Self::SIZE as u32;
        h.protocol = VSR_VERSION;
        h
    }

    /// Returns the header as a byte slice for serialization or checksumming.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self as *const Self as *const u8, Self::SIZE) }
    }

    /// Calculates the header checksum (bytes 16-255).
    #[inline]
    pub fn calculate_checksum(&self) -> Checksum128 {
        let bytes = self.as_bytes();
        let checksum_field_size = mem::size_of::<Checksum128>();

        assert!(bytes.len() > checksum_field_size);
        checksum(&bytes[checksum_field_size..])
    }

    /// Updates `self.checksum` with the calculated value.
    ///
    /// Call this after [`Self::set_checksum_body()`] since the header checksum
    /// covers the `checksum_body` field.
    #[inline]
    pub fn set_checksum(&mut self) {
        let checksum = self.calculate_checksum();
        self.checksum = checksum;
        assert!(self.valid_checksum())
    }

    /// Returns true if `self.checksum` matches the content.
    #[inline]
    pub fn valid_checksum(&self) -> bool {
        self.checksum == self.calculate_checksum()
    }

    /// Calculates the checksum of the provided body.
    ///
    /// Panics if `self.size` does not match `Header::SIZE + body.len()`.
    #[inline]
    pub fn calculate_checksum_body(&self, body: &[u8]) -> Checksum128 {
        let expected_size = Self::SIZE + body.len();
        assert_eq!(self.size as usize, expected_size);
        checksum(body)
    }

    /// Updates `self.checksum_body` with the checksum of `body`.
    #[inline]
    pub fn set_checksum_body(&mut self, body: &[u8]) {
        let checksum = self.calculate_checksum_body(body);
        self.checksum_body = checksum;
        assert!(self.valid_checksum_body(body))
    }

    /// Returns true if `self.checksum_body` matches the provided `body`.
    #[inline]
    pub fn valid_checksum_body(&self, body: &[u8]) -> bool {
        self.calculate_checksum_body(body) == self.checksum_body
    }

    /// Checks protocol invariants and structural validity.
    ///
    /// Validates:
    /// - Protocol version and message size.
    /// - Zero padding (reserved fields).
    /// - Operation-specific logic (e.g. `Pulse` vs `Register` vs `Root`).
    ///
    /// NOTE: Does NOT validate `checksum`. `Root` and `Reserved` entries expect
    /// `checksum_body` to match an empty body due to their fixed size.
    pub fn invalid(&self) -> Option<&'static str> {
        // Protocol version check (likely to catch version mismatches early)
        if self.protocol != VSR_VERSION {
            return Some("protocol != Version");
        }

        // Size bounds
        if self.size < Self::SIZE as u32 {
            return Some("size < @sizeOf(Header)");
        }
        if self.size > constants::MESSAGE_SIZE_MAX {
            return Some("size > message_size_max");
        }

        // Padding fields must be zero
        if self.checksum_padding != 0 {
            return Some("checksum_padding != 0");
        }
        if self.checksum_body_padding != 0 {
            return Some("checksum_body_padding != 0");
        }
        if self.nonce_reserved != 0 {
            return Some("nonce_reserved != 0");
        }

        // Epoch must be zero (for now)
        if self.epoch != 0 {
            return Some("epoch != 0");
        }

        // Reserved frame bytes must be zero
        if !self.reserved_frame.iter().all(|&b| b == 0) {
            return Some("reserved_frame != 0");
        }

        if self.command != Command::Prepare {
            return Some("command != Prepare");
        }

        // Delegate to operation-specific validation
        self.invalid_header()
    }

    /// Validates operation-specific invariants (Prepare command logic).
    pub fn invalid_header(&self) -> Option<&'static str> {
        // Precondition: must be a Prepare command
        assert!(
            self.command == Command::Prepare,
            "invalid_header called on non-Prepare command"
        );

        // Padding fields
        if self.parent_padding != 0 {
            return Some("parent_padding != 0");
        }
        if self.request_checksum_padding != 0 {
            return Some("request_checksum_padding != 0");
        }

        // Operation-specific validation
        let result = if self.operation == Operation::RESERVED {
            self.validate_reserved_operation()
        } else if self.operation == Operation::ROOT {
            self.validate_root_operation()
        } else {
            self.validate_normal_operation()
        };

        if result.is_some() {
            return result;
        }

        // Final check: reserved bytes
        if !self.reserved.iter().all(|&b| b == 0) {
            return Some("reserved != 0");
        }

        None
    }

    fn validate_reserved_operation(&self) -> Option<&'static str> {
        let checksum_body_empty = checksum(&[]);

        if self.size != Self::SIZE as u32 {
            return Some("reserved: size != @sizeOf(Header)");
        }
        if self.checksum_body != checksum_body_empty {
            return Some("reserved: checksum_body != expected");
        }
        if self.view != 0 {
            return Some("reserved: view != 0");
        }
        if self.release.0 != 0 {
            return Some("release != 0");
        }
        if self.replica != 0 {
            return Some("reserved: replica != 0");
        }
        if self.parent != 0 {
            return Some("reserved: parent != 0");
        }
        if self.client != 0 {
            return Some("reserved: client != 0");
        }
        if self.request_checksum != 0 {
            return Some("reserved: request_checksum != 0");
        }
        if self.checkpoint_id != 0 {
            return Some("reserved: checkpoint_id != 0");
        }
        // op may be 0 or non-zero (slot index)
        if self.commit != 0 {
            return Some("reserved: commit != 0");
        }
        if self.request != 0 {
            return Some("reserved: request != 0");
        }
        if self.timestamp != 0 {
            return Some("reserved: timestamp != 0");
        }

        None
    }

    fn validate_root_operation(&self) -> Option<&'static str> {
        let checksum_body_empty = checksum(&[]);

        if self.size != Self::SIZE as u32 {
            return Some("root: size != @sizeOf(Header)");
        }
        if self.checksum_body != checksum_body_empty {
            return Some("root: checksum_body != expected");
        }
        if self.view != 0 {
            return Some("root: view != 0");
        }
        if self.release.0 != 0 {
            return Some("release != 0");
        }
        if self.replica != 0 {
            return Some("root: replica != 0");
        }
        if self.parent != 0 {
            return Some("root: parent != 0");
        }
        if self.client != 0 {
            return Some("root: client != 0");
        }
        if self.request_checksum != 0 {
            return Some("root: request_checksum != 0");
        }
        if self.checkpoint_id != 0 {
            return Some("root: checkpoint_id != 0");
        }
        if self.op != 0 {
            return Some("root: op != 0");
        }
        if self.commit != 0 {
            return Some("root: commit != 0");
        }
        if self.timestamp != 0 {
            return Some("root: timestamp != 0");
        }
        if self.request != 0 {
            return Some("root: request != 0");
        }

        None
    }

    fn validate_normal_operation(&self) -> Option<&'static str> {
        debug_assert!(self.operation != Operation::RESERVED && self.operation != Operation::ROOT);

        if self.operation.vsr_reserved()
            && self.operation != Operation::REGISTER
            && self.operation != Operation::RECONFIGURE
            && self.operation != Operation::PULSE
            && self.operation != Operation::UPGRADE
            && self.operation != Operation::NOOP
        {
            return Some("operation reserved");
        }

        // Release must be set for normal operations
        if self.release.value() == 0 {
            return Some("release == 0");
        }

        // Normal operations must link to the previous log entry.
        if self.parent == 0 {
            return Some("parent == 0");
        }

        // Client ID validation depends on operation type
        if self.operation == Operation::PULSE || self.operation == Operation::UPGRADE {
            if self.client != 0 {
                return Some("client != 0");
            }
        } else if self.client == 0 {
            return Some("client == 0");
        }

        // Op/commit/timestamp invariants
        if self.op == 0 {
            return Some("op == 0");
        }
        if self.op <= self.commit {
            return Some("op <= commit");
        }
        if self.timestamp == 0 {
            return Some("timestamp == 0");
        }

        // Request validation depends on operation type
        if self.operation == Operation::REGISTER
            || self.operation == Operation::PULSE
            || self.operation == Operation::UPGRADE
        {
            if self.request != 0 {
                return Some("request != 0");
            }
        } else if self.request == 0 {
            return Some("request == 0");
        }

        None
    }

    /// Creates a placeholder header to mark a journal slot as occupied but uncommitted.
    ///
    ///  Used during log initialization or slot pre-allocation where the actual
    /// operation hasn't been received yet. The header passes validation but carries
    /// no meaningful payload.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= JOURNAL_SLOT_COUNT`.
    pub fn reserve(cluster: u128, slot: u64) -> Self {
        assert!(slot < JOURNAL_SLOT_COUNT as u64);

        let mut h = Self::new();

        h.command = Command::Prepare;
        h.cluster = cluster;
        h.release = Release(0);
        h.op = slot;
        h.operation = Operation::RESERVED;
        h.view = 0;

        h.set_checksum_body(&[]);
        h.set_checksum();

        assert!(h.invalid().is_none());

        h
    }

    /// Creates the genesis entry (op=0) that anchors the hash chain.
    ///
    /// Every replica's log starts with this deterministic header. Subsequent
    /// entries link back via [`Self::parent`], ensuring log integrity from
    /// the first operation onward.
    pub fn root(cluster: u128) -> Self {
        let mut h = Self::new();

        h.cluster = cluster;
        h.size = Self::SIZE as u32;
        h.release = Release(0);
        h.command = Command::Prepare;
        h.operation = Operation::ROOT;
        h.op = 0;
        h.view = 0;

        h.request_checksum = 0;
        h.checkpoint_id = 0;
        h.parent = 0;
        h.client = 0;
        h.commit = 0;
        h.timestamp = 0;
        h.request = 0;

        h.set_checksum_body(&[]);
        h.set_checksum();

        assert!(h.invalid().is_none());

        h
    }
}

impl Default for HeaderPrepare {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for HeaderPrepare {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HeaderPrepare")
            .field("checksum", &format_args!("{:#034x}", self.checksum))
            .field(
                "checksum_body",
                &format_args!("{:#034x}", self.checksum_body),
            )
            .field("cluster", &format_args!("{:#034x}", self.cluster))
            .field("size", &self.size)
            .field("view", &self.view)
            .field("protocol", &self.protocol)
            .field("command", &self.command)
            .field("operation", &self.operation)
            .field("op", &self.op)
            .field("commit", &self.commit)
            .field("timestamp", &self.timestamp)
            .field("client", &format_args!("{:#034x}", self.client))
            .field("request", &self.request)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsr::Operation;

    #[test]
    fn test_zeroed() {
        let header = HeaderPrepare::zeroed();
        assert_eq!(header.checksum, 0);
        assert_eq!(header.size, 0);
        assert!(header.reserved_frame.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_new() {
        let header = HeaderPrepare::new();
        assert_eq!(header.size, HeaderPrepare::SIZE as u32);
        assert_eq!(header.protocol, VSR_VERSION);
    }

    #[test]
    fn test_default() {
        let from_new = HeaderPrepare::new();
        let from_default = HeaderPrepare::default();
        assert_eq!(from_new.as_bytes(), from_default.as_bytes());
    }

    #[test]
    fn test_as_bytes_layout() {
        let mut header = HeaderPrepare::new();
        header.cluster = 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0;
        let bytes = header.as_bytes();
        let cluster_bytes = &bytes[80..96];
        let cluster_value = u128::from_le_bytes(cluster_bytes.try_into().unwrap());
        assert_eq!(cluster_value, 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0);
    }

    #[test]
    fn test_constants() {
        assert_eq!(HeaderPrepare::SIZE, 256);
        assert_eq!(HeaderPrepare::SIZE_MIN, 256);
        assert_eq!(HeaderPrepare::SIZE, core::mem::size_of::<HeaderPrepare>());
    }

    #[test]
    fn test_checksum_tamper() {
        let mut header = HeaderPrepare::new();
        header.set_checksum();
        assert!(header.valid_checksum());
        header.view = 999;
        assert!(!header.valid_checksum());
        header.set_checksum();
        assert!(header.valid_checksum());
    }

    #[test]
    fn test_checksum_coverage() {
        let h1 = HeaderPrepare::new();
        let mut h2 = HeaderPrepare::new();
        h2.checksum_padding = 1;
        assert_ne!(h1.calculate_checksum(), h2.calculate_checksum());
    }

    #[test]
    fn test_checksum_body_empty() {
        let header = HeaderPrepare::new();
        let cs = header.calculate_checksum_body(&[]);
        assert_ne!(cs, 0);
    }

    #[test]
    fn test_checksum_body_data() {
        let mut header = HeaderPrepare::new();
        let body = [1u8, 2, 3];
        header.size = (HeaderPrepare::SIZE + body.len()) as u32;
        let cs1 = header.calculate_checksum_body(&body);
        let cs2 = header.calculate_checksum_body(&body);
        assert_eq!(cs1, cs2);
    }

    #[test]
    fn test_checksum_body_update() {
        let mut header = HeaderPrepare::new();
        let body = [42u8; 64];
        header.size = (HeaderPrepare::SIZE + body.len()) as u32;
        header.set_checksum_body(&body);
        assert!(header.valid_checksum_body(&body));
    }

    #[test]
    fn test_checksum_body_tamper() {
        let mut header = HeaderPrepare::new();
        let body = [1u8, 2, 3];
        header.size = (HeaderPrepare::SIZE + body.len()) as u32;
        header.set_checksum_body(&body);
        assert!(!header.valid_checksum_body(&[1, 2, 4]));
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_checksum_body_size_mismatch() {
        let mut header = HeaderPrepare::new();
        header.size = HeaderPrepare::SIZE as u32;
        let _ = header.calculate_checksum_body(&[1]);
    }

    #[test]
    fn test_checksum_order() {
        let mut header = HeaderPrepare::new();
        let body = [0xABu8; 32];
        header.size = (HeaderPrepare::SIZE + body.len()) as u32;
        header.set_checksum_body(&body);
        header.set_checksum();
        assert!(header.valid_checksum());
        assert!(header.valid_checksum_body(&body));
    }

    #[test]
    fn test_valid_register() {
        let mut header = HeaderPrepare::new();
        header.command = Command::Prepare;
        header.operation = Operation::REGISTER;
        header.client = 123;
        header.request = 0;
        header.op = 10;
        header.commit = 9;
        header.timestamp = 100;
        header.release = Release(1);
        header.parent = 1;
        header.set_checksum_body(&[]);
        header.set_checksum();
        assert_eq!(header.invalid(), None);
    }

    #[test]
    fn test_protocol_version() {
        let mut header = HeaderPrepare::new();
        header.protocol += 1;
        assert_eq!(header.invalid(), Some("protocol != Version"));
    }

    #[test]
    fn test_msg_size() {
        let mut header = HeaderPrepare::new();
        header.size = HeaderPrepare::SIZE as u32 - 1;
        assert_eq!(header.invalid(), Some("size < @sizeOf(Header)"));
        header.size = constants::MESSAGE_SIZE_MAX + 1;
        assert_eq!(header.invalid(), Some("size > message_size_max"));
    }

    #[test]
    fn test_command_prepare() {
        let mut header = HeaderPrepare::new();
        header.command = Command::Ping;
        assert_eq!(header.invalid(), Some("command != Prepare"));
    }

    #[test]
    fn test_padding() {
        let mut header = HeaderPrepare::new();
        header.command = Command::Prepare;

        header.checksum_padding = 1;
        assert_eq!(header.invalid(), Some("checksum_padding != 0"));
        header.checksum_padding = 0;

        header.checksum_body_padding = 1;
        assert_eq!(header.invalid(), Some("checksum_body_padding != 0"));
        header.checksum_body_padding = 0;

        header.nonce_reserved = 1;
        assert_eq!(header.invalid(), Some("nonce_reserved != 0"));
        header.nonce_reserved = 0;

        header.epoch = 1;
        assert_eq!(header.invalid(), Some("epoch != 0"));
        header.epoch = 0;

        header.reserved_frame[0] = 1;
        assert_eq!(header.invalid(), Some("reserved_frame != 0"));
        header.reserved_frame[0] = 0;

        header.parent_padding = 1;
        assert_eq!(header.invalid(), Some("parent_padding != 0"));
        header.parent_padding = 0;

        header.request_checksum_padding = 1;
        assert_eq!(header.invalid(), Some("request_checksum_padding != 0"));
    }

    fn valid_reserved_header() -> HeaderPrepare {
        let mut header = HeaderPrepare::new();
        header.command = Command::Prepare;
        header.operation = Operation::RESERVED;
        header.op = 0;
        header.commit = 0;
        header.timestamp = 0;
        header.request = 0;
        header.view = 0;
        header.release = Release(0);
        header.set_checksum_body(&[]);
        header
    }

    #[test]
    fn test_reserved_invariants() {
        let mut header = valid_reserved_header();
        header.view = 1;
        assert_eq!(header.invalid(), Some("reserved: view != 0"));

        let mut header = valid_reserved_header();
        header.release = Release(1);
        assert_eq!(header.invalid(), Some("release != 0"));

        let mut header = valid_reserved_header();
        header.commit = 1;
        assert_eq!(header.invalid(), Some("reserved: commit != 0"));

        let mut header = valid_reserved_header();
        header.size += 1;
        assert_eq!(header.invalid(), Some("reserved: size != @sizeOf(Header)"));

        let mut header = valid_reserved_header();
        header.checksum_body = 123;
        assert_eq!(
            header.invalid(),
            Some("reserved: checksum_body != expected")
        );

        let mut header = valid_reserved_header();
        header.replica = 1;
        assert_eq!(header.invalid(), Some("reserved: replica != 0"));

        let mut header = valid_reserved_header();
        header.parent = 1;
        assert_eq!(header.invalid(), Some("reserved: parent != 0"));

        let mut header = valid_reserved_header();
        header.client = 1;
        assert_eq!(header.invalid(), Some("reserved: client != 0"));

        let mut header = valid_reserved_header();
        header.request_checksum = 1;
        assert_eq!(header.invalid(), Some("reserved: request_checksum != 0"));

        let mut header = valid_reserved_header();
        header.checkpoint_id = 1;
        assert_eq!(header.invalid(), Some("reserved: checkpoint_id != 0"));

        let mut header = valid_reserved_header();
        header.request = 1;
        assert_eq!(header.invalid(), Some("reserved: request != 0"));

        let mut header = valid_reserved_header();
        header.timestamp = 1;
        assert_eq!(header.invalid(), Some("reserved: timestamp != 0"));
    }

    fn valid_root_header() -> HeaderPrepare {
        let mut header = HeaderPrepare::new();
        header.command = Command::Prepare;
        header.operation = Operation::ROOT;
        header.op = 0;
        header.commit = 0;
        header.timestamp = 0;
        header.request = 0;
        header.view = 0;
        header.release = Release(0);
        header.set_checksum_body(&[]);
        header
    }

    #[test]
    fn test_root_invariants() {
        let mut header = valid_root_header();
        header.op = 1;
        assert_eq!(header.invalid(), Some("root: op != 0"));

        let mut header = valid_root_header();
        header.commit = 1;
        assert_eq!(header.invalid(), Some("root: commit != 0"));

        let mut header = valid_root_header();
        header.size += 1;
        assert_eq!(header.invalid(), Some("root: size != @sizeOf(Header)"));

        let mut header = valid_root_header();
        header.checksum_body = 123;
        assert_eq!(header.invalid(), Some("root: checksum_body != expected"));

        let mut header = valid_root_header();
        header.view = 1;
        assert_eq!(header.invalid(), Some("root: view != 0"));

        let mut header = valid_root_header();
        header.release = Release(1);
        assert_eq!(header.invalid(), Some("release != 0"));

        let mut header = valid_root_header();
        header.replica = 1;
        assert_eq!(header.invalid(), Some("root: replica != 0"));

        let mut header = valid_root_header();
        header.parent = 1;
        assert_eq!(header.invalid(), Some("root: parent != 0"));

        let mut header = valid_root_header();
        header.client = 1;
        assert_eq!(header.invalid(), Some("root: client != 0"));

        let mut header = valid_root_header();
        header.request_checksum = 1;
        assert_eq!(header.invalid(), Some("root: request_checksum != 0"));

        let mut header = valid_root_header();
        header.checkpoint_id = 1;
        assert_eq!(header.invalid(), Some("root: checkpoint_id != 0"));

        let mut header = valid_root_header();
        header.timestamp = 1;
        assert_eq!(header.invalid(), Some("root: timestamp != 0"));

        let mut header = valid_root_header();
        header.request = 1;
        assert_eq!(header.invalid(), Some("root: request != 0"));
    }

    #[test]
    fn test_normal_op_invariants() {
        let mut header = HeaderPrepare::new();
        header.command = Command::Prepare;
        header.operation = Operation::NOOP;

        header.op = 10;
        header.commit = 9;
        header.timestamp = 100;
        header.client = 123;
        header.request = 1;
        header.release = Release(1);
        header.parent = 1;
        header.set_checksum_body(&[]);

        let mut h = header;
        h.parent = 0;
        assert_eq!(h.invalid(), Some("parent == 0"));

        let mut h = header;
        h.release = Release(0);
        assert_eq!(h.invalid(), Some("release == 0"));

        let mut h = header;
        h.op = 0;
        assert_eq!(h.invalid(), Some("op == 0"));

        let mut h = header;
        h.op = 9;
        assert_eq!(h.invalid(), Some("op <= commit"));

        let mut h = header;
        h.timestamp = 0;
        assert_eq!(h.invalid(), Some("timestamp == 0"));

        let mut h = header;
        h.client = 0;
        assert_eq!(h.invalid(), Some("client == 0"));

        let mut h = header;
        h.request = 0;
        assert_eq!(h.invalid(), Some("request == 0"));
    }

    #[test]
    fn test_reserved_op_code() {
        let mut header = HeaderPrepare::new();
        header.command = Command::Prepare;
        header.operation = Operation::from_u8(7); // Undefined reserved
        header.op = 10;
        header.commit = 9;
        header.timestamp = 100;
        header.client = 123;
        header.request = 1;
        header.release = Release(1);
        header.parent = 1;
        assert_eq!(header.invalid(), Some("operation reserved"));
    }

    #[test]
    fn test_valid_upgrade() {
        let mut header = HeaderPrepare::new();
        header.command = Command::Prepare;
        header.operation = Operation::UPGRADE;
        header.client = 0;
        header.request = 0;
        header.op = 10;
        header.commit = 9;
        header.timestamp = 100;
        header.release = Release(1);
        header.parent = 1;
        header.set_checksum_body(&[]);
        header.set_checksum();
        assert_eq!(header.invalid(), None);
    }

    #[test]
    fn test_upgrade_invariants() {
        let mut header = HeaderPrepare::new();
        header.command = Command::Prepare;
        header.operation = Operation::UPGRADE;
        header.op = 10;
        header.commit = 9;
        header.timestamp = 100;
        header.release = Release(1);
        header.client = 0;
        header.request = 0;
        header.parent = 1;
        header.set_checksum_body(&[]);

        let mut h = header;
        h.client = 1;
        assert_eq!(h.invalid(), Some("client != 0"));

        let mut h = header;
        h.request = 1;
        assert_eq!(h.invalid(), Some("request != 0"));
    }

    #[test]
    fn test_debug_format() {
        let mut header = HeaderPrepare::new();
        header.command = Command::Prepare;
        header.operation = Operation::PULSE;
        header.cluster = 0x1234;
        header.op = 42;
        let debug_str = format!("{:?}", header);
        assert!(debug_str.contains("HeaderPrepare"));
        assert!(debug_str.contains("cluster"));
        assert!(debug_str.contains("op"));
    }

    #[test]
    fn test_boundary_max_size() {
        let mut header = HeaderPrepare::new();
        header.command = Command::Prepare;
        header.operation = Operation::NOOP;
        header.release = Release(1);
        header.client = 1;
        header.request = 1;
        header.timestamp = 1;
        header.op = 2;
        header.commit = 1;
        header.parent = 1;
        header.size = constants::MESSAGE_SIZE_MAX;
        assert_ne!(header.invalid(), Some("size > message_size_max"));
    }

    #[test]
    fn test_register_invariants() {
        let mut header = HeaderPrepare::new();
        header.command = Command::Prepare;
        header.operation = Operation::REGISTER;
        header.release = Release(1);
        header.op = 2;
        header.commit = 1;
        header.timestamp = 1;
        header.parent = 1;
        header.set_checksum_body(&[]);

        let mut h = header;
        h.client = 0;
        h.request = 0;
        assert_eq!(h.invalid(), Some("client == 0"));

        let mut h = header;
        h.client = 123;
        h.request = 0;
        assert!(h.invalid().is_none());

        let mut h = header;
        h.client = 123;
        h.request = 1;
        assert_eq!(h.invalid(), Some("request != 0"));
    }

    #[test]
    fn test_reserve_factory() {
        let cluster = 0x1234_5678_9ABC_DEF0;
        let slot = 7;
        let header = HeaderPrepare::reserve(cluster, slot);

        assert_eq!(header.cluster, cluster);
        assert_eq!(header.op, slot);
        assert_eq!(header.command, Command::Prepare);
        assert_eq!(header.operation, Operation::RESERVED);
        assert_eq!(header.view, 0);
        assert_eq!(header.release, Release(0));
        assert!(header.valid_checksum());
        assert!(header.invalid().is_none());
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_reserve_panic() {
        let cluster = 0x1234_5678_9ABC_DEF0;
        let slot = crate::constants::JOURNAL_SLOT_COUNT as u64;
        HeaderPrepare::reserve(cluster, slot);
    }

    #[test]
    fn test_root_factory() {
        let cluster = 0xDEAD_BEEF;
        let header = HeaderPrepare::root(cluster);

        assert_eq!(header.cluster, cluster);
        assert_eq!(header.op, 0);
        assert_eq!(header.command, Command::Prepare);
        assert_eq!(header.operation, Operation::ROOT);
        assert_eq!(header.view, 0);
        assert_eq!(header.release, Release(0));
        assert_eq!(header.parent, 0);
        assert_eq!(header.client, 0);
        assert_eq!(header.commit, 0);
        assert_eq!(header.timestamp, 0);
        assert_eq!(header.request, 0);
        assert!(header.valid_checksum());
        assert!(header.invalid().is_none());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::vsr::Operation;
    use proptest::prelude::*;

    fn valid_normal_header_strategy() -> impl Strategy<Value = HeaderPrepare> {
        (
            1u64..=u64::MAX,
            0u64..=u64::MAX - 1,
            1u64..=u64::MAX,
            1u128..=u128::MAX,
            1u32..=u32::MAX,
            1u32..=u32::MAX,
            any::<u128>(),
            1u128..=u128::MAX,
        )
            .prop_filter_map(
                "op must be greater than commit",
                |(op, commit, ts, client, req, rel, cluster, parent)| {
                    if op > commit {
                        Some((op, commit, ts, client, req, rel, cluster, parent))
                    } else {
                        None
                    }
                },
            )
            .prop_map(
                |(op, commit, timestamp, client, request, release, cluster, parent)| {
                    let mut header = HeaderPrepare::new();
                    header.command = Command::Prepare;
                    header.operation = Operation::NOOP;
                    header.op = op;
                    header.commit = commit;
                    header.timestamp = timestamp;
                    header.client = client;
                    header.request = request;
                    header.release = Release(release);
                    header.cluster = cluster;
                    header.parent = parent;
                    header.set_checksum_body(&[]);
                    header.set_checksum();
                    header
                },
            )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn prop_valid_normal_headers_pass_validation(header in valid_normal_header_strategy()) {
            prop_assert!(header.invalid().is_none(), "Valid header should pass: {:?}", header);
        }

        #[test]
        fn prop_checksum_is_deterministic(header in valid_normal_header_strategy()) {
            let cs1 = header.calculate_checksum();
            let cs2 = header.calculate_checksum();
            prop_assert_eq!(cs1, cs2);
        }

        #[test]
        fn prop_valid_checksum_after_set(header in valid_normal_header_strategy()) {
            prop_assert!(header.valid_checksum());
        }

        #[test]
        fn prop_checksum_detects_view_changes(
            mut header in valid_normal_header_strategy(),
            new_view in any::<u32>()
        ) {
            let original_view = header.view;
            if new_view != original_view {
                header.view = new_view;
                prop_assert!(!header.valid_checksum(), "Checksum should detect view change");
            }
        }

        #[test]
        fn prop_checksum_detects_cluster_changes(
            mut header in valid_normal_header_strategy(),
            new_cluster in any::<u128>()
        ) {
            let original_cluster = header.cluster;
            if new_cluster != original_cluster {
                header.cluster = new_cluster;
                prop_assert!(!header.valid_checksum(), "Checksum should detect cluster change");
            }
        }

        #[test]
        fn prop_as_bytes_length_is_256(header in valid_normal_header_strategy()) {
            prop_assert_eq!(header.as_bytes().len(), 256);
        }

        #[test]
        fn prop_op_must_be_greater_than_commit(
            commit in 0u64..=u64::MAX - 1,
            timestamp in 1u64..=u64::MAX,
            client in 1u128..=u128::MAX,
            request in 1u32..=u32::MAX,
            release in 1u32..=u32::MAX,
        ) {
            let mut header = HeaderPrepare::new();
            header.command = Command::Prepare;
            header.operation = Operation::NOOP;
            header.commit = commit;
            header.timestamp = timestamp;
            header.client = client;
            header.request = request;
            header.release = Release(release);
            header.parent = 1;
            header.set_checksum_body(&[]);

            header.op = commit;
            prop_assert_eq!(header.invalid(), Some("op <= commit"));

            if commit > 0 {
                header.op = commit - 1;
                prop_assert_eq!(header.invalid(), Some("op <= commit"));
            }

            header.op = commit + 1;
            let result = header.invalid();
            prop_assert!(result.is_none() || result == Some("op == 0"));
        }
    }

    fn valid_pulse_header_strategy() -> impl Strategy<Value = HeaderPrepare> {
        (
            1u64..=u64::MAX,
            0u64..=u64::MAX - 1,
            1u64..=u64::MAX,
            1u32..=u32::MAX,
            1u128..=u128::MAX,
        )
            .prop_filter_map(
                "op must be greater than commit",
                |(op, commit, ts, rel, parent)| {
                    if op > commit {
                        Some((op, commit, ts, rel, parent))
                    } else {
                        None
                    }
                },
            )
            .prop_map(|(op, commit, timestamp, release, parent)| {
                let mut header = HeaderPrepare::new();
                header.command = Command::Prepare;
                header.operation = Operation::PULSE;
                header.op = op;
                header.commit = commit;
                header.timestamp = timestamp;
                header.release = Release(release);
                header.client = 0;
                header.request = 0;
                header.parent = parent;
                header.set_checksum_body(&[]);
                header.set_checksum();
                header
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_valid_pulse_headers_pass_validation(header in valid_pulse_header_strategy()) {
            prop_assert!(header.invalid().is_none(), "Valid pulse header should pass: {:?}", header);
        }

        #[test]
        fn prop_pulse_rejects_nonzero_client(
            mut header in valid_pulse_header_strategy(),
            client in 1u128..=u128::MAX
        ) {
            header.client = client;
            prop_assert_eq!(header.invalid(), Some("client != 0"));
        }

        #[test]
        fn prop_pulse_rejects_nonzero_request(
            mut header in valid_pulse_header_strategy(),
            request in 1u32..=u32::MAX
        ) {
            header.request = request;
            prop_assert_eq!(header.invalid(), Some("request != 0"));
        }
    }

    fn valid_reserved_header_strategy() -> impl Strategy<Value = HeaderPrepare> {
        any::<u64>().prop_map(|op| {
            let mut header = HeaderPrepare::new();
            header.command = Command::Prepare;
            header.operation = Operation::RESERVED;
            header.op = op;
            header.commit = 0;
            header.timestamp = 0;
            header.request = 0;
            header.view = 0;
            header.release = Release(0);
            header.set_checksum_body(&[]);
            header.set_checksum();
            header
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_valid_reserved_headers_pass_validation(header in valid_reserved_header_strategy()) {
            prop_assert!(header.invalid().is_none(), "Valid reserved header should pass: {:?}", header);
        }

        #[test]
        fn prop_reserved_rejects_nonzero_commit(
            mut header in valid_reserved_header_strategy(),
            commit in 1u64..=u64::MAX
        ) {
            header.commit = commit;
            prop_assert_eq!(header.invalid(), Some("reserved: commit != 0"));
        }

        #[test]
        fn prop_reserved_rejects_nonzero_view(
            mut header in valid_reserved_header_strategy(),
            view in 1u32..=u32::MAX
        ) {
            header.view = view;
            prop_assert_eq!(header.invalid(), Some("reserved: view != 0"));
        }
    }

    fn valid_root_header_strategy() -> impl Strategy<Value = HeaderPrepare> {
        Just(()).prop_map(|_| {
            let mut header = HeaderPrepare::new();
            header.command = Command::Prepare;
            header.operation = Operation::ROOT;
            header.op = 0;
            header.commit = 0;
            header.timestamp = 0;
            header.request = 0;
            header.view = 0;
            header.release = Release(0);
            header.set_checksum_body(&[]);
            header.set_checksum();
            header
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_valid_root_header_passes_validation(_unused in valid_root_header_strategy()) {
            let header = {
                let mut h = HeaderPrepare::new();
                h.command = Command::Prepare;
                h.operation = Operation::ROOT;
                h.set_checksum_body(&[]);
                h.set_checksum();
                h
            };
            prop_assert!(header.invalid().is_none());
        }

        #[test]
        fn prop_root_rejects_nonzero_op(op in 1u64..=u64::MAX) {
            let mut header = HeaderPrepare::new();
            header.command = Command::Prepare;
            header.operation = Operation::ROOT;
            header.op = op;
            header.set_checksum_body(&[]);
            prop_assert_eq!(header.invalid(), Some("root: op != 0"));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_checksum_padding_must_be_zero(
            padding in 1u128..=u128::MAX
        ) {
            let mut header = HeaderPrepare::new();
            header.checksum_padding = padding;
            prop_assert_eq!(header.invalid(), Some("checksum_padding != 0"));
        }

        #[test]
        fn prop_checksum_body_padding_must_be_zero(
            padding in 1u128..=u128::MAX
        ) {
            let mut header = HeaderPrepare::new();
            header.checksum_body_padding = padding;
            prop_assert_eq!(header.invalid(), Some("checksum_body_padding != 0"));
        }

        #[test]
        fn prop_nonce_reserved_must_be_zero(
            nonce in 1u128..=u128::MAX
        ) {
            let mut header = HeaderPrepare::new();
            header.nonce_reserved = nonce;
            prop_assert_eq!(header.invalid(), Some("nonce_reserved != 0"));
        }
    }
}
