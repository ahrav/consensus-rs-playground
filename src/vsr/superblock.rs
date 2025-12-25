//! Superblock: durable replica state for crash recovery.
//!
//! The superblock stores critical VSR state that must survive crashes and power failures.
//! Multiple copies are written to tolerate partial disk corruption. On recovery, replicas
//! read all copies and use quorum logic to select the authoritative state.
//!
//! # Layout
//!
//! The superblock zone contains [`SUPERBLOCK_COPIES`](constants::SUPERBLOCK_COPIES) copies,
//! each [`SUPERBLOCK_COPY_SIZE`] bytes, sector-aligned for direct I/O.

#![allow(dead_code)]
#[allow(unused_imports)]
use core::mem::size_of;
use std::ptr::NonNull;

#[allow(unused_imports)]
use crate::{
    constants,
    util::{AlignedBox, Zeroable, align_up, as_bytes_unchecked, as_bytes_unchecked_mut},
    vsr::{
        Header, ViewChangeArray, ViewChangeCommand, ViewChangeSlice, VsrState, storage,
        wire::Checksum128,
    },
};
use crate::{
    constants::SUPERBLOCK_VERSION,
    vsr::{
        HeaderPrepare,
        members::{Members, member_index},
        state::RootOptions,
        superblock_quorum::Threshold,
        wire::{checksum, header::Release},
    },
};
// use crate::vsr::superblock_quorums::{Quorums, RepairIterator, Threshold};

/// Extra space reserved in each superblock copy for future client session tracking.
///
/// Sized to hold headers for the difference between max clients and pipeline depth,
/// allowing session state to grow without breaking the wire format.
pub const SUPERBLOCK_COPY_PADDING: usize =
    (constants::CLIENTS_MAX - constants::PIPELINE_PREPARE_QUEUE_MAX) * size_of::<Header>();

/// Total size of one superblock copy on disk, sector-aligned.
pub const SUPERBLOCK_COPY_SIZE: usize = align_up(
    size_of::<SuperBlockHeader>() + SUPERBLOCK_COPY_PADDING,
    constants::SECTOR_SIZE,
);

/// Max concurrent I/O operations to the superblock zone.
const MAX_QUEUE_DEPTH: usize = 16;

/// Upper bound on repair attempts before giving up.
///
/// Set to 2x copies to allow re-reading each copy after corrective writes.
const MAX_REPAIR_ITERATIONS: usize = constants::SUPERBLOCK_COPIES * 2;

// Compile-time invariant checks.
#[allow(clippy::manual_is_multiple_of)]
const _: () = {
    assert!(constants::SUPERBLOCK_VERSION > 0);

    assert!(constants::SUPERBLOCK_ZONE_SIZE > 0);
    assert!(
        constants::SUPERBLOCK_ZONE_SIZE
            >= SUPERBLOCK_COPY_SIZE as u64 * constants::SUPERBLOCK_COPIES as u64
    );

    assert!(SUPERBLOCK_COPY_SIZE > 0);
    assert!(SUPERBLOCK_COPY_SIZE % constants::SECTOR_SIZE == 0);

    assert!(size_of::<SuperBlockHeader>() > 0);
    assert!(size_of::<SuperBlockHeader>() <= SUPERBLOCK_COPY_SIZE);

    // 4 copies minimum for fault tolerance (tolerates 1 corrupted copy).
    // 8 copies maximum to bound storage overhead.
    assert!(constants::SUPERBLOCK_COPIES >= 4);
    assert!(constants::SUPERBLOCK_COPIES <= 8);

    assert!(MAX_QUEUE_DEPTH > 0);
    assert!(MAX_QUEUE_DEPTH <= 256);

    assert!(MAX_REPAIR_ITERATIONS >= constants::SUPERBLOCK_COPIES);
};

/// Validates that an I/O region falls within the superblock zone and is sector-aligned.
///
/// # Panics
///
/// Panics if the region is empty, exceeds zone bounds, or is not sector-aligned.
pub fn assert_bounds(offset: u64, len: usize) {
    assert!(len > 0);
    assert!(len <= constants::SUPERBLOCK_ZONE_SIZE as usize);
    assert!(offset < constants::SUPERBLOCK_ZONE_SIZE);
    assert!(offset.checked_add(len as u64).is_some());
    assert!(offset + (len as u64) <= constants::SUPERBLOCK_ZONE_SIZE);
    assert!(offset.is_multiple_of(constants::SECTOR_SIZE as u64));
    assert!(len.is_multiple_of(constants::SECTOR_SIZE));
}

/// Persistent header written to each superblock copy.
///
/// Uses `#[repr(C)]` for stable wire layout. Field order matches disk format exactly.
/// Padding fields ensure future extensibility and must always be zero.
///
/// # Checksum Coverage
///
/// The checksum covers bytes starting after `copy` (offset 34) through end of struct.
/// This excludes `checksum`, `checksum_padding`, and `copy` since:
/// - `checksum` obviously can't cover itself
/// - `copy` differs between copies but shouldn't affect logical equality
#[repr(C, align(4096))]
#[derive(Clone, Copy)]
pub struct SuperBlockHeader {
    /// AEGIS-128L MAC covering bytes after `copy` field.
    pub checksum: Checksum128,
    /// Reserved for checksum extension; must be zero.
    pub checksum_padding: Checksum128,
    /// Which of the N redundant copies this is (0..SUPERBLOCK_COPIES).
    pub copy: u16,

    /// Schema version for forward compatibility.
    pub version: u16,
    /// Cluster identifier; must match across all replicas.
    pub cluster: u128,
    /// Monotonically increasing write sequence number.
    pub sequence: u64,
    /// Checksum of the previous superblock (chain integrity).
    pub parent: u128,
    /// Reserved; must be zero.
    pub parent_padding: u128,

    /// Current VSR protocol state (view, commit index, etc).
    pub vsr_state: VsrState,

    /// Number of valid entries in `view_headers_all`.
    pub view_headers_count: u32,
    /// Headers from recent view changes for view-change recovery.
    pub view_headers_all: ViewChangeArray,
}

// SAFETY: SuperBlockHeader is #[repr(C)] with only primitive types and arrays thereof.
// All-zeros is a valid bit pattern for every field, as validated by `SuperBlockHeader::zeroed()`.
unsafe impl Zeroable for SuperBlockHeader {}

impl SuperBlockHeader {
    /// Byte offset where checksum coverage begins.
    ///
    /// Excludes: checksum (16) + checksum_padding (16) + copy (2) = 34 bytes.
    const CHECKSUM_EXCLUDE_SIZE: usize = size_of::<u128>() + size_of::<u128>() + size_of::<u16>();

    /// Creates a zero-initialized header.
    ///
    /// # Safety
    ///
    /// Safe because `SuperBlockHeader` contains only primitive types and arrays thereof;
    /// zero is a valid bit pattern for all fields.
    pub fn zeroed() -> Self {
        // SAFETY: All fields are primitives or arrays of primitives where zero is valid.
        let header: Self = unsafe { core::mem::zeroed() };

        assert_eq!(header.checksum, 0);
        assert_eq!(header.sequence, 0);
        assert_eq!(header.copy, 0);

        header
    }

    /// Computes the checksum over all fields except checksum, padding, and copy.
    ///
    /// # Safety
    ///
    /// Relies on the invariant that headers are created via [`zeroed()`](Self::zeroed)
    /// or read from storage with all bytes initialized. `ViewChangeArray` constructors
    /// zero-fill unused slots and padding to keep the header byte-initialized.
    pub fn calculate_checksum(&self) -> Checksum128 {
        // SAFETY: SuperBlockHeader instances are created via `zeroed()` or read from
        // storage, and ViewChangeArray constructors zero-fill unused slots/padding.
        let bytes = unsafe { as_bytes_unchecked(self) };
        assert!(bytes.len() > Self::CHECKSUM_EXCLUDE_SIZE);

        checksum(&bytes[Self::CHECKSUM_EXCLUDE_SIZE..])
    }

    /// Returns true if the stored checksum matches the computed value.
    ///
    /// Also validates that `checksum_padding` is zero.
    pub fn valid_checksum(&self) -> bool {
        if self.checksum_padding != 0 {
            return false;
        }
        self.checksum == self.calculate_checksum()
    }

    /// Computes and stores the checksum.
    ///
    /// # Panics
    ///
    /// Panics if `checksum_padding` is non-zero (indicates corruption or misuse).
    pub fn set_checksum(&mut self) {
        assert_eq!(self.checksum_padding, 0);
        self.checksum = self.calculate_checksum();
        assert!(self.valid_checksum())
    }

    /// Logical equality ignoring `checksum` and `copy` fields.
    ///
    /// Two headers are equal if they represent the same VSR state, even if
    /// stored in different copy slots or with different checksums.
    pub fn equal(&self, other: &SuperBlockHeader) -> bool {
        self.checksum_padding == other.checksum_padding
            && self.version == other.version
            && self.cluster == other.cluster
            && self.sequence == other.sequence
            && self.parent == other.parent
            && self.parent_padding == other.parent_padding
            // SAFETY: VsrState is Pod, and ViewChangeArray constructors zero-fill bytes.
            && unsafe {
                as_bytes_unchecked(&self.vsr_state) == as_bytes_unchecked(&other.vsr_state)
            }
            && self.view_headers_count == other.view_headers_count
            && unsafe {
                as_bytes_unchecked(&self.view_headers_all)
                    == as_bytes_unchecked(&other.view_headers_all)
            }
    }

    /// Validates structural invariants (Tiger Style defense-in-depth).
    ///
    /// Call after reading from storage or before critical operations.
    fn assert_invariants(&self) {
        assert_eq!(self.checksum_padding, 0);
        assert_eq!(self.parent_padding, 0);
        assert!(self.version == 0 || self.version == constants::SUPERBLOCK_VERSION);
        assert!((self.copy as usize) < constants::SUPERBLOCK_COPIES || self.copy == 0);
    }
}

/// Identifies which superblock operation initiated a context.
///
/// Used for state machine validation: each caller has specific I/O expectations
/// (read-only, write-only, or read-then-write) that are checked at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Caller {
    /// Context is idle and available for reuse.
    None,
    /// Reading superblock copies during replica startup.
    Open,
    /// Writing initial superblock state for a new replica.
    Format,
    /// Persisting state after committing a batch of operations.
    Checkpoint,
    /// Persisting state after a view change completes.
    ViewChange,
}

impl Caller {
    /// Returns `true` if this operation writes superblock copies.
    pub(super) fn expects_write(&self) -> bool {
        matches!(
            self,
            Caller::Format | Caller::Checkpoint | Caller::ViewChange
        )
    }

    /// Returns `true` if this operation reads superblock copies.
    pub(super) fn expects_read(&self) -> bool {
        matches!(
            self,
            Caller::Open | Caller::Format | Caller::Checkpoint | Caller::ViewChange
        )
    }
}

/// Completion callback invoked when a superblock operation finishes.
pub type Callback<S> = fn(&mut Context<S>);

/// State for an in-flight superblock operation.
///
/// Tracks which copy is being accessed, the completion callback, and any
/// state updates to persist. Contexts are pooled and reused to avoid allocation.
///
/// # Layout
///
/// `#[repr(C)]` with `read`/`write` embedded by value enables `container_of!`-style
/// pointer recovery: when the storage engine completes an I/O and hands back
/// `&mut Read`, we can compute the containing `Context` address. This avoids
/// heap allocation for the callback context.
#[repr(C)]
pub struct Context<S: storage::Storage> {
    /// Which operation owns this context, or `None` if idle.
    pub(super) caller: Caller,
    /// Invoked when the operation completes (success or failure).
    pub(super) callback: Option<Callback<S>>,
    /// Which superblock copy (0..SUPERBLOCK_COPIES) is being accessed.
    pub(super) copy: Option<u8>,

    /// Embedded read iocb for `container_of!` recovery.
    pub(super) read: S::Read,
    /// Embedded write iocb for `container_of!` recovery.
    pub(super) write: S::Write,

    /// Updated VSR state to persist (for write operations).
    pub(super) vsr_state: Option<VsrState>,
    /// View change headers to persist (for view change operations).
    pub(super) view_headers: Option<ViewChangeArray>,
    /// Format options to persist (for format operations).
    pub(super) format_opts: Option<FormatOptions>,
    // pub(super) repairs: Option<RepairIterator< { constants::SUPERBLOCK_COPIES }>>,
    /// Intrusive linked list pointer for context pooling.
    pub(super) next: Option<NonNull<Context<S>>>,
    pub(super) sb: *mut SuperBlock<S>,
}

impl<S: storage::Storage> Context<S> {
    /// Creates an idle context with the given I/O control blocks.
    pub fn new(read: S::Read, write: S::Write) -> Self {
        Self {
            caller: Caller::None,
            callback: None,
            copy: None,
            read,
            write,
            vsr_state: None,
            view_headers: None,
            format_opts: None,
            next: None,
            sb: core::ptr::null_mut(),
        }
    }

    /// Panics if context is already linked into a queue.
    pub(super) fn assert_valid_for_enqueue(&self) {
        assert!(self.next.is_none());
    }

    /// Panics if context is not ready to issue a read.
    pub(super) fn assert_ready_for_read(&self) {
        assert!(self.caller != Caller::None);
        assert!(self.caller.expects_read());
    }

    /// Panics if context is not ready to issue a write.
    pub(super) fn assert_ready_for_write(&self) {
        assert!(self.caller != Caller::None);
        assert!(self.caller.expects_write() || self.caller == Caller::Open);
    }

    /// Panics if `copy` is out of bounds.
    pub(super) fn assert_valid_copy(&self) {
        if let Some(copy) = self.copy {
            assert!((copy as usize) < constants::SUPERBLOCK_COPIES)
        }
    }

    /// Returns `true` if this context is in use by an operation.
    pub fn is_active(&self) -> bool {
        self.caller != Caller::None
    }
}

/// Manages durable superblock state with copy-on-write semantics.
///
/// Maintains three header buffers:
/// - `working`: current committed state (read-only during normal operation)
/// - `staging`: next state being prepared (write target)
/// - `reading`: scratch space for reading all copies during open/verify
///
/// Operations are queued and processed sequentially via an intrusive linked list.
pub struct SuperBlock<S: storage::Storage> {
    storage: S,
    /// Last committed superblock state. Updated only after successful quorum verification.
    pub working: AlignedBox<SuperBlockHeader>,
    /// Next superblock state being prepared. Becomes `working` after commit.
    pub staging: AlignedBox<SuperBlockHeader>,
    /// Buffer for reading all copies during open/verification.
    reading: AlignedBox<[SuperBlockHeader; constants::SUPERBLOCK_COPIES]>,
    // quorums: Quorums< {constants::SUPERBLOCK_COPIES}>,
    /// Head of the pending operation queue (intrusive linked list).
    queue_head: Option<NonNull<Context<S>>>,
    /// Tail of the pending operation queue.
    queue_tail: Option<NonNull<Context<S>>>,
    /// This replica's index in the cluster, set during `open`.
    replica_index: Option<u8>,
    queue_depth: usize,
}

impl<S: storage::Storage> SuperBlock<S> {
    /// Creates an uninitialized superblock. Call `open` or `format` before use.
    pub fn new(storage: S) -> Self {
        // SAFETY: SuperBlockHeader implements Zeroable; all-zeros is valid.
        let working = unsafe { AlignedBox::new_zeroed() };
        let staging = unsafe { AlignedBox::new_zeroed() };
        let reading = unsafe { AlignedBox::new_zeroed() };
        let queue_head = None;
        let queue_tail = None;
        let replica_index = None;
        let queue_depth = 0;

        let sb = Self {
            storage,
            working,
            staging,
            reading,
            // quorums: Quorums::default(),
            queue_head,
            queue_tail,
            replica_index,
            queue_depth,
        };
        sb.assert_invariants();

        sb
    }

    /// Validates queue consistency and replica index bounds.
    fn assert_invariants(&self) {
        if self.queue_head.is_none() {
            assert!(self.queue_tail.is_none());
            assert_eq!(self.queue_depth, 0);
        }
        if self.queue_tail.is_none() {
            assert!(self.queue_head.is_none());
        }
        assert!(self.queue_depth <= MAX_QUEUE_DEPTH);
        if let Some(idx) = self.replica_index {
            assert!((idx as usize) < constants::REPLICAS_MAX);
        }
    }

    /// Validates staging header forms a valid chain from working.
    ///
    /// For `Open`: staging equals working (no pending changes).
    /// For other callers: staging.sequence = working.sequence + 1, parent links correctly.
    fn assert_staging_for_write(&self, caller: Caller) {
        if caller == Caller::Open {
            assert_eq!(self.staging.sequence, self.working.sequence);
        } else {
            assert_eq!(self.staging.sequence, self.working.sequence + 1);
            assert_eq!(self.staging.parent, self.working.checksum);
        }
    }

    /// Appends a context to the operation queue.
    ///
    /// The context's `sb` pointer is set to this superblock for callback access.
    fn enqueue(&mut self, ctx: &mut Context<S>) {
        ctx.assert_valid_for_enqueue();
        assert!(self.queue_depth < MAX_QUEUE_DEPTH);

        let nn = NonNull::from(&mut *ctx);
        ctx.next = None;
        ctx.sb = self as *mut SuperBlock<S>;

        match self.queue_tail {
            None => {
                assert!(self.queue_head.is_none());
                self.queue_tail = Some(nn);
                self.queue_head = Some(nn);
            }
            Some(mut tail) => {
                assert!(self.queue_head.is_some());
                // SAFETY: tail is valid while in the queue; we own the queue structure.
                unsafe {
                    let tail_ref = tail.as_mut();
                    assert!(tail_ref.next.is_none());
                    tail_ref.next = Some(nn);
                }
                self.queue_tail = Some(nn);
            }
        }

        self.queue_depth += 1;

        assert!(self.queue_head.is_some());
        assert!(self.queue_tail.is_some());
        assert!(self.queue_depth > 0);
    }

    /// Removes the head context from the queue after operation completion.
    ///
    /// # Panics
    /// Panics if `ctx` is not the current queue head.
    fn dequeue(&mut self, ctx: &mut Context<S>) {
        assert!(self.is_queue_head(ctx));
        assert!(self.queue_depth > 0);

        let next = ctx.next;
        ctx.next = None;

        self.queue_head = next;
        self.queue_depth -= 1;

        if self.queue_head.is_none() {
            self.queue_tail = None;
            assert_eq!(self.queue_depth, 0);
        }

        assert!(ctx.next.is_none());
    }

    /// Returns true if `ctx` is at the head of the operation queue.
    fn is_queue_head(&self, ctx: &Context<S>) -> bool {
        self.queue_head
            .is_some_and(|nn| core::ptr::eq(nn.as_ptr(), ctx))
    }

    /// Opens an existing superblock by reading all copies and selecting via quorum.
    ///
    /// Reads all [`SUPERBLOCK_COPIES`](constants::SUPERBLOCK_COPIES), selects the
    /// authoritative state, and repairs any corrupted copies. Invokes `cb` on completion.
    ///
    /// # Panics
    /// Panics if `ctx` is already active or `replica_index` exceeds [`REPLICAS_MAX`](constants::REPLICAS_MAX).
    pub fn open(&mut self, cb: Callback<S>, ctx: &mut Context<S>, replica_index: u8) {
        assert!(!ctx.is_active());
        assert!((replica_index as usize) < constants::REPLICAS_MAX);

        ctx.caller = Caller::Open;
        ctx.callback = Some(cb);
        ctx.copy = None;
        // ctx.repairs = None;

        self.replica_index = Some(replica_index);
        self.enqueue(ctx);

        if self.is_queue_head(ctx) {
            self.read_working(ctx, Threshold::Open);
        }
    }

    /// Initializes a fresh superblock with the given cluster configuration.
    ///
    /// Creates sequence 0 (genesis) in `working`, then sequence 1 with VSR state in `staging`.
    /// Writes all copies and verifies via read-back. Invokes `cb` on completion.
    ///
    /// # Panics
    /// Panics if `ctx` is already active or options are invalid.
    pub fn format(&mut self, cb: Callback<S>, ctx: &mut Context<S>, opts: FormatOptions) {
        assert!(!ctx.is_active());
        self.assert_format_options(&opts);

        ctx.caller = Caller::Format;
        ctx.callback = Some(cb);
        ctx.copy = Some(0);
        ctx.format_opts = Some(opts);
        // ctx.repairs = None;

        self.enqueue(ctx);

        if self.is_queue_head(ctx) {
            self.prepare_format(ctx);
        }
    }

    fn assert_format_options(&self, opts: &FormatOptions) {
        assert!(opts.replica_count > 0);
        assert!((opts.replica_count as usize) <= constants::REPLICAS_MAX);
    }

    /// Builds genesis (seq 0) and initial (seq 1) superblock headers, then writes.
    fn prepare_format(&mut self, ctx: &mut Context<S>) {
        let opts = ctx.format_opts.take().expect("format requires format_opts");

        // Genesis header: sequence 0, no parent.
        *self.working = SuperBlockHeader::zeroed();
        self.working.version = SUPERBLOCK_VERSION;
        self.working.cluster = opts.cluster;
        self.working.sequence = 0;
        self.working.parent = 0;
        self.working.vsr_state.checkpoint.header = HeaderPrepare::zeroed();
        self.working.set_checksum();

        let members = Members(opts.members);
        let replica_index =
            member_index(&members, opts.replica_id).expect("replica_id not found in members");

        let vsr_state = VsrState::root(&RootOptions {
            cluster: opts.cluster,
            release: opts.release,
            replica_index,
            members,
            replica_count: opts.replica_count,
            view: opts.view,
        });

        let view_headers = ctx
            .view_headers
            .take()
            .expect("format requires view_headers");
        assert!(!view_headers.is_empty());

        // Initial header: sequence 1, chains from genesis.
        *self.staging = *self.working;
        self.staging.sequence = 1;
        self.staging.parent = self.working.checksum;
        self.staging.vsr_state = vsr_state;
        self.staging.view_headers_count = view_headers.len() as u32;
        self.staging.view_headers_all = view_headers;
        self.staging.set_checksum();

        assert!(self.staging.valid_checksum());
        self.assert_staging_for_write(Caller::Format);

        self.write_headers(ctx)
    }

    /// Persists a checkpoint by writing a new superblock with incremented sequence.
    ///
    /// Queues the operation if another is in progress; executes immediately if at queue head.
    ///
    /// # Panics
    /// Panics if `ctx` is already active (reentrant use).
    pub fn checkpoint(&mut self, cb: Callback<S>, ctx: &mut Context<S>, opts: CheckpointOptions) {
        assert!(!ctx.is_active());

        // Calculate VSR state immediately to capture intent.
        let mut vsr_state = self.working.vsr_state;
        vsr_state.update_for_checkpoint(&opts);
        ctx.vsr_state = Some(vsr_state);

        if let Some(view_attrs) = &opts.view_attributes {
            assert!(!view_attrs.headers.is_empty());
            ctx.view_headers = Some(*view_attrs.headers);
        }

        ctx.caller = Caller::Checkpoint;
        ctx.callback = Some(cb);
        // ctx.repairs = None;
        ctx.copy = Some(0);

        self.enqueue(ctx);

        if self.is_queue_head(ctx) {
            self.prepare_checkpoint(ctx);
        }
    }

    /// Prepares staging header for checkpoint write.
    ///
    /// Chains from working header: copies state, increments sequence, updates parent checksum.
    fn prepare_checkpoint(&mut self, ctx: &mut Context<S>) {
        // Chain from current working header.
        *self.staging = *self.working;
        self.staging.sequence = self
            .working
            .sequence
            .checked_add(1)
            .expect("sequence overflow");
        self.staging.parent = self.working.checksum;

        // Apply captured VSR state.
        self.staging.vsr_state = ctx.vsr_state.take().expect("checkpoint requires vsr_state");

        // Apply view headers if provided.
        if let Some(view_headers) = ctx.view_headers.take() {
            self.staging.view_headers_count = view_headers.len() as u32;
            self.staging.view_headers_all = view_headers;
        }

        self.staging.set_checksum();

        assert!(self.staging.valid_checksum());
        self.assert_staging_for_write(Caller::Checkpoint);

        self.write_headers(ctx)
    }

    /// Reads the next superblock copy into `self.reading[ctx.copy]`.
    ///
    /// Iterates through copies 0..SUPERBLOCK_COPIES across successive callbacks.
    fn read_working(&mut self, ctx: &mut Context<S>, _threshold: Threshold) {
        ctx.assert_ready_for_read();

        if ctx.copy.is_none() {
            ctx.copy = Some(0);
        }

        let copy = ctx.copy.expect("copy must be Some after init");
        assert!((copy as usize) < constants::SUPERBLOCK_COPIES);

        let offset = (SUPERBLOCK_COPY_SIZE as u64)
            .checked_mul(copy as u64)
            .expect("offset overflow");

        let header_ref = &mut self.reading.as_mut()[copy as usize];
        // SAFETY: SuperBlockHeader is repr(C) with no padding requirements violated.
        let buf = unsafe { as_bytes_unchecked_mut(header_ref) };

        assert_bounds(offset, buf.len());

        self.storage.read_sectors(
            Self::read_header_callback,
            &mut ctx.read,
            buf,
            S::SUPERBLOCK_ZONE,
            offset,
        );
    }

    /// Completes the current operation and returns control to the caller.
    ///
    /// Extracts the completion callback, resets context state, removes the context
    /// from the queue, invokes the callback, then kicks off the next queued operation.
    ///
    /// # Callback Invocation Order
    ///
    /// The callback is invoked *after* dequeue but *before* `kick_next`. This ensures
    /// the caller can inspect final state before the next operation potentially mutates it.
    ///
    /// # Panics
    ///
    /// Panics if `ctx` has no callback or is not the current queue head.
    fn release(&mut self, ctx: &mut Context<S>) {
        assert!(ctx.callback.is_some());
        assert!(self.is_queue_head(ctx));

        let callback = ctx.callback.take().expect("callback must be Some");
        ctx.caller = Caller::None;

        self.dequeue(ctx);
        assert!(!ctx.is_active());

        // Invoke callback while context is still valid but no longer queued.
        callback(ctx);

        // Start the next pending operation, if any.
        self.kick_next();
    }

    /// Starts the next queued operation, if any.
    ///
    /// Called after `release` completes an operation. Inspects the queue head's
    /// `caller` to determine whether to start a read (Open) or write (Format,
    /// Checkpoint, ViewChange) operation.
    ///
    /// # Panics
    ///
    /// Panics if the queue head has `Caller::None`, which indicates a bug in
    /// queue management (contexts should only be enqueued with a valid caller).
    fn kick_next(&mut self) {
        let Some(mut head_nn) = self.queue_head else {
            // Queue is empty; nothing to do.
            return;
        };

        // SAFETY: head_nn is valid while in the queue; we own the queue structure.
        let head_ctx = unsafe { head_nn.as_mut() };

        // Dispatch based on operation type.
        match head_ctx.caller {
            Caller::Open => self.read_working(head_ctx, Threshold::Open),
            Caller::Format => self.prepare_format(head_ctx),
            Caller::Checkpoint => self.prepare_checkpoint(head_ctx),
            Caller::ViewChange => self.write_headers(head_ctx),
            Caller::None => {
                panic!("Caller::None should not be reached");
            }
        }
    }

    /// Writes staging header to the next copy slot.
    ///
    /// Iterates through copies 0..SUPERBLOCK_COPIES across successive callbacks.
    fn write_headers(&mut self, ctx: &mut Context<S>) {
        ctx.assert_ready_for_write();

        let copy = ctx.copy.expect("copy must be Some for write");
        assert!((copy as usize) < constants::SUPERBLOCK_COPIES);

        self.assert_staging_for_write(ctx.caller);

        self.staging.copy = copy as u16;
        assert!(self.staging.valid_checksum());

        // SAFETY: SuperBlockHeader is repr(C) with no padding requirements violated.
        let buf = unsafe { as_bytes_unchecked_mut(&mut *self.staging) };
        let offset = (SUPERBLOCK_COPY_SIZE as u64)
            .checked_mul(copy as u64)
            .expect("offset overflow");

        assert_bounds(offset, buf.len());

        self.storage.write_sectors(
            Self::write_header_callback,
            &mut ctx.write,
            buf,
            S::SUPERBLOCK_ZONE,
            offset,
        );
    }

    /// Callback invoked after each sector read completes.
    ///
    /// Continues reading remaining copies or transitions to quorum processing.
    fn read_header_callback(read: &mut S::Read) {
        // SAFETY: Storage guarantees context pointer validity for callback duration.
        let ctx = unsafe { S::context_from_read(read) };

        assert!(!ctx.sb.is_null());

        // SAFETY: ctx.sb set by enqueue(); valid while operation is in-flight.
        let sb = unsafe { &mut *ctx.sb };

        let threshold = if ctx.caller == Caller::Open {
            Threshold::Open
        } else {
            Threshold::Verify
        };

        let copy = ctx.copy.expect("copy must be Some in read callback");
        assert!((copy as usize) < constants::SUPERBLOCK_COPIES);

        // Continue reading remaining copies.
        if (copy as usize) + 1 < constants::SUPERBLOCK_COPIES {
            ctx.copy = Some(copy + 1);
            sb.read_working(ctx, threshold);
            return;
        }

        // All copies read; proceed to quorum selection.
        ctx.copy = None;
        // sb.process_quorum(ctx, threshold);
    }

    /// Callback invoked after each sector write completes.
    ///
    /// For format/update: writes remaining copies, then verifies via read-back.
    /// For open (repair): transitions to repair completion.
    fn write_header_callback(write: &mut S::Write) {
        // SAFETY: Storage guarantees context pointer validity for callback duration.
        let ctx = unsafe { S::context_from_write(write) };
        assert!(!ctx.sb.is_null());

        // SAFETY: ctx.sb set by enqueue(); valid while operation is in-flight.
        let sb = unsafe { &mut *ctx.sb };

        let copy = ctx.copy.expect("copy must be Some in write callback");
        assert!((copy as usize) < constants::SUPERBLOCK_COPIES);

        if ctx.caller == Caller::Open {
            // Repair write completed.
            ctx.copy = None;
            // sb.repair(ctx);
            return;
        }

        if (copy as usize) + 1 == constants::SUPERBLOCK_COPIES {
            // All copies written; verify by reading back.
            ctx.copy = None;
            sb.read_working(ctx, Threshold::Verify);
        } else {
            ctx.copy = Some(copy + 1);
            sb.write_headers(ctx);
        }
    }
}

/// Configuration for initializing a new superblock.
pub struct FormatOptions {
    /// Cluster identifier (must match all replicas).
    pub cluster: u128,
    /// This replica's unique identifier within `members`.
    pub replica_id: u128,
    /// All cluster member IDs; `replica_id` must be present.
    pub members: [u128; constants::MEMBERS_MAX],
    /// Number of active replicas (1..=REPLICAS_MAX).
    pub replica_count: u8,
    /// Initial view number.
    pub view: u32,
    /// Software release version for compatibility checks.
    pub release: Release,
}

/// View state captured during checkpoint for view change recovery.
///
/// When a replica checkpoints during a view change, it must persist the current
/// view headers to enable recovery. Without this, a restarted replica cannot
/// determine which operations were committed in the new view.
pub struct ViewAttributes<'a> {
    /// Prepare headers from the current view (used to reconstruct commit state).
    pub headers: &'a ViewChangeArray,
    /// Current view number.
    pub view: u32,
    /// View in which the last log entry was created.
    pub log_view: u32,
}

/// Configuration for persisting a checkpoint to the superblock.
///
/// Captures all state needed to resume from this checkpoint after crash recovery:
/// block references, commit bounds, and optionally view change state.
///
/// # Tuple Field Layout
/// Reference tuples follow consistent patterns:
/// - `manifest_references`: (oldest_checksum, oldest_addr, newest_checksum, newest_addr, block_count)
/// - `free_set_*_references`: (checksum, size, last_block_checksum, last_block_addr)
/// - `client_sessions_references`: (checksum, size, last_block_checksum, last_block_addr)
pub struct CheckpointOptions<'a> {
    /// Prepare header that triggered this checkpoint.
    pub header: HeaderPrepare,
    /// View state if checkpointing during/after a view change.
    pub view_attributes: Option<ViewAttributes<'a>>,
    /// Highest committed operation number.
    pub commit_max: u64,
    /// Sync target range: minimum operation to sync from.
    pub sync_op_min: u64,
    /// Sync target range: maximum operation to sync to.
    pub sync_op_max: u64,
    /// LSM manifest block chain: (oldest_cs, oldest_addr, newest_cs, newest_addr, count).
    pub manifest_references: (u128, u64, u128, u64, u64),
    /// Acquired free set blocks: (checksum, size, last_block_cs, last_block_addr).
    pub free_set_acquired_references: (u128, u64, u128, u64),
    /// Released free set blocks: (checksum, size, last_block_cs, last_block_addr).
    pub free_set_released_references: (u128, u64, u128, u64),
    /// Client session state: (checksum, size, last_block_cs, last_block_addr).
    pub client_sessions_references: (u128, u64, u128, u64),
    /// Total data file size in bytes.
    pub storage_size: u64,
    /// Software release for compatibility validation on recovery.
    pub release: Release,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use core::mem;
    use proptest::prelude::*;

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Creates a minimal valid SuperBlockHeader for testing.
    fn make_header() -> SuperBlockHeader {
        let mut header = SuperBlockHeader::zeroed();
        header.version = constants::SUPERBLOCK_VERSION;
        header.cluster = 1;
        header.sequence = 1;
        header.view_headers_count = 1;
        header.view_headers_all = ViewChangeArray::root(header.cluster);
        header
    }

    /// Creates a header with a specific copy number.
    fn make_header_with_copy(copy: u16) -> SuperBlockHeader {
        let mut header = make_header();
        header.copy = copy;
        header
    }

    // =========================================================================
    // Compile-Time Constants Tests
    // =========================================================================

    #[test]
    #[allow(clippy::manual_is_multiple_of)]
    fn test_superblock_copy_size_aligned() {
        const { assert!(SUPERBLOCK_COPY_SIZE > 0) };
        const { assert!(SUPERBLOCK_COPY_SIZE % constants::SECTOR_SIZE == 0) };
    }

    #[test]
    fn test_superblock_zone_size_sufficient() {
        let total_needed = SUPERBLOCK_COPY_SIZE as u64 * constants::SUPERBLOCK_COPIES as u64;
        assert!(constants::SUPERBLOCK_ZONE_SIZE >= total_needed);
    }

    #[test]
    fn test_checksum_exclude_size() {
        assert_eq!(
            SuperBlockHeader::CHECKSUM_EXCLUDE_SIZE,
            size_of::<u128>() + size_of::<u128>() + size_of::<u16>()
        );
        assert_eq!(SuperBlockHeader::CHECKSUM_EXCLUDE_SIZE, 34);
    }

    #[test]
    fn test_bounds_valid_aligned_region() {
        // First sector, one sector length
        assert_bounds(0, constants::SECTOR_SIZE);

        // Second sector
        assert_bounds(constants::SECTOR_SIZE as u64, constants::SECTOR_SIZE);

        // Multiple sectors
        assert_bounds(0, constants::SECTOR_SIZE * 4);

        // Maximum valid region
        let max_offset = constants::SUPERBLOCK_ZONE_SIZE - constants::SECTOR_SIZE as u64;
        assert_bounds(max_offset, constants::SECTOR_SIZE);
    }

    #[test]
    #[should_panic]
    fn test_bounds_zero_length() {
        assert_bounds(0, 0);
    }

    #[test]
    #[should_panic]
    fn test_bounds_unaligned_offset() {
        assert_bounds(1, constants::SECTOR_SIZE);
    }

    #[test]
    #[should_panic]
    fn test_bounds_unaligned_length() {
        assert_bounds(0, constants::SECTOR_SIZE + 1);
    }

    #[test]
    #[should_panic]
    fn test_bounds_exceeds_zone() {
        assert_bounds(
            0,
            constants::SUPERBLOCK_ZONE_SIZE as usize + constants::SECTOR_SIZE,
        );
    }

    #[test]
    #[should_panic]
    fn test_bounds_offset_exceeds_zone() {
        assert_bounds(constants::SUPERBLOCK_ZONE_SIZE, constants::SECTOR_SIZE);
    }

    #[test]
    #[should_panic]
    fn test_bounds_offset_plus_len_exceeds_zone() {
        let offset = constants::SUPERBLOCK_ZONE_SIZE - (constants::SECTOR_SIZE as u64 / 2);
        assert_bounds(offset, constants::SECTOR_SIZE);
    }

    #[test]
    #[should_panic]
    fn test_bounds_length_exceeds_zone() {
        assert_bounds(0, constants::SUPERBLOCK_ZONE_SIZE as usize + 1);
    }

    #[test]
    fn test_zeroed_initialization() {
        let header = SuperBlockHeader::zeroed();

        assert_eq!(header.checksum, 0);
        assert_eq!(header.checksum_padding, 0);
        assert_eq!(header.copy, 0);
        assert_eq!(header.version, 0);
        assert_eq!(header.cluster, 0);
        assert_eq!(header.sequence, 0);
        assert_eq!(header.parent, 0);
        assert_eq!(header.parent_padding, 0);
        assert_eq!(header.view_headers_count, 0);
    }

    #[test]
    fn test_checksum_excludes_checksum_field() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.checksum = 123;
        header2.checksum = 456;

        // Changing checksum field shouldn't affect calculated checksum
        assert_eq!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_includes_version() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.version = 1;
        header2.version = 2;

        assert_ne!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_includes_cluster() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.cluster = 1;
        header2.cluster = 2;

        assert_ne!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_includes_parent() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.parent = 1;
        header2.parent = 2;

        assert_ne!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_includes_parent_padding() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        // Note: This violates invariants but tests checksum coverage
        header1.parent_padding = 0;
        header2.parent_padding = 1;

        assert_ne!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_checksum_includes_view_headers_count() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.view_headers_count = 1;
        header2.view_headers_count = 2;

        assert_ne!(header1.calculate_checksum(), header2.calculate_checksum());
    }

    #[test]
    fn test_valid_checksum_detects_tampering() {
        let mut header = make_header();
        header.set_checksum();

        // Tamper with sequence
        header.sequence += 1;

        assert!(!header.valid_checksum());
    }

    #[test]
    fn test_valid_checksum_rejects_nonzero_padding() {
        let mut header = make_header();
        header.set_checksum();

        // Corrupt padding
        header.checksum_padding = 1;

        assert!(!header.valid_checksum());
    }

    #[test]
    #[should_panic]
    fn test_set_checksum_requires_zero_padding() {
        let mut header = make_header();
        header.checksum_padding = 1;
        header.set_checksum();
    }

    #[test]
    fn test_equal_ignores_checksum() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.checksum = 100;
        header2.checksum = 200;

        assert!(header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_version_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.version = 1;
        header2.version = 2;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_cluster_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.cluster = 1;
        header2.cluster = 2;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_sequence_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.sequence = 1;
        header2.sequence = 2;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_parent_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.parent = 1;
        header2.parent = 2;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_view_headers_count_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        header1.view_headers_count = 1;
        header2.view_headers_count = 2;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_parent_padding_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        // This violates invariants but tests equal() behavior
        header1.parent_padding = 0;
        header2.parent_padding = 1;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_equal_detects_checksum_padding_difference() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        // This violates invariants but tests equal() behavior
        header1.checksum_padding = 0;
        header2.checksum_padding = 1;

        assert!(!header1.equal(&header2));
    }

    #[test]
    fn test_invariants_zeroed_header_passes() {
        let header = SuperBlockHeader::zeroed();
        header.assert_invariants();
    }

    #[test]
    fn test_invariants_valid_header_passes() {
        let header = make_header();
        header.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_invariants_nonzero_checksum_padding() {
        let mut header = make_header();
        header.checksum_padding = 1;
        header.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_invariants_nonzero_parent_padding() {
        let mut header = make_header();
        header.parent_padding = 1;
        header.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_invariants_invalid_version() {
        let mut header = make_header();
        header.version = 99;
        header.assert_invariants();
    }

    #[test]
    fn test_invariants_version_zero_allowed() {
        let mut header = make_header();
        header.version = 0;
        header.assert_invariants();
    }

    #[test]
    fn test_invariants_version_current_allowed() {
        let mut header = make_header();
        header.version = constants::SUPERBLOCK_VERSION;
        header.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_invariants_copy_out_of_range() {
        let mut header = make_header();
        header.copy = constants::SUPERBLOCK_COPIES as u16;
        header.assert_invariants();
    }

    #[test]
    fn test_invariants_copy_max_valid() {
        let mut header = make_header();
        header.copy = (constants::SUPERBLOCK_COPIES - 1) as u16;
        header.assert_invariants();
    }

    #[test]
    fn test_invariants_copy_zero_allowed() {
        let mut header = make_header();
        header.copy = 0;
        header.assert_invariants();
    }

    proptest! {
        #[test]
        fn prop_checksum_deterministic(
            sequence in any::<u64>(),
            cluster in any::<u128>(),
        ) {
            let mut header = make_header();
            header.sequence = sequence;
            header.cluster = cluster;

            let checksum1 = header.calculate_checksum();
            let checksum2 = header.calculate_checksum();

            prop_assert_eq!(checksum1, checksum2);
        }

        #[test]
        fn prop_equal_is_reflexive(
            sequence in any::<u64>(),
            cluster in any::<u128>(),
            version in prop::option::of(Just(constants::SUPERBLOCK_VERSION)),
        ) {
            let mut header = make_header();
            header.sequence = sequence;
            header.cluster = cluster;
            if let Some(v) = version {
                header.version = v;
            }

            prop_assert!(header.equal(&header));
        }

        #[test]
        fn prop_equal_is_symmetric(
            seq1 in any::<u64>(),
            seq2 in any::<u64>(),
        ) {
            let mut header1 = make_header();
            let mut header2 = make_header();
            header1.sequence = seq1;
            header2.sequence = seq2;

            let eq_12 = header1.equal(&header2);
            let eq_21 = header2.equal(&header1);

            prop_assert_eq!(eq_12, eq_21);
        }

        #[test]
        fn prop_changing_covered_field_changes_checksum(
            sequence1 in any::<u64>(),
            sequence2 in any::<u64>(),
        ) {
            if sequence1 == sequence2 {
                return Ok(());
            }

            let mut header1 = make_header();
            let mut header2 = make_header();

            header1.sequence = sequence1;
            header2.sequence = sequence2;

            prop_assert_ne!(
                header1.calculate_checksum(),
                header2.calculate_checksum()
            );
        }

        #[test]
        fn prop_changing_excluded_field_preserves_checksum(
            copy1 in 0u16..(constants::SUPERBLOCK_COPIES as u16),
            copy2 in 0u16..(constants::SUPERBLOCK_COPIES as u16),
        ) {
            let mut header1 = make_header();
            let mut header2 = make_header();

            header1.copy = copy1;
            header2.copy = copy2;

            prop_assert_eq!(
                header1.calculate_checksum(),
                header2.calculate_checksum()
            );
        }

        #[test]
        fn prop_set_checksum_makes_valid(
            sequence in any::<u64>(),
            cluster in any::<u128>(),
        ) {
            let mut header = make_header();
            header.sequence = sequence;
            header.cluster = cluster;
            header.set_checksum();

            prop_assert!(header.valid_checksum());
        }

        #[test]
        fn prop_equal_ignores_copy_values(
            copy1 in 0u16..(constants::SUPERBLOCK_COPIES as u16),
            copy2 in 0u16..(constants::SUPERBLOCK_COPIES as u16),
            sequence in any::<u64>(),
        ) {
            let mut header1 = make_header();
            let mut header2 = make_header();

            header1.copy = copy1;
            header2.copy = copy2;
            header1.sequence = sequence;
            header2.sequence = sequence;

            prop_assert!(header1.equal(&header2));
        }
    }

    #[test]
    fn test_checksum_survives_clone() {
        let mut original = make_header();
        original.set_checksum();

        let cloned = original;

        assert!(cloned.valid_checksum());
        assert_eq!(cloned.checksum, original.checksum);
    }

    #[test]
    fn test_equal_after_clone() {
        let original = make_header();
        let cloned = original;

        assert!(original.equal(&cloned));
    }

    #[test]
    fn test_max_sequence_value() {
        let mut header = make_header();
        header.sequence = u64::MAX;
        header.set_checksum();

        assert!(header.valid_checksum());
    }

    #[test]
    fn test_max_cluster_value() {
        let mut header = make_header();
        header.cluster = u128::MAX;
        header.set_checksum();

        assert!(header.valid_checksum());
    }

    #[test]
    fn test_max_copy_index() {
        let mut header = make_header();
        header.copy = (constants::SUPERBLOCK_COPIES - 1) as u16;
        header.assert_invariants();
    }

    #[test]
    fn test_different_view_headers_arrays() {
        let mut header1 = make_header();
        let mut header2 = make_header();

        // Create different view change arrays
        header1.view_headers_all = ViewChangeArray::root(1);
        header2.view_headers_all = ViewChangeArray::root(2);

        assert!(!header1.equal(&header2));
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_lifecycle() {
        // Create header
        let mut header = make_header();
        header.copy = 0;
        header.sequence = 1;
        header.cluster = 42;

        // Set checksum
        header.set_checksum();

        // Verify invariants
        header.assert_invariants();

        // Verify checksum
        assert!(header.valid_checksum());

        // Clone to another copy
        let mut copy = header;
        copy.copy = 1;

        // Copies should be logically equal
        assert!(header.equal(&copy));

        // But checksums should be identical (copy field not covered)
        assert_eq!(header.checksum, copy.checksum);
    }

    #[test]
    fn test_progression_sequence() {
        let mut header1 = make_header();
        header1.sequence = 1;
        header1.set_checksum();

        let mut header2 = make_header();
        header2.sequence = 2;
        header2.parent = header1.checksum;
        header2.set_checksum();

        // Headers should not be equal (different sequence)
        assert!(!header1.equal(&header2));

        // Parent chain should link
        assert_eq!(header2.parent, header1.checksum);
    }
    #[test]
    fn test_superblock_header_size_alignment() {
        assert_eq!(mem::align_of::<SuperBlockHeader>(), 4096);
        assert!(mem::size_of::<SuperBlockHeader>() <= SUPERBLOCK_COPY_SIZE);
        assert_eq!(
            mem::size_of::<SuperBlockHeader>() % constants::SECTOR_SIZE,
            0,
            "SuperBlockHeader size {} not multiple of SECTOR_SIZE",
            mem::size_of::<SuperBlockHeader>()
        );
    }

    // =========================================================================
    // Context & Caller Tests
    // =========================================================================

    use crate::container_of;

    struct MockStorage {
        pub disk: Box<[u8]>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                disk: vec![0; constants::SUPERBLOCK_ZONE_SIZE as usize].into_boxed_slice(),
            }
        }
    }

    impl crate::vsr::storage::Storage for MockStorage {
        type Read = u64;
        type Write = u64;
        type Zone = u8;
        const SUPERBLOCK_ZONE: Self::Zone = 0;

        fn read_sectors(
            &mut self,
            cb: fn(&mut Self::Read),
            read: &mut Self::Read,
            buf: &mut [u8],
            _zone: Self::Zone,
            offset: u64,
        ) {
            let offset = offset as usize;
            assert!(offset + buf.len() <= self.disk.len());
            buf.copy_from_slice(&self.disk[offset..offset + buf.len()]);
            cb(read);
        }

        fn write_sectors(
            &mut self,
            cb: fn(&mut Self::Write),
            write: &mut Self::Write,
            buf: &[u8],
            _zone: Self::Zone,
            offset: u64,
        ) {
            let offset = offset as usize;
            assert!(offset + buf.len() <= self.disk.len());
            self.disk[offset..offset + buf.len()].copy_from_slice(buf);
            cb(write);
        }

        unsafe fn context_from_read(read: &mut Self::Read) -> &mut Context<Self> {
            let ptr = read as *mut Self::Read;
            let ctx_ptr = container_of!(ptr, Context<Self>, read);
            unsafe { &mut *ctx_ptr }
        }

        unsafe fn context_from_write(write: &mut Self::Write) -> &mut Context<Self> {
            let ptr = write as *mut Self::Write;
            let ctx_ptr = container_of!(ptr, Context<Self>, write);
            unsafe { &mut *ctx_ptr }
        }
    }

    #[test]
    fn test_format_hangs_due_to_missing_quorum() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::new(0, 0);

        // Required for format
        ctx.view_headers = Some(ViewChangeArray::root(1));

        let cb: Callback<MockStorage> = |_| {
            panic!("Callback should not be called because logic is incomplete");
        };

        let mut members = [0u128; constants::MEMBERS_MAX];
        members[0] = 1;
        members[1] = 2;
        members[2] = 3;

        let opts = FormatOptions {
            cluster: 1,
            replica_id: 1,
            members,
            replica_count: 3,
            view: 0,
            release: Release(0),
        };

        sb.format(cb, &mut ctx, opts);

        // Verification:
        // 1. Context should still be in queue (active) because the operation chain stops
        //    before calling the completion callback.
        assert!(sb.is_queue_head(&ctx));
        assert!(ctx.is_active());

        // 2. Data should have been written to disk (since MockStorage is sync)
        let disk = &sb.storage.disk;

        let mut header = SuperBlockHeader::zeroed();
        let header_bytes = unsafe { as_bytes_unchecked_mut(&mut header) };
        let len = header_bytes.len();
        header_bytes.copy_from_slice(&disk[0..len]);

        assert_eq!(header.cluster, 1);
        assert_eq!(header.sequence, 1);
    }

    #[test]
    fn test_caller_expectations() {
        // None expects nothing
        assert!(!Caller::None.expects_read());
        assert!(!Caller::None.expects_write());

        // Open expects read
        assert!(Caller::Open.expects_read());
        // Open does not "expect" write (in the sense of being a write op),
        // but can write (repair).
        assert!(!Caller::Open.expects_write());

        // Write operations
        for caller in [Caller::Format, Caller::Checkpoint, Caller::ViewChange] {
            // Updated behavior: write operations verify their writes by reading back
            assert!(caller.expects_read());
            assert!(caller.expects_write());
        }
    }

    #[test]
    fn test_context_lifecycle_and_invariants() {
        let mut ctx = Context::<MockStorage>::new(100, 200);

        // Initial state
        assert_eq!(ctx.caller, Caller::None);
        assert!(ctx.callback.is_none());
        assert!(ctx.copy.is_none());
        assert_eq!(ctx.read, 100);
        assert_eq!(ctx.write, 200);
        assert!(!ctx.is_active());

        // Enqueue check
        ctx.assert_valid_for_enqueue();

        // Activation
        ctx.caller = Caller::Open;
        assert!(ctx.is_active());

        // IO Readiness
        ctx.assert_ready_for_read();
        ctx.assert_ready_for_write();

        // Copy bounds
        ctx.copy = Some(0);
        ctx.assert_valid_copy();

        ctx.copy = Some((constants::SUPERBLOCK_COPIES - 1) as u8);
        ctx.assert_valid_copy();
    }

    #[test]
    #[should_panic]
    fn test_context_panic_on_invalid_read_state() {
        let ctx = Context::<MockStorage>::new(0, 0);
        // Caller is None
        ctx.assert_ready_for_read();
    }

    #[test]
    #[should_panic]
    fn test_context_panic_on_invalid_write_state() {
        let ctx = Context::<MockStorage>::new(0, 0);
        // Caller is None
        ctx.assert_ready_for_write();
    }

    #[test]
    #[should_panic]
    fn test_context_panic_on_invalid_enqueue() {
        let mut ctx = Context::<MockStorage>::new(0, 0);
        // Simulate already linked
        ctx.next = NonNull::new(&mut ctx as *mut _);
        ctx.assert_valid_for_enqueue();
    }

    #[test]
    #[should_panic]
    fn test_context_panic_on_invalid_copy() {
        let mut ctx = Context::<MockStorage>::new(0, 0);
        ctx.copy = Some(constants::SUPERBLOCK_COPIES as u8);
        ctx.assert_valid_copy();
    }

    // =========================================================================
    // SuperBlock Construction Tests
    // =========================================================================

    #[test]
    fn test_superblock_new_initial_state() {
        let storage = MockStorage::new();
        let sb = SuperBlock::new(storage);

        // Queue starts empty.
        assert!(sb.queue_head.is_none());
        assert!(sb.queue_tail.is_none());
        assert_eq!(sb.queue_depth, 0);

        // Replica index unset until open().
        assert!(sb.replica_index.is_none());

        // Headers are zeroed.
        assert_eq!(sb.working.sequence, 0);
        assert_eq!(sb.staging.sequence, 0);
    }

    #[test]
    fn test_superblock_new_invariants_hold() {
        let storage = MockStorage::new();
        let sb = SuperBlock::new(storage);
        // Should not panic.
        sb.assert_invariants();
    }

    // =========================================================================
    // Queue Operation Tests (enqueue/dequeue/is_queue_head)
    // =========================================================================

    #[test]
    fn test_enqueue_single_context() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        sb.enqueue(&mut ctx);

        assert!(sb.queue_head.is_some());
        assert!(sb.queue_tail.is_some());
        assert_eq!(sb.queue_depth, 1);
        assert!(sb.is_queue_head(&ctx));
    }

    #[test]
    fn test_enqueue_sets_context_sb_pointer() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        sb.enqueue(&mut ctx);

        // Context should point back to superblock.
        assert!(!ctx.sb.is_null());
        assert!(core::ptr::eq(ctx.sb, &sb as *const _ as *mut _));
    }

    #[test]
    fn test_enqueue_multiple_contexts_fifo_order() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx1 = Context::<MockStorage>::new(1, 1);
        let mut ctx2 = Context::<MockStorage>::new(2, 2);

        sb.enqueue(&mut ctx1);
        sb.enqueue(&mut ctx2);

        assert_eq!(sb.queue_depth, 2);
        // First enqueued is head.
        assert!(sb.is_queue_head(&ctx1));
        assert!(!sb.is_queue_head(&ctx2));
    }

    #[test]
    fn test_dequeue_single_context() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        sb.enqueue(&mut ctx);
        sb.dequeue(&mut ctx);

        assert!(sb.queue_head.is_none());
        assert!(sb.queue_tail.is_none());
        assert_eq!(sb.queue_depth, 0);
        assert!(ctx.next.is_none());
    }

    #[test]
    fn test_dequeue_preserves_remaining_queue() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx1 = Context::<MockStorage>::new(1, 1);
        let mut ctx2 = Context::<MockStorage>::new(2, 2);

        sb.enqueue(&mut ctx1);
        sb.enqueue(&mut ctx2);
        sb.dequeue(&mut ctx1);

        assert_eq!(sb.queue_depth, 1);
        assert!(sb.is_queue_head(&ctx2));
    }

    #[test]
    #[should_panic]
    fn test_dequeue_panics_if_not_head() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx1 = Context::<MockStorage>::new(1, 1);
        let mut ctx2 = Context::<MockStorage>::new(2, 2);

        sb.enqueue(&mut ctx1);
        sb.enqueue(&mut ctx2);
        // ctx2 is not head, should panic.
        sb.dequeue(&mut ctx2);
    }

    #[test]
    #[should_panic]
    fn test_dequeue_panics_on_empty_queue() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        // Never enqueued, should panic.
        sb.dequeue(&mut ctx);
    }

    #[test]
    fn test_is_queue_head_empty_queue() {
        let storage = MockStorage::new();
        let sb = SuperBlock::new(storage);
        let ctx = Context::<MockStorage>::new(0, 0);

        assert!(!sb.is_queue_head(&ctx));
    }

    #[test]
    #[should_panic]
    fn test_enqueue_panics_at_max_depth() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);

        // Create MAX_QUEUE_DEPTH + 1 contexts.
        let mut contexts: Vec<Context<MockStorage>> = (0..=MAX_QUEUE_DEPTH)
            .map(|i| Context::new(i as u64, i as u64))
            .collect();

        // Enqueue up to max should succeed.
        for ctx in contexts.iter_mut().take(MAX_QUEUE_DEPTH) {
            sb.enqueue(ctx);
        }

        // One more should panic.
        sb.enqueue(contexts.last_mut().unwrap());
    }

    // =========================================================================
    // assert_invariants() Tests
    // =========================================================================

    #[test]
    fn test_assert_invariants_valid_replica_index() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        sb.replica_index = Some(0);
        sb.assert_invariants();

        sb.replica_index = Some((constants::REPLICAS_MAX - 1) as u8);
        sb.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_assert_invariants_panics_invalid_replica_index() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        sb.replica_index = Some(constants::REPLICAS_MAX as u8);
        sb.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_assert_invariants_panics_head_without_tail() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        // Manually corrupt state: head set, tail None.
        sb.queue_head = Some(NonNull::from(&mut ctx));
        sb.queue_tail = None;
        sb.queue_depth = 1;

        sb.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_assert_invariants_panics_tail_without_head() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        // Manually corrupt state: tail set, head None.
        sb.queue_head = None;
        sb.queue_tail = Some(NonNull::from(&mut ctx));
        sb.queue_depth = 1;

        sb.assert_invariants();
    }

    #[test]
    #[should_panic]
    fn test_assert_invariants_panics_depth_exceeds_max() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        sb.queue_depth = MAX_QUEUE_DEPTH + 1;
        sb.assert_invariants();
    }

    // =========================================================================
    // assert_staging_for_write() Tests
    // =========================================================================

    #[test]
    fn test_assert_staging_for_write_open_caller() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);

        sb.working.sequence = 5;
        sb.staging.sequence = 5;

        // Should not panic for Open when sequences match.
        sb.assert_staging_for_write(Caller::Open);
    }

    #[test]
    #[should_panic]
    fn test_assert_staging_for_write_open_sequence_mismatch() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);

        sb.working.sequence = 5;
        sb.staging.sequence = 6;

        sb.assert_staging_for_write(Caller::Open);
    }

    #[test]
    fn test_assert_staging_for_write_format_caller() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);

        sb.working.sequence = 0;
        sb.working.set_checksum();
        sb.staging.sequence = 1;
        sb.staging.parent = sb.working.checksum;

        // Should not panic for Format with proper chain.
        sb.assert_staging_for_write(Caller::Format);
    }

    #[test]
    #[should_panic]
    fn test_assert_staging_for_write_format_wrong_sequence() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);

        sb.working.sequence = 0;
        sb.staging.sequence = 2; // Should be 1.

        sb.assert_staging_for_write(Caller::Format);
    }

    #[test]
    #[should_panic]
    fn test_assert_staging_for_write_format_wrong_parent() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);

        sb.working.sequence = 0;
        sb.working.set_checksum();
        sb.staging.sequence = 1;
        sb.staging.parent = 0xDEADBEEF; // Wrong parent.

        sb.assert_staging_for_write(Caller::Format);
    }

    // =========================================================================
    // open() Tests
    // =========================================================================

    #[test]
    #[should_panic]
    fn test_open_panics_on_active_context() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        // Mark context as active.
        ctx.caller = Caller::Open;

        let cb: Callback<MockStorage> = |_| {};
        sb.open(cb, &mut ctx, 0);
    }

    #[test]
    #[should_panic]
    fn test_open_panics_on_invalid_replica_index() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        let cb: Callback<MockStorage> = |_| {};
        sb.open(cb, &mut ctx, constants::REPLICAS_MAX as u8);
    }

    #[test]
    fn test_open_sets_replica_index() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        let cb: Callback<MockStorage> = |_| {};
        sb.open(cb, &mut ctx, 2);

        assert_eq!(sb.replica_index, Some(2));
    }

    #[test]
    fn test_open_enqueues_context() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        let cb: Callback<MockStorage> = |_| {};
        sb.open(cb, &mut ctx, 0);

        assert!(sb.is_queue_head(&ctx));
        assert_eq!(ctx.caller, Caller::Open);
        assert!(ctx.callback.is_some());
    }

    // =========================================================================
    // format() and assert_format_options() Tests
    // =========================================================================

    #[test]
    #[should_panic]
    fn test_format_panics_on_active_context() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);
        ctx.view_headers = Some(ViewChangeArray::root(1));

        // Mark context as active.
        ctx.caller = Caller::Format;

        let mut members = [0u128; constants::MEMBERS_MAX];
        members[0] = 1;

        let cb: Callback<MockStorage> = |_| {};
        let opts = FormatOptions {
            cluster: 1,
            replica_id: 1,
            members,
            replica_count: 1,
            view: 0,
            release: Release(0),
        };
        sb.format(cb, &mut ctx, opts);
    }

    #[test]
    #[should_panic]
    fn test_format_panics_on_zero_replica_count() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);
        ctx.view_headers = Some(ViewChangeArray::root(1));

        let mut members = [0u128; constants::MEMBERS_MAX];
        members[0] = 1;

        let cb: Callback<MockStorage> = |_| {};
        let opts = FormatOptions {
            cluster: 1,
            replica_id: 1,
            members,
            replica_count: 0, // Invalid.
            view: 0,
            release: Release(0),
        };
        sb.format(cb, &mut ctx, opts);
    }

    #[test]
    #[should_panic]
    fn test_format_panics_on_replica_count_exceeds_max() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);
        ctx.view_headers = Some(ViewChangeArray::root(1));

        let mut members = [0u128; constants::MEMBERS_MAX];
        members[0] = 1;

        let cb: Callback<MockStorage> = |_| {};
        let opts = FormatOptions {
            cluster: 1,
            replica_id: 1,
            members,
            replica_count: (constants::REPLICAS_MAX + 1) as u8,
            view: 0,
            release: Release(0),
        };
        sb.format(cb, &mut ctx, opts);
    }

    #[test]
    fn test_format_enqueues_context() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);
        ctx.view_headers = Some(ViewChangeArray::root(1));

        let mut members = [0u128; constants::MEMBERS_MAX];
        members[0] = 1;

        let cb: Callback<MockStorage> = |_| {};
        let opts = FormatOptions {
            cluster: 1,
            replica_id: 1,
            members,
            replica_count: 1,
            view: 0,
            release: Release(0),
        };
        sb.format(cb, &mut ctx, opts);

        // Context remains queued (callback not invoked due to incomplete quorum logic).
        assert!(sb.is_queue_head(&ctx));
        assert_eq!(ctx.caller, Caller::Format);
        // Note: With sync MockStorage, copy advances through write/read cycles.
        // Final state depends on how far the operation progressed.
    }

    // =========================================================================
    // prepare_format() Tests
    // =========================================================================

    #[test]
    fn test_prepare_format_genesis_header() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);
        ctx.view_headers = Some(ViewChangeArray::root(42));

        let mut members = [0u128; constants::MEMBERS_MAX];
        members[0] = 100;

        let cb: Callback<MockStorage> = |_| {};
        let opts = FormatOptions {
            cluster: 42,
            replica_id: 100,
            members,
            replica_count: 1,
            view: 0,
            release: Release(1),
        };
        sb.format(cb, &mut ctx, opts);

        // Working should have genesis state.
        assert_eq!(sb.working.version, SUPERBLOCK_VERSION);
        assert_eq!(sb.working.cluster, 42);
        assert_eq!(sb.working.sequence, 0);
        assert_eq!(sb.working.parent, 0);
        assert!(sb.working.valid_checksum());
    }

    #[test]
    fn test_prepare_format_staging_chains_from_working() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);
        ctx.view_headers = Some(ViewChangeArray::root(42));

        let mut members = [0u128; constants::MEMBERS_MAX];
        members[0] = 100;

        let cb: Callback<MockStorage> = |_| {};
        let opts = FormatOptions {
            cluster: 42,
            replica_id: 100,
            members,
            replica_count: 1,
            view: 0,
            release: Release(1),
        };
        sb.format(cb, &mut ctx, opts);

        // Staging should chain from working.
        assert_eq!(sb.staging.sequence, 1);
        assert_eq!(sb.staging.parent, sb.working.checksum);
        assert_eq!(sb.staging.cluster, 42);
        assert!(sb.staging.valid_checksum());
    }

    #[test]
    #[should_panic]
    fn test_prepare_format_panics_without_view_headers() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);
        // No view_headers set.

        let mut members = [0u128; constants::MEMBERS_MAX];
        members[0] = 1;

        let cb: Callback<MockStorage> = |_| {};
        let opts = FormatOptions {
            cluster: 1,
            replica_id: 1,
            members,
            replica_count: 1,
            view: 0,
            release: Release(0),
        };
        sb.format(cb, &mut ctx, opts);
    }

    #[test]
    #[should_panic]
    fn test_prepare_format_panics_replica_not_in_members() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);
        ctx.view_headers = Some(ViewChangeArray::root(1));

        let mut members = [0u128; constants::MEMBERS_MAX];
        members[0] = 1;
        members[1] = 2;
        members[2] = 3;

        let cb: Callback<MockStorage> = |_| {};
        let opts = FormatOptions {
            cluster: 1,
            replica_id: 999, // Not in members.
            members,
            replica_count: 3,
            view: 0,
            release: Release(0),
        };
        sb.format(cb, &mut ctx, opts);
    }

    // =========================================================================
    // I/O Callback Iteration Tests
    // =========================================================================

    #[test]
    fn test_format_writes_all_copies() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);
        ctx.view_headers = Some(ViewChangeArray::root(1));

        let mut members = [0u128; constants::MEMBERS_MAX];
        members[0] = 1;

        let cb: Callback<MockStorage> = |_| {};
        let opts = FormatOptions {
            cluster: 1,
            replica_id: 1,
            members,
            replica_count: 1,
            view: 0,
            release: Release(0),
        };
        sb.format(cb, &mut ctx, opts);

        // With MockStorage (sync), all copies should be written.
        // Verify each copy location has valid data.
        for copy_idx in 0..constants::SUPERBLOCK_COPIES {
            let offset = SUPERBLOCK_COPY_SIZE * copy_idx;
            let mut header = SuperBlockHeader::zeroed();
            let header_bytes = unsafe { as_bytes_unchecked_mut(&mut header) };
            header_bytes.copy_from_slice(&sb.storage.disk[offset..offset + header_bytes.len()]);

            assert_eq!(header.cluster, 1);
            assert_eq!(header.sequence, 1);
            assert_eq!(header.copy, copy_idx as u16);
        }
    }

    #[test]
    fn test_open_reads_all_copies() {
        // Pre-populate disk with valid headers.
        let mut storage = MockStorage::new();
        for copy_idx in 0..constants::SUPERBLOCK_COPIES {
            let offset = SUPERBLOCK_COPY_SIZE * copy_idx;
            let mut header = make_header();
            header.copy = copy_idx as u16;
            header.set_checksum();
            let header_bytes = unsafe { as_bytes_unchecked(&header) };
            storage.disk[offset..offset + header_bytes.len()].copy_from_slice(header_bytes);
        }

        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        let cb: Callback<MockStorage> = |_| {};
        sb.open(cb, &mut ctx, 0);

        // After open completes reads (sync mock), all copies should be in reading buffer.
        for copy_idx in 0..constants::SUPERBLOCK_COPIES {
            let read_header = &sb.reading[copy_idx];
            assert_eq!(read_header.copy, copy_idx as u16);
            assert!(read_header.valid_checksum());
        }
    }

    // =========================================================================
    // Property-Based Tests for SuperBlock
    // =========================================================================

    proptest! {
        #[test]
        fn prop_format_staging_always_chains(
            cluster in any::<u128>(),
            view in any::<u32>(),
        ) {
            let storage = MockStorage::new();
            let mut sb = SuperBlock::new(storage);
            let mut ctx = Context::<MockStorage>::new(0, 0);
            ctx.view_headers = Some(ViewChangeArray::root(cluster));

            let mut members = [0u128; constants::MEMBERS_MAX];
            members[0] = 1;

            let cb: Callback<MockStorage> = |_| {};
            let opts = FormatOptions {
                cluster,
                replica_id: 1,
                members,
                replica_count: 1,
                view,
                release: Release(0),
            };
            sb.format(cb, &mut ctx, opts);

            // Staging always chains from working.
            prop_assert_eq!(sb.staging.sequence, sb.working.sequence + 1);
            prop_assert_eq!(sb.staging.parent, sb.working.checksum);
            prop_assert!(sb.staging.valid_checksum());
        }

        #[test]
        fn prop_enqueue_dequeue_preserves_depth(
            count in 1usize..=MAX_QUEUE_DEPTH,
        ) {
            let storage = MockStorage::new();
            let mut sb = SuperBlock::new(storage);

            let mut contexts: Vec<Context<MockStorage>> = (0..count)
                .map(|i| Context::new(i as u64, i as u64))
                .collect();

            // Enqueue all.
            for ctx in contexts.iter_mut() {
                sb.enqueue(ctx);
            }
            prop_assert_eq!(sb.queue_depth, count);

            // Dequeue all.
            for ctx in contexts.iter_mut() {
                sb.dequeue(ctx);
            }
            prop_assert_eq!(sb.queue_depth, 0);
            prop_assert!(sb.queue_head.is_none());
            prop_assert!(sb.queue_tail.is_none());
        }
    }

    // =========================================================================
    // Checkpoint Tests
    // =========================================================================

    use crate::vsr::{HeaderPrepare, Members, ViewChangeArray, wire::checksum};

    fn make_prepare_header(cluster: u128, op: u64) -> HeaderPrepare {
        use crate::vsr::wire::{Command, Operation};

        assert!(op > 0);

        let mut header = HeaderPrepare::new();
        header.cluster = cluster;
        header.command = Command::Prepare;
        header.operation = Operation::NOOP;
        header.op = op;
        header.commit = op - 1;
        header.timestamp = 1;
        header.parent = 1;
        header.client = 1;
        header.request = 1;
        header.release = Release(1);

        header.set_checksum_body(&[]);
        header.set_checksum();

        debug_assert!(header.invalid().is_none());

        header
    }

    /// Creates a minimal valid CheckpointOptions for testing.
    ///
    /// `op` must be > 0 to pass VsrState::update_for_checkpoint validation.
    fn make_checkpoint_options(op: u64, commit_max: u64) -> CheckpointOptions<'static> {
        let header = make_prepare_header(1, op);

        CheckpointOptions {
            header,
            view_attributes: None,
            commit_max,
            sync_op_min: 0,
            sync_op_max: 0,
            manifest_references: (0, 0, 0, 0, 0),
            free_set_acquired_references: (checksum(&[]), 0, 0, 0),
            free_set_released_references: (checksum(&[]), 0, 0, 0),
            client_sessions_references: (checksum(&[]), 0, 0, 0),
            storage_size: constants::DATA_FILE_SIZE_MIN,
            release: Release::ZERO,
        }
    }

    /// Initializes a SuperBlock with a formatted working header for checkpoint testing.
    fn setup_formatted_superblock() -> SuperBlock<MockStorage> {
        use crate::vsr::wire::header::Release;

        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);

        // Initialize working header to simulate post-format state.
        sb.working.version = constants::SUPERBLOCK_VERSION;
        sb.working.cluster = 1;
        sb.working.sequence = 1;
        sb.working.view_headers_count = 1;
        sb.working.view_headers_all = ViewChangeArray::root(1);

        // Initialize VsrState with valid replica config.
        let members = Members([1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let root_opts = RootOptions {
            cluster: 1,
            replica_index: 0,
            replica_count: 3,
            members,
            release: Release::ZERO,
            view: 0,
        };
        sb.working.vsr_state = VsrState::root(&root_opts);

        sb.working.set_checksum();
        sb
    }

    fn prepare_checkpoint_helper(
        sb: &mut SuperBlock<MockStorage>,
        ctx: &mut Context<MockStorage>,
        opts: CheckpointOptions,
    ) {
        let mut vsr_state = sb.working.vsr_state;
        vsr_state.update_for_checkpoint(&opts);
        ctx.vsr_state = Some(vsr_state);

        if let Some(view_attrs) = &opts.view_attributes {
            ctx.view_headers = Some(*view_attrs.headers);
        }
        sb.prepare_checkpoint(ctx);
    }

    #[test]
    fn test_checkpoint_sets_context_caller() {
        let mut sb = setup_formatted_superblock();
        let mut ctx = Context::<MockStorage>::new(0, 0);

        let cb: Callback<MockStorage> = |_| {};
        let opts = make_checkpoint_options(1, 1);

        sb.checkpoint(cb, &mut ctx, opts);

        assert_eq!(ctx.caller, Caller::Checkpoint);
    }

    #[test]
    fn test_checkpoint_sets_callback() {
        let mut sb = setup_formatted_superblock();
        let mut ctx = Context::<MockStorage>::new(0, 0);

        let cb: Callback<MockStorage> = |_| {};
        let opts = make_checkpoint_options(1, 1);

        sb.checkpoint(cb, &mut ctx, opts);

        assert!(ctx.callback.is_some());
    }

    #[test]
    fn test_checkpoint_initializes_copy_to_zero() {
        let mut sb = setup_formatted_superblock();
        let mut ctx1 = Context::<MockStorage>::new(1, 1);
        let mut ctx2 = Context::<MockStorage>::new(2, 2);

        let cb: Callback<MockStorage> = |_| {};

        // First checkpoint takes the head position.
        let opts1 = make_checkpoint_options(1, 1);
        sb.checkpoint(cb, &mut ctx1, opts1);

        // Second checkpoint queues behind first - prepare_checkpoint not called.
        let opts2 = make_checkpoint_options(2, 2);
        sb.checkpoint(cb, &mut ctx2, opts2);

        // ctx2.copy should still be Some(0) since it hasn't started writing.
        assert_eq!(ctx2.copy, Some(0));
    }

    #[test]
    fn test_checkpoint_enqueues_context() {
        let mut sb = setup_formatted_superblock();
        let mut ctx = Context::<MockStorage>::new(0, 0);

        let cb: Callback<MockStorage> = |_| {};
        let opts = make_checkpoint_options(1, 1);

        sb.checkpoint(cb, &mut ctx, opts);

        assert_eq!(sb.queue_depth, 1);
        assert!(sb.is_queue_head(&ctx));
    }

    #[test]
    #[should_panic]
    fn test_checkpoint_panics_on_active_context() {
        let mut sb = setup_formatted_superblock();
        let mut ctx = Context::<MockStorage>::new(0, 0);

        // Simulate active context by setting caller.
        ctx.caller = Caller::Open;

        let cb: Callback<MockStorage> = |_| {};
        let opts = make_checkpoint_options(1, 1);

        // Should panic: context already active.
        sb.checkpoint(cb, &mut ctx, opts);
    }

    #[test]
    fn test_checkpoint_queues_when_not_head() {
        let mut sb = setup_formatted_superblock();
        let mut ctx1 = Context::<MockStorage>::new(1, 1);
        let mut ctx2 = Context::<MockStorage>::new(2, 2);

        // First checkpoint becomes head.
        let cb: Callback<MockStorage> = |_| {};
        let opts1 = make_checkpoint_options(1, 1);
        sb.checkpoint(cb, &mut ctx1, opts1);

        // Second checkpoint queued but not executed.
        let opts2 = make_checkpoint_options(2, 2);
        sb.checkpoint(cb, &mut ctx2, opts2);

        assert_eq!(sb.queue_depth, 2);
        assert!(sb.is_queue_head(&ctx1));
        assert!(!sb.is_queue_head(&ctx2));
    }

    /// Sets up context for prepare_checkpoint tests (mimics what checkpoint() does).
    fn setup_checkpoint_context(sb: &mut SuperBlock<MockStorage>, ctx: &mut Context<MockStorage>) {
        ctx.caller = Caller::Checkpoint;
        ctx.copy = Some(0);
        sb.enqueue(ctx);
    }

    #[test]
    fn test_prepare_checkpoint_increments_sequence() {
        let mut sb = setup_formatted_superblock();
        let original_sequence = sb.working.sequence;

        let mut ctx = Context::<MockStorage>::new(0, 0);
        setup_checkpoint_context(&mut sb, &mut ctx);
        let opts = make_checkpoint_options(1, 1);

        prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

        assert_eq!(sb.staging.sequence, original_sequence + 1);
    }

    #[test]
    fn test_prepare_checkpoint_chains_parent_checksum() {
        let mut sb = setup_formatted_superblock();
        let working_checksum = sb.working.checksum;

        let mut ctx = Context::<MockStorage>::new(0, 0);
        setup_checkpoint_context(&mut sb, &mut ctx);
        let opts = make_checkpoint_options(1, 1);

        prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

        assert_eq!(sb.staging.parent, working_checksum);
    }

    #[test]
    fn test_prepare_checkpoint_copies_working_state() {
        let mut sb = setup_formatted_superblock();

        let mut ctx = Context::<MockStorage>::new(0, 0);
        setup_checkpoint_context(&mut sb, &mut ctx);
        let opts = make_checkpoint_options(1, 1);

        prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

        // Core fields should be inherited from working.
        assert_eq!(sb.staging.cluster, sb.working.cluster);
        assert_eq!(sb.staging.version, sb.working.version);
    }

    #[test]
    fn test_prepare_checkpoint_updates_vsr_state() {
        let mut sb = setup_formatted_superblock();

        let mut ctx = Context::<MockStorage>::new(0, 0);
        setup_checkpoint_context(&mut sb, &mut ctx);
        let opts = make_checkpoint_options(100, 100);

        prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

        // VsrState should reflect checkpoint options.
        assert_eq!(sb.staging.vsr_state.checkpoint.header.op, 100);
        assert_eq!(sb.staging.vsr_state.commit_max, 100);
    }

    #[test]
    fn test_prepare_checkpoint_with_view_attributes() {
        let mut sb = setup_formatted_superblock();

        let view_headers = ViewChangeArray::root(1);
        let view_attrs = ViewAttributes {
            headers: &view_headers,
            view: 42,
            log_view: 40,
        };

        let header = make_prepare_header(1, 1);

        let opts = CheckpointOptions {
            header,
            view_attributes: Some(view_attrs),
            commit_max: 1,
            sync_op_min: 0,
            sync_op_max: 0,
            manifest_references: (0, 0, 0, 0, 0),
            free_set_acquired_references: (checksum(&[]), 0, 0, 0),
            free_set_released_references: (checksum(&[]), 0, 0, 0),
            client_sessions_references: (checksum(&[]), 0, 0, 0),
            storage_size: constants::DATA_FILE_SIZE_MIN,
            release: crate::vsr::wire::header::Release::ZERO,
        };

        let mut ctx = Context::<MockStorage>::new(0, 0);
        setup_checkpoint_context(&mut sb, &mut ctx);

        prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

        assert_eq!(sb.staging.view_headers_count, 1);
        assert_eq!(sb.staging.vsr_state.view, 42);
        assert_eq!(sb.staging.vsr_state.log_view, 40);
    }

    #[test]
    fn test_prepare_checkpoint_without_view_attributes_preserves_headers() {
        let mut sb = setup_formatted_superblock();
        let original_count = sb.working.view_headers_count;

        let mut ctx = Context::<MockStorage>::new(0, 0);
        setup_checkpoint_context(&mut sb, &mut ctx);
        let opts = make_checkpoint_options(1, 1);

        prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

        // View headers should remain unchanged when view_attributes is None.
        assert_eq!(sb.staging.view_headers_count, original_count);
    }

    #[test]
    fn test_prepare_checkpoint_sets_valid_checksum() {
        let mut sb = setup_formatted_superblock();

        let mut ctx = Context::<MockStorage>::new(0, 0);
        setup_checkpoint_context(&mut sb, &mut ctx);
        let opts = make_checkpoint_options(1, 1);

        prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

        assert!(sb.staging.valid_checksum());
    }

    #[test]
    fn test_prepare_checkpoint_staging_differs_from_working() {
        let mut sb = setup_formatted_superblock();

        let mut ctx = Context::<MockStorage>::new(0, 0);
        setup_checkpoint_context(&mut sb, &mut ctx);
        let opts = make_checkpoint_options(1, 1);

        prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

        // Staging should have advanced sequence and different checksum.
        assert!(sb.staging.sequence > sb.working.sequence);
        assert_ne!(sb.staging.checksum, sb.working.checksum);
    }

    #[test]
    #[should_panic(expected = "sequence overflow")]
    fn test_prepare_checkpoint_panics_on_sequence_overflow() {
        let mut sb = setup_formatted_superblock();
        sb.working.sequence = u64::MAX;
        sb.working.set_checksum();

        let mut ctx = Context::<MockStorage>::new(0, 0);
        setup_checkpoint_context(&mut sb, &mut ctx);
        let opts = make_checkpoint_options(1, 1);

        // Should panic on sequence overflow.
        prepare_checkpoint_helper(&mut sb, &mut ctx, opts);
    }

    // NOTE: test_prepare_checkpoint_panics_on_empty_view_headers is not implemented
    // because ViewChangeArray cannot be constructed with zero length through safe APIs.
    // The assertion `view_attrs.headers.len() > 0` is a defensive check against invariant
    // violations that cannot occur through normal usage.

    proptest! {
        #[test]
        fn prop_checkpoint_sequence_always_advances(
            initial_seq in 1u64..u64::MAX - 1
        ) {
            let mut sb = setup_formatted_superblock();
            sb.working.sequence = initial_seq;
            sb.working.set_checksum();

            let mut ctx = Context::<MockStorage>::new(0, 0);
            setup_checkpoint_context(&mut sb, &mut ctx);
            let opts = make_checkpoint_options(1, 1);

            prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

            prop_assert_eq!(sb.staging.sequence, initial_seq + 1);
        }

        #[test]
        fn prop_checkpoint_parent_chain_integrity(
            initial_seq in 1u64..1000u64
        ) {
            let mut sb = setup_formatted_superblock();
            sb.working.sequence = initial_seq;
            sb.working.set_checksum();
            let expected_parent = sb.working.checksum;

            let mut ctx = Context::<MockStorage>::new(0, 0);
            setup_checkpoint_context(&mut sb, &mut ctx);
            let opts = make_checkpoint_options(1, 1);

            prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

            prop_assert_eq!(sb.staging.parent, expected_parent);
            prop_assert!(sb.staging.valid_checksum());
        }

        #[test]
        fn prop_checkpoint_preserves_cluster(
            cluster in any::<u128>()
        ) {
            let mut sb = setup_formatted_superblock();
            sb.working.cluster = cluster;
            sb.working.view_headers_all = ViewChangeArray::root(cluster);
            sb.working.set_checksum();

            let mut ctx = Context::<MockStorage>::new(0, 0);
            setup_checkpoint_context(&mut sb, &mut ctx);
            let opts = make_checkpoint_options(1, 1);

            prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

            prop_assert_eq!(sb.staging.cluster, cluster);
        }

        #[test]
        fn prop_checkpoint_staging_checksum_valid(
            op in 1u64..10000u64
        ) {
            let mut sb = setup_formatted_superblock();

            let mut ctx = Context::<MockStorage>::new(0, 0);
            setup_checkpoint_context(&mut sb, &mut ctx);
            let opts = make_checkpoint_options(op, op);

            prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

            prop_assert!(sb.staging.valid_checksum());
            prop_assert!(sb.staging.checksum != 0);
        }

        #[test]
        fn prop_checkpoint_staging_inherits_version(
            version in Just(constants::SUPERBLOCK_VERSION)
        ) {
            let mut sb = setup_formatted_superblock();
            sb.working.version = version;
            sb.working.set_checksum();

            let mut ctx = Context::<MockStorage>::new(0, 0);
            setup_checkpoint_context(&mut sb, &mut ctx);
            let opts = make_checkpoint_options(1, 1);

            prepare_checkpoint_helper(&mut sb, &mut ctx, opts);

            prop_assert_eq!(sb.staging.version, version);
        }
    }

    // =========================================================================
    // release() and kick_next() Tests
    // =========================================================================

    #[test]
    fn test_release_dequeues_and_resets_caller() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        // Set up context with callback.
        ctx.caller = Caller::Open;
        ctx.callback = Some(|_ctx| {
            // Callback is invoked; context should already be dequeued.
        });

        sb.enqueue(&mut ctx);
        assert!(sb.is_queue_head(&ctx));
        assert_eq!(sb.queue_depth, 1);

        // Release should dequeue, reset caller, and invoke callback.
        sb.release(&mut ctx);

        // Post-conditions.
        assert_eq!(ctx.caller, Caller::None);
        assert!(!ctx.is_active());
        assert!(ctx.callback.is_none());
        assert_eq!(sb.queue_depth, 0);
        assert!(sb.queue_head.is_none());
    }

    #[test]
    #[should_panic(expected = "ctx.callback.is_some()")]
    fn test_release_panics_without_callback() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        ctx.caller = Caller::Open;
        ctx.callback = None; // Missing callback.

        sb.enqueue(&mut ctx);

        // Should panic: callback is None.
        sb.release(&mut ctx);
    }

    #[test]
    #[should_panic]
    fn test_release_panics_if_not_head() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx1 = Context::<MockStorage>::new(1, 1);
        let mut ctx2 = Context::<MockStorage>::new(2, 2);

        ctx1.caller = Caller::Open;
        ctx1.callback = Some(|_| {});
        ctx2.caller = Caller::Open;
        ctx2.callback = Some(|_| {});

        sb.enqueue(&mut ctx1);
        sb.enqueue(&mut ctx2);

        // ctx2 is not head; should panic.
        sb.release(&mut ctx2);
    }

    #[test]
    fn test_kick_next_empty_queue_is_noop() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);

        // Queue is empty.
        assert!(sb.queue_head.is_none());
        assert_eq!(sb.queue_depth, 0);

        // Should not panic or modify state.
        sb.kick_next();

        assert!(sb.queue_head.is_none());
        assert_eq!(sb.queue_depth, 0);
    }

    #[test]
    fn test_kick_next_dispatches_open() {
        // Pre-populate disk with valid headers.
        let mut storage = MockStorage::new();
        for copy_idx in 0..constants::SUPERBLOCK_COPIES {
            let offset = SUPERBLOCK_COPY_SIZE * copy_idx;
            let mut header = make_header();
            header.copy = copy_idx as u16;
            header.set_checksum();
            let header_bytes = unsafe { as_bytes_unchecked(&header) };
            storage.disk[offset..offset + header_bytes.len()].copy_from_slice(header_bytes);
        }

        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        ctx.caller = Caller::Open;
        ctx.callback = Some(|_| {});
        ctx.copy = None; // Not yet started.

        sb.enqueue(&mut ctx);

        // kick_next should dispatch read_working for Caller::Open.
        sb.kick_next();

        // With sync MockStorage, all reads complete immediately.
        // Verify read_working was called by checking reading buffer has data.
        for copy_idx in 0..constants::SUPERBLOCK_COPIES {
            let read_header = &sb.reading[copy_idx];
            assert_eq!(read_header.copy, copy_idx as u16);
            assert!(read_header.valid_checksum());
        }
    }

    #[test]
    fn test_kick_next_dispatches_write_operations() {
        let mut sb = setup_formatted_superblock();
        let mut ctx = Context::<MockStorage>::new(0, 0);

        // Set up context with state to write.
        ctx.caller = Caller::Checkpoint;
        ctx.callback = Some(|_| {});
        ctx.copy = Some(0);

        let mut vsr_state = sb.working.vsr_state;
        vsr_state.commit_max = 12345;
        ctx.vsr_state = Some(vsr_state);

        sb.enqueue(&mut ctx);

        // kick_next should dispatch prepare_checkpoint -> write_headers.
        sb.kick_next();

        // With MockStorage (sync), data should be written to disk.
        let mut header = SuperBlockHeader::zeroed();
        let header_bytes = unsafe { as_bytes_unchecked_mut(&mut header) };
        header_bytes.copy_from_slice(&sb.storage.disk[0..header_bytes.len()]);

        // Staging should have been updated by prepare_checkpoint.
        assert_eq!(sb.staging.sequence, sb.working.sequence + 1);
        assert_eq!(sb.staging.vsr_state.commit_max, 12345);

        assert_eq!(header.sequence, sb.staging.sequence);
        assert_eq!(header.vsr_state.commit_max, 12345);
    }

    #[test]
    #[should_panic(expected = "Caller::None should not be reached")]
    fn test_kick_next_panics_on_caller_none() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        // Corrupt state: enqueued context has Caller::None.
        ctx.caller = Caller::None;
        ctx.callback = Some(|_| {});

        sb.enqueue(&mut ctx);

        // Should panic: Caller::None is invalid for queued context.
        sb.kick_next();
    }

    #[test]
    fn test_release_then_kick_next_continues_queue() {
        // Pre-populate disk with valid headers so reading produces valid state.
        let mut storage = MockStorage::new();
        for copy_idx in 0..constants::SUPERBLOCK_COPIES {
            let offset = SUPERBLOCK_COPY_SIZE * copy_idx;
            let mut header = make_header();
            header.copy = copy_idx as u16;
            header.set_checksum();
            let header_bytes = unsafe { as_bytes_unchecked(&header) };
            storage.disk[offset..offset + header_bytes.len()].copy_from_slice(header_bytes);
        }

        let mut sb = SuperBlock::new(storage);
        let mut ctx1 = Context::<MockStorage>::new(1, 1);
        let mut ctx2 = Context::<MockStorage>::new(2, 2);

        ctx1.caller = Caller::Open;
        ctx1.callback = Some(|_| {});
        ctx2.caller = Caller::Open;
        ctx2.callback = Some(|_| {});

        sb.enqueue(&mut ctx1);
        sb.enqueue(&mut ctx2);

        assert_eq!(sb.queue_depth, 2);
        assert!(sb.is_queue_head(&ctx1));

        // Release first context; should kick_next and start ctx2.
        sb.release(&mut ctx1);

        // ctx1 should be dequeued and inactive.
        assert!(!ctx1.is_active());
        assert_eq!(ctx1.caller, Caller::None);

        // ctx2 should now be head. With MockStorage (sync), the entire read
        // sequence completes, so copy ends up None after all copies are read.
        // We verify kick_next ran by checking the reading buffer was populated.
        assert_eq!(sb.queue_depth, 1);
        assert!(sb.is_queue_head(&ctx2));

        // Verify read_working was called by checking reading buffer has data.
        // (With sync MockStorage, all copies are read immediately.)
        for copy_idx in 0..constants::SUPERBLOCK_COPIES {
            let read_header = &sb.reading[copy_idx];
            assert_eq!(read_header.copy, copy_idx as u16);
            assert!(read_header.valid_checksum());
        }
    }

    #[test]
    fn test_callback_invoked_after_dequeue() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        // Use atomic to track callback state since fn pointers can't capture.
        static CALLBACK_CTX_WAS_INACTIVE: AtomicBool = AtomicBool::new(false);

        ctx.caller = Caller::Open;
        ctx.callback = Some(|ctx| {
            // At callback time, context should already be dequeued and inactive.
            CALLBACK_CTX_WAS_INACTIVE.store(!ctx.is_active(), Ordering::SeqCst);
        });

        sb.enqueue(&mut ctx);
        sb.release(&mut ctx);

        // Verify callback saw inactive context.
        assert!(CALLBACK_CTX_WAS_INACTIVE.load(Ordering::SeqCst));
    }

    #[test]
    fn test_multiple_operations_fifo_sequence() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);

        const COUNT: usize = 4;
        let mut contexts: [Context<MockStorage>; COUNT] = [
            Context::new(0, 0),
            Context::new(1, 1),
            Context::new(2, 2),
            Context::new(3, 3),
        ];

        // Track completion order using atomics.
        static COMPLETION_ORDER: [AtomicUsize; COUNT] = [
            AtomicUsize::new(usize::MAX),
            AtomicUsize::new(usize::MAX),
            AtomicUsize::new(usize::MAX),
            AtomicUsize::new(usize::MAX),
        ];
        static COMPLETION_IDX: AtomicUsize = AtomicUsize::new(0);

        fn make_callback(id: usize) -> Callback<MockStorage> {
            match id {
                0 => |_| {
                    let idx = COMPLETION_IDX.fetch_add(1, Ordering::SeqCst);
                    COMPLETION_ORDER[idx].store(0, Ordering::SeqCst);
                },
                1 => |_| {
                    let idx = COMPLETION_IDX.fetch_add(1, Ordering::SeqCst);
                    COMPLETION_ORDER[idx].store(1, Ordering::SeqCst);
                },
                2 => |_| {
                    let idx = COMPLETION_IDX.fetch_add(1, Ordering::SeqCst);
                    COMPLETION_ORDER[idx].store(2, Ordering::SeqCst);
                },
                3 => |_| {
                    let idx = COMPLETION_IDX.fetch_add(1, Ordering::SeqCst);
                    COMPLETION_ORDER[idx].store(3, Ordering::SeqCst);
                },
                _ => panic!("unexpected id"),
            }
        }

        // Enqueue all contexts.
        for (i, ctx) in contexts.iter_mut().enumerate() {
            ctx.caller = Caller::Open;
            ctx.callback = Some(make_callback(i));
            sb.enqueue(ctx);
        }

        assert_eq!(sb.queue_depth, COUNT);

        // Release all in FIFO order.
        for ctx in contexts.iter_mut() {
            if sb.is_queue_head(ctx) {
                sb.release(ctx);
            }
        }

        // All should be completed in FIFO order.
        let order: [usize; COUNT] = [
            COMPLETION_ORDER[0].load(Ordering::SeqCst),
            COMPLETION_ORDER[1].load(Ordering::SeqCst),
            COMPLETION_ORDER[2].load(Ordering::SeqCst),
            COMPLETION_ORDER[3].load(Ordering::SeqCst),
        ];
        assert_eq!(order, [0, 1, 2, 3]);
        assert_eq!(COMPLETION_IDX.load(Ordering::SeqCst), COUNT);
        assert_eq!(sb.queue_depth, 0);
    }

    #[test]
    fn test_context_reuse_after_release() {
        let storage = MockStorage::new();
        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        // First operation.
        ctx.caller = Caller::Open;
        ctx.callback = Some(|_| {});
        sb.enqueue(&mut ctx);
        sb.release(&mut ctx);

        // Context should be fully reset.
        assert!(!ctx.is_active());
        assert_eq!(ctx.caller, Caller::None);
        assert!(ctx.callback.is_none());
        assert!(ctx.next.is_none());

        // Reuse same context for second operation.
        ctx.caller = Caller::Open;
        ctx.callback = Some(|_| {});
        sb.enqueue(&mut ctx);

        assert!(ctx.is_active());
        assert!(sb.is_queue_head(&ctx));
        assert_eq!(sb.queue_depth, 1);
    }

    proptest! {
        #[test]
        fn prop_release_always_deactivates_context(
            caller in prop::sample::select(vec![
                Caller::Open,
                Caller::Format,
                Caller::Checkpoint,
                Caller::ViewChange,
            ])
        ) {
            let storage = MockStorage::new();
            let mut sb = SuperBlock::new(storage);
            let mut ctx = Context::<MockStorage>::new(0, 0);

            ctx.caller = caller;
            ctx.callback = Some(|_| {});

            sb.enqueue(&mut ctx);
            sb.release(&mut ctx);

            prop_assert_eq!(ctx.caller, Caller::None);
            prop_assert!(!ctx.is_active());
        }

        #[test]
        fn prop_queue_depth_consistent_after_release(
            enqueue_count in 1usize..=MAX_QUEUE_DEPTH,
            release_count in 0usize..=MAX_QUEUE_DEPTH,
        ) {
            let release_count = release_count.min(enqueue_count);

            let storage = MockStorage::new();
            let mut sb = SuperBlock::new(storage);
            let mut contexts: Vec<Context<MockStorage>> = (0..enqueue_count)
                .map(|i| Context::new(i as u64, i as u64))
                .collect();

            // Enqueue all.
            for ctx in contexts.iter_mut() {
                ctx.caller = Caller::Open;
                ctx.callback = Some(|_| {});
                sb.enqueue(ctx);
            }

            // Release some.
            for ctx in contexts.iter_mut().take(release_count) {
                if sb.is_queue_head(ctx) {
                    sb.release(ctx);
                }
            }

            prop_assert_eq!(sb.queue_depth, enqueue_count - release_count);
        }
    }

    #[test]
    fn test_queued_checkpoint_loses_data() {
        let mut sb = setup_formatted_superblock();
        let mut ctx1 = Context::<MockStorage>::new(1, 1);
        let mut ctx2 = Context::<MockStorage>::new(2, 2);

        let cb: Callback<MockStorage> = |_| {};

        // 1. Start first checkpoint.
        let opts1 = make_checkpoint_options(1, 1);
        sb.checkpoint(cb, &mut ctx1, opts1);

        // 2. Queue second checkpoint.
        let mut opts2 = make_checkpoint_options(2, 2);
        opts2.commit_max = 999;
        sb.checkpoint(cb, &mut ctx2, opts2);

        // Verify ctx2 captured the state intent
        assert!(sb.is_queue_head(&ctx1));
        assert!(!sb.is_queue_head(&ctx2));

        // This assertion will FAIL until the bug is fixed:
        assert!(
            ctx2.vsr_state.is_some(),
            "Context should have captured vsr_state"
        );
        assert_eq!(
            ctx2.vsr_state.unwrap().commit_max,
            999,
            "Context should have captured commit_max"
        );

        // 3. Complete first checkpoint.
        sb.release(&mut ctx1);

        // 4. Now ctx2 becomes head and is kicked.
        // It should have written sequence 3 (assuming sequence 2 was written by ctx1)
        // and correct commit_max.

        // Note: In this mock, ctx1 didn't actually update working because we didn't run the full callbacks.
        // So staging.sequence might be 2 (if it based on working=1).
        // But vsr_state should be from ctx2.

        assert_eq!(
            sb.staging.vsr_state.commit_max, 999,
            "Staging should reflect ctx2 state"
        );
    }

    #[test]
    fn test_release_reentrant_enqueue() {
        // Pre-populate disk with valid headers.
        let mut storage = MockStorage::new();
        for copy_idx in 0..constants::SUPERBLOCK_COPIES {
            let offset = SUPERBLOCK_COPY_SIZE * copy_idx;
            let mut header = make_header();
            header.copy = copy_idx as u16;
            header.set_checksum();
            let header_bytes = unsafe { as_bytes_unchecked(&header) };
            storage.disk[offset..offset + header_bytes.len()].copy_from_slice(header_bytes);
        }

        let mut sb = SuperBlock::new(storage);
        let mut ctx = Context::<MockStorage>::new(0, 0);

        // Setup a callback that re-enqueues the context.
        ctx.caller = Caller::Open;
        ctx.callback = Some(|c| {
            // Re-use context for a new operation immediately.
            c.caller = Caller::Open;
            c.callback = Some(|_| {});
        });

        // Manual simulation of re-entrant enqueue sequence:
        sb.enqueue(&mut ctx);

        // 1. Release starts
        assert!(ctx.callback.is_some());
        let _cb = ctx.callback.take().unwrap();
        ctx.caller = Caller::None;
        sb.dequeue(&mut ctx);
        assert!(!ctx.is_active());

        // 2. Callback runs (simulated) and enqueues ctx again
        ctx.caller = Caller::Open;
        ctx.callback = Some(|_| {});
        sb.enqueue(&mut ctx);

        // 3. Release continues -> kick_next
        sb.kick_next();

        // Verification: ctx should be running again
        assert!(sb.is_queue_head(&ctx));
        // Check if read was dispatched (MockStorage reading buffer populated)
        assert!(sb.reading[0].valid_checksum());
    }
}
