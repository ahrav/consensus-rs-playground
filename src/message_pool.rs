//! Zero-copy, reference-counted message buffer pool.
//!
//! This module provides a pre-allocated pool of fixed-size message buffers
//! optimized for the VSR protocol. Buffers are sector-aligned for direct I/O
//! and use intrusive reference counting for efficient sharing without heap
//! allocation per clone.
//!
//! # Design
//!
//! The pool pre-allocates `capacity` buffers at construction time, each sized
//! to [`constants::MESSAGE_SIZE_MAX`] and aligned to [`constants::SECTOR_SIZE`]
//! for direct I/O compatibility. Messages are acquired from the pool and
//! automatically returned when all references are dropped.
//!
//! # Thread Safety
//!
//! [`MessagePool`] and [`Message`] are **not** thread-safe. They use [`Cell`]
//! and [`Rc`] internally, restricting usage to a single thread. For
//! multi-threaded scenarios, use one pool per thread.
//!
//! # Reference Counting
//!
//! [`Message`] implements [`Clone`] with explicit reference counting:
//! - Cloning increments the reference count (panics on `u32` overflow)
//! - Dropping decrements the count; when it reaches zero, the buffer returns
//!   to the pool's free list
//! - Mutable access requires exclusive ownership ([`Message::is_unique`] == true)

use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    cell::Cell,
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
};

use crate::{
    constants,
    vsr::{command::CommandMarker, header::ProtoHeader},
};

/// Internal state for a pooled message buffer.
///
/// Contains the reference count, free-list linkage, and the actual buffer pointer.
/// This struct is never exposed publicly; users interact via [`Message`].
struct MessageInner {
    /// Current reference count. Zero means the buffer is on the free list.
    refs: Cell<u32>,
    /// Intrusive free-list pointer. Only valid when `refs == 0`.
    next_free: Cell<Option<NonNull<MessageInner>>>,
    /// Pointer to the sector-aligned message buffer.
    buf: NonNull<u8>,
}

impl MessageInner {
    #[inline]
    fn buf_ptr(&self) -> *mut u8 {
        self.buf.as_ptr()
    }
}

impl Drop for MessageInner {
    fn drop(&mut self) {
        // SAFETY: `self.buf` was allocated via `alloc_zeroed` with this exact layout
        // in `MessagePool::new`. The buffer has not been deallocated because
        // `MessageInner` owns it exclusively and this is the Drop impl.
        unsafe {
            let layout =
                Layout::from_size_align(constants::MESSAGE_SIZE_MAX_USIZE, constants::SECTOR_SIZE)
                    .expect("layout failed");
            dealloc(self.buf_ptr(), layout);
        }
    }
}

/// Internal pool storage with intrusive free-list management.
struct MessagePoolInner {
    /// Head of the intrusive free-list. `None` when all buffers are in use.
    free_head: Cell<Option<NonNull<MessageInner>>>,
    /// Owns all `MessageInner` instances for the pool's lifetime.
    ///
    /// We use `Box<[T]>` (boxed slice) instead of `Vec<T>` to strictly enforce
    /// fixed capacity and ensure that `MessageInner` instances have stable
    /// memory addresses for the intrusive linked list.
    _storage: Box<[MessageInner]>,
}

impl MessagePoolInner {
    /// Removes and returns a message from the free list, or `None` if empty.
    #[inline]
    fn pop_free(&self) -> Option<NonNull<MessageInner>> {
        let head = self.free_head.get()?;
        // SAFETY: `head` is a valid pointer to a `MessageInner` in `storage`.
        // All pointers in the free list are valid because:
        // 1. They originate from the stable `Box<[MessageInner]>` in `MessagePool::new`
        // 2. The slice lives in `_storage` and is never moved/dropped while pool exists
        // 3. Single-threaded access (not Send/Sync)
        let next = unsafe { head.as_ref().next_free.get() };
        self.free_head.set(next);
        unsafe { head.as_ref().next_free.set(None) };
        Some(head)
    }

    /// Returns a message to the free list.
    #[inline]
    fn push_free(&self, msg: NonNull<MessageInner>) {
        let head = self.free_head.get();
        // SAFETY: `msg` is a valid pointer to a `MessageInner` in `storage`.
        // The caller (`Message::drop`) ensures `msg` was obtained from this pool.
        unsafe { msg.as_ref().next_free.set(head) };
        self.free_head.set(Some(msg));
    }
}

/// A pool of pre-allocated, sector-aligned message buffers.
///
/// `MessagePool` manages a fixed-capacity collection of message buffers for
/// zero-copy I/O operations. Buffers are allocated once at construction and
/// recycled through an intrusive free list.
///
/// # Cloning
///
/// `MessagePool` is cheaply cloneable (wraps an [`Rc`]). All clones share the
/// same underlying storage and free list.
///
/// # Thread Safety
///
/// This type is `!Send` and `!Sync`. Use one pool per thread.
#[derive(Clone)]
pub struct MessagePool(Rc<MessagePoolInner>);

impl MessagePool {
    /// Creates a new message pool with the specified capacity.
    ///
    /// Allocates `capacity` message buffers upfront. Each buffer is:
    /// - Sized to [`constants::MESSAGE_SIZE_MAX`]
    /// - Aligned to [`constants::SECTOR_SIZE`] for direct I/O
    /// - Zero-initialized
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of message buffers to pre-allocate
    ///
    /// # Panics
    ///
    /// - Panics if memory allocation fails
    /// - Panics if `HEADER_SIZE > MESSAGE_SIZE_MAX` (compile-time invariant)
    /// - Panics if `SECTOR_SIZE` is not a power of two (compile-time invariant)
    pub fn new(capacity: usize) -> Self {
        const _: () = assert!(constants::HEADER_SIZE <= constants::MESSAGE_SIZE_MAX);
        const _: () = assert!(constants::SECTOR_SIZE.is_power_of_two());

        // Allocate metadata and buffers upfront so pool operations are O(1) and
        // buffer addresses remain stable for the intrusive free list.
        let mut storage = Vec::with_capacity(capacity);

        for _ in 0..capacity {
            let layout =
                Layout::from_size_align(constants::MESSAGE_SIZE_MAX_USIZE, constants::SECTOR_SIZE)
                    .expect("layout bad");
            // SAFETY: Layout is valid (size > 0, alignment is power of 2).
            // We check the returned pointer for null below.
            let ptr = unsafe { alloc_zeroed(layout) };
            let buf = NonNull::new(ptr).expect("alloc failed");

            storage.push(MessageInner {
                refs: Cell::new(0),
                next_free: Cell::new(None), // Linked in Phase 2
                buf,
            });
        }

        // Freeze layout to keep `MessageInner` addresses stable for raw pointers.
        let storage = storage.into_boxed_slice();

        // Link all nodes into the free list; initial order is LIFO.
        let mut head: Option<NonNull<MessageInner>> = None;

        for msg in storage.iter() {
            let msg_ptr = NonNull::from(msg);
            msg.next_free.set(head);
            head = Some(msg_ptr);
        }

        Self(Rc::new(MessagePoolInner {
            free_head: Cell::new(head),
            _storage: storage,
        }))
    }

    /// Attempts to acquire a message buffer from the pool.
    ///
    /// Returns `None` if the pool is exhausted (all buffers are in use).
    /// The returned message is initialized with:
    /// - Header bytes zeroed
    /// - Command set to `C::COMMAND`
    /// - Size set to [`constants::HEADER_SIZE`] (header only, no body)
    /// - Protocol set to [`constants::VSR_VERSION`]
    ///
    /// The message has a reference count of 1 (unique ownership).
    pub fn try_get<C: CommandMarker>(&self) -> Option<Message<C>> {
        let inner = self.0.pop_free()?;

        let m = unsafe { inner.as_ref() };
        m.refs.set(1);

        let buf = m.buf_ptr();
        unsafe { std::ptr::write_bytes(buf, 0, constants::HEADER_SIZE_USIZE) };

        let header = unsafe { &mut *(buf as *mut C::Header) };
        header.set_command(C::COMMAND);
        header.set_size(constants::HEADER_SIZE);
        header.set_protocol(constants::VSR_VERSION);

        Some(Message {
            inner,
            pool: self.0.clone(),
            _marker: PhantomData,
        })
    }

    /// Acquires a message buffer from the pool.
    ///
    /// This is a convenience wrapper around [`try_get`](Self::try_get) that panics
    /// if no buffers are available.
    ///
    /// # Panics
    ///
    /// Panics if the pool is exhausted (all buffers are in use).
    pub fn get<C: CommandMarker>(&self) -> Message<C> {
        self.try_get().expect("message pool is empty")
    }
}

/// A reference-counted handle to a pooled message buffer.
///
/// `Message<C>` provides zero-copy access to a pre-allocated buffer from a
/// [`MessagePool`]. The type parameter `C` implements [`CommandMarker`] and
/// determines the header type for type-safe protocol message handling.
///
/// # Ownership Semantics
///
/// - **Cloning**: Increments the internal reference count. The clone shares
///   the same underlying buffer.
/// - **Dropping**: Decrements the reference count. When it reaches zero, the
///   buffer is returned to the pool's free list for reuse.
/// - **Mutable Access**: Methods like [`header_mut`](Self::header_mut) require
///   exclusive ownership ([`is_unique`](Self::is_unique) == true) and panic otherwise.
///
/// # Memory Layout
///
/// The buffer contains:
/// - Bytes `0..HEADER_SIZE`: Protocol header (type determined by `C::Header`)
/// - Bytes `HEADER_SIZE..size`: Message body
/// - Bytes `size..MESSAGE_SIZE_MAX`: Unused padding
pub struct Message<C: CommandMarker> {
    inner: NonNull<MessageInner>,
    pool: Rc<MessagePoolInner>,
    _marker: PhantomData<C>,
}

impl<C: CommandMarker> std::fmt::Debug for Message<C>
where
    C::Header: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Message")
            .field("header", self.header())
            .field("used_len", &self.used_len())
            .field("is_unique", &self.is_unique())
            .finish()
    }
}

/// Cloning a `Message` increments the reference count, creating a shared
/// handle to the same underlying buffer.
impl<C: CommandMarker> Clone for Message<C> {
    /// Creates a new handle to the same message buffer.
    ///
    /// # Panics
    ///
    /// Panics if the reference count would overflow `u32::MAX`.
    fn clone(&self) -> Self {
        // SAFETY: `self.inner` is valid because we hold a reference to the pool
        // which keeps all `MessageInner` instances alive.
        let m = unsafe { self.inner.as_ref() };
        let refs = m.refs.get();
        m.refs
            .set(refs.checked_add(1).expect("message refs overflow"));
        Self {
            inner: self.inner,
            pool: self.pool.clone(),
            _marker: PhantomData,
        }
    }
}

/// Dropping a `Message` decrements the reference count. When the count
/// reaches zero, the buffer is returned to the pool's free list for reuse.
impl<C: CommandMarker> Drop for Message<C> {
    fn drop(&mut self) {
        // SAFETY: `self.inner` is valid because we hold a reference to the pool.
        let m = unsafe { self.inner.as_ref() };
        let refs = m.refs.get();
        if refs == 1 {
            m.refs.set(0);
            self.pool.push_free(self.inner);
        } else {
            m.refs.set(refs - 1);
        }
    }
}

impl<C: CommandMarker> Message<C> {
    /// Returns `true` if this is the only handle to the underlying buffer.
    ///
    /// Mutable access methods require `is_unique() == true`. If this returns
    /// `false`, the message is shared and mutation would be unsafe.
    #[inline]
    pub fn is_unique(&self) -> bool {
        // SAFETY: `self.inner` is valid because we hold a reference to the pool.
        unsafe { self.inner.as_ref().refs.get() == 1 }
    }

    /// Returns a raw pointer to the underlying buffer.
    ///
    /// The pointer is valid for the lifetime of the pool and points to
    /// [`constants::MESSAGE_SIZE_MAX`] bytes of sector-aligned memory.
    ///
    /// # Safety
    ///
    /// The returned pointer is mutable but callers must ensure exclusive
    /// access before writing. Use [`is_unique`](Self::is_unique) to verify.
    #[inline]
    pub fn buffer_ptr(&self) -> *mut u8 {
        // SAFETY: `self.inner` is valid because we hold a reference to the pool.
        unsafe { self.inner.as_ref().buf_ptr() }
    }

    /// Returns a shared reference to the message header.
    ///
    /// The header type is determined by the [`CommandMarker`] type parameter.
    #[inline]
    pub fn header(&self) -> &C::Header {
        // SAFETY: The buffer is properly aligned for `C::Header` (sector-aligned)
        // and initialized. The header region (0..HEADER_SIZE) is always valid.
        unsafe { &*(self.buffer_ptr() as *const C::Header) }
    }

    /// Returns a mutable reference to the header if this message is unique.
    ///
    /// Returns `None` if the message is shared (reference count > 1).
    #[inline]
    pub fn try_header_mut(&mut self) -> Option<&mut C::Header> {
        if !self.is_unique() {
            return None;
        }
        // SAFETY: We verified exclusive ownership above. The buffer is properly
        // aligned and the header region is always valid.
        Some(unsafe { &mut *(self.buffer_ptr() as *mut C::Header) })
    }

    /// Returns a mutable reference to the message header.
    ///
    /// # Panics
    ///
    /// Panics if the message is shared (reference count > 1).
    #[inline]
    pub fn header_mut(&mut self) -> &mut C::Header {
        self.try_header_mut()
            .expect("message is shared; cannot mutably borrow header")
    }

    /// Returns the used portion of the message buffer in bytes.
    ///
    /// This is the value stored in the header's `size` field, which includes
    /// both the header and body.
    ///
    /// # Panics
    ///
    /// Panics if the header's size field is outside the valid range
    /// `[HEADER_SIZE, MESSAGE_SIZE_MAX]`.
    #[inline]
    pub fn used_len(&self) -> usize {
        let size = self.header().size() as usize;
        assert!((constants::HEADER_SIZE_USIZE..=constants::MESSAGE_SIZE_MAX_USIZE).contains(&size));
        size
    }

    /// Returns the used portion of the message as a byte slice.
    ///
    /// The slice includes both the header and body (`0..used_len()`).
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        let n = self.used_len();
        // SAFETY: The buffer is valid for `MESSAGE_SIZE_MAX` bytes and
        // `used_len()` is verified to be within bounds.
        unsafe { std::slice::from_raw_parts(self.buffer_ptr(), n) }
    }

    /// Returns the used portion as a mutable slice if this message is unique.
    ///
    /// Returns `None` if the message is shared (reference count > 1).
    #[inline]
    pub fn try_as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        if !self.is_unique() {
            return None;
        }
        let n = self.used_len();
        // SAFETY: We verified exclusive ownership. Buffer is valid and within bounds.
        Some(unsafe { std::slice::from_raw_parts_mut(self.buffer_ptr(), n) })
    }

    /// Returns the used portion of the message as a mutable byte slice.
    ///
    /// # Panics
    ///
    /// Panics if the message is shared (reference count > 1).
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.try_as_bytes_mut()
            .expect("message is shared; cannot mutably borrow bytes")
    }

    /// Returns the message body (excluding the header) as a byte slice.
    ///
    /// The slice covers `HEADER_SIZE..used_len()`.
    #[inline]
    pub fn body_used(&self) -> &[u8] {
        let n = self.used_len();
        // SAFETY: Buffer is valid for `MESSAGE_SIZE_MAX` bytes. We offset by
        // `HEADER_SIZE` and read `n - HEADER_SIZE` bytes, which is within bounds.
        unsafe {
            std::slice::from_raw_parts(
                self.buffer_ptr().add(constants::HEADER_SIZE as usize),
                n - constants::HEADER_SIZE as usize,
            )
        }
    }

    /// Returns the message body as a mutable slice if this message is unique.
    ///
    /// Returns `None` if the message is shared (reference count > 1).
    #[inline]
    pub fn try_body_used_mut(&mut self) -> Option<&mut [u8]> {
        if !self.is_unique() {
            return None;
        }
        let n = self.used_len();
        // SAFETY: We verified exclusive ownership. Offset and length are within bounds.
        Some(unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer_ptr().add(constants::HEADER_SIZE as usize),
                n - constants::HEADER_SIZE as usize,
            )
        })
    }

    /// Returns the message body as a mutable byte slice.
    ///
    /// # Panics
    ///
    /// Panics if the message is shared (reference count > 1).
    #[inline]
    pub fn body_used_mut(&mut self) -> &mut [u8] {
        self.try_body_used_mut()
            .expect("message is shared; cannot mutably borrow body")
    }

    /// Sets the used length of the message (header + body).
    ///
    /// Updates the header's `size` field to `total`.
    ///
    /// # Panics
    ///
    /// - Panics if `total < HEADER_SIZE` or `total > MESSAGE_SIZE_MAX`
    /// - Panics if the message is shared (via [`header_mut`](Self::header_mut))
    pub fn set_used_len(&mut self, total: usize) {
        assert!(
            (constants::HEADER_SIZE_USIZE..=constants::MESSAGE_SIZE_MAX_USIZE).contains(&total)
        );
        self.header_mut().set_size(total as u32);
    }

    /// Resets the header to a zero-initialized state with default values.
    ///
    /// After calling this method:
    /// - All header bytes are zeroed
    /// - `command` is set to `C::COMMAND`
    /// - `size` is set to `HEADER_SIZE` (header only, no body)
    /// - `protocol` is set to [`constants::VSR_VERSION`]
    ///
    /// # Panics
    ///
    /// Panics if the message is shared (reference count > 1).
    pub fn reset_header(&mut self) {
        assert!(self.is_unique(), "message is shared; cannot reset header");
        let buf = self.buffer_ptr();
        // SAFETY: Buffer is valid for at least `HEADER_SIZE` bytes.
        // We verified exclusive ownership above via is_unique().
        unsafe { std::ptr::write_bytes(buf, 0, constants::HEADER_SIZE_USIZE) };
        // SAFETY: We have exclusive ownership (verified above), so header_mut() won't panic.
        let h = self.header_mut();
        h.set_command(C::COMMAND);
        h.set_size(constants::HEADER_SIZE);
        h.set_protocol(constants::VSR_VERSION);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsr::{Command, command::PrepareCmd, header::ProtoHeader};

    #[test]
    fn pool_zero_capacity() {
        let pool = MessagePool::new(0);
        // Pool with 0 capacity should immediately return None
        assert!(pool.try_get::<PrepareCmd>().is_none());
    }

    #[test]
    fn pool_minimum_capacity() {
        let pool = MessagePool::new(1);
        let msg = pool.try_get::<PrepareCmd>();
        assert!(msg.is_some());
        // Second acquire should fail
        assert!(pool.try_get::<PrepareCmd>().is_none());
    }

    #[test]
    fn buffer_zero_initialized_on_acquire() {
        let pool = MessagePool::new(1);
        let msg: Message<PrepareCmd> = pool.get();
        // The header region should be zeroed except for the fields set by try_get
        // We can't easily check all bytes, but we can verify size/command/protocol are set correctly
        assert_eq!(msg.header().size(), constants::HEADER_SIZE);
        assert_eq!(msg.header().command(), Command::Prepare);
        assert_eq!(msg.header().protocol(), constants::VSR_VERSION);
    }

    #[test]
    fn pool_clone_shares_storage() {
        let pool1 = MessagePool::new(2);
        let pool2 = pool1.clone();

        // Acquire from pool1
        let msg1: Message<PrepareCmd> = pool1.get();
        // pool2 should see reduced availability
        let _msg2: Message<PrepareCmd> = pool2.get();

        // Both pools should now be exhausted
        assert!(pool1.try_get::<PrepareCmd>().is_none());
        assert!(pool2.try_get::<PrepareCmd>().is_none());

        // Drop one message
        drop(msg1);

        // Both pools should now have one available
        assert!(pool1.try_get::<PrepareCmd>().is_some());
    }

    #[test]
    fn acquire_sets_refcount_to_one() {
        let pool = MessagePool::new(1);
        let msg: Message<PrepareCmd> = pool.get();
        assert!(
            msg.is_unique(),
            "Newly acquired message should have refcount 1"
        );
    }

    #[test]
    fn clone_increments_refcount() {
        let pool = MessagePool::new(1);
        let msg1: Message<PrepareCmd> = pool.get();
        assert!(msg1.is_unique());

        let msg2 = msg1.clone();
        assert!(
            !msg1.is_unique(),
            "After clone, original should not be unique"
        );
        assert!(!msg2.is_unique(), "After clone, clone should not be unique");

        drop(msg2);
        assert!(
            msg1.is_unique(),
            "After dropping clone, original should be unique again"
        );
    }

    #[test]
    fn drop_returns_to_pool_at_zero() {
        let pool = MessagePool::new(1);

        let msg: Message<PrepareCmd> = pool.get();
        assert!(pool.try_get::<PrepareCmd>().is_none());

        drop(msg);
        // After drop, buffer should be back in pool
        assert!(pool.try_get::<PrepareCmd>().is_some());
    }

    #[test]
    fn multiple_clones_all_dropped_recycled() {
        let pool = MessagePool::new(1);
        let msg1: Message<PrepareCmd> = pool.get();
        let msg2 = msg1.clone();
        let msg3 = msg1.clone();
        let msg4 = msg2.clone();

        assert!(pool.try_get::<PrepareCmd>().is_none());

        drop(msg1);
        assert!(pool.try_get::<PrepareCmd>().is_none()); // Still 3 refs

        drop(msg2);
        assert!(pool.try_get::<PrepareCmd>().is_none()); // Still 2 refs

        drop(msg3);
        assert!(pool.try_get::<PrepareCmd>().is_none()); // Still 1 ref

        drop(msg4);
        // All refs dropped, buffer should be recycled
        assert!(pool.try_get::<PrepareCmd>().is_some());
    }

    #[test]
    fn exhausted_pool_try_get_returns_none() {
        let pool = MessagePool::new(2);
        let _m1: Message<PrepareCmd> = pool.get();
        let _m2: Message<PrepareCmd> = pool.get();

        assert!(pool.try_get::<PrepareCmd>().is_none());
    }

    #[test]
    fn buffer_reacquirable_after_release() {
        let pool = MessagePool::new(1);

        for _ in 0..100 {
            let msg: Message<PrepareCmd> = pool.get();
            drop(msg);
        }
        // Should still work after many cycles
        assert!(pool.try_get::<PrepareCmd>().is_some());
    }

    #[test]
    #[should_panic(expected = "message pool is empty")]
    fn exhausted_pool_get_panics() {
        let pool = MessagePool::new(1);
        let _m1: Message<PrepareCmd> = pool.get();
        let _m2: Message<PrepareCmd> = pool.get(); // Should panic
    }

    #[test]
    #[should_panic(expected = "message refs overflow")]
    fn refcount_overflow_panics() {
        let pool = MessagePool::new(1);
        let msg: Message<PrepareCmd> = pool.get();

        // Artificially set refcount to MAX
        unsafe {
            msg.inner.as_ref().refs.set(u32::MAX);
        }

        // Clone should panic
        let _clone = msg.clone();
    }

    #[test]
    #[should_panic(expected = "message is shared")]
    fn header_mut_panics_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();

        // Should panic because message is shared
        let _ = msg1.header_mut();
    }

    #[test]
    #[should_panic(expected = "message is shared")]
    fn as_bytes_mut_panics_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();

        let _ = msg1.as_bytes_mut();
    }

    #[test]
    #[should_panic(expected = "message is shared")]
    fn body_used_mut_panics_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();

        let _ = msg1.body_used_mut();
    }

    #[test]
    #[should_panic(expected = "message is shared; cannot mutably borrow header")]
    fn set_used_len_panics_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();

        msg1.set_used_len(constants::HEADER_SIZE_USIZE);
    }

    #[test]
    #[should_panic(expected = "message is shared; cannot reset header")]
    fn reset_header_panics_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();

        // Should panic because message is shared
        msg1.reset_header();
    }

    #[test]
    #[should_panic]
    fn used_len_panics_below_header_size() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        // Force an invalid size
        msg.header_mut().set_size(constants::HEADER_SIZE - 1);

        // used_len should panic
        let _ = msg.used_len();
    }

    #[test]
    #[should_panic]
    fn used_len_panics_above_max_size() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        // Force an invalid size
        msg.header_mut().set_size(constants::MESSAGE_SIZE_MAX + 1);

        // used_len should panic
        let _ = msg.used_len();
    }

    #[test]
    #[should_panic]
    fn set_used_len_panics_below_header_size() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        msg.set_used_len(constants::HEADER_SIZE_USIZE - 1);
    }

    #[test]
    #[should_panic]
    fn set_used_len_panics_above_max_size() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        msg.set_used_len(constants::MESSAGE_SIZE_MAX_USIZE + 1);
    }

    #[test]
    fn body_used_returns_correct_slice() {
        let pool = MessagePool::new(1);
        let msg: Message<PrepareCmd> = pool.get();

        // Default: header only, no body
        assert_eq!(msg.body_used().len(), 0);
    }

    #[test]
    fn try_header_mut_returns_some_when_unique() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();
        assert!(msg.try_header_mut().is_some());
    }

    #[test]
    fn try_header_mut_returns_none_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();
        assert!(msg1.try_header_mut().is_none());
    }

    #[test]
    fn try_as_bytes_mut_returns_some_when_unique() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();
        let bytes = msg.try_as_bytes_mut();
        assert!(bytes.is_some());
        assert_eq!(bytes.unwrap().len(), constants::HEADER_SIZE_USIZE);
    }

    #[test]
    fn try_as_bytes_mut_returns_none_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        let _msg2 = msg1.clone();
        assert!(msg1.try_as_bytes_mut().is_none());
    }

    #[test]
    fn try_body_used_mut_returns_some_when_unique() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();
        msg.set_used_len(constants::HEADER_SIZE_USIZE + 4);
        let body = msg.try_body_used_mut();
        assert!(body.is_some());
        assert_eq!(body.unwrap().len(), 4);
    }

    #[test]
    fn try_body_used_mut_returns_none_when_shared() {
        let pool = MessagePool::new(1);
        let mut msg1: Message<PrepareCmd> = pool.get();
        msg1.set_used_len(constants::HEADER_SIZE_USIZE + 4);
        let _msg2 = msg1.clone();
        assert!(msg1.try_body_used_mut().is_none());
    }

    #[test]
    fn body_used_returns_correct_slice_after_resize() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        let body_len = 256;
        msg.set_used_len(constants::HEADER_SIZE_USIZE + body_len);

        let body = msg.body_used();
        assert_eq!(body.len(), body_len);

        // Verify body starts at the right offset
        let full = msg.as_bytes();
        assert_eq!(&full[constants::HEADER_SIZE_USIZE..], body);
    }

    #[test]
    fn stress_acquire_release_cycles() {
        let pool = MessagePool::new(8);

        for _ in 0..100 {
            let mut handles: Vec<Message<PrepareCmd>> = Vec::new();

            // Acquire all
            while let Some(m) = pool.try_get::<PrepareCmd>() {
                handles.push(m);
            }

            // Release all (drop)
            handles.clear();
        }

        // Pool should be in clean state
        assert!(pool.try_get::<PrepareCmd>().is_some());
    }

    #[test]
    fn stress_clone_drop_interleaved() {
        let pool = MessagePool::new(4);

        for _ in 0..10 {
            let msg1: Message<PrepareCmd> = pool.get();
            let msg2 = msg1.clone();
            let msg3 = msg2.clone();

            drop(msg1);
            let msg4 = msg3.clone();
            drop(msg2);
            drop(msg4);
            drop(msg3);
        }

        // All messages should be returned
        let mut handles = Vec::new();
        while let Some(m) = pool.try_get::<PrepareCmd>() {
            handles.push(m);
        }
        assert_eq!(handles.len(), 4);
    }

    #[test]
    fn pool_drop_after_messages_dropped() {
        let msg: Message<PrepareCmd>;
        {
            let pool = MessagePool::new(1);
            msg = pool.get();
            // Pool dropped here, but msg holds Rc to pool inner
        }
        // Message should still be usable because Rc keeps pool alive
        assert!(msg.is_unique());
        assert_eq!(msg.header().command(), Command::Prepare);
    }

    #[test]
    fn message_with_header_prepare_full_lifecycle() {
        let pool = MessagePool::new(2);

        // Acquire
        let mut msg: Message<PrepareCmd> = pool.get();

        // Verify initialization
        assert_eq!(msg.header().command(), Command::Prepare);
        assert_eq!(msg.header().size(), constants::HEADER_SIZE);
        assert_eq!(msg.header().protocol(), constants::VSR_VERSION);

        // Modify header
        msg.header_mut().set_size(constants::HEADER_SIZE + 64);

        // Clone and verify sharing
        let msg2 = msg.clone();
        assert!(!msg.is_unique());
        assert_eq!(msg.header().size(), msg2.header().size());

        // Drop clone, verify original is unique again
        drop(msg2);
        assert!(msg.is_unique());

        // Reset header
        msg.reset_header();
        assert_eq!(msg.header().size(), constants::HEADER_SIZE);

        // Drop and reacquire
        let ptr = msg.buffer_ptr();
        drop(msg);

        let msg3: Message<PrepareCmd> = pool.get();
        // Due to LIFO, should get same buffer back
        assert_eq!(msg3.buffer_ptr(), ptr);
    }

    #[test]
    fn pool_clone_message_lifecycle() {
        let pool1 = MessagePool::new(4);
        let pool2 = pool1.clone();

        // Acquire from pool1
        let msg1: Message<PrepareCmd> = pool1.get();
        let msg2: Message<PrepareCmd> = pool2.get();

        // Both should be unique initially
        assert!(msg1.is_unique());
        assert!(msg2.is_unique());

        // Different buffers
        assert_ne!(msg1.buffer_ptr(), msg2.buffer_ptr());

        // Release both
        drop(msg1);
        drop(msg2);

        // Both pools see the freed buffers
        let msg3 = pool1.try_get::<PrepareCmd>();
        let msg4 = pool2.try_get::<PrepareCmd>();
        assert!(msg3.is_some());
        assert!(msg4.is_some());
    }

    struct RequestCmd;
    impl crate::vsr::command::CommandMarker for RequestCmd {
        const COMMAND: Command = Command::Request;
        type Header = crate::vsr::HeaderPrepare; // Reuse layout for test
    }

    #[test]
    fn reuse_with_different_command_type() {
        let pool = MessagePool::new(1);

        // 1. Get as Prepare
        let mut msg1: Message<PrepareCmd> = pool.get();
        msg1.header_mut().set_size(100);
        drop(msg1);

        // 2. Get as Request
        let msg2: Message<RequestCmd> = pool.get();

        // Header should be re-initialized correctly
        assert_eq!(msg2.header().command(), Command::Request);
        assert_eq!(msg2.header().size(), constants::HEADER_SIZE);
        assert_eq!(msg2.header().protocol(), constants::VSR_VERSION);
    }

    #[test]
    fn body_persistence_dirty_reuse() {
        let pool = MessagePool::new(1);
        let mut msg: Message<PrepareCmd> = pool.get();

        // Write to body
        let body_len = 32;
        msg.set_used_len(constants::HEADER_SIZE_USIZE + body_len);
        msg.body_used_mut().fill(0xAA);

        // Verify write
        assert_eq!(msg.body_used()[0], 0xAA);

        drop(msg);

        // Reacquire
        let mut msg2: Message<PrepareCmd> = pool.get();

        // Header is reset (size == HEADER_SIZE)
        assert_eq!(msg2.used_len(), constants::HEADER_SIZE_USIZE);

        // But underlying buffer still has data if we look
        // We need to extend size to see it safely via API
        msg2.set_used_len(constants::HEADER_SIZE_USIZE + body_len);

        // Data should still be there (Zero-copy / dirty reuse)
        assert_eq!(msg2.body_used()[0], 0xAA);
    }

    #[test]
    fn message_debug_impl() {
        let pool = MessagePool::new(1);
        let msg: Message<PrepareCmd> = pool.get();
        let s = format!("{:?}", msg);
        assert!(s.contains("Message"));
        assert!(s.contains("Prepare"));
        assert!(s.contains("used_len"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::vsr::command::PrepareCmd;
    use proptest::prelude::*;

    // Keep proptests small: each buffer is 1 MiB, so allocations scale quickly.
    const PROPTEST_CASES: u32 = 8;
    const PROPTEST_CAPACITY_MAX: usize = 4;
    const PROPTEST_HANDLES_MAX: usize = 8;
    const PROPTEST_OPS_MAX: usize = 16;

    /// Operations that can be performed on a pool
    #[derive(Debug, Clone)]
    enum PoolOp {
        Acquire,
        Release(usize),     // Index into held handles
        CloneHandle(usize), // Clone handle at index
    }

    fn pool_op_strategy(max_handles: usize) -> impl Strategy<Value = PoolOp> {
        prop_oneof![
            3 => Just(PoolOp::Acquire),
            2 => (0..max_handles).prop_map(PoolOp::Release),
            1 => (0..max_handles).prop_map(PoolOp::CloneHandle),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        /// Property: After any sequence of acquire/release/clone operations,
        /// the pool maintains the invariant that all buffers are either in-use or available.
        #[test]
        fn prop_pool_conservation(
            capacity in 1usize..=PROPTEST_CAPACITY_MAX,
            ops in prop::collection::vec(pool_op_strategy(PROPTEST_HANDLES_MAX), 0..=PROPTEST_OPS_MAX)
        ) {
            let pool = MessagePool::new(capacity);
            let mut handles: Vec<Option<Message<PrepareCmd>>> = Vec::new();

            for op in ops {
                match op {
                    PoolOp::Acquire => {
                        if let Some(m) = pool.try_get::<PrepareCmd>() {
                            handles.push(Some(m));
                        }
                    }
                    PoolOp::Release(idx) => {
                        if idx < handles.len() {
                            handles[idx] = None;
                        }
                    }
                    PoolOp::CloneHandle(idx) => {
                        if idx < handles.len()
                            && let Some(ref h) = handles[idx]
                        {
                            handles.push(Some(h.clone()));
                        }
                    }
                }
            }

            // Drop all and verify conservation
            handles.clear();

            // All buffers should be available now
            let mut available = Vec::new();
            while let Some(m) = pool.try_get::<PrepareCmd>() {
                available.push(m);
            }
            prop_assert_eq!(available.len(), capacity, "Pool capacity not conserved");
        }

        /// Property: Header is always correctly initialized after try_get
        #[test]
        fn prop_header_initialized_correctly(capacity in 1usize..=PROPTEST_CAPACITY_MAX) {
            let pool = MessagePool::new(capacity);

            for _ in 0..capacity {
                let msg: Message<PrepareCmd> = pool.get();
                prop_assert_eq!(
                    msg.header().command(),
                    crate::vsr::Command::Prepare,
                    "Command not set correctly"
                );
                prop_assert_eq!(
                    msg.header().size(),
                    constants::HEADER_SIZE,
                    "Size not set correctly"
                );
                prop_assert_eq!(
                    msg.header().protocol(),
                    constants::VSR_VERSION,
                    "Protocol not set correctly"
                );
            }
        }

        /// Property: set_used_len and used_len are inverse operations
        #[test]
        fn prop_used_len_roundtrip(
            len in constants::HEADER_SIZE_USIZE..=constants::MESSAGE_SIZE_MAX_USIZE
        ) {
            let pool = MessagePool::new(1);
            let mut msg: Message<PrepareCmd> = pool.get();

            msg.set_used_len(len);
            prop_assert_eq!(msg.used_len(), len);
            prop_assert_eq!(msg.as_bytes().len(), len);
        }

        /// Property: Free-list is LIFO (last released = first acquired)
        #[test]
        fn prop_free_list_lifo(_seed in 0u32..100) {
            let pool = MessagePool::new(4);

            // Acquire all 4
            let msg0: Message<PrepareCmd> = pool.get();
            let msg1: Message<PrepareCmd> = pool.get();
            let msg2: Message<PrepareCmd> = pool.get();
            let msg3: Message<PrepareCmd> = pool.get();

            let ptr0 = msg0.buffer_ptr();
            let ptr1 = msg1.buffer_ptr();
            let ptr2 = msg2.buffer_ptr();
            let ptr3 = msg3.buffer_ptr();

            // Release in order: 0, 1, 2, 3
            drop(msg0);
            drop(msg1);
            drop(msg2);
            drop(msg3);

            // LIFO: should get back in reverse order: 3, 2, 1, 0
            let re0: Message<PrepareCmd> = pool.get();
            let re1: Message<PrepareCmd> = pool.get();
            let re2: Message<PrepareCmd> = pool.get();
            let re3: Message<PrepareCmd> = pool.get();

            prop_assert_eq!(re0.buffer_ptr(), ptr3, "First reacquired should be last released");
            prop_assert_eq!(re1.buffer_ptr(), ptr2);
            prop_assert_eq!(re2.buffer_ptr(), ptr1);
            prop_assert_eq!(re3.buffer_ptr(), ptr0, "Last reacquired should be first released");
        }

        /// Property: Buffer alignment is always SECTOR_SIZE
        #[test]
        fn prop_buffer_alignment(capacity in 1usize..=PROPTEST_CAPACITY_MAX) {
            let pool = MessagePool::new(capacity);
            let mut handles: Vec<Message<PrepareCmd>> = Vec::new();

            for _ in 0..capacity {
                let msg = pool.get();
                let ptr = msg.buffer_ptr() as usize;
                prop_assert_eq!(
                    ptr % constants::SECTOR_SIZE,
                    0,
                    "Buffer not sector-aligned: {:#x}",
                    ptr
                );
                handles.push(msg);
            }
        }

        /// Property: Clone preserves buffer pointer identity
        #[test]
        fn prop_clone_preserves_pointer(_seed in 0u32..1000) {
            let pool = MessagePool::new(4);
            let msg1: Message<PrepareCmd> = pool.get();
            let ptr1 = msg1.buffer_ptr();

            let msg2 = msg1.clone();
            let msg3 = msg2.clone();

            prop_assert_eq!(msg1.buffer_ptr(), ptr1);
            prop_assert_eq!(msg2.buffer_ptr(), ptr1);
            prop_assert_eq!(msg3.buffer_ptr(), ptr1);
        }

        /// Property: After reset_header, header is in initial state
        #[test]
        fn prop_reset_header_restores_initial(
            new_size in constants::HEADER_SIZE_USIZE..=constants::MESSAGE_SIZE_MAX_USIZE
        ) {
            let pool = MessagePool::new(1);
            let mut msg: Message<PrepareCmd> = pool.get();

            // Modify header
            msg.header_mut().set_size(new_size as u32);

            // Reset
            msg.reset_header();

            // Verify initial state
            prop_assert_eq!(msg.header().size(), constants::HEADER_SIZE);
            prop_assert_eq!(msg.header().command(), crate::vsr::Command::Prepare);
            prop_assert_eq!(msg.header().protocol(), constants::VSR_VERSION);
        }
    }
}
