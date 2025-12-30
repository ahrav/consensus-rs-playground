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
#[path = "message_pool_tests.rs"]
mod message_pool_tests;
