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

struct MessageInner {
    refs: Cell<u32>,
    next_free: Cell<Option<NonNull<MessageInner>>>,
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
        unsafe {
            let layout =
                Layout::from_size_align(constants::MESSAGE_SIZE_MAX_USIZE, constants::SECTOR_SIZE)
                    .expect("layout failed");
            dealloc(self.buf_ptr(), layout);
        }
    }
}

struct MessagePoolInner {
    free_head: Cell<Option<NonNull<MessageInner>>>,
    storage: Vec<Box<MessageInner>>,
}

impl MessagePoolInner {
    #[inline]
    fn pop_free(&self) -> Option<NonNull<MessageInner>> {
        let head = self.free_head.get()?;
        let next = unsafe { head.as_ref().next_free.get() };
        self.free_head.set(next);
        unsafe { head.as_ref().next_free.set(None) };
        Some(head)
    }

    #[inline]
    fn push_free(&self, msg: NonNull<MessageInner>) {
        let head = self.free_head.get();
        unsafe { msg.as_ref().next_free.set(head) };
        self.free_head.set(Some(msg));
    }
}

#[derive(Clone)]
pub struct MessagePool(Rc<MessagePoolInner>);

impl MessagePool {
    pub fn new(capacity: usize) -> Self {
        assert!(constants::HEADER_SIZE <= constants::MESSAGE_SIZE_MAX);
        assert!(constants::SECTOR_SIZE.is_power_of_two());

        let mut storage = Vec::with_capacity(capacity);
        let mut head: Option<NonNull<MessageInner>> = None;

        for _ in 0..capacity {
            let layout =
                Layout::from_size_align(constants::MESSAGE_SIZE_MAX_USIZE, constants::SECTOR_SIZE)
                    .expect("layout bad");
            let ptr = unsafe { alloc_zeroed(layout) };
            let buf = NonNull::new(ptr).expect("alloc failed");

            let msg = Box::new(MessageInner {
                refs: Cell::new(0),
                next_free: Cell::new(None),
                buf,
            });

            let msg_ptr = NonNull::from(msg.as_ref());

            msg.next_free.set(head);
            head = Some(msg_ptr);
            storage.push(msg);
        }

        Self(Rc::new(MessagePoolInner {
            free_head: Cell::new(head),
            storage,
        }))
    }

    // pub fn try_get<C: CommandMarker>(&self) -> Option<>
}

pub struct Message<C: CommandMarker> {
    inner: NonNull<MessageInner>,
    pool: Rc<MessagePoolInner>,
    _marker: PhantomData<C>,
}

impl<C: CommandMarker> Clone for Message<C> {
    fn clone(&self) -> Self {
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

impl<C: CommandMarker> Drop for Message<C> {
    fn drop(&mut self) {
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
    #[inline]
    pub fn is_unique(&self) -> bool {
        unsafe { self.inner.as_ref().refs.get() == 1 }
    }

    #[inline]
    pub fn buffer_ptr(&self) -> *mut u8 {
        unsafe { self.inner.as_ref().buf_ptr() }
    }

    #[inline]
    pub fn header(&self) -> &C::Header {
        unsafe { &*(self.buffer_ptr() as *const C::Header) }
    }

    #[inline]
    pub fn try_header_mut(&mut self) -> Option<&mut C::Header> {
        if !self.is_unique() {
            return None;
        }
        Some(unsafe { &mut *(self.buffer_ptr() as *mut C::Header) })
    }

    #[inline]
    pub fn header_mut(&mut self) -> &mut C::Header {
        self.try_header_mut()
            .expect("message is shared; cannot mutably borrow header")
    }

    #[inline]
    pub fn used_len(&self) -> usize {
        let size = self.header().size() as usize;
        assert!(
            size >= constants::HEADER_SIZE as usize && size <= constants::MESSAGE_SIZE_MAX_USIZE
        );
        size
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        let n = self.used_len();
        unsafe { std::slice::from_raw_parts(self.buffer_ptr(), n) }
    }

    #[inline]
    pub fn try_as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        if !self.is_unique() {
            return None;
        }
        let n = self.used_len();
        Some(unsafe { std::slice::from_raw_parts_mut(self.buffer_ptr(), n) })
    }

    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.try_as_bytes_mut()
            .expect("message is shared; cannot mutably borrow bytes")
    }

    #[inline]
    pub fn body_used(&self) -> &[u8] {
        let n = self.used_len();
        unsafe {
            std::slice::from_raw_parts(
                self.buffer_ptr().add(constants::HEADER_SIZE as usize),
                n - constants::HEADER_SIZE as usize,
            )
        }
    }

    #[inline]
    pub fn try_body_used_mut(&mut self) -> Option<&mut [u8]> {
        if !self.is_unique() {
            return None;
        }
        let n = self.used_len();
        Some(unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer_ptr().add(constants::HEADER_SIZE as usize),
                n - constants::HEADER_SIZE as usize,
            )
        })
    }

    #[inline]
    pub fn body_used_mut(&mut self) -> &mut [u8] {
        self.try_body_used_mut()
            .expect("message is shared; cannot mutably borrow body")
    }

    pub fn set_used_len(&mut self, total: usize) {
        assert!(
            total >= constants::HEADER_SIZE_USIZE && total <= constants::MESSAGE_SIZE_MAX_USIZE
        );
        self.header_mut().set_size(total as u32);
    }

    pub fn reset_header(&mut self) {
        let buf = self.buffer_ptr();
        unsafe { std::ptr::write_bytes(buf, 0, constants::HEADER_SIZE_USIZE) };
        let h = self.header_mut();
        h.set_command(C::COMMAND);
        h.set_size(constants::HEADER_SIZE);
    }
}
