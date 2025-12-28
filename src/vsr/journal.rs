use core::mem::MaybeUninit;
use core::ptr;

use crate::container_of;
use crate::vsr::iops::IOPSType;
use crate::vsr::journal_primitives::Ring;
use crate::vsr::storage::{Storage, Synchronicity};

#[repr(C)]
pub struct Range<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> {
    pub completion: MaybeUninit<S::Write>,
    pub callback: fn(*mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>),
    pub buffer_ptr: *const u8,
    pub buffer_len: usize,
    pub ring: Ring,
    pub offset: u64,

    /// Linked-list of other writes waiting for this range to complete.
    pub next: *mut Range<S, WRITE_OPS, WRITE_OPS_WORDS>,
    /// True if Storage.write_sectors() operation is in progress for this buffer/offset.
    pub locked: bool,
}

impl<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize>
    Range<S, WRITE_OPS, WRITE_OPS_WORDS>
{
    #[inline]
    pub fn new(
        callback: fn(*mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>),
        buffer: &[u8],
        ring: Ring,
        offset: u64,
    ) -> Self {
        Self {
            completion: MaybeUninit::uninit(),
            callback,
            buffer_ptr: buffer.as_ptr(),
            buffer_len: buffer.len(),
            ring,
            offset,
            next: ptr::null_mut(),
            locked: false,
        }
    }

    #[inline]
    pub fn buffer(&self) -> &[u8] {
        assert!(self.buffer_ptr.is_aligned());
        assert!(!self.buffer_ptr.is_null());
        assert!(self.buffer_len > 0);
        unsafe { core::slice::from_raw_parts(self.buffer_ptr, self.buffer_len) }
    }

    #[inline]
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.ring != other.ring {
            return false;
        }

        let len_a = self.buffer_len as u64;
        let len_b = other.buffer_len as u64;

        if self.offset < other.offset {
            self.offset + len_a > other.offset
        } else {
            other.offset + len_b > self.offset
        }
    }
}

/// Minimal write object for the locking machinery.
///
/// In the full Journal port, `Write` will have more fields (message pointer, high-level callback, etc),
/// but for this piece we only need:
/// - `journal` back-pointer (for completion callback)
/// - `range` (for overlap locking & write_sectors callback)
/// - a "slot identity" for the assert that we never write the same slot concurrently
#[repr(C)]
pub struct Write<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> {
    pub journal: *mut Journal<S, WRITE_OPS, WRITE_OPS_WORDS>,
    // TODO: This will be replaced.
    pub slot_index: u32,
    pub range: Range<S, WRITE_OPS, WRITE_OPS_WORDS>,
}

/// Journal skeleton that only includes what we need for range-locking.
///
/// In the full Journal port, this will include lots more state (headers/prepares index arrays, bitsets, etc).
pub struct Journal<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize> {
    pub storage: *mut S,
    pub replica: u8,

    pub writes: IOPSType<Write<S, WRITE_OPS, WRITE_OPS_WORDS>, WRITE_OPS, WRITE_OPS_WORDS>,
}

impl<S: Storage, const WRITE_OPS: usize, const WRITE_OPS_WORDS: usize>
    Journal<S, WRITE_OPS, WRITE_OPS_WORDS>
{
    pub fn new(storage: *mut S, replica: u8) -> Self {
        Self {
            storage,
            replica,
            writes: IOPSType::default(),
        }
    }

    pub unsafe fn write_sectors(
        &mut self,
        write: *mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>,
        callback: fn(*mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>),
        buffer: &[u8],
        ring: Ring,
        offset: u64,
    ) {
        unsafe { ptr::addr_of_mut!((*write).journal).write(self as *mut _) };
        unsafe {
            ptr::addr_of_mut!((*write).range).write(Range::new(callback, buffer, ring, offset))
        };
        self.lock_sectors(write);
    }

    fn lock_sectors(&mut self, write: *mut Write<S, WRITE_OPS, WRITE_OPS_WORDS>) {
        unsafe {
            assert!(!(*write).range.locked);
            assert!((*write).range.next.is_null());
        }

        let mut it = self.writes.iterate();
        while let Some(other) = it.next() {
            if other == write {
                continue;
            }

            unsafe {
                assert!((*other).slot_index != (*write).slot_index);

                if !(*other).range.locked {
                    continue;
                }

                if (*other).range.overlaps(&(*write).range) {
                    assert!((*other).range.offset == (*write).range.offset);
                    assert!((*other).range.buffer_len == (*write).range.buffer_len);
                    assert!((*other).range.ring == (*write).range.ring);
                    assert!((*other).range.ring == Ring::Headers);

                    // Append to tail of wait list.
                    let mut tail: *mut Range<S, WRITE_OPS, WRITE_OPS_WORDS> =
                        ptr::addr_of_mut!((*other).range);
                    while !(*tail).next.is_null() {
                        tail = (*tail).next;
                    }
                    (*tail).next = ptr::addr_of_mut!((*write).range);
                    return;
                }
            }
        }

        unsafe {
            (*write).range.locked = true;

            let zone = match (*write).range.ring {
                Ring::Headers => S::WAL_HEADERS_ZONE,
                Ring::Prepares => S::WAL_PREPARES_ZONE,
            };

            let storage = &mut *self.storage;
            storage.write_sectors(
                Self::write_sectors_on_write,
                &mut *(*write).range.completion.as_mut_ptr(),
                (*write).range.buffer(),
                zone,
                (*write).range.offset,
            );

            match S::SYNCHRONICITY {
                Synchronicity::AlwaysAsynchronous => assert!(!(*write).range.locked),
                Synchronicity::AlwaysSynchronous => assert!((*write).range.locked),
            }
        };
    }

    fn write_sectors_on_write(completion: &mut S::Write) {
        let range: *mut Range<S, WRITE_OPS, WRITE_OPS_WORDS> =
            container_of!(completion, Range<S, WRITE_OPS, WRITE_OPS_WORDS>, completion);
        let write: *mut Write<S, WRITE_OPS, WRITE_OPS_WORDS> =
            container_of!(range, Write<S, WRITE_OPS, WRITE_OPS_WORDS>, range);

        unsafe {
            let journal: *mut Journal<S, WRITE_OPS, WRITE_OPS_WORDS> = (*write).journal;

            assert!((*write).range.locked);
            (*write).range.locked = false;

            // Drain waiters.
            let mut current = (*range).next;
            (*range).next = ptr::null_mut();

            while !current.is_null() {
                let waiting = current;
                assert!((*waiting).locked == false);

                current = (*waiting).next;
                (*waiting).next = ptr::null_mut();

                let waiting_write: *mut Write<S, WRITE_OPS, WRITE_OPS_WORDS> =
                    container_of!(waiting, Write<S, WRITE_OPS, WRITE_OPS_WORDS>, range);

                (&mut *journal).lock_sectors(waiting_write);
            }

            (((*range).callback)(write));
        };
    }
}
