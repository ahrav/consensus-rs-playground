use core::marker::PhantomData;
use core::ptr::NonNull;

// Compile-time: verify u32 fits in usize
const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

/// Intrusive single-link for a FIFO queue.
///
/// A node can be in at most one `Queue<_, Tag>` at a time for a given `Tag`.
#[derive(Debug)]
pub struct QueueLink<T, Tag> {
    next: Option<NonNull<T>>,
    _tag: PhantomData<Tag>,
}

impl<T, Tag> QueueLink<T, Tag> {
    /// Create a new `QueueLink`.
    pub const fn new() -> Self {
        Self {
            next: None,
            _tag: PhantomData,
        }
    }

    #[inline]
    pub fn is_unlinked(&self) -> bool {
        self.next.is_none()
    }

    pub fn reset(&mut self) {
        self.next = None;

        assert!(self.is_unlinked());
    }
}
