use std::{mem::MaybeUninit, usize};

// Compile-time proof that u32 -> usize is safe on this platform.
// This fails to compile on 16-bit platforms.
const _: () = assert!(
    std::mem::size_of::<usize>() >= std::mem::size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

#[inline(always)]
fn index(i: u32) -> usize {
    i as usize
}

/// Fixed-capacity ring buffer backed by stack-allocated storage.
///
/// Values are stored in-place using `MaybeUninit` so no heap allocation is
/// required. Capacity is a const generic known at compile time; insertion past
/// capacity is a logic error unless handled via `push_back`.
pub struct RingBuffer<T, const N: usize> {
    buf: [MaybeUninit<T>; N],
    head: u32,
    len: u32,
}

fn uninit_array<T, const N: usize>() -> [MaybeUninit<T>; N] {
    // SAFETY: An uninitialized MaybeUninit<T> is valid.
    unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() }
}

impl<T, const N: usize> RingBuffer<T, N> {
    const CAPACITY: u32 = {
        assert!(N > 0, "RingBuffer capacity must be > 0");
        assert!(
            N <= u32::MAX as usize / 2,
            "N must fit in u32 and not risk overflow"
        );
        N as u32
    };

    /// Constructs an empty ring buffer with capacity `N` without heap
    /// allocation.
    pub fn new() -> Self {
        let _ = Self::CAPACITY;

        let ring = Self {
            buf: uninit_array(),
            head: 0,
            len: 0,
        };

        assert!(ring.len == 0);
        assert!(ring.head == 0);

        ring
    }

    #[inline]
    pub fn capacity(&self) -> u32 {
        Self::CAPACITY
    }

    /// Number of initialized elements currently stored.
    #[inline]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Returns true when no elements are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true when `len == capacity`.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == Self::CAPACITY
    }

    /// Attempts to append `value`, returning `Err(value)` if the buffer is
    /// already full.
    ///
    /// This keeps ownership with the caller on overflow instead of dropping
    /// silently.
    pub fn push_back(&mut self, value: T) -> Result<(), T> {
        if self.is_full() {
            return Err(value);
        }
        self.push_back_assume_capacity(value);
        Ok(())
    }

    /// Appends `value` assuming spare capacity exists.
    ///
    /// # Panics
    ///
    /// Panics (debug-asserts) if the buffer is full. Use `push_back` when the
    /// caller cannot guarantee capacity.
    pub fn push_back_assume_capacity(&mut self, value: T) {
        assert!(self.len < Self::CAPACITY);
        assert!(self.head < Self::CAPACITY);

        let tail = (self.head + self.len) % Self::CAPACITY;

        assert!(tail < Self::CAPACITY);
        assert!(self.len < Self::CAPACITY);

        self.buf[index(tail)].write(value);
        self.len += 1;

        assert!(self.len <= Self::CAPACITY);
        assert!(self.len > 0);
    }

    /// Removes and returns the oldest element, or `None` when empty.
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        assert!(self.len > 0);
        assert!(self.head <= Self::CAPACITY);

        let idx = self.head;
        assert!(idx < Self::CAPACITY);

        // SAFETY: idx < CAPACITY proven, element initialized because len > 0
        let value = unsafe { self.buf[index(idx)].as_ptr().read() };

        self.head = (self.head + 1) % Self::CAPACITY;
        self.len -= 1;

        assert!(self.head < Self::CAPACITY);

        Some(value)
    }

    /// Borrows the oldest element without removal.
    pub fn front(&mut self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        assert!(self.len > 0);
        assert!(self.head <= Self::CAPACITY);

        let idx = self.head;
        assert!(idx < Self::CAPACITY);

        // SAFETY: idx < CAPACITY proven, element initialized because len > 0
        Some(unsafe { &*self.buf[index(idx)].as_ptr() })
    }

    /// Mutably borrows the oldest element without removal.
    ///
    /// Useful for in-place updates while preserving position.
    pub fn front_mut(&mut self) -> Option<&mut T> {
        if self.is_empty() {
            return None;
        }

        assert!(self.len > 0);
        assert!(self.head <= Self::CAPACITY);

        let idx = self.head;
        assert!(idx < Self::CAPACITY);

        // SAFETY: idx < CAPACITY proven, element initialized because len > 0
        Some(unsafe { &mut *self.buf[index(idx)].as_mut_ptr() })
    }

    /// Returns a reference to the element at logical index `logical_idx`.
    ///
    /// Index `0` refers to the current front; indices grow toward the back,
    /// even after wraparound.
    pub fn get(&self, logical_idx: u32) -> Option<&T> {
        if logical_idx >= self.len {
            return None;
        }

        assert!(logical_idx < self.len);
        assert!(self.head < Self::CAPACITY);
        assert!(self.len <= Self::CAPACITY);

        let idx = (self.head + logical_idx) % Self::CAPACITY;
        assert!(idx < Self::CAPACITY);

        // SAFETY: idx < CAPACITY proven, element initialized because len > 0
        Some(unsafe { &*self.buf[index(idx)].as_ptr() })
    }

    /// Returns a mutable reference to the element at logical index `logical_idx`.
    ///
    /// Indexing semantics mirror `get`; callers can update values in place
    /// without changing ordering.
    pub fn get_mut(&mut self, logical_idx: u32) -> Option<&mut T> {
        if logical_idx >= self.len {
            return None;
        }

        assert!(logical_idx < self.len);
        assert!(self.head < Self::CAPACITY);
        assert!(self.len <= Self::CAPACITY);

        let idx = (self.head + logical_idx) % Self::CAPACITY;
        assert!(idx < Self::CAPACITY);

        // SAFETY: idx < CAPACITY proven, element initialized because len > 0
        Some(unsafe { &mut *self.buf[index(idx)].as_mut_ptr() })
    }

    /// Removes all elements, dropping them in FIFO order.
    ///
    /// Buffer remains usable afterwards without reallocating.
    pub fn clear(&mut self) {
        while self.pop_front().is_some() {}

        assert!(self.len == 0);
        assert!(self.is_empty());
    }
}

impl<T, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for RingBuffer<T, N> {
    fn drop(&mut self) {
        self.clear();
        assert!(self.len == 0);
    }
}

#[cfg(test)]
mod tests {
    use super::RingBuffer;
    use std::cell::Cell;
    use std::rc::Rc;

    #[derive(Debug)]
    struct DropTracker {
        value: i32,
        drops: Rc<Cell<usize>>,
    }

    impl DropTracker {
        fn new(value: i32, drops: Rc<Cell<usize>>) -> Self {
            Self { value, drops }
        }
    }

    impl Drop for DropTracker {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[test]
    fn ring_buffer_construction() {
        let r: RingBuffer<u32, 8> = RingBuffer::new();
        assert_eq!(r.head, 0);
        assert_eq!(r.len, 0);
    }

    #[test]
    fn basic_push_pop() {
        let mut rb: RingBuffer<i32, 4> = RingBuffer::new();
        assert!(rb.is_empty());
        rb.push_back_assume_capacity(1);
        rb.push_back_assume_capacity(2);
        rb.push_back_assume_capacity(3);
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.front(), Some(&1));
        assert_eq!(rb.pop_front(), Some(1));
        assert_eq!(rb.pop_front(), Some(2));
        assert_eq!(rb.pop_front(), Some(3));
        assert_eq!(rb.pop_front(), None);
        assert!(rb.is_empty());
    }

    #[test]
    fn wraparound() {
        let mut rb: RingBuffer<i32, 3> = RingBuffer::new();
        rb.push_back_assume_capacity(1);
        rb.push_back_assume_capacity(2);
        assert_eq!(rb.pop_front(), Some(1));
        rb.push_back_assume_capacity(3);
        rb.push_back_assume_capacity(4);
        // Now buffer is full: contents [2,3,4]
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.front(), Some(&2));
        assert_eq!(rb.get(0), Some(&2));
        assert_eq!(rb.get(1), Some(&3));
        assert_eq!(rb.get(2), Some(&4));
    }

    #[test]
    fn get_mut_wraparound_updates_correct_slot() {
        let mut rb: RingBuffer<i32, 3> = RingBuffer::new();
        rb.push_back_assume_capacity(10);
        rb.push_back_assume_capacity(20);
        assert_eq!(rb.pop_front(), Some(10));
        rb.push_back_assume_capacity(30);
        rb.push_back_assume_capacity(40);

        if let Some(elem) = rb.get_mut(1) {
            *elem += 1;
        }

        assert_eq!(rb.get(0), Some(&20));
        assert_eq!(rb.get(1), Some(&31));
        assert_eq!(rb.get(2), Some(&40));
    }

    #[test]
    fn front_mut_allows_in_place_update() {
        let mut rb: RingBuffer<i32, 2> = RingBuffer::new();
        rb.push_back_assume_capacity(5);
        rb.push_back_assume_capacity(6);

        if let Some(front) = rb.front_mut() {
            *front *= 2;
        }

        assert_eq!(rb.front(), Some(&10));
        assert_eq!(rb.pop_front(), Some(10));
        assert_eq!(rb.pop_front(), Some(6));
    }

    #[test]
    fn push_returns_error_when_full() {
        let mut rb: RingBuffer<i32, 2> = RingBuffer::new();
        assert!(rb.push_back(1).is_ok());
        assert!(rb.push_back(2).is_ok());
        assert!(rb.push_back(3).is_err());
        assert_eq!(rb.len(), 2);
        assert_eq!(rb.front(), Some(&1));
    }

    #[test]
    fn clear_drops_elements_and_allows_reuse() {
        let drops = Rc::new(Cell::new(0));
        {
            let mut rb: RingBuffer<DropTracker, 3> = RingBuffer::new();
            rb.push_back_assume_capacity(DropTracker::new(1, Rc::clone(&drops)));
            rb.push_back_assume_capacity(DropTracker::new(2, Rc::clone(&drops)));
            rb.push_back_assume_capacity(DropTracker::new(3, Rc::clone(&drops)));

            rb.clear();

            assert_eq!(drops.get(), 3);
            assert!(rb.is_empty());

            rb.push_back_assume_capacity(DropTracker::new(4, Rc::clone(&drops)));
            assert_eq!(rb.len(), 1);
            assert_eq!(rb.front().map(|t| t.value), Some(4));
        }
        // Drop should also clear any remaining elements.
        assert_eq!(drops.get(), 4);
    }
}
