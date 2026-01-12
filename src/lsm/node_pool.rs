//! Fixed-size node pool for LSM tree memory management.
//!
//! This module provides a memory pool that pre-allocates a contiguous buffer of
//! fixed-size nodes and manages their allocation state via a bitset. This design
//! eliminates per-node allocation overhead and provides O(1) acquire/release
//! operations with predictable memory usage.
//!
//! # Design Rationale
//!
//! LSM trees require frequent allocation and deallocation of fixed-size nodes
//! (e.g., for memtables, manifest entries). Using the global allocator for each
//! node would incur significant overhead and memory fragmentation. The node pool:
//!
//! - Pre-allocates all memory upfront, enabling memory budgeting
//! - Uses a bitset to track free slots, providing O(1) allocation
//! - Guarantees proper alignment for all nodes
//! - Validates all nodes are released before destruction (leak detection)
//!
//! # Memory Layout
//!
//! The pool allocates a single contiguous buffer:
//!
//! ```text
//! +----------+----------+----------+-----+----------+
//! |  Node 0  |  Node 1  |  Node 2  | ... |  Node N  |
//! +----------+----------+----------+-----+----------+
//! |<-------- NODE_SIZE * node_count --------------->|
//! ```

use std::{
    alloc::{Layout, alloc, dealloc, handle_alloc_error},
    ptr::NonNull,
};

use crate::stdx::DynamicBitSet;

/// Trait for fixed-size node memory pools.
///
/// Implementors provide a pool of pre-allocated, uniformly-sized memory nodes
/// that can be acquired and released without invoking the global allocator.
///
/// # Contract
///
/// - [`acquire`](NodePool::acquire) returns a pointer to an unused node of exactly
///   [`NODE_SIZE`](NodePool::NODE_SIZE) bytes with [`NODE_ALIGNMENT`](NodePool::NODE_ALIGNMENT)
///   alignment.
/// - Each acquired node must be released exactly once via [`release`](NodePool::release).
/// - Releasing a node that was not acquired from this pool is undefined behavior.
/// - The returned pointer is valid until the node is released or the pool is dropped.
///
/// # Implementor Requirements
///
/// Implementations must ensure:
/// - All returned pointers are properly aligned to `NODE_ALIGNMENT`
/// - Acquired nodes do not overlap
/// - Released nodes can be reacquired
pub trait NodePool {
    /// The size in bytes of each node in the pool.
    ///
    /// Must be a power of two and a multiple of [`NODE_ALIGNMENT`](NodePool::NODE_ALIGNMENT).
    const NODE_SIZE: usize;

    /// The alignment in bytes for each node in the pool.
    ///
    /// Must be a power of two and at most 4096.
    const NODE_ALIGNMENT: usize;

    /// Acquires an unused node from the pool.
    ///
    /// Returns a non-null pointer to a node of [`NODE_SIZE`](NodePool::NODE_SIZE) bytes
    /// with [`NODE_ALIGNMENT`](NodePool::NODE_ALIGNMENT) alignment. The memory contents
    /// are uninitialized.
    ///
    /// # Panics
    ///
    /// Panics if no free nodes are available in the pool.
    fn acquire(&mut self) -> NonNull<u8>;

    /// Releases a previously acquired node back to the pool.
    ///
    /// After release, the node may be returned by future [`acquire`](NodePool::acquire) calls.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `node` was not acquired from this pool
    /// - `node` is not properly aligned to [`NODE_SIZE`](NodePool::NODE_SIZE) boundaries
    /// - `node` has already been released (double-free)
    fn release(&mut self, node: NonNull<u8>);
}

/// A fixed-size memory pool backed by a contiguous buffer and bitset allocator.
///
/// `NodePoolType` pre-allocates a buffer capable of holding `node_count` nodes,
/// each of size `NODE_SIZE` bytes with `NODE_ALIGNMENT` alignment. Node allocation
/// is tracked via a [`DynamicBitSet`] where set bits indicate free nodes.
///
/// # Type Parameters
///
/// * `NODE_SIZE` - Size in bytes of each node. Must be a power of two and a
///   multiple of `NODE_ALIGNMENT`.
/// * `NODE_ALIGNMENT` - Alignment in bytes for each node. Must be a power of two
///   and at most 4096.
///
/// # Invariants
///
/// - `buffer` points to a valid allocation of `len` bytes (or is dangling if `len == 0`)
/// - `len` equals `NODE_SIZE * node_count` from initialization
/// - `free.bit_length()` equals `node_count`
/// - A bit is set in `free` if and only if that node slot is available for acquisition
///
/// # Memory Management
///
/// The pool owns its backing buffer and deallocates it on drop. The [`Drop`]
/// implementation asserts that all nodes have been released, panicking on memory
/// leaks to catch bugs early.
pub struct NodePoolType<const NODE_SIZE: usize, const NODE_ALIGNMENT: usize> {
    /// Pointer to the base of the allocated buffer.
    buffer: NonNull<u8>,
    /// Total size of the buffer in bytes (`NODE_SIZE * node_count`).
    len: usize,
    /// Bitset tracking free nodes. A set bit indicates the node is available.
    free: DynamicBitSet,
}

impl<const NODE_SIZE: usize, const NODE_ALIGNMENT: usize> NodePoolType<NODE_SIZE, NODE_ALIGNMENT> {
    /// The size in bytes of each node in the pool.
    pub const NODE_SIZE: usize = NODE_SIZE;

    /// The alignment in bytes for each node in the pool.
    pub const NODE_ALIGNMENT: usize = NODE_ALIGNMENT;

    /// Creates a new node pool with the specified capacity.
    ///
    /// Allocates a contiguous buffer capable of holding `node_count` nodes and
    /// initializes all nodes as free.
    ///
    /// # Arguments
    ///
    /// * `node_count` - The number of nodes to allocate. Must be greater than zero.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `node_count` is zero
    /// - `NODE_SIZE` is zero or not a power of two
    /// - `NODE_ALIGNMENT` is zero, not a power of two, or greater than 4096
    /// - `NODE_SIZE` is not a multiple of `NODE_ALIGNMENT`
    /// - `NODE_SIZE * node_count` overflows `usize`
    /// - Memory allocation fails
    pub fn init(node_count: u32) -> Self {
        assert!(node_count > 0);
        Self::assert_layout();

        let size = NODE_SIZE
            .checked_mul(node_count as usize)
            .expect("node buffer size overflow");
        let layout = Layout::from_size_align(size, NODE_ALIGNMENT)
            .unwrap_or_else(|_| panic!("invalid layout size"));

        // SAFETY: `layout` was validated by `Layout::from_size_align` which ensures
        // non-zero size and valid alignment. The returned pointer is checked for null
        // via `NonNull::new`, and allocation failure is handled by `handle_alloc_error`.
        let raw = unsafe { alloc(layout) };
        let buffer = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));

        let mut free = DynamicBitSet::empty(node_count as usize);
        Self::set_all(&mut free);

        Self {
            buffer,
            len: size,
            free,
        }
    }

    /// Explicitly deallocates the pool's backing buffer.
    ///
    /// This method performs the same cleanup as [`Drop`], but can be called
    /// explicitly when you need to control deallocation timing.
    ///
    /// After calling `deinit`, the pool is in an empty state and cannot be used.
    /// Calling `deinit` multiple times is safe (subsequent calls are no-ops).
    ///
    /// # Panics
    ///
    /// Panics if any nodes have not been released. This is intentional leak
    /// detection to catch bugs during development.
    pub fn deinit(&mut self) {
        self.deinit_internal(true);
    }

    /// Resets the pool, marking all nodes as free without deallocation.
    ///
    /// This is a fast operation that simply sets all bits in the free bitset.
    /// It does **not** zero or reinitialize the node memory contents.
    ///
    /// # Safety Considerations
    ///
    /// Calling `reset` while holding pointers to acquired nodes creates dangling
    /// pointers. The caller must ensure no references to acquired nodes exist
    /// before calling this method.
    pub fn reset(&mut self) {
        Self::set_all(&mut self.free);
    }

    /// Acquires an unused node from the pool.
    ///
    /// Returns a non-null pointer to an uninitialized node of [`NODE_SIZE`](Self::NODE_SIZE)
    /// bytes with [`NODE_ALIGNMENT`](Self::NODE_ALIGNMENT) alignment. The caller is responsible
    /// for initializing the memory before use and releasing it via [`release`](Self::release)
    /// when done.
    ///
    /// # Panics
    ///
    /// Panics with a descriptive message if no free nodes are available.
    pub fn acquire(&mut self) -> NonNull<u8> {
        let node_index = Self::find_first_set(&self.free).unwrap_or_else(|| {
            panic!(
                "out of memory for manifest, restart the replica increasing '--memory-lsm-manifest"
            )
        });
        assert!(self.free.is_set(node_index));
        self.free.unset(node_index);

        let offset = node_index * NODE_SIZE;
        assert!(offset + NODE_SIZE <= self.len);

        // SAFETY: The offset is within bounds because:
        // 1. `node_index` was obtained from `find_first_set` on our bitset, so it's < node_count
        // 2. `offset = node_index * NODE_SIZE` is thus < self.len
        // 3. The assertion above verifies this explicitly
        // The pointer is non-null because `self.buffer` is non-null and offset doesn't wrap.
        unsafe { NonNull::new_unchecked(self.buffer.as_ptr().add(offset)) }
    }

    /// Releases a previously acquired node back to the pool.
    ///
    /// After release, the node's slot becomes available for future [`acquire`](Self::acquire)
    /// calls. The memory contents are not cleared.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `node` points outside this pool's buffer
    /// - `node` is not aligned to a `NODE_SIZE` boundary within the buffer
    /// - `node` was already released (double-free detection)
    pub fn release(&mut self, node: NonNull<u8>) {
        let base = self.buffer.as_ptr() as usize;
        let ptr = node.as_ptr() as usize;

        assert!(ptr >= base);
        assert!(ptr + NODE_SIZE <= base + self.len);

        let node_offset = ptr - base;
        assert!(node_offset.is_multiple_of(NODE_SIZE));

        let node_index = node_offset / NODE_SIZE;
        assert!(!self.free.is_set(node_index));
        self.free.set(node_index);
    }

    /// Validates type parameter constraints at runtime.
    fn assert_layout() {
        assert!(NODE_SIZE > 0);
        assert!(NODE_ALIGNMENT > 0);
        assert!(NODE_ALIGNMENT <= 4096);
        assert!(NODE_SIZE.is_power_of_two());
        assert!(NODE_ALIGNMENT.is_power_of_two());
        assert!(NODE_SIZE.is_multiple_of(NODE_ALIGNMENT));
    }

    /// Internal deallocation with optional leak checking.
    fn deinit_internal(&mut self, verify_free: bool) {
        if self.len == 0 {
            return;
        }

        if verify_free {
            assert_eq!(self.free.count(), self.free.bit_length());
        }

        let layout = Layout::from_size_align(self.len, NODE_ALIGNMENT)
            .unwrap_or_else(|_| panic!("invalid layout"));

        // SAFETY: `self.buffer` was allocated with the same layout (size = self.len,
        // alignment = NODE_ALIGNMENT) via `alloc` in `init`. The buffer has not been
        // deallocated yet (we check `self.len != 0` at function entry, and set it to 0
        // after deallocation to prevent double-free).
        unsafe {
            dealloc(self.buffer.as_ptr(), layout);
        }

        self.buffer = NonNull::dangling();
        self.len = 0;
        self.free = DynamicBitSet::empty(0);
    }

    /// Sets all bits in the bitset to mark all nodes as free.
    fn set_all(bits: &mut DynamicBitSet) {
        bits.clear();
        bits.toggle_all();
    }

    /// Returns the index of the first set bit, or `None` if all bits are clear.
    fn find_first_set(bits: &DynamicBitSet) -> Option<usize> {
        bits.iter_set().next()
    }
}

impl<const NODE_SIZE: usize, const NODE_ALIGNMENT: usize> NodePool
    for NodePoolType<NODE_SIZE, NODE_ALIGNMENT>
{
    const NODE_SIZE: usize = NODE_SIZE;
    const NODE_ALIGNMENT: usize = NODE_ALIGNMENT;

    #[inline]
    fn acquire(&mut self) -> NonNull<u8> {
        NodePoolType::acquire(self)
    }

    #[inline]
    fn release(&mut self, ptr: NonNull<u8>) {
        NodePoolType::release(self, ptr)
    }
}

impl<const NODE_SIZE: usize, const NODE_ALIGNMENT: usize> Drop
    for NodePoolType<NODE_SIZE, NODE_ALIGNMENT>
{
    fn drop(&mut self) {
        self.deinit_internal(true);
    }
}

#[cfg(test)]
mod tests {
    use super::NodePoolType;
    use std::{mem, ptr::NonNull, slice};

    #[derive(Clone)]
    struct Prng {
        s: [u64; 4],
    }

    impl Prng {
        fn from_seed(seed: u64) -> Self {
            let mut state = seed;
            Self {
                s: [
                    Self::splitmix64(&mut state),
                    Self::splitmix64(&mut state),
                    Self::splitmix64(&mut state),
                    Self::splitmix64(&mut state),
                ],
            }
        }

        fn splitmix64(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        }

        fn next_u64(&mut self) -> u64 {
            let result = (self.s[0].wrapping_add(self.s[3]))
                .rotate_left(23)
                .wrapping_add(self.s[0]);
            let t = self.s[1] << 17;

            self.s[2] ^= self.s[0];
            self.s[3] ^= self.s[1];
            self.s[1] ^= self.s[2];
            self.s[0] ^= self.s[3];
            self.s[2] ^= t;
            self.s[3] = self.s[3].rotate_left(45);

            result
        }

        fn int_inclusive_u64(&mut self, max: u64) -> u64 {
            if max == u64::MAX {
                return self.next_u64();
            }

            let less_than = max + 1;
            let mut x = self.next_u64();
            let mut m = (x as u128) * (less_than as u128);
            let mut l = m as u64;
            if l < less_than {
                let mut t = less_than.wrapping_neg();
                if t >= less_than {
                    t = t.wrapping_sub(less_than);
                    if t >= less_than {
                        t %= less_than;
                    }
                }
                while l < t {
                    x = self.next_u64();
                    m = (x as u128) * (less_than as u128);
                    l = m as u64;
                }
            }
            (m >> 64) as u64
        }

        fn index(&mut self, len: usize) -> usize {
            assert!(len > 0);
            self.int_inclusive_u64(len as u64 - 1) as usize
        }
    }

    #[derive(Copy, Clone)]
    enum Action {
        Acquire,
        Release,
    }

    struct TestContext<const NODE_SIZE: usize, const NODE_ALIGNMENT: usize> {
        node_count: usize,
        sentinel: u64,
        node_pool: NodePoolType<NODE_SIZE, NODE_ALIGNMENT>,
        nodes: Vec<(NonNull<u8>, u64)>,
        acquires: u64,
        releases: u64,
    }

    impl<const NODE_SIZE: usize, const NODE_ALIGNMENT: usize> TestContext<NODE_SIZE, NODE_ALIGNMENT> {
        fn init(prng: &mut Prng, node_count: u32) -> Self {
            let node_pool = NodePoolType::<NODE_SIZE, NODE_ALIGNMENT>::init(node_count);
            let sentinel = prng.next_u64();
            assert!(NODE_ALIGNMENT >= mem::align_of::<u64>());
            assert_eq!(NODE_SIZE % mem::size_of::<u64>(), 0);

            let mut context = Self {
                node_count: node_count as usize,
                sentinel,
                node_pool,
                nodes: Vec::new(),
                acquires: 0,
                releases: 0,
            };
            context.fill_buffer();
            context
        }

        fn run(&mut self, prng: &mut Prng) {
            for _ in 0..self.node_count * 4 {
                match Self::choose_action(prng, 60, 40) {
                    Action::Acquire => self.acquire(prng),
                    Action::Release => self.release(prng),
                }
            }

            for _ in 0..self.node_count * 4 {
                match Self::choose_action(prng, 40, 60) {
                    Action::Acquire => self.acquire(prng),
                    Action::Release => self.release(prng),
                }
            }

            self.release_all(prng);
        }

        fn acquire(&mut self, prng: &mut Prng) {
            if self.nodes.len() == self.node_count {
                return;
            }

            let node = self.node_pool.acquire();
            assert!(self.nodes.iter().all(|(ptr, _)| *ptr != node));

            unsafe {
                let words = Self::node_as_u64_slice(node);
                for &word in words.iter() {
                    assert_eq!(self.sentinel, word);
                }
                let id = prng.next_u64();
                for word in words.iter_mut() {
                    *word = id;
                }
                self.nodes.push((node, id));
            }

            self.acquires += 1;
        }

        fn release(&mut self, prng: &mut Prng) {
            if self.nodes.is_empty() {
                return;
            }

            let index = prng.index(self.nodes.len());
            let (node, id) = self.nodes[index];

            unsafe {
                let words = Self::node_as_u64_slice(node);
                for &word in words.iter() {
                    assert_eq!(id, word);
                }
                for word in words.iter_mut() {
                    *word = self.sentinel;
                }
            }

            self.node_pool.release(node);
            self.nodes.swap_remove(index);
            self.releases += 1;
        }

        fn release_all(&mut self, prng: &mut Prng) {
            while !self.nodes.is_empty() {
                self.release(prng);
            }

            let sentinel = self.sentinel;
            unsafe {
                let words = self.buffer_as_u64_slice();
                for &word in words.iter() {
                    assert_eq!(sentinel, word);
                }
            }

            assert!(self.acquires > 0);
            assert_eq!(self.acquires, self.releases);
        }

        fn choose_action(prng: &mut Prng, acquire_weight: u64, release_weight: u64) -> Action {
            let total = acquire_weight + release_weight;
            let pick = prng.int_inclusive_u64(total - 1);
            if pick < acquire_weight {
                Action::Acquire
            } else {
                Action::Release
            }
        }

        fn fill_buffer(&mut self) {
            let sentinel = self.sentinel;
            unsafe {
                let words = self.buffer_as_u64_slice();
                for word in words.iter_mut() {
                    *word = sentinel;
                }
            }
        }

        unsafe fn buffer_as_u64_slice(&mut self) -> &mut [u64] {
            // Safety: buffer is aligned and sized for u64 access in these tests.
            let bytes = self.node_pool.len;
            assert_eq!(bytes % mem::size_of::<u64>(), 0);
            unsafe {
                slice::from_raw_parts_mut(
                    self.node_pool.buffer.as_ptr() as *mut u64,
                    bytes / mem::size_of::<u64>(),
                )
            }
        }

        unsafe fn node_as_u64_slice<'a>(node: NonNull<u8>) -> &'a mut [u64] {
            // Safety: node is aligned and sized for u64 access in these tests.
            unsafe {
                slice::from_raw_parts_mut(
                    node.as_ptr() as *mut u64,
                    NODE_SIZE / mem::size_of::<u64>(),
                )
            }
        }
    }

    fn run_for<const NODE_SIZE: usize, const NODE_ALIGNMENT: usize>(prng: &mut Prng) {
        let mut node_count: u32 = 1;
        while node_count < 64 {
            let mut context = TestContext::<NODE_SIZE, NODE_ALIGNMENT>::init(prng, node_count);
            context.run(prng);
            node_count += 1;
        }
    }

    #[test]
    fn node_pool() {
        let mut prng = Prng::from_seed(42);
        run_for::<8, 8>(&mut prng);
        run_for::<16, 8>(&mut prng);
        run_for::<64, 8>(&mut prng);
        run_for::<16, 16>(&mut prng);
        run_for::<32, 16>(&mut prng);
        run_for::<128, 16>(&mut prng);
    }
}
