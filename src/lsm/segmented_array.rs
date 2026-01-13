#![allow(dead_code)]

use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    slice,
};

use crate::lsm::{
    binary_search::{Config, binary_search_keys, binary_search_values_upsert_index},
    node_pool::NodePool,
};

/// Traversal direction for segmented array iteration.
///
/// `Ascending` walks from the start toward the end; `Descending` walks in reverse.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum Direction {
    /// Iterate from low to high indices.
    Ascending = 0,
    /// Iterate from high to low indices.
    Descending = 1,
}

impl Direction {
    #[inline]
    /// Returns the opposite direction.
    pub fn reverse(self) -> Self {
        match self {
            Direction::Ascending => Direction::Descending,
            Direction::Descending => Direction::Ascending,
        }
    }
}

/// Identifies a position within a segmented array node.
///
/// `node` is the node index; `relative_index` is the offset within that node.
/// For the last node, `relative_index` may equal the node's element count to
/// represent a one-past-the-end insertion point.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Cursor {
    pub node: u32,
    pub relative_index: u32,
}

/// A segmented array that distributes elements across multiple pooled nodes.
///
/// Elements are stored across nodes allocated from a [`NodePool`]. Since nodes are usually
/// not full, computing the absolute index of an element would be O(N) over the number of nodes.
/// To avoid this cost, we precompute the absolute index of the first element of each node
/// in the `indexes` array.
pub struct SegmentedArray<
    T: Copy,
    P: NodePool,
    const ELEMENT_COUNT_MAX: u32,
    const VERIFY: bool = false,
> {
    node_count: u32,
    /// The segmented array storage. The first `node_count` pointers are non-null;
    /// the rest are null. We use optional pointers here to get safety checks.
    nodes: Vec<Option<NonNull<u8>>>,
    /// Precomputed absolute index of the first element of each node.
    ///
    /// To avoid a separate `counts` field, we derive the number of elements in a node
    /// from the difference between consecutive indexes: `indexes[i+1] - indexes[i]`.
    ///
    /// To avoid special-casing the count function for the last node, the array length
    /// is `node_count_max + 1`, with the total element count stored in the last slot.
    indexes: Vec<u32>,
    _marker: PhantomData<(T, P)>,
}

impl<T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32, const VERIFY: bool> Default
    for SegmentedArray<T, P, ELEMENT_COUNT_MAX, VERIFY>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32, const VERIFY: bool>
    SegmentedArray<T, P, ELEMENT_COUNT_MAX, VERIFY>
{
    /// Maximum number of elements that fit in a single node.
    ///
    /// We can't use exact division here as we may store structs of various sizes
    /// (e.g., `TableInfo`) in this data structure, meaning there may be padding
    /// at the end of the node.
    ///
    /// We require that the node capacity is evenly divisible by 2 to simplify
    /// code that splits/joins nodes at the midpoint.
    pub const NODE_CAPACITY: u32 = {
        let max = (P::NODE_SIZE / size_of::<T>()) as u32;
        let capacity = if max.is_multiple_of(2) { max } else { max - 1 };
        assert!(capacity >= 2);
        assert!(capacity % 2 == 0);
        capacity
    };

    // Compile-time invariant checks.
    const _INVARIANTS: () = {
        // If this assert fails, we should be using a non-segmented array instead!
        assert!(ELEMENT_COUNT_MAX > Self::NODE_CAPACITY);

        // The buffers returned from the node pool must be able to store T with correct alignment.
        assert!(P::NODE_ALIGNMENT >= align_of::<T>());
    };

    /// Naive upper bound on the number of nodes needed.
    ///
    /// When a node fills up, it is divided into two new nodes. Therefore, the worst
    /// possible space overhead is when all nodes are half full. This uses ceiling
    /// division to examine that worst case.
    pub const NODE_COUNT_MAX_NAIVE: u32 = {
        // Minimum elements per node in the worst case (half full).
        let elements_per_node_min = Self::NODE_CAPACITY / 2;
        ELEMENT_COUNT_MAX.div_ceil(elements_per_node_min)
    };

    /// Actual maximum number of nodes, accounting for configurations where we can't
    /// reach [`NODE_COUNT_MAX_NAIVE`].
    ///
    /// We can't always reach `NODE_COUNT_MAX_NAIVE` in all configurations. If we're at
    /// `NODE_COUNT_MAX_NAIVE - 1` nodes and want to split one more node to reach the naive
    /// maximum, the following conditions must all be met:
    ///
    /// - The node we split must be full: `NODE_CAPACITY`
    /// - The last node must have at least one element: `+ 1`
    /// - All other nodes must be at least half-full: `(NODE_COUNT_MAX_NAIVE - 3) * (NODE_CAPACITY / 2)`
    /// - And then we insert one more element into the full node: `+ 1`
    ///
    /// If `ELEMENT_COUNT_MAX` is below this threshold, we can never actually reach the
    /// naive maximum and must use `NODE_COUNT_MAX_NAIVE - 1` instead.
    pub const NODE_COUNT_MAX: u32 = {
        let half = (Self::NODE_CAPACITY / 2) as u64;
        let naive = Self::NODE_COUNT_MAX_NAIVE as u64;
        // Threshold: minimum elements needed to potentially reach NODE_COUNT_MAX_NAIVE nodes.
        let threshold = (Self::NODE_CAPACITY as u64) // Node we split must be full
            + 1 // Last node must have at least one element
            + ((Self::NODE_COUNT_MAX_NAIVE.saturating_sub(3) as u64) * half) // Other nodes half-full
            + 1; // Insert one more element to trigger the split
        if (ELEMENT_COUNT_MAX as u64) >= threshold {
            naive as u32
        } else {
            (naive as u32) - 1
        }
    };

    /// Creates an empty segmented array with no nodes allocated yet.
    ///
    /// The backing vectors are sized to the maximum node count to avoid
    /// dynamic growth during insertions.
    pub fn new() -> Self {
        assert!(size_of::<T>() > 0);
        assert!(Self::NODE_CAPACITY >= 2);
        assert!(Self::NODE_CAPACITY.is_multiple_of(2));
        assert!(P::NODE_ALIGNMENT >= align_of::<T>());
        assert!(ELEMENT_COUNT_MAX > Self::NODE_CAPACITY);

        let nodes = vec![None; Self::NODE_COUNT_MAX as usize];
        let mut indexes = vec![0u32; Self::NODE_COUNT_MAX as usize + 1];
        indexes[0] = 0;

        let array = Self {
            node_count: 0,
            nodes,
            indexes,
            _marker: PhantomData,
        };

        if VERIFY {
            array.verify();
        }

        array
    }

    /// Releases all allocated nodes back to the pool and consumes `self`.
    ///
    /// The pool owns node memory, so cleanup must be explicit and needs a
    /// `&mut P` handle rather than relying on `Drop`.
    pub fn deinit(mut self, pool: &mut P) {
        if VERIFY {
            self.verify();
        }

        self.nodes[..self.node_count as usize]
            .iter_mut()
            .filter_map(Option::take)
            .for_each(|ptr| pool.release(ptr));
    }

    /// Releases nodes back to the pool and resets bookkeeping to the empty state.
    ///
    /// Intended for teardown paths or reuse where `VERIFY` should see an empty
    /// structure before the value is dropped.
    pub fn clear(&mut self, pool: &mut P) {
        if VERIFY {
            self.verify();
        }

        self.nodes[..self.node_count as usize]
            .iter_mut()
            .filter_map(Option::take)
            .for_each(|ptr| pool.release(ptr));

        self.node_count = 0;
        self.nodes.fill(None);
        self.indexes[0] = 0;

        if VERIFY {
            self.verify();
        }
    }

    /// Resets bookkeeping without returning nodes to the pool.
    ///
    /// Use only when the pool is reset separately (e.g. bulk teardown), otherwise
    /// unreleased nodes will be treated as leaks by the pool.
    ///
    /// Consumes `self` because after a pool reset any outstanding node pointers
    /// become invalid, so keeping this handle around would invite use-after-free.
    pub fn reset(mut self) {
        self.node_count = 0;
        self.nodes.fill(None);
        self.indexes.fill(0);

        if VERIFY {
            self.verify();
        }
    }

    /// Asserts internal invariants (node counts, indexes, and fill levels).
    pub fn verify(&self) {
        assert!(self.node_count <= Self::NODE_COUNT_MAX);

        assert!(
            self.nodes
                .iter()
                .enumerate()
                .all(|(k, v)| { ((k as u32) < self.node_count) == v.is_some() })
        );

        let half = Self::NODE_CAPACITY / 2;
        for node in 0..self.node_count {
            let count = self.count(node);
            assert!(count <= Self::NODE_CAPACITY);

            if node < self.node_count.saturating_sub(1) {
                assert!(count >= half);
            }
        }
    }

    /// Inserts `elements` starting at `absolute_index`, allocating/splitting nodes as needed.
    ///
    /// This is the public bulk-insert entry point: it delegates to the batching logic
    /// and double-checks the length change (and optional invariants) around the insert.
    ///
    /// # Panics
    /// - if `elements` is empty.
    /// - if `absolute_index + elements.len()` exceeds `ELEMENT_COUNT_MAX`.
    pub fn insert_elements(&mut self, pool: &mut P, absolute_index: u32, elements: &[T]) {
        if VERIFY {
            self.verify();
        }

        let count_before = self.len();
        self.insert_elements_at_absolute_index(pool, absolute_index, elements);
        let count_after = self.len();
        assert_eq!(count_after, count_before + elements.len() as u32);

        if VERIFY {
            self.verify();
        }
    }

    fn insert_elements_at_absolute_index(
        &mut self,
        pool: &mut P,
        absolute_index: u32,
        elements: &[T],
    ) {
        assert!(!elements.is_empty());
        assert!((absolute_index as u64 + elements.len() as u64) <= ELEMENT_COUNT_MAX as u64);

        let mut i: usize = 0;
        let node_capacity = Self::NODE_CAPACITY as usize;

        while i < elements.len() {
            let remaining = elements.len() - i;
            let batch = remaining.min(node_capacity);
            let abs = absolute_index + (i as u32);
            self.insert_elements_batch(pool, abs, &elements[i..i + batch]);
            i += batch;
        }
        assert_eq!(i, elements.len());
    }

    /// Inserts a contiguous batch at `absolute_index`, splitting nodes if needed.
    ///
    /// This is the "bulk" insert path for `elements.len() <= NODE_CAPACITY`. It either:
    /// - shifts within a single node when everything fits, or
    /// - splits the current node `a` into `[a|b]` and rebalance both halves, then
    ///   stitches `elements` into the correct gap across the two buffers.
    ///
    /// The split path is careful about overlap: we treat `a` and `b` as one logical
    /// buffer and use `copy_backwards_raw` to move tails without clobbering source
    /// elements when the source and destination overlap within `a`.
    fn insert_elements_batch(&mut self, pool: &mut P, absolute_index: u32, elements: &[T]) {
        assert!(!elements.is_empty());
        assert!(elements.len() <= Self::NODE_CAPACITY as usize);
        assert!((absolute_index as u64 + elements.len() as u64) <= ELEMENT_COUNT_MAX as u64);

        if self.node_count == 0 {
            assert_eq!(absolute_index, 0);
            self.insert_empty_node_at(pool, 0);
            assert_eq!(self.node_count, 1);
            assert!(self.nodes[0].is_some());
            assert_eq!(self.indexes[0], 0);
            assert_eq!(self.indexes[1], 0);
        }

        let cursor = self.cursor_for_absolute_index(absolute_index);
        assert!(cursor.node < self.node_count);
        let a = cursor.node;

        let a_count = self.count(a);
        assert!(cursor.relative_index <= a_count);

        let total = (a_count as u64) + (elements.len() as u64);
        let node_capacity = Self::NODE_CAPACITY as u64;

        if total <= node_capacity {
            // Simple in-node insert, all elements fit nicely :)
            let a_ptr = self.nodes[a as usize].expect("node missing");
            let buf = unsafe { Self::node_buf_from_ptr_mut(a_ptr) };

            let rel = cursor.relative_index as usize;
            let count = a_count as usize;
            let n = elements.len();

            let src = unsafe { buf.as_ptr().add(rel) };
            let dst = unsafe { buf.as_mut_ptr().add(rel + n) };
            let len_to_move = count - rel;
            unsafe { ptr::copy(src, dst, len_to_move) };

            let dst_elems = unsafe { buf.as_mut_ptr().add(rel) };
            let src_elems = elements.as_ptr() as *const MaybeUninit<T>;
            unsafe { ptr::copy_nonoverlapping(src_elems, dst_elems, n) };
            self.increment_indexes_after(a, elements.len() as u32);
            return;
        }

        let b = a + 1;
        self.insert_empty_node_at(pool, b);

        let a_ptr = self.nodes[a as usize].expect("node missing");
        let b_ptr = self.nodes[b as usize].expect("node missing");

        let total_u32 = total as u32;
        // Split total count as evenly as possible so both nodes stay >= half full.
        let a_half = total_u32.div_ceil(2);
        let b_half = total_u32 - a_half;

        let a_buf = unsafe { Self::node_buf_from_ptr_mut(a_ptr) };
        let b_buf = unsafe { Self::node_buf_from_ptr_mut(b_ptr) };

        let rel = cursor.relative_index as usize;
        let count = a_count as usize;
        let a_half_usize = a_half as usize;
        let b_half_usize = b_half as usize;

        // Step 1: Copy source_tail (a_buf[rel..count]) to position rel + elements.len()
        // across a_buf[..a_half] and b_buf[..b_half].
        // Use raw pointers since source and destination regions may overlap within a_buf.
        Self::copy_backwards_raw(
            a_buf.as_mut_ptr(),
            a_half_usize,
            b_buf.as_mut_ptr(),
            b_half_usize,
            rel + elements.len(),
            unsafe { a_buf.as_ptr().add(rel) },
            count - rel,
        );

        // Step 2: If the split point falls before the insert gap, move the "middle"
        // range from the tail of a into the front of b.
        if a_half_usize < rel {
            let mid_len = rel - a_half_usize;
            unsafe {
                ptr::copy_nonoverlapping(
                    a_buf.as_ptr().add(a_half_usize),
                    b_buf.as_mut_ptr(),
                    mid_len,
                )
            };
        }

        // Step 3: Copy elements into the gap, spanning a and b if needed.
        Self::copy_backwards_raw(
            a_buf.as_mut_ptr(),
            a_half_usize,
            b_buf.as_mut_ptr(),
            b_half_usize,
            rel,
            elements.as_ptr() as *const MaybeUninit<T>,
            elements.len(),
        );

        self.indexes[b as usize] = self.indexes[a as usize] + a_half;
        self.increment_indexes_after(b, elements.len() as u32);
    }

    /// Copies `source_len` elements from `source_ptr` into a logical destination
    /// spanning two buffers: `a[0..a_len]` followed by `b[0..b_len]`.
    /// The copy starts at logical index `target` within this combined span.
    ///
    /// "Backwards" here means the destination may straddle `a` and `b`, so we
    /// materialize the tail in `b` first, then the head in `a`. This allows
    /// safe use with overlapping regions inside `a` (via `ptr::copy`) while
    /// still supporting a logical buffer that crosses the node boundary.
    fn copy_backwards_raw(
        a_ptr: *mut MaybeUninit<T>,
        a_len: usize,
        b_ptr: *mut MaybeUninit<T>,
        b_len: usize,
        target: usize,
        source_ptr: *const MaybeUninit<T>,
        source_len: usize,
    ) {
        assert!(target + source_len <= a_len + b_len);

        let target_a_start = target.min(a_len);
        let target_a_end = (target + source_len).min(a_len);
        let target_a_len = target_a_end - target_a_start;

        let target_b_start = target.saturating_sub(a_len);
        let target_b_end = (target + source_len).saturating_sub(a_len);
        let target_b_len = target_b_end - target_b_start;

        assert_eq!(target_a_len + target_b_len, source_len);

        // Copy to b first (backwards order), then to a
        if target_b_len > 0 {
            unsafe {
                ptr::copy(
                    source_ptr.add(target_a_len),
                    b_ptr.add(target_b_start),
                    target_b_len,
                )
            };
        }

        if target_a_len > 0 {
            unsafe { ptr::copy(source_ptr, a_ptr.add(target_a_start), target_a_len) };
        }
    }

    /// Removes `remove_count` elements starting at `absolute_index`.
    ///
    /// The work is chunked into half-node batches so each step only needs to touch
    /// `a` and its neighbor `b`, preserving the half-full invariant locally and
    /// avoiding rebalancing cascades across many nodes.
    pub fn remove_elements(&mut self, pool: &mut P, absolute_index: u32, remove_count: u32) {
        if VERIFY {
            self.verify();
        }

        assert!(self.node_count > 0);
        assert!(remove_count > 0);
        assert!((absolute_index as u64 + remove_count as u64) <= ELEMENT_COUNT_MAX as u64);
        assert!(absolute_index + remove_count <= self.len());

        let half = Self::NODE_CAPACITY / 2;
        let mut remaining = remove_count;

        while remaining > 0 {
            let batch = remaining.min(half);
            // Always remove at the same absolute index; the next element shifts into place.
            self.remove_elements_batch(pool, absolute_index, batch);
            remaining -= batch;
        }

        if VERIFY {
            self.verify();
        }
    }

    /// Removes a small batch starting at `absolute_index`.
    ///
    /// The batch size is capped at `NODE_CAPACITY / 2` so we only touch `a` and
    /// its right neighbor `b`, and can restore the half-full invariant locally
    /// (shift within a node, rebalance, or merge) without cascading updates.
    fn remove_elements_batch(&mut self, pool: &mut P, absolute_index: u32, remove_count: u32) {
        let half = Self::NODE_CAPACITY / 2;

        assert!(self.node_count > 0);
        assert!(remove_count > 0);
        assert!(remove_count <= half);
        assert!((absolute_index as u64 + remove_count as u64) <= ELEMENT_COUNT_MAX as u64);
        assert!(absolute_index + remove_count <= self.len());

        let cursor = self.cursor_for_absolute_index(absolute_index);
        assert!(cursor.node < self.node_count);
        let a = cursor.node;
        let a_ptr = self.nodes[a as usize].expect("node missing");

        let a_remaining = cursor.relative_index;
        let a_count = self.count(a);

        if a_remaining + remove_count <= a_count {
            // Removal stays within `a`: shift the tail left to fill the gap.
            let buf = unsafe { Self::node_buf_from_ptr_mut(a_ptr) };

            let start = a_remaining as usize;
            let src_start = (a_remaining + remove_count) as usize;
            let count = a_count as usize;

            let len_to_move = count - src_start;
            unsafe {
                ptr::copy(
                    buf.as_ptr().add(src_start),
                    buf.as_mut_ptr().add(start),
                    len_to_move,
                )
            };

            self.decrement_indexes_after(a, remove_count);
            // `a` might have dropped below half full; rebalance with its neighbor.
            self.maybe_remove_or_merge_node_with_next(pool, a);
            return;
        }

        // Removal crosses the boundary: drop the tail of `a` and the head of `b`.
        let b = a + 1;
        let b_ptr = self.nodes[b as usize].expect("missing node");

        let removed_from_a = a_count - a_remaining;
        let removed_from_b = remove_count - removed_from_a;
        assert!(removed_from_b > 0);

        let b_count = self.count(b);
        let b_remaining_len = b_count - removed_from_b;
        assert!(a_remaining > 0 || b_remaining_len > 0);

        if a_remaining >= half {
            // Keep 'a' (still >= half full), shift remaining of 'b' to left.
            let buf_b = unsafe { Self::node_buf_from_ptr_mut(b_ptr) };

            let src_start = removed_from_b as usize;
            let count = b_count as usize;
            let len_to_move = count - src_start;

            unsafe {
                ptr::copy(
                    buf_b.as_ptr().add(src_start),
                    buf_b.as_mut_ptr(),
                    len_to_move,
                )
            };

            self.indexes[b as usize] = self.indexes[a as usize] + a_remaining;
            self.decrement_indexes_after(b, remove_count);
            // `b` may be underfull after shifting; fix by merging/rebalancing with its next.
            self.maybe_remove_or_merge_node_with_next(pool, b);
            return;
        }

        // Snapshot the logical remainder of `b` after removing its prefix.
        let b_remaining_slice = {
            let buf_b = unsafe { Self::node_buf_from_ptr(b_ptr) };
            let start = removed_from_b as usize;
            let end = (removed_from_b + b_remaining_len) as usize;
            &buf_b[start..end]
        };

        if b_remaining_len >= half {
            // Keep 'b' (still >= half full), maybe rebalance by moving some of 'b' into 'a'.
            self.indexes[b as usize] = self.indexes[a as usize] + a_remaining;
            self.decrement_indexes_after(b, remove_count);
            // Use the sliced remainder as the logical contents of `b` while rebalancing.
            self.maybe_merge_nodes(pool, a, b_remaining_slice);
        } else {
            // Merge all remaining of 'b' into 'a' then remove empty 'b'.
            let buf_a = unsafe { Self::node_buf_from_ptr_mut(a_ptr) };
            let dst_start = a_remaining as usize;
            let dst_end = (a_remaining + b_remaining_len) as usize;
            let dst = &mut buf_a[dst_start..dst_end];

            unsafe {
                ptr::copy_nonoverlapping(
                    b_remaining_slice.as_ptr(),
                    dst.as_mut_ptr(),
                    b_remaining_slice.len(),
                )
            };

            self.indexes[b as usize] = self.indexes[a as usize] + a_remaining + b_remaining_len;
            self.decrement_indexes_after(b, remove_count);
            self.remove_empty_node_at(pool, b);

            assert!(b == self.node_count || self.count(a) >= half);
        }
    }

    /// Attempts to merge or rebalance `node` with its right neighbor.
    ///
    /// Removes `node` when it becomes empty, otherwise returns early if it is the last node.
    /// The next node's elements are sliced from its backing buffer and forwarded to
    /// `maybe_merge_nodes`.
    fn maybe_remove_or_merge_node_with_next(&mut self, pool: &mut P, node: u32) {
        assert!(node < self.node_count);

        if self.count(node) == 0 {
            self.remove_empty_node_at(pool, node);
            return;
        }

        if node == self.node_count - 1 {
            return;
        }

        let next = node + 1;
        let b_ptr = self.nodes[next as usize].expect("node missing");
        let b_count = self.count(next) as usize;

        let next_elements = {
            let b_buf = unsafe { Self::node_buf_from_ptr(b_ptr) };
            &b_buf[..b_count]
        };

        self.maybe_merge_nodes(pool, node, next_elements);
    }

    /// Attempts to merge or rebalance node `a` with its right neighbor `b`.
    ///
    /// `elements_next_node` represents the logical contents of `b` after a caller
    /// mutation; it may alias `b`'s buffer or be a temporary slice. The method keeps
    /// the "all but last nodes are at least half full" invariant, either by:
    /// - fully merging `b` into `a` when the combined total fits, or
    /// - stealing enough from the front of `b` to make both nodes >= half full.
    fn maybe_merge_nodes(
        &mut self,
        pool: &mut P,
        node: u32,
        elements_next_node: &[MaybeUninit<T>],
    ) {
        // Merge/rebalance `a` with its right neighbor `b`.
        // `elements_next_node` is the logical contents of `b` after the caller's edits:
        // it may alias `b`'s buffer or be a temporary slice that has not been written back.
        let half = Self::NODE_CAPACITY / 2;

        let a = node;
        let b = node + 1;
        assert!(b < self.node_count);

        let a_ptr = self.nodes[a as usize].expect("node missing");
        let b_ptr = self.nodes[b as usize].expect("node missing");

        let a_count = self.count(a);

        assert!(!elements_next_node.is_empty());
        // Only the last node is allowed to be below half full.
        assert!((elements_next_node.len() as u32) >= half || b == (self.node_count) - 1);

        let b_buf = unsafe { Self::node_buf_from_ptr(b_ptr) };
        // If `a` is empty and `elements_next_node` already lives in `b`, the caller should
        // drop `a` instead of forcing a full copy into `a`.
        assert!(!(a_count == 0 && ptr::eq(elements_next_node.as_ptr(), b_buf.as_ptr())));

        let total = a_count + (elements_next_node.len() as u32);

        if total <= Self::NODE_CAPACITY {
            // Full merged: move all of 'b' into the end of 'a' then remove 'b'.
            let buf_a = unsafe { Self::node_buf_from_ptr_mut(a_ptr) };
            let dst = &mut buf_a[(a_count as usize)..(total as usize)];
            unsafe {
                ptr::copy_nonoverlapping(elements_next_node.as_ptr(), dst.as_mut_ptr(), dst.len())
            };

            // Make `b` empty so remove_empty_node_at can validate its precondition.
            self.indexes[b as usize] = self.indexes[(b + 1) as usize];
            self.remove_empty_node_at(pool, b);

            return;
        }

        if a_count < half {
            // Re-balance by stealing from the front of 'b'.
            let a_half = total.div_ceil(2);

            let need_from_b = (a_half - a_count) as usize;

            // Copy some elements from the front of 'b_elements' into the end of 'a'.
            let buf_a = unsafe { Self::node_buf_from_ptr_mut(a_ptr) };
            let dst = &mut buf_a[(a_count as usize)..(a_half as usize)];
            unsafe {
                ptr::copy_nonoverlapping(elements_next_node.as_ptr(), dst.as_mut_ptr(), need_from_b)
            };

            // Shift the remaining 'b' elements to the start of 'b' buffer.
            // Use ptr::copy because the source and destination can overlap in `b`.
            let buf_b = unsafe { Self::node_buf_from_ptr_mut(b_ptr) };
            let remaining = &elements_next_node[need_from_b..];
            unsafe { ptr::copy(remaining.as_ptr(), buf_b.as_mut_ptr(), remaining.len()) };

            self.indexes[b as usize] = self.indexes[a as usize] + a_half;
            assert!(self.count(a) >= half);
            assert!(self.count(b) >= half);

            return;
        }

        // Nothing to do: both nodes already satisfy the half-full invariant.
        // If the caller supplied a temporary slice, they must have written it back first.
        let b_buf = unsafe { Self::node_buf_from_ptr(b_ptr) };
        assert!(ptr::eq(elements_next_node.as_ptr(), b_buf.as_ptr()));
    }

    fn insert_empty_node_at(&mut self, pool: &mut P, node: u32) {
        assert!(node <= self.node_count);
        assert!(self.node_count < Self::NODE_COUNT_MAX);

        let node = node as usize;
        let node_count = self.node_count as usize;

        // Shift nodes right: [node..node_count] -> [node+1..node_count+1]
        self.nodes.copy_within(node..node_count, node + 1);

        // Shift indexes right: [node..node_count] -> [node+1..node_count+1]
        self.indexes.copy_within(node..(node_count + 1), node + 1);

        self.node_count += 1;

        let ptr = pool.acquire();
        self.nodes[node] = Some(ptr);
        assert_eq!(self.indexes[node], self.indexes[node + 1]);
    }

    fn remove_empty_node_at(&mut self, pool: &mut P, node: u32) {
        assert!(self.node_count > 0);
        assert!(node < self.node_count);
        assert_eq!(self.count(node), 0);

        let node = node as usize;
        let node_count = self.node_count as usize;

        let ptr = self.nodes[node].take().expect("node missing");
        pool.release(ptr);

        // Shift nodes left: [node+1..node_count] -> [node..node_count-1]
        self.nodes.copy_within(node + 1..node_count, node);

        // Shift indexes left: [node+1..node_count+1] -> [node..node_count]
        self.indexes.copy_within(node + 1..node_count + 1, node);

        self.node_count -= 1;

        let new_count = self.node_count as usize;
        self.nodes[new_count] = None;
    }

    #[inline]
    fn count(&self, node: u32) -> u32 {
        let n = node as usize;
        let result = self.indexes[n + 1] - self.indexes[n];
        assert!(result <= Self::NODE_CAPACITY);
        result
    }

    #[inline]
    fn increment_indexes_after(&mut self, node: u32, delta: u32) {
        let start = (node as usize) + 1;
        let end = (self.node_count as usize) + 1;
        self.indexes[start..end]
            .iter_mut()
            .for_each(|index| *index += delta);
    }

    #[inline]
    fn decrement_indexes_after(&mut self, node: u32, delta: u32) {
        let start = (node as usize) + 1;
        let end = (self.node_count as usize) + 1;
        self.indexes[start..end]
            .iter_mut()
            .for_each(|index| *index -= delta);
    }

    fn cursor_for_absolute_index(&self, absolute_index: u32) -> Cursor {
        assert!(absolute_index < ELEMENT_COUNT_MAX);
        assert!(absolute_index <= self.len());
        assert!(self.node_count > 0);

        let keys = &self.indexes[..(self.node_count as usize)];
        let result = binary_search_keys(keys, absolute_index, Config::default());

        if result.exact {
            return Cursor {
                node: result.index,
                relative_index: 0,
            };
        }

        let node = result.index - 1;
        let rel = absolute_index - self.indexes[node as usize];

        // Only the last node allows insertion at one-past-the-end.
        let count = self.count(node);
        if node == self.node_count - 1 {
            assert!(rel <= count);
        } else {
            assert!(rel < count);
        }

        Cursor {
            node,
            relative_index: rel,
        }
    }

    #[inline]
    unsafe fn node_buf_from_ptr<'a>(ptr: NonNull<u8>) -> &'a [MaybeUninit<T>] {
        unsafe {
            slice::from_raw_parts(
                ptr.as_ptr() as *const MaybeUninit<T>,
                Self::NODE_CAPACITY as usize,
            )
        }
    }

    #[inline]
    unsafe fn node_buf_from_ptr_mut<'a>(ptr: NonNull<u8>) -> &'a mut [MaybeUninit<T>] {
        unsafe {
            slice::from_raw_parts_mut(
                ptr.as_ptr() as *mut MaybeUninit<T>,
                Self::NODE_CAPACITY as usize,
            )
        }
    }

    /// Returns the initialized elements slice for `node`.
    ///
    /// Panics if `node` is out of bounds.
    pub fn node_elements(&self, node: u32) -> &[T] {
        assert!(node < self.node_count);
        let ptr = self.nodes[node as usize].as_ref().expect("node missing");
        let count = self.count(node) as usize;

        let buf = unsafe { Self::node_buf_from_ptr(*ptr) };
        let init = &buf[..count];
        // SAFETY: The first `count` elements are initialized (tracked by indexes array).
        unsafe { &*(init as *const [MaybeUninit<T>] as *const [T]) }
    }

    /// Returns the last element in `node`.
    ///
    /// Panics if the node is empty or out of bounds.
    pub fn node_last_element(&self, node: u32) -> T {
        let elems = self.node_elements(node);
        assert!(!elems.is_empty());
        elems[elems.len() - 1]
    }

    /// Returns the element at `cursor`.
    ///
    /// Panics if the cursor is out of bounds.
    pub fn element_at_cursor(&self, cursor: Cursor) -> T {
        let elems = self.node_elements(cursor.node);
        elems[cursor.relative_index as usize]
    }

    #[inline]
    /// Returns a cursor to the first element (or the start when empty).
    pub fn first(&self) -> Cursor {
        Cursor {
            node: 0,
            relative_index: 0,
        }
    }

    /// Returns a cursor to the last element (or `first()` when empty).
    pub fn last(&self) -> Cursor {
        if self.node_count == 0 {
            return self.first();
        }

        let node = self.node_count - 1;
        let rel = self.count(node) - 1;

        Cursor {
            node,
            relative_index: rel,
        }
    }

    #[inline]
    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    #[inline]
    /// Returns the total number of elements across all nodes.
    pub fn len(&self) -> u32 {
        let result = self.indexes[self.node_count as usize];
        assert!(result <= ELEMENT_COUNT_MAX);
        result
    }

    /// Converts a cursor into its absolute index in the logical array.
    ///
    /// Panics if the cursor does not point into the current structure.
    pub fn absolute_index_for_cursor(&self, cursor: Cursor) -> u32 {
        if self.node_count == 0 {
            assert_eq!(cursor.node, 0);
            assert_eq!(cursor.relative_index, 0);
            return 0;
        }

        let count = self.count(cursor.node);
        if cursor.node == self.node_count - 1 {
            assert!(cursor.relative_index <= count);
        } else {
            assert!(cursor.relative_index < count);
        }

        self.indexes[cursor.node as usize] + cursor.relative_index
    }

    /// Creates an iterator starting at `cursor` and walking in `direction`.
    ///
    /// This is meant for callers that already have a [`Cursor`], such as a sorted
    /// wrapper that performed a search and wants to scan forward/backward from an
    /// insertion point. A cursor that is one-past-the-end is only valid for the
    /// last node; in that case an ascending iterator is immediately done, while a
    /// descending iterator starts at the last element.
    ///
    /// # Panics
    /// Panics if `cursor` does not refer to the current structure.
    pub fn iterator_from_cursor<'a>(
        &'a self,
        mut cursor: Cursor,
        direction: Direction,
    ) -> Iterator<'a, T, P, ELEMENT_COUNT_MAX, VERIFY> {
        if self.node_count == 0 {
            assert_eq!(cursor.node, 0);
            assert_eq!(cursor.relative_index, 0);

            return Iterator {
                array: self as *const _,
                direction,
                cursor,
                done: true,
                _marker: PhantomData,
            };
        }

        let last_node = self.node_count - 1;

        if cursor.node == last_node && cursor.relative_index == self.count(last_node) {
            let done = direction == Direction::Ascending;

            if done {
                return Iterator {
                    array: self as *const _,
                    direction,
                    cursor,
                    done: true,
                    _marker: PhantomData,
                };
            }

            cursor.relative_index -= 1;
            return Iterator {
                array: self as *const _,
                direction,
                cursor,
                done: false,
                _marker: PhantomData,
            };
        }

        assert!(cursor.node < self.node_count);
        assert!(cursor.relative_index < self.count(cursor.node));

        Iterator {
            array: self as *const _,
            direction,
            cursor,
            done: false,
            _marker: PhantomData,
        }
    }

    /// Creates an iterator starting at the element with `absolute_index`.
    ///
    /// This is a convenience for search APIs that operate on absolute indexes
    /// (e.g. binary search in a sorted wrapper) and then need to scan in either
    /// direction without converting back to a cursor manually.
    ///
    /// # Panics
    /// Panics if `absolute_index` is out of bounds.
    pub fn iterator_from_index<'a>(
        &'a self,
        absolute_index: u32,
        direction: Direction,
    ) -> Iterator<'a, T, P, ELEMENT_COUNT_MAX, VERIFY> {
        assert!(absolute_index < ELEMENT_COUNT_MAX);

        if self.node_count == 0 {
            assert_eq!(absolute_index, 0);

            return Iterator {
                array: self as *const _,
                direction,
                cursor: Cursor {
                    node: 0,
                    relative_index: 0,
                },
                done: true,
                _marker: PhantomData,
            };
        }

        assert!(absolute_index < self.len());
        let cursor = self.cursor_for_absolute_index(absolute_index);

        Iterator {
            array: self as *const _,
            direction,
            cursor,
            done: false,
            _marker: PhantomData,
        }
    }
}

/// Cursor-based traversal over a [`SegmentedArray`] that yields raw element pointers.
///
/// This iterator exists so higher-level wrappers (like `SortedSegmentArray`) can walk
/// elements in order while still performing in-place value updates. Holding a raw
/// pointer avoids borrowing the array for the duration of the traversal, which keeps
/// APIs flexible but means callers must not structurally modify the array while
/// iterating (insert/remove/split/merge).
pub struct Iterator<'a, T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32, const VERIFY: bool> {
    // Raw pointer avoids tying up a borrow of the array during iteration.
    array: *const SegmentedArray<T, P, ELEMENT_COUNT_MAX, VERIFY>,
    direction: Direction,
    cursor: Cursor,
    /// True once the iterator has reached the terminal element.
    pub done: bool,
    _marker: PhantomData<&'a SegmentedArray<T, P, ELEMENT_COUNT_MAX, VERIFY>>,
}

impl<'a, T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32, const VERIFY: bool>
    Iterator<'a, T, P, ELEMENT_COUNT_MAX, VERIFY>
{
    /// Returns a pointer to the current element and advances the cursor.
    ///
    /// The returned pointer remains valid as long as the underlying array is not
    /// structurally modified. Mutating the element's value in place is expected.
    ///
    /// The `std::iter::Iterator` implementation delegates to this method.
    pub fn next_ptr(&mut self) -> Option<*mut T> {
        if self.done {
            return None;
        }

        let array = unsafe { &*self.array };
        assert!(array.node_count > 0);
        assert!(self.cursor.node < array.node_count);

        let node = self.cursor.node;
        let rel = self.cursor.relative_index as usize;

        let count = array.count(node) as usize;
        assert!(rel < count);

        let node_ptr = array.nodes[node as usize].expect("node missing");
        let element_ptr = unsafe { (node_ptr.as_ptr() as *mut MaybeUninit<T>).add(rel) as *mut T };

        match self.direction {
            Direction::Ascending => {
                if rel == count - 1 {
                    if node == array.node_count - 1 {
                        self.done = true;
                    } else {
                        self.cursor.node += 1;
                        self.cursor.relative_index = 0;
                    }
                } else {
                    self.cursor.relative_index += 1;
                }
            }
            Direction::Descending => {
                if rel == 0 {
                    if node == 0 {
                        self.done = true;
                    } else {
                        self.cursor.node -= 1;
                        self.cursor.relative_index = array.count(self.cursor.node) - 1;
                    }
                } else {
                    self.cursor.relative_index -= 1;
                }
            }
        }

        Some(element_ptr)
    }
}

impl<'a, T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32, const VERIFY: bool> std::iter::Iterator
    for Iterator<'a, T, P, ELEMENT_COUNT_MAX, VERIFY>
{
    type Item = *mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_ptr()
    }
}

/// Defines how to extract an ordered key from a value stored in a segmented array.
///
/// This keeps the core `SegmentedArray` generic while enabling sorted wrappers
/// (like `SortedSegmentArray`) to compare by key without storing a parallel key
/// buffer or requiring a bespoke value type.
pub trait KeyFromValue<T: Copy> {
    /// Key type used to order values stored in the segmented array.
    type Key: Ord + Copy;

    /// Extracts the key that determines the value's sorted position.
    ///
    /// This indirection lets sorted wrappers (e.g. `SortedSegmentArray`) perform
    /// binary searches and range scans without storing a parallel key array.
    fn key_from_value(value: &T) -> Self::Key;
}

/// Sorted wrapper around [`SegmentedArray`] keyed by [`KeyFromValue`].
///
/// Ordering is derived from values, so no parallel key buffer is needed.
pub struct SortedSegmentedArray<
    T: Copy,
    P: NodePool,
    const ELEMENT_COUNT_MAX: u32,
    KF: KeyFromValue<T>,
    const VERIFY: bool = false,
> {
    base: SegmentedArray<T, P, ELEMENT_COUNT_MAX, VERIFY>,
    _marker: PhantomData<KF>,
}

impl<T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32, KF: KeyFromValue<T>, const VERIFY: bool>
    SortedSegmentedArray<T, P, ELEMENT_COUNT_MAX, KF, VERIFY>
{
    /// Maximum number of elements per node, inherited from `SegmentedArray`.
    pub const NODE_CAPACITY: u32 = SegmentedArray::<T, P, ELEMENT_COUNT_MAX, false>::NODE_CAPACITY;
    /// Naive upper bound on the number of nodes, inherited from `SegmentedArray`.
    pub const NODE_COUNT_MAX_NAIVE: u32 =
        SegmentedArray::<T, P, ELEMENT_COUNT_MAX, false>::NODE_COUNT_MAX_NAIVE;
    /// Actual maximum number of nodes, inherited from `SegmentedArray`.
    pub const NODE_COUNT_MAX: u32 =
        SegmentedArray::<T, P, ELEMENT_COUNT_MAX, VERIFY>::NODE_COUNT_MAX;

    /// Creates an empty sorted segmented array.
    pub fn new() -> Self {
        let s = Self {
            base: SegmentedArray::<T, P, ELEMENT_COUNT_MAX, VERIFY>::new(),
            _marker: PhantomData,
        };

        if VERIFY {
            s.verify();
        }

        s
    }

    /// Releases all nodes back to the pool and consumes `self`.
    pub fn deinit(self, pool: &mut P) {
        self.base.deinit(pool);
    }

    /// Clears all elements and returns nodes to the pool.
    pub fn clear(&mut self, pool: &mut P) {
        if VERIFY {
            self.verify();
        }
        self.base.clear(pool);
        if VERIFY {
            self.verify();
        }
    }

    /// Returns the total number of elements across all nodes.
    pub fn len(&self) -> u32 {
        self.base.len()
    }

    /// Returns the initialized elements slice for `node`.
    pub fn node_elements(&self, node: u32) -> &[T] {
        self.base.node_elements(node)
    }

    /// Converts a cursor into its absolute index in the logical array.
    pub fn absolute_index_for_cursor(&self, cursor: Cursor) -> u32 {
        self.base.absolute_index_for_cursor(cursor)
    }

    /// Creates an iterator starting at `cursor` and walking in `direction`.
    pub fn iterator_from_cursor<'a>(
        &'a self,
        cursor: Cursor,
        direction: Direction,
    ) -> Iterator<'a, T, P, ELEMENT_COUNT_MAX, VERIFY> {
        self.base.iterator_from_cursor(cursor, direction)
    }

    /// Creates an iterator starting at the element with `index`.
    pub fn iterator_from_index<'a>(
        &'a self,
        index: u32,
        direction: Direction,
    ) -> Iterator<'a, T, P, ELEMENT_COUNT_MAX, VERIFY> {
        self.base.iterator_from_index(index, direction)
    }

    /// Asserts base invariants and that keys are globally sorted.
    pub fn verify(&self) {
        self.base.verify();

        assert!(
            (0..self.base.node_count)
                .flat_map(|node| self.base.node_elements(node))
                .is_sorted_by_key(|v| KF::key_from_value(v))
        );
    }

    /// Finds the cursor where `key` should be inserted.
    ///
    /// Binary searches node boundaries by each node's first key, then searches
    /// within that node.
    pub fn search(&self, key: KF::Key) -> Cursor {
        if self.base.node_count == 0 {
            return Cursor {
                node: 0,
                relative_index: 0,
            };
        }

        let mut offset: usize = 0;
        let mut length: usize = self.base.node_count as usize;

        while length > 1 {
            let half = length / 2;
            let mid = offset + half;

            let node_first = &self.base.node_elements(mid as u32)[0];
            if KF::key_from_value(node_first) < key {
                offset = mid;
            }
            length -= half;
        }

        let node = offset as u32;
        assert!(node < self.base.node_count);
        let key_from_value = |value: &T| KF::key_from_value(value);
        let rel = binary_search_values_upsert_index(
            self.base.node_elements(node),
            key,
            Config::default(),
            &key_from_value,
        );

        if node + 1 < self.base.node_count && rel == self.base.count(node) {
            Cursor {
                node: node + 1,
                relative_index: 0,
            }
        } else {
            Cursor {
                node,
                relative_index: rel,
            }
        }
    }

    /// Inserts `element` while maintaining key order and returns its absolute index.
    pub fn insert_element(&mut self, pool: &mut P, element: T) -> u32 {
        if VERIFY {
            self.verify();
        }

        let count_before = self.len();
        let key = KF::key_from_value(&element);
        let cursor = self.search(key);
        let absolute_index = self.base.absolute_index_for_cursor(cursor);

        self.base.insert_elements_at_absolute_index(
            pool,
            absolute_index,
            slice::from_ref(&element),
        );
        if VERIFY {
            self.verify();
        }

        let count_after = self.len();
        assert_eq!(count_after, count_before + 1);
        absolute_index
    }

    /// Removes `remove_count` elements starting at `absolute_index`.
    pub fn remove_elements(&mut self, pool: &mut P, absolute_index: u32, remove_count: u32) {
        if VERIFY {
            self.verify();
        }

        self.base
            .remove_elements(pool, absolute_index, remove_count);
        if VERIFY {
            self.verify();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SegmentedArray;
    use crate::lsm::node_pool::{NodePool, NodePoolType};
    use crate::stdx::fastrange::fast_range;
    use proptest::prelude::*;
    use proptest::test_runner::{RngAlgorithm, TestRng};
    use std::fmt::Debug;

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct TestTableInfo {
        words: [u64; 12],
    }

    const DEFAULT_PROPTEST_CASES: u32 = 2;
    const DEFAULT_FUZZ_STEPS_MULTIPLIER: u32 = 1;

    trait TestElement: Copy + Eq + Debug {
        fn random(rng: &mut TestRng) -> Self;
    }

    impl TestElement for u32 {
        fn random(rng: &mut TestRng) -> Self {
            rng.next_u32()
        }
    }

    impl TestElement for TestTableInfo {
        fn random(rng: &mut TestRng) -> Self {
            let mut words = [0u64; 12];
            for word in &mut words {
                *word = rng.next_u64();
            }
            TestTableInfo { words }
        }
    }

    fn rng_from_seed(seed: u64, salt: u64) -> TestRng {
        let mixed = seed ^ salt.wrapping_mul(0x9e3779b97f4a7c15);
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&mixed.to_le_bytes());
        bytes[8..16].copy_from_slice(&seed.rotate_left(17).to_le_bytes());
        bytes[16..24].copy_from_slice(&salt.to_le_bytes());
        bytes[24..32].copy_from_slice(&mixed.rotate_left(32).to_le_bytes());
        TestRng::from_seed(RngAlgorithm::ChaCha, &bytes)
    }

    fn fuzz_steps(element_count_max: u32) -> usize {
        let multiplier = crate::test_utils::proptest_fuzz_multiplier(DEFAULT_FUZZ_STEPS_MULTIPLIER);
        (element_count_max as usize)
            .saturating_mul(multiplier as usize)
            .max(1)
    }

    fn range_inclusive(rng: &mut TestRng, max: u32) -> u32 {
        if max == u32::MAX {
            return rng.next_u32();
        }
        let p = (max as u64) + 1;
        fast_range(rng.next_u64(), p) as u32
    }

    struct FuzzHarness<T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32> {
        rng: TestRng,
        pool: P,
        array: Option<SegmentedArray<T, P, ELEMENT_COUNT_MAX, true>>,
        reference: Vec<T>,
        inserts: u64,
        removes: u64,
    }

    impl<T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32> Drop
        for FuzzHarness<T, P, ELEMENT_COUNT_MAX>
    {
        fn drop(&mut self) {
            if let Some(array) = self.array.take() {
                array.deinit(&mut self.pool);
            }
        }
    }

    impl<T: TestElement, P: NodePool, const ELEMENT_COUNT_MAX: u32>
        FuzzHarness<T, P, ELEMENT_COUNT_MAX>
    {
        fn new(
            rng: TestRng,
            pool: P,
            array: SegmentedArray<T, P, ELEMENT_COUNT_MAX, true>,
        ) -> Self {
            Self {
                rng,
                pool,
                array: Some(array),
                reference: Vec::with_capacity(ELEMENT_COUNT_MAX as usize),
                inserts: 0,
                removes: 0,
            }
        }

        fn array_ref(&self) -> &SegmentedArray<T, P, ELEMENT_COUNT_MAX, true> {
            self.array.as_ref().expect("array missing")
        }

        fn array_mut(&mut self) -> &mut SegmentedArray<T, P, ELEMENT_COUNT_MAX, true> {
            self.array.as_mut().expect("array missing")
        }

        fn run_phase(&mut self, steps: usize, insert_weight: u32, remove_weight: u32) {
            let total = insert_weight + remove_weight;
            for _ in 0..steps {
                let roll = range_inclusive(&mut self.rng, total - 1);
                if roll < insert_weight {
                    self.insert();
                } else {
                    self.remove();
                }
            }
        }

        fn insert(&mut self) {
            let reference_len = self.reference.len() as u32;
            let count_free = ELEMENT_COUNT_MAX - reference_len;

            if count_free == 0 {
                return;
            }

            let node_capacity = SegmentedArray::<T, P, ELEMENT_COUNT_MAX, true>::NODE_CAPACITY;
            let count_max = count_free.min(node_capacity.saturating_mul(3));
            let count = range_inclusive(&mut self.rng, count_max - 1) + 1;
            let index = range_inclusive(&mut self.rng, reference_len);

            let mut elements = Vec::with_capacity(count as usize);
            for _ in 0..count {
                elements.push(T::random(&mut self.rng));
            }

            let array = self.array.as_mut().expect("array missing");
            array.insert_elements(&mut self.pool, index, &elements);
            let insert_at = index as usize;
            let _ = self
                .reference
                .splice(insert_at..insert_at, elements.iter().copied());

            self.inserts += count as u64;
            self.verify();
        }

        fn remove(&mut self) {
            let reference_len = self.reference.len() as u32;
            if reference_len == 0 {
                return;
            }

            let node_capacity = SegmentedArray::<T, P, ELEMENT_COUNT_MAX, true>::NODE_CAPACITY;
            let count_max = reference_len.min(node_capacity.saturating_mul(3));
            let count = range_inclusive(&mut self.rng, count_max - 1) + 1;
            let index = range_inclusive(&mut self.rng, reference_len - count);

            let array = self.array.as_mut().expect("array missing");
            array.remove_elements(&mut self.pool, index, count);
            let start = index as usize;
            let end = start + count as usize;
            self.reference.drain(start..end);

            self.removes += count as u64;
            self.verify();
        }

        fn insert_before_first(&mut self) {
            let insert_index = self
                .array_ref()
                .absolute_index_for_cursor(self.array_ref().first());
            let element = T::random(&mut self.rng);

            let array = self.array.as_mut().expect("array missing");
            array.insert_elements(&mut self.pool, insert_index, &[element]);
            self.reference.insert(insert_index as usize, element);

            self.inserts += 1;
            self.verify();
        }

        fn remove_last(&mut self) {
            let remove_index = self
                .array_ref()
                .absolute_index_for_cursor(self.array_ref().last());

            let array = self.array.as_mut().expect("array missing");
            array.remove_elements(&mut self.pool, remove_index, 1);
            self.reference.remove(remove_index as usize);

            self.removes += 1;
            self.verify();
        }

        fn remove_all(&mut self) {
            while !self.reference.is_empty() {
                self.remove();
            }

            assert_eq!(0, self.array_ref().len());
            assert!(self.inserts > 0);
            assert_eq!(self.inserts, self.removes);

            self.verify();
        }

        fn verify(&self) {
            let array = self.array_ref();
            array.verify();

            assert_eq!(self.reference.len() as u32, array.len());

            let mut actual = Vec::with_capacity(self.reference.len());
            for node in 0..array.node_count {
                actual.extend_from_slice(array.node_elements(node));
            }
            assert_eq!(self.reference, actual);

            let mut actual_rev = Vec::with_capacity(self.reference.len());
            let mut node = array.node_count;
            while node > 0 {
                node -= 1;
                let elements = array.node_elements(node);
                for element in elements.iter().rev() {
                    actual_rev.push(*element);
                }
            }
            let mut expected_rev = self.reference.clone();
            expected_rev.reverse();
            assert_eq!(expected_rev, actual_rev);

            if !self.reference.is_empty() {
                for (index, _) in self.reference.iter().enumerate() {
                    let absolute = index as u32;
                    let cursor = array.cursor_for_absolute_index(absolute);
                    assert_eq!(absolute, array.absolute_index_for_cursor(cursor));
                }
            }

            if array.is_empty() {
                assert_eq!(array.node_count, 0);
            }

            for node in &array.nodes[array.node_count as usize..] {
                assert!(node.is_none());
            }

            let half = SegmentedArray::<T, P, ELEMENT_COUNT_MAX, true>::NODE_CAPACITY / 2;
            if array.node_count > 1 {
                for node in 0..array.node_count - 1 {
                    assert!(array.count(node) >= half);
                }
            }
        }
    }

    fn run_unsorted_case<
        T: TestElement,
        const NODE_SIZE: usize,
        const NODE_ALIGNMENT: usize,
        const ELEMENT_COUNT_MAX: u32,
    >(
        seed: u64,
        salt: u64,
    ) {
        let rng = rng_from_seed(seed, salt);
        let node_count_max = SegmentedArray::<
            T,
            NodePoolType<{ NODE_SIZE }, { NODE_ALIGNMENT }>,
            { ELEMENT_COUNT_MAX },
            true,
        >::NODE_COUNT_MAX;
        let pool = NodePoolType::<{ NODE_SIZE }, { NODE_ALIGNMENT }>::init(node_count_max);
        let array = SegmentedArray::<
            T,
            NodePoolType<{ NODE_SIZE }, { NODE_ALIGNMENT }>,
            { ELEMENT_COUNT_MAX },
            true,
        >::new();

        let mut context = FuzzHarness::new(rng, pool, array);
        let steps = fuzz_steps(ELEMENT_COUNT_MAX);

        context.run_phase(steps, 60, 40);
        context.run_phase(steps, 40, 60);

        if context.inserts > 0 {
            context.remove_all();
        }

        while context.array_ref().len() < ELEMENT_COUNT_MAX {
            context.insert_before_first();
        }

        {
            let array = context.array_ref();
            assert!(array.node_count >= node_count_max.saturating_sub(1));
        }

        let last_count = context
            .array_ref()
            .count(context.array_ref().node_count - 1);
        for _ in 0..(last_count.saturating_sub(1) as usize) {
            context.remove_last();
            context.insert_before_first();
        }

        {
            let array = context.array_ref();
            assert_eq!(array.node_count, node_count_max);
        }

        context.remove_all();
    }

    fn run_all_unsorted_cases(seed: u64) {
        let mut tested_padding = false;
        let mut tested_node_capacity_min = false;
        let mut salt = 0u64;

        macro_rules! run_case {
            ($ty:ty, $node_size:expr, $node_alignment:expr, $element_count_max:expr) => {{
                salt = salt.wrapping_add(1);
                run_unsorted_case::<$ty, $node_size, $node_alignment, $element_count_max>(
                    seed, salt,
                );
                if $node_size % std::mem::size_of::<$ty>() != 0 {
                    tested_padding = true;
                }
                type Array = SegmentedArray<
                    $ty,
                    NodePoolType<$node_size, $node_alignment>,
                    $element_count_max,
                    true,
                >;
                if Array::NODE_CAPACITY == 2 {
                    tested_node_capacity_min = true;
                }
            }};
        }

        run_case!(u32, 8, 8, 3);
        run_case!(u32, 8, 8, 4);
        run_case!(u32, 8, 8, 5);
        run_case!(u32, 8, 8, 6);
        run_case!(u32, 8, 8, 1024);
        run_case!(u32, 16, 8, 1024);
        run_case!(u32, 32, 8, 1024);
        run_case!(u32, 64, 8, 1024);
        run_case!(TestTableInfo, 256, 32, 3);
        run_case!(TestTableInfo, 256, 32, 4);
        run_case!(TestTableInfo, 256, 32, 1024);
        run_case!(TestTableInfo, 512, 32, 1024);
        run_case!(TestTableInfo, 1024, 32, 1024);

        assert!(tested_padding);
        assert!(tested_node_capacity_min);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(
            crate::test_utils::proptest_cases(DEFAULT_PROPTEST_CASES)
        ))]

        #[test]
        fn segmented_array_unsorted_fuzz(seed in any::<u64>()) {
            run_all_unsorted_cases(seed);
        }
    }
}
