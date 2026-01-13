#![allow(dead_code)]

use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    slice,
};

use crate::lsm::{
    binary_search::{Config, binary_search_keys},
    node_pool::NodePool,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum Direction {
    Ascending = 0,
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
        assert!((P::NODE_ALIGNMENT as usize) >= align_of::<T>());
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
    /// Consumes `self`; intended for teardown paths where `VERIFY` should see an
    /// empty structure before the value is dropped.
    pub fn clear(mut self, pool: &mut P) {
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

        while i < node_capacity {
            let batch = (elements.len() - 1).min(node_capacity);
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
    /// Returns early if `node` is empty (caller handles removal separately). The
    /// next node's elements are sliced from its backing buffer and forwarded to
    /// `maybe_merge_nodes`. Caller must ensure `node + 1` is in-bounds.
    fn maybe_remove_or_merge_node_with_next(&mut self, pool: &mut P, node: u32) {
        assert!(node < self.node_count);

        if self.count(node) == 0 {
            return;
        }

        let next = node + 1;
        let b_ptr = self.nodes[node as usize].expect("node missing");
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
        assert!(self.node_count + 1 <= Self::NODE_COUNT_MAX);

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

        // Shift nodes left: [node+1..node_count+1] -> [node..node_count]
        self.nodes.copy_within(node + 1..node_count + 1, node);

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
}
