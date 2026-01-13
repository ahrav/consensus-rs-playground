// Port of segmented_array.zig to Rust.
// Notes:
// - This implementation intentionally mirrors Zig's "bitwise move" semantics by requiring `T: Copy`.
// - Node memory comes from a user-supplied `NodePool` that hands out fixed-size, aligned buffers.
// - `SegmentedArray` (unsorted) exposes `insert_elements` / `remove_elements`.
// - `SortedSegmentedArray` adds `search` and `insert_element` (left-biased for duplicates).

#![allow(dead_code)]

#[path = "node_pool.rs"]
mod node_pool;

use crate::lsm::binary_search::{binary_search_keys, binary_search_values_upsert_index, Config};
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::ptr::{self, NonNull};
use std::slice;
use node_pool::NodePool;
#[cfg(test)]
use node_pool::NodePoolType;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Direction {
    Ascending = 0,
    Descending = 1,
}

impl Direction {
    #[inline]
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

#[inline]
const fn div_ceil_u32(n: u32, d: u32) -> u32 {
    // Equivalent to Zig's stdx.div_ceil for u32, using wider arithmetic to avoid overflow.
    let n64 = n as u64;
    let d64 = d as u64;
    ((n64 + d64 - 1) / d64) as u32
}

/// Unsorted segmented array (a packed unrolled linked-list backed by a node pool).
///
/// This mirrors `SegmentedArrayType(T, NodePool, element_count_max, options)` when `Key == null`.
pub struct SegmentedArray<
    T: Copy,
    P: NodePool,
    const ELEMENT_COUNT_MAX: u32,
    const VERIFY: bool = false,
> {
    node_count: u32,
    nodes: Vec<Option<NonNull<u8>>>, // length = NODE_COUNT_MAX
    indexes: Vec<u32>,               // length = NODE_COUNT_MAX + 1
    _marker: PhantomData<(T, P)>,
}

impl<T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32, const VERIFY: bool>
    SegmentedArray<T, P, ELEMENT_COUNT_MAX, VERIFY>
{
    /// Rounded down to an even number, like Zig.
    pub const NODE_CAPACITY: u32 = {
        let max = (P::NODE_SIZE / mem::size_of::<T>()) as u32;
        if max % 2 == 0 { max } else { max - 1 }
    };

    pub const NODE_COUNT_MAX_NAIVE: u32 = div_ceil_u32(ELEMENT_COUNT_MAX, Self::NODE_CAPACITY / 2);

    pub const NODE_COUNT_MAX: u32 = {
        let half = (Self::NODE_CAPACITY / 2) as u64;
        let naive = Self::NODE_COUNT_MAX_NAIVE as u64;
        let threshold = (Self::NODE_CAPACITY as u64)
            + 1
            + ((Self::NODE_COUNT_MAX_NAIVE.saturating_sub(3) as u64) * half)
            + 1;

        if (ELEMENT_COUNT_MAX as u64) >= threshold {
            naive as u32
        } else {
            // Safe because if NODE_CAPACITY is sane, NODE_COUNT_MAX_NAIVE >= 1.
            (naive as u32) - 1
        }
    };

    pub fn new() -> Self {
        // Mirrors Zig's comptime asserts (as runtime asserts here).
        assert!(mem::size_of::<T>() > 0, "ZSTs are not supported");
        assert!(Self::NODE_CAPACITY >= 2, "node capacity too small");
        assert!(Self::NODE_CAPACITY % 2 == 0, "node capacity must be even");
        assert!(
            (P::NODE_ALIGNMENT as usize) >= mem::align_of::<T>(),
            "NodePool alignment too small for T"
        );
        assert!(
            ELEMENT_COUNT_MAX > Self::NODE_CAPACITY,
            "use a normal array: element_count_max must be > node_capacity"
        );

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

    /// Mirrors Zig's `deinit(array, allocator, node_pool)`.
    /// Consumes `self`, releases nodes back to `pool`, and drops the index tables.
    pub fn deinit(mut self, pool: &mut P) {
        if VERIFY {
            self.verify();
        }

        for i in 0..(self.node_count as usize) {
            if let Some(ptr) = self.nodes[i].take() {
                pool.release(ptr);
            }
        }

        // `Vec` drops itself.
    }

    /// Safer "reset": releases all nodes and returns to empty state while keeping tables allocated.
    pub fn clear(&mut self, pool: &mut P) {
        if VERIFY {
            self.verify();
        }

        for i in 0..(self.node_count as usize) {
            if let Some(ptr) = self.nodes[i].take() {
                pool.release(ptr);
            }
        }

        self.node_count = 0;
        for slot in self.nodes.iter_mut() {
            *slot = None;
        }
        self.indexes[0] = 0;

        if VERIFY {
            self.verify();
        }
    }

    /// Mirrors Zig's `reset()` as written: it does NOT return nodes to the pool.
    /// Only call this when the array is already empty (or you intentionally want to leak nodes).
    pub fn reset(&mut self) {
        self.node_count = 0;
        for slot in self.nodes.iter_mut() {
            *slot = None;
        }
        self.indexes.fill(0);
        self.indexes[0] = 0;

        if VERIFY {
            self.verify();
        }
    }

    pub fn verify(&self) {
        assert!(self.node_count <= Self::NODE_COUNT_MAX);

        for (i, node) in self.nodes.iter().enumerate() {
            if (i as u32) < self.node_count {
                assert!(node.is_some());
            } else {
                assert!(node.is_none());
            }
        }

        let half = Self::NODE_CAPACITY / 2;

        for node in 0..self.node_count {
            let count = self.count(node);
            assert!(count <= Self::NODE_CAPACITY);

            if node < self.node_count.saturating_sub(1) {
                assert!(count >= half);
            }
        }
    }

    /// Insert elements at an absolute index. Available only in the unsorted variant in Zig.
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
        assert!(
            (absolute_index as u64 + elements.len() as u64) <= (ELEMENT_COUNT_MAX as u64),
            "would exceed element_count_max"
        );

        let mut i: usize = 0;
        let node_capacity = Self::NODE_CAPACITY as usize;

        while i < elements.len() {
            let batch = (elements.len() - i).min(node_capacity);
            let abs = absolute_index + (i as u32);
            self.insert_elements_batch(pool, abs, &elements[i..i + batch]);
            i += batch;
        }
        assert_eq!(i, elements.len());
    }

    fn insert_elements_batch(&mut self, pool: &mut P, absolute_index: u32, elements: &[T]) {
        assert!(!elements.is_empty());
        assert!(elements.len() <= (Self::NODE_CAPACITY as usize));
        assert!(
            (absolute_index as u64 + elements.len() as u64) <= (ELEMENT_COUNT_MAX as u64),
            "would exceed element_count_max"
        );

        if self.node_count == 0 {
            assert_eq!(absolute_index, 0);
            self.insert_empty_node_at(pool, 0);
            debug_assert_eq!(self.node_count, 1);
            debug_assert!(self.nodes[0].is_some());
            debug_assert_eq!(self.indexes[0], 0);
            debug_assert_eq!(self.indexes[1], 0);
        }

        let cursor = self.cursor_for_absolute_index(absolute_index);
        assert!(cursor.node < self.node_count);
        let a = cursor.node;

        let a_count = self.count(a);
        debug_assert!(cursor.relative_index <= a_count);

        let total = (a_count as u64) + (elements.len() as u64);
        let node_capacity = Self::NODE_CAPACITY as u64;

        if total <= node_capacity {
            // Simple in-node insert.
            let a_ptr = self.nodes[a as usize].expect("node missing");
            unsafe {
                let buf = Self::node_buf_from_ptr_mut(a_ptr);

                let rel = cursor.relative_index as usize;
                let count = a_count as usize;
                let n = elements.len();

                // Shift tail right by n.
                let src = buf.as_ptr().add(rel);
                let dst = buf.as_mut_ptr().add(rel + n);
                let len_to_move = count - rel;
                ptr::copy(src, dst, len_to_move);

                // Write new elements into the gap.
                let dst_elems = buf.as_mut_ptr().add(rel);
                let src_elems = elements.as_ptr() as *const MaybeUninit<T>;
                ptr::copy_nonoverlapping(src_elems, dst_elems, n);
            }

            self.increment_indexes_after(a, elements.len() as u32);
            return;
        }

        // Split a into {a,b} and insert across both halves.
        let b = a + 1;
        self.insert_empty_node_at(pool, b);

        let a_ptr = self.nodes[a as usize].expect("node missing");
        let b_ptr = self.nodes[b as usize].expect("node missing");

        let total_u32 = total as u32;
        let a_half = div_ceil_u32(total_u32, 2);
        let b_half = total_u32 - a_half;

        debug_assert!(a_half >= b_half);
        debug_assert_eq!(a_half + b_half, total_u32);

        unsafe {
            let a_full = Self::node_buf_from_ptr_mut(a_ptr);
            let b_full = Self::node_buf_from_ptr_mut(b_ptr);

            let a_half_slice = &mut a_full[..(a_half as usize)];
            let b_half_slice = &mut b_full[..(b_half as usize)];

            let rel = cursor.relative_index as usize;
            let count = a_count as usize;

            // Move the tail of `a` to make room for `elements`, potentially spilling into `b`.
            let source_tail = &a_full[rel..count];
            Self::copy_backwards(
                a_half_slice,
                b_half_slice,
                rel + elements.len(),
                source_tail,
            );

            // If the insertion point is in the `b` half, move the middle chunk into `b`.
            if (a_half as usize) < rel {
                let mid = &a_full[(a_half as usize)..rel];
                ptr::copy_nonoverlapping(mid.as_ptr(), b_half_slice.as_mut_ptr(), mid.len());
            }

            // Copy inserted elements into the concatenated halves.
            let elements_mu =
                slice::from_raw_parts(elements.as_ptr() as *const MaybeUninit<T>, elements.len());
            Self::copy_backwards(a_half_slice, b_half_slice, rel, elements_mu);
        }

        self.indexes[b as usize] = self.indexes[a as usize] + a_half;
        self.increment_indexes_after(b, elements.len() as u32);
    }

    fn copy_backwards(
        a: &mut [MaybeUninit<T>],
        b: &mut [MaybeUninit<T>],
        target: usize,
        source: &[MaybeUninit<T>],
    ) {
        debug_assert!(target + source.len() <= a.len() + b.len());

        let target_a_start = target.min(a.len());
        let target_a_end = (target + source.len()).min(a.len());
        let target_a = &mut a[target_a_start..target_a_end];

        let target_b_start = target.saturating_sub(a.len());
        let target_b_end = (target + source.len()).saturating_sub(a.len());
        let target_b = &mut b[target_b_start..target_b_end];

        debug_assert_eq!(target_a.len() + target_b.len(), source.len());

        let (source_a, source_b) = source.split_at(target_a.len());

        unsafe {
            if !target_b.is_empty() {
                ptr::copy(source_b.as_ptr(), target_b.as_mut_ptr(), source_b.len());
            }
            if !target_a.is_empty() {
                ptr::copy(source_a.as_ptr(), target_a.as_mut_ptr(), source_a.len());
            }
        }
    }

    fn insert_empty_node_at(&mut self, pool: &mut P, node: u32) {
        assert!(node <= self.node_count);
        assert!(
            self.node_count + 1 <= Self::NODE_COUNT_MAX,
            "exceeds node_count_max"
        );

        let node = node as usize;
        let node_count = self.node_count as usize;

        // Shift nodes right: [node..node_count) -> [node+1..node_count+1)
        self.nodes.copy_within(node..node_count, node + 1);

        // Shift indexes right: [node..node_count+1] -> [node+1..node_count+2]
        self.indexes.copy_within(node..(node_count + 1), node + 1);

        self.node_count += 1;

        let ptr = pool.acquire();
        self.nodes[node] = Some(ptr);

        debug_assert_eq!(self.indexes[node], self.indexes[node + 1]);
    }

    /// Remove `remove_count` elements starting at `absolute_index`.
    pub fn remove_elements(&mut self, pool: &mut P, absolute_index: u32, remove_count: u32) {
        if VERIFY {
            self.verify();
        }

        assert!(self.node_count > 0, "cannot remove from empty array");
        assert!(remove_count > 0);
        assert!(
            (absolute_index as u64 + remove_count as u64) <= (ELEMENT_COUNT_MAX as u64),
            "would exceed element_count_max"
        );
        assert!(
            absolute_index + remove_count <= self.len(),
            "remove out of bounds"
        );

        let half = Self::NODE_CAPACITY / 2;
        let mut remaining = remove_count;

        while remaining > 0 {
            let batch = remaining.min(half);
            self.remove_elements_batch(pool, absolute_index, batch);
            remaining -= batch;
        }

        if VERIFY {
            self.verify();
        }
    }

    fn remove_elements_batch(&mut self, pool: &mut P, absolute_index: u32, remove_count: u32) {
        let half = Self::NODE_CAPACITY / 2;

        assert!(self.node_count > 0);
        assert!(remove_count > 0);
        assert!(remove_count <= half);
        assert!(
            (absolute_index as u64 + remove_count as u64) <= (ELEMENT_COUNT_MAX as u64),
            "would exceed element_count_max"
        );
        assert!(
            absolute_index + remove_count <= self.len(),
            "remove out of bounds"
        );

        let cursor = self.cursor_for_absolute_index(absolute_index);
        assert!(cursor.node < self.node_count);
        let a = cursor.node;
        let a_ptr = self.nodes[a as usize].expect("node missing");

        let a_remaining = cursor.relative_index;
        let a_count = self.count(a);

        if a_remaining + remove_count <= a_count {
            // Removal is fully within node `a`.
            unsafe {
                let buf = Self::node_buf_from_ptr_mut(a_ptr);

                let start = a_remaining as usize;
                let src_start = (a_remaining + remove_count) as usize;
                let count = a_count as usize;

                let len_to_move = count - src_start;
                ptr::copy(
                    buf.as_ptr().add(src_start),
                    buf.as_mut_ptr().add(start),
                    len_to_move,
                );
            }

            self.decrement_indexes_after(a, remove_count);
            self.maybe_remove_or_merge_node_with_next(pool, a);
            return;
        }

        // Removal spans exactly two nodes: `a` and `b = a+1`.
        let b = a + 1;
        let b_ptr = self.nodes[b as usize].expect("node missing");

        let removed_from_a = a_count - a_remaining;
        let removed_from_b = remove_count - removed_from_a;
        debug_assert!(removed_from_b > 0);

        let b_count = self.count(b);
        let b_remaining_len = b_count - removed_from_b;

        // With `remove_count <= half`, we can only empty at most one of the nodes.
        debug_assert!(a_remaining > 0 || b_remaining_len > 0);

        if a_remaining >= half {
            // Keep `a` (still >= half full), shift remaining of `b` left to start.
            unsafe {
                let buf_b = Self::node_buf_from_ptr_mut(b_ptr);

                let src_start = removed_from_b as usize;
                let count = b_count as usize;
                let len_to_move = count - src_start;

                ptr::copy(
                    buf_b.as_ptr().add(src_start),
                    buf_b.as_mut_ptr(),
                    len_to_move,
                );
            }

            self.indexes[b as usize] = self.indexes[a as usize] + a_remaining;
            self.decrement_indexes_after(b, remove_count);
            self.maybe_remove_or_merge_node_with_next(pool, b);
            return;
        }

        // Build a view of the remaining elements in `b` (may start after index 0).
        let b_remaining_slice = unsafe {
            let buf_b = Self::node_buf_from_ptr(b_ptr);
            let start = removed_from_b as usize;
            let end = (removed_from_b + b_remaining_len) as usize;
            &buf_b[start..end]
        };

        if b_remaining_len >= half {
            // Keep `b` (still >= half full), maybe rebalance by moving some of `b` into `a`.
            self.indexes[b as usize] = self.indexes[a as usize] + a_remaining;
            self.decrement_indexes_after(b, remove_count);
            self.maybe_merge_nodes(pool, a, b_remaining_slice);
        } else {
            // Merge all remaining from `b` into `a`, then remove empty `b`.
            debug_assert!(a_remaining < half && b_remaining_len < half);
            debug_assert!(a_remaining + b_remaining_len <= Self::NODE_CAPACITY);

            unsafe {
                let buf_a = Self::node_buf_from_ptr_mut(a_ptr);
                let dst_start = a_remaining as usize;
                let dst_end = (a_remaining + b_remaining_len) as usize;
                let dst = &mut buf_a[dst_start..dst_end];

                ptr::copy_nonoverlapping(
                    b_remaining_slice.as_ptr(),
                    dst.as_mut_ptr(),
                    b_remaining_slice.len(),
                );
            }

            self.indexes[b as usize] = self.indexes[a as usize] + a_remaining + b_remaining_len;
            self.decrement_indexes_after(b, remove_count);
            self.remove_empty_node_at(pool, b);

            debug_assert!(b == self.node_count || self.count(a) >= half);
        }
    }

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

        let next_elements = unsafe {
            let buf_b = Self::node_buf_from_ptr(b_ptr);
            &buf_b[..b_count]
        };

        self.maybe_merge_nodes(pool, node, next_elements);
    }

    fn maybe_merge_nodes(
        &mut self,
        pool: &mut P,
        node: u32,
        elements_next_node: &[MaybeUninit<T>],
    ) {
        let half = Self::NODE_CAPACITY / 2;

        let a = node;
        let b = node + 1;
        assert!(b < self.node_count);

        let a_ptr = self.nodes[a as usize].expect("node missing");
        let b_ptr = self.nodes[b as usize].expect("node missing");

        let a_count = self.count(a);
        let b_count = self.count(b);

        debug_assert_eq!(elements_next_node.len(), b_count as usize);
        debug_assert!(!elements_next_node.is_empty());
        debug_assert!(
            (elements_next_node.len() as u32) >= half || b == (self.node_count - 1),
            "next node should be >= half full unless it is the last node"
        );
        debug_assert!(elements_next_node.len() <= Self::NODE_CAPACITY as usize);
        unsafe {
            let b_full = Self::node_buf_from_ptr(b_ptr);
            debug_assert!(
                (elements_next_node.as_ptr() as usize) >= (b_full.as_ptr() as usize),
                "elements_next_node must point into node b"
            );
        }

        // If `a` is empty and `b_elements` starts at `b[0]`, it would be faster to delete `a`.
        // Zig asserts this to catch performance bugs; keep as debug_assert.
        unsafe {
            let b_full = Self::node_buf_from_ptr(b_ptr);
            debug_assert!(
                !(a_count == 0 && ptr::eq(elements_next_node.as_ptr(), b_full.as_ptr())),
                "prefer removing empty node instead of merging into it"
            );
        }

        let total = a_count + (elements_next_node.len() as u32);

        if total <= Self::NODE_CAPACITY {
            // Full merge: move all of `b` into the end of `a`, then remove `b`.
            unsafe {
                let buf_a = Self::node_buf_from_ptr_mut(a_ptr);
                let dst = &mut buf_a[(a_count as usize)..(total as usize)];
                ptr::copy_nonoverlapping(elements_next_node.as_ptr(), dst.as_mut_ptr(), dst.len());
            }

            // Make `b` empty and remove it.
            self.indexes[b as usize] = self.indexes[(b + 1) as usize];
            self.remove_empty_node_at(pool, b);

            debug_assert!(self.count(a) >= half || a == (self.node_count - 1));
            return;
        }

        if a_count < half {
            // Rebalance by stealing from the front of `b`.
            let a_half = div_ceil_u32(total, 2);
            let b_half = total - a_half;

            debug_assert!(a_half >= b_half);
            debug_assert_eq!(a_half + b_half, total);

            let need_from_b = (a_half - a_count) as usize;

            unsafe {
                // Copy some elements from the front of `b_elements` into the end of `a`.
                let buf_a = Self::node_buf_from_ptr_mut(a_ptr);
                let dst = &mut buf_a[(a_count as usize)..(a_half as usize)];
                ptr::copy_nonoverlapping(
                    elements_next_node.as_ptr(),
                    dst.as_mut_ptr(),
                    need_from_b,
                );

                // Shift the remaining `b` elements left to the start of the `b` buffer.
                let buf_b = Self::node_buf_from_ptr_mut(b_ptr);
                let remaining = &elements_next_node[need_from_b..];
                ptr::copy(remaining.as_ptr(), buf_b.as_mut_ptr(), remaining.len());
            }

            self.indexes[b as usize] = self.indexes[a as usize] + a_half;

            debug_assert!(self.count(a) >= half);
            debug_assert!(self.count(b) >= half);
            return;
        }

        // Otherwise `a` is already >= half-full and `b` is assumed fine.
        // In this case, `elements_next_node` must start at `b[0]` (Zig asserts this).
        unsafe {
            let b_full = Self::node_buf_from_ptr(b_ptr);
            debug_assert!(ptr::eq(elements_next_node.as_ptr(), b_full.as_ptr()));
        }
    }

    fn remove_empty_node_at(&mut self, pool: &mut P, node: u32) {
        assert!(self.node_count > 0);
        assert!(node < self.node_count);
        assert_eq!(self.count(node), 0);

        let node = node as usize;
        let node_count = self.node_count as usize;

        let ptr = self.nodes[node].take().expect("node missing");
        pool.release(ptr);

        // Shift nodes left: [node+1..node_count) -> [node..node_count-1)
        self.nodes.copy_within((node + 1)..node_count, node);

        // Shift indexes left: [node+1..node_count+1] -> [node..node_count]
        self.indexes.copy_within((node + 1)..(node_count + 1), node);

        self.node_count -= 1;

        let new_count = self.node_count as usize;
        self.nodes[new_count] = None;
        // The tail index slot is unused; Zig leaves it undefined.
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
        for idx in &mut self.indexes[start..end] {
            *idx += delta;
        }
    }

    #[inline]
    fn decrement_indexes_after(&mut self, node: u32, delta: u32) {
        let start = (node as usize) + 1;
        let end = (self.node_count as usize) + 1;
        for idx in &mut self.indexes[start..end] {
            *idx -= delta;
        }
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
        slice::from_raw_parts(
            ptr.as_ptr() as *const MaybeUninit<T>,
            Self::NODE_CAPACITY as usize,
        )
    }

    #[inline]
    unsafe fn node_buf_from_ptr_mut<'a>(ptr: NonNull<u8>) -> &'a mut [MaybeUninit<T>] {
        slice::from_raw_parts_mut(
            ptr.as_ptr() as *mut MaybeUninit<T>,
            Self::NODE_CAPACITY as usize,
        )
    }

    pub fn node_elements(&self, node: u32) -> &[T] {
        assert!(node < self.node_count);
        let ptr = self.nodes[node as usize].expect("node missing");
        let count = self.count(node) as usize;

        unsafe {
            let buf = Self::node_buf_from_ptr(ptr);
            let init = &buf[..count];
            MaybeUninit::slice_assume_init_ref(init)
        }
    }

    pub fn node_last_element(&self, node: u32) -> T {
        let elems = self.node_elements(node);
        assert!(!elems.is_empty());
        elems[elems.len() - 1]
    }

    pub fn element_at_cursor(&self, cursor: Cursor) -> T {
        let elems = self.node_elements(cursor.node);
        elems[cursor.relative_index as usize]
    }

    #[inline]
    pub fn first(&self) -> Cursor {
        Cursor {
            node: 0,
            relative_index: 0,
        }
    }

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
    pub fn len(&self) -> u32 {
        let result = self.indexes[self.node_count as usize];
        assert!(result <= ELEMENT_COUNT_MAX);
        result
    }

    pub fn absolute_index_for_cursor(&self, cursor: Cursor) -> u32 {
        if self.node_count == 0 {
            assert_eq!(cursor.node, 0);
            assert_eq!(cursor.relative_index, 0);
            return 0;
        }

        assert!(cursor.node < self.node_count);

        let count = self.count(cursor.node);
        if cursor.node == self.node_count - 1 {
            assert!(cursor.relative_index <= count);
        } else {
            assert!(cursor.relative_index < count);
        }

        self.indexes[cursor.node as usize] + cursor.relative_index
    }

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

        // Cursor may point one past the end (only legal in the last node).
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

            // Descending: start at the last element.
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

/// Iterator that mirrors the Zig iterator behavior:
/// - It is created from `&SegmentedArray`
/// - `next()` yields a raw `*mut T` (like Zig's `?*T`)
/// - You may stop early by setting `done = true`.
pub struct Iterator<'a, T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32, const VERIFY: bool> {
    array: *const SegmentedArray<T, P, ELEMENT_COUNT_MAX, VERIFY>,
    direction: Direction,
    cursor: Cursor,
    pub done: bool,
    _marker: PhantomData<&'a SegmentedArray<T, P, ELEMENT_COUNT_MAX, VERIFY>>,
}

impl<'a, T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32, const VERIFY: bool>
    Iterator<'a, T, P, ELEMENT_COUNT_MAX, VERIFY>
{
    pub fn next(&mut self) -> Option<*mut T> {
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
        let element_ptr =
            unsafe { (node_ptr.as_ptr() as *mut MaybeUninit<T>).add(rel) as *mut T };

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

/// Compile-time key function, matching Zig's `key_from_value`.
pub trait KeyFromValue<T: Copy> {
    type Key: Ord + Copy;
    fn key_from_value(value: &T) -> Self::Key;
}

/// Sorted wrapper that mirrors `SortedSegmentedArrayType`.
///
/// Internally it uses a `SegmentedArray<..., VERIFY=false>` and performs verification (including order)
/// in this wrapper when `VERIFY == true`.
pub struct SortedSegmentedArray<
    T: Copy,
    P: NodePool,
    const ELEMENT_COUNT_MAX: u32,
    KF: KeyFromValue<T>,
    const VERIFY: bool = false,
> {
    base: SegmentedArray<T, P, ELEMENT_COUNT_MAX, false>,
    _marker: PhantomData<KF>,
}

impl<T: Copy, P: NodePool, const ELEMENT_COUNT_MAX: u32, KF: KeyFromValue<T>, const VERIFY: bool>
    SortedSegmentedArray<T, P, ELEMENT_COUNT_MAX, KF, VERIFY>
{
    pub const NODE_CAPACITY: u32 = SegmentedArray::<T, P, ELEMENT_COUNT_MAX, false>::NODE_CAPACITY;
    pub const NODE_COUNT_MAX_NAIVE: u32 =
        SegmentedArray::<T, P, ELEMENT_COUNT_MAX, false>::NODE_COUNT_MAX_NAIVE;
    pub const NODE_COUNT_MAX: u32 =
        SegmentedArray::<T, P, ELEMENT_COUNT_MAX, false>::NODE_COUNT_MAX;

    pub fn new() -> Self {
        let s = Self {
            base: SegmentedArray::<T, P, ELEMENT_COUNT_MAX, false>::new(),
            _marker: PhantomData,
        };

        if VERIFY {
            s.verify();
        }

        s
    }

    pub fn deinit(self, pool: &mut P) {
        self.base.deinit(pool);
    }

    pub fn clear(&mut self, pool: &mut P) {
        if VERIFY {
            self.verify();
        }
        self.base.clear(pool);
        if VERIFY {
            self.verify();
        }
    }

    pub fn len(&self) -> u32 {
        self.base.len()
    }

    pub fn node_elements(&self, node: u32) -> &[T] {
        self.base.node_elements(node)
    }

    pub fn absolute_index_for_cursor(&self, cursor: Cursor) -> u32 {
        self.base.absolute_index_for_cursor(cursor)
    }

    pub fn iterator_from_cursor<'a>(
        &'a self,
        cursor: Cursor,
        direction: Direction,
    ) -> Iterator<'a, T, P, ELEMENT_COUNT_MAX, false> {
        self.base.iterator_from_cursor(cursor, direction)
    }

    pub fn iterator_from_index<'a>(
        &'a self,
        absolute_index: u32,
        direction: Direction,
    ) -> Iterator<'a, T, P, ELEMENT_COUNT_MAX, false> {
        self.base.iterator_from_index(absolute_index, direction)
    }

    pub fn verify(&self) {
        self.base.verify();

        // Verify non-decreasing key order across all elements.
        let mut prior: Option<KF::Key> = None;

        for node in 0..self.base.node_count {
            for v in self.base.node_elements(node) {
                let k = KF::key_from_value(v);
                if let Some(p) = prior {
                    assert!(p <= k);
                }
                prior = Some(k);
            }
        }
    }

    /// Left-biased search (lower bound) over the whole structure.
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

            // Compare against the first element of the mid node.
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

        // Canonicalize "end of node" cursor to "start of next node" for non-last nodes.
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

    /// Insert a single element, preserving sort order. Returns absolute index inserted at.
    pub fn insert_element(&mut self, pool: &mut P, element: T) -> u32 {
        if VERIFY {
            self.verify();
        }

        let count_before = self.len();
        let key = KF::key_from_value(&element);
        let cursor = self.search(key);
        let absolute_index = self.base.absolute_index_for_cursor(cursor);

        // Call the base's internal insertion to avoid exposing `insert_elements` on the sorted wrapper.
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
    use super::*;

    struct U32Key;
    impl KeyFromValue<u32> for U32Key {
        type Key = u32;
        fn key_from_value(value: &u32) -> Self::Key {
            *value
        }
    }

    #[test]
    fn sorted_duplicate_elements_left_biased() {
        // Mirrors the Zig test in segmented_array.zig.
        type Pool = NodePoolType<{ 128 * 4 }, { 2 * 4 }>; // node_size=512, alignment=8
        type Array = SortedSegmentedArray<u32, Pool, 1024, U32Key, true>;

        let mut pool = Pool::init(Array::NODE_COUNT_MAX);
        let mut array = Array::new();

        for i in 0..3u32 {
            let inserted_at = array.insert_element(&mut pool, 0);
            assert_eq!(inserted_at, 0);

            let inserted_at = array.insert_element(&mut pool, 100);
            assert_eq!(inserted_at, i + 1);

            let inserted_at = array.insert_element(&mut pool, u32::MAX);
            assert_eq!(inserted_at, (i + 1) * 2);
        }

        assert_eq!(array.len(), 9);

        let c0 = array.search(0);
        assert_eq!(array.absolute_index_for_cursor(c0), 0);

        let c100 = array.search(100);
        assert_eq!(array.absolute_index_for_cursor(c100), 3);

        let cmax = array.search(u32::MAX);
        assert_eq!(array.absolute_index_for_cursor(cmax), 6);

        {
            let target: u32 = 0;
            let mut it = array.iterator_from_cursor(array.search(target), Direction::Ascending);
            unsafe {
                assert_eq!(*it.next().unwrap(), 0);
                assert_eq!(*it.next().unwrap(), 0);
                assert_eq!(*it.next().unwrap(), 0);
                assert_eq!(*it.next().unwrap(), 100);
            }

            let mut it = array.iterator_from_cursor(array.search(target), Direction::Descending);
            unsafe {
                assert_eq!(*it.next().unwrap(), 0);
            }
            assert!(it.next().is_none());
        }

        {
            let target: u32 = 100;
            let mut it = array.iterator_from_cursor(array.search(target), Direction::Ascending);
            unsafe {
                assert_eq!(*it.next().unwrap(), 100);
                assert_eq!(*it.next().unwrap(), 100);
                assert_eq!(*it.next().unwrap(), 100);
                assert_eq!(*it.next().unwrap(), u32::MAX);
            }

            let mut it = array.iterator_from_cursor(array.search(target), Direction::Descending);
            unsafe {
                assert_eq!(*it.next().unwrap(), 100);
                assert_eq!(*it.next().unwrap(), 0);
            }
        }

        {
            let target: u32 = u32::MAX;
            let mut it = array.iterator_from_cursor(array.search(target), Direction::Ascending);
            unsafe {
                assert_eq!(*it.next().unwrap(), u32::MAX);
                assert_eq!(*it.next().unwrap(), u32::MAX);
                assert_eq!(*it.next().unwrap(), u32::MAX);
            }
            assert!(it.next().is_none());

            let mut it = array.iterator_from_cursor(array.search(target), Direction::Descending);
            unsafe {
                assert_eq!(*it.next().unwrap(), u32::MAX);
                assert_eq!(*it.next().unwrap(), 100);
            }
        }

        array.deinit(&mut pool);
        pool.deinit();
    }
}
