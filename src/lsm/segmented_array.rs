#![allow(dead_code)]

use std::{marker::PhantomData, ptr::NonNull};

use crate::lsm::node_pool::NodePool;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C)]
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

    pub fn deinit(mut self, pool: &mut P) {
        if VERIFY {
            self.verify();
        }

        self.nodes[..self.node_count as usize]
            .iter_mut()
            .filter_map(Option::take)
            .for_each(|ptr| pool.release(ptr));
    }

    pub fn clear(mut self, pool: &mut P) {
        if VERIFY {
            self.verify();
        }

        self.nodes[..self.node_count as usize]
            .iter_mut()
            .filter_map(Option::take)
            .for_each(|ptr| pool.release(ptr));

        self.node_count = 0;
        self.nodes.iter_mut().map(|s| *s = None);
        self.indexes[0] = 0;

        if VERIFY {
            self.verify();
        }
    }

    pub fn reset(mut self) {
        self.node_count = 0;
        self.nodes.iter_mut().map(|s| *s = None);
        self.indexes.fill(0);

        if VERIFY {
            self.verify();
        }
    }

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

    #[inline]
    fn count(&self, node: u32) -> u32 {
        let n = node as usize;
        let result = self.indexes[n + 1] - self.indexes[n];
        assert!(result <= Self::NODE_CAPACITY);
        result
    }
}
