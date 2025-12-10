use core::marker::PhantomData;
use core::ptr::NonNull;

// Compile-time validations
const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

/// Intrusive double-link for a doubly linked list.
#[derive(Debug)]
pub struct ListLink<T, Tag> {
    prev: Option<NonNull<T>>,
    next: Option<NonNull<T>>,
    /// Tracks whether this node is currently in a list.
    /// Required because head/tail nodes have None prev/next respectively,
    /// which would otherwise be indistinguishable from an unlinked node.
    linked: bool,
    _tag: PhantomData<Tag>,
}

impl<T, Tag> ListLink<T, Tag> {
    pub const fn new() -> Self {
        Self {
            prev: None,
            next: None,
            linked: false,
            _tag: PhantomData,
        }
    }

    /// Returns true if this node is currently in a list.
    #[inline]
    pub fn is_linked(&self) -> bool {
        self.linked
    }

    /// Reset link to unlinked state.
    pub fn reset(&mut self) {
        self.prev = None;
        self.next = None;
        self.linked = false;

        // Postconditions
        assert!(!self.is_linked());
        assert!(self.prev.is_none());
        assert!(self.next.is_none());
    }
}

impl<T, Tag> Default for ListLink<T, Tag> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for nodes that can be placed in an intrusive doubly linked list.
pub trait ListNode<Tag>: Sized {
    fn list_link(&mut self) -> &mut ListLink<Self, Tag>;
    fn list_link_ref(&self) -> &ListLink<Self, Tag>;
}

/// Intrusive doubly linked list.
#[derive(Debug)]
pub struct DoublyLinkedList<T, Tag>
where
    T: ListNode<Tag>,
{
    head: Option<NonNull<T>>,
    tail: Option<NonNull<T>>,
    len: u32,
    _tag: PhantomData<Tag>,
}

impl<T, Tag> Default for DoublyLinkedList<T, Tag>
where
    T: ListNode<Tag>,
{
    fn default() -> Self {
        Self::init()
    }
}

impl<T, Tag> DoublyLinkedList<T, Tag>
where
    T: ListNode<Tag>,
{
    pub const fn init() -> Self {
        Self {
            head: None,
            tail: None,
            len: 0,
            _tag: PhantomData,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        let empty = self.head.is_none();

        assert!(empty == (self.len == 0));
        assert!(empty == self.tail.is_none());

        empty
    }

    #[inline]
    pub fn len(&self) -> u32 {
        // Invariant: len == 0 iff head/tail are None
        assert!((self.len == 0) == self.head.is_none());
        assert!((self.len == 0) == self.tail.is_none());

        self.len
    }

    #[inline]
    pub fn peek_front(&self) -> Option<NonNull<T>> {
        self.head
    }

    #[inline]
    pub fn peek_back(&self) -> Option<NonNull<T>> {
        self.tail
    }

    /// Push node to back of list.
    pub fn push_back(&mut self, node: &mut T) {
        let old_len = self.len;

        assert!(
            !node.list_link_ref().is_linked(),
            "push_back: node already linked"
        );

        assert!(old_len < u32::MAX, "list length overflow");

        let node_ptr = NonNull::from(&mut *node);

        // Set up node's links
        {
            let link = node.list_link();
            link.prev = self.tail;
            link.next = None;
            link.linked = true;
        }

        match self.tail {
            None => {
                // List was empty
                assert!(self.head.is_none());
                assert!(self.len == 0);

                self.head = Some(node_ptr);
                self.tail = Some(node_ptr);
            }
            Some(mut tail_ptr) => {
                // List had elements
                assert!(self.head.is_some());
                assert!(self.len > 0);

                // SAFETY: tail_ptr is valid (we own the list structure)
                unsafe {
                    let tail = tail_ptr.as_mut();

                    assert!(
                        tail.list_link_ref().next.is_none(),
                        "tail node has non-null next"
                    );
                    assert!(tail.list_link_ref().is_linked());

                    tail.list_link().next = Some(node_ptr);
                }

                self.tail = Some(node_ptr);
            }
        }

        self.len += 1;

        assert!(self.len == old_len + 1);
        assert!(self.tail == Some(node_ptr));
        assert!(!self.is_empty());
        assert!(node.list_link_ref().is_linked());
        assert!(node.list_link_ref().next.is_none()); // It's the new tail
    }

    /// Push node to front of list.
    pub fn push_front(&mut self, node: &mut T) {
        let old_len = self.len;

        assert!(
            !node.list_link_ref().is_linked(),
            "push_front: node already linked"
        );

        assert!(old_len < u32::MAX, "list length overflow");

        let node_ptr = NonNull::from(&mut *node);

        // Set up node's links
        {
            let link = node.list_link();
            link.prev = None;
            link.next = self.head;
            link.linked = true;
        }

        match self.head {
            None => {
                // List was empty
                assert!(self.tail.is_none());
                assert!(self.len == 0);

                self.head = Some(node_ptr);
                self.tail = Some(node_ptr);
            }
            Some(mut head_ptr) => {
                // List had elements
                assert!(self.tail.is_some());
                assert!(self.len > 0);

                // SAFETY: head_ptr is valid (we own the list structure)
                unsafe {
                    let head = head_ptr.as_mut();

                    assert!(
                        head.list_link_ref().prev.is_none(),
                        "head node has non-null prev"
                    );
                    assert!(head.list_link_ref().is_linked());

                    head.list_link().prev = Some(node_ptr);
                }

                self.head = Some(node_ptr);
            }
        }

        self.len += 1;

        assert!(self.len == old_len + 1);
        assert!(self.head == Some(node_ptr));
        assert!(!self.is_empty());
        assert!(node.list_link_ref().is_linked());
        assert!(node.list_link_ref().prev.is_none()); // It's the new head
    }

    /// Pop node from front of list.
    pub fn pop_front(&mut self) -> Option<NonNull<T>> {
        let mut head_ptr = self.head?;

        let old_len = self.len;

        assert!(old_len > 0, "head exists but len is 0");

        // SAFETY: head_ptr is valid (we own the list structure)
        let next = unsafe {
            let head = head_ptr.as_ref();

            assert!(head.list_link_ref().is_linked(), "head is not linked");
            assert!(
                head.list_link_ref().prev.is_none(),
                "head has non-null prev"
            );

            head.list_link_ref().next
        };

        self.head = next;

        match next {
            Some(mut next_ptr) => {
                // SAFETY: next_ptr is valid
                unsafe {
                    let next_node = next_ptr.as_mut();

                    // Invariant: next's prev should point to old head
                    assert!(next_node.list_link_ref().prev == Some(head_ptr));

                    next_node.list_link().prev = None;

                    assert!(next_node.list_link_ref().prev.is_none());
                }
            }
            None => {
                // List is now empty
                self.tail = None;
            }
        }

        assert!(self.len > 0, "underflow in pop_front");
        self.len -= 1;

        // Reset the popped node's link
        unsafe {
            head_ptr.as_mut().list_link().reset();
        }

        assert!(self.len == old_len - 1);
        assert!(self.is_empty() == (self.len == 0));
        assert!((self.head.is_none()) == (self.tail.is_none()));

        Some(head_ptr)
    }

    /// Pop node from back of list.
    pub fn pop_back(&mut self) -> Option<NonNull<T>> {
        let mut tail_ptr = self.tail?;

        let old_len = self.len;

        assert!(old_len > 0, "tail exists but len is 0");

        // SAFETY: tail_ptr is valid (we own the list structure)
        let prev = unsafe {
            let tail = tail_ptr.as_ref();

            assert!(tail.list_link_ref().is_linked(), "tail is not linked");
            assert!(
                tail.list_link_ref().next.is_none(),
                "tail has non-null next"
            );

            tail.list_link_ref().prev
        };

        self.tail = prev;

        match prev {
            Some(mut prev_ptr) => {
                // SAFETY: prev_ptr is valid
                unsafe {
                    let prev_node = prev_ptr.as_mut();

                    // Invariant: prev's next should point to old tail
                    assert!(prev_node.list_link_ref().next == Some(tail_ptr));

                    prev_node.list_link().next = None;

                    assert!(prev_node.list_link_ref().next.is_none());
                }
            }
            None => {
                // List is now empty
                self.head = None;
            }
        }

        assert!(self.len > 0, "underflow in pop_back");
        self.len -= 1;

        // Reset the popped node's link
        unsafe {
            tail_ptr.as_mut().list_link().reset();
        }

        assert!(self.len == old_len - 1);
        assert!(self.is_empty() == (self.len == 0));
        assert!((self.head.is_none()) == (self.tail.is_none()));

        Some(tail_ptr)
    }

    /// Remove `node` in O(1).
    ///
    /// # Safety
    /// `node` must currently be in this list.
    pub unsafe fn remove(&mut self, mut node: NonNull<T>) {
        let old_len = self.len;

        assert!(old_len > 0, "remove from empty list");

        assert!(
            unsafe { node.as_ref().list_link_ref().is_linked() },
            "remove: node not linked"
        );

        let prev = unsafe { node.as_ref().list_link_ref().prev };
        let next = unsafe { node.as_ref().list_link_ref().next };

        // Update previous node's next pointer (or head)
        match prev {
            Some(mut prev_ptr) => {
                assert!(
                    unsafe { prev_ptr.as_ref().list_link_ref().next } == Some(node),
                    "broken prev->next link"
                );

                unsafe { prev_ptr.as_mut().list_link().next = next };
            }
            None => {
                // Node is the head
                assert!(self.head == Some(node), "node has no prev but is not head");

                self.head = next;
            }
        }

        // Update next node's prev pointer (or tail)
        match next {
            Some(mut next_ptr) => {
                assert!(
                    unsafe { next_ptr.as_ref().list_link_ref().prev } == Some(node),
                    "broken next->prev link"
                );

                unsafe { next_ptr.as_mut().list_link().prev = prev };
            }
            None => {
                // Node is the tail
                assert!(self.tail == Some(node), "node has no next but is not tail");

                self.tail = prev;
            }
        }

        assert!(self.len > 0, "underflow in remove");
        self.len -= 1;

        // Reset the removed node's link
        unsafe { node.as_mut().list_link().reset() };

        assert!(self.len == old_len - 1);
        assert!(unsafe { !node.as_ref().list_link_ref().is_linked() });
        assert!(self.is_empty() == (self.len == 0));
        assert!((self.head.is_none()) == (self.tail.is_none()));
    }

    /// Returns true if the list contains the given node.
    /// O(n) traversal - intended for debugging/assertions only.
    pub fn contains(&self, node: &T) -> bool {
        let target = node as *const T;
        let mut current = self.head;
        let mut visited: u32 = 0;

        while let Some(ptr) = current {
            // Safety check: detect infinite loops
            visited += 1;
            assert!(visited <= self.len, "cycle detected in list");

            if ptr.as_ptr() == target as *mut T {
                return true;
            }

            // SAFETY: ptr is valid (part of our list)
            current = unsafe { ptr.as_ref().list_link_ref().next };
        }

        false
    }

    /// Verify list invariants. For debugging/testing.
    #[cfg(debug_assertions)]
    pub fn check_invariants(&self) {
        // Empty list checks
        if self.len == 0 {
            assert!(self.head.is_none(), "len=0 but head is Some");
            assert!(self.tail.is_none(), "len=0 but tail is Some");
            return;
        }

        // Non-empty list checks
        assert!(self.head.is_some(), "len>0 but head is None");
        assert!(self.tail.is_some(), "len>0 but tail is None");

        // Check head's prev is None
        unsafe {
            let head = self.head.unwrap();
            assert!(
                head.as_ref().list_link_ref().prev.is_none(),
                "head has non-null prev"
            );
            assert!(head.as_ref().list_link_ref().is_linked(), "head not linked");
        }

        // Check tail's next is None
        unsafe {
            let tail = self.tail.unwrap();
            assert!(
                tail.as_ref().list_link_ref().next.is_none(),
                "tail has non-null next"
            );
            assert!(tail.as_ref().list_link_ref().is_linked(), "tail not linked");
        }

        // Walk forward and count
        let mut count: u32 = 0;
        let mut current = self.head;
        let mut last: Option<NonNull<T>> = None;

        while let Some(ptr) = current {
            count += 1;
            assert!(count <= self.len, "forward: more nodes than len indicates");

            unsafe {
                let node = ptr.as_ref();

                // Every node must be linked
                assert!(node.list_link_ref().is_linked(), "node not marked linked");

                // Check backward link consistency
                if let Some(prev) = node.list_link_ref().prev {
                    assert!(
                        prev.as_ref().list_link_ref().next == Some(ptr),
                        "broken prev->next link"
                    );
                }

                // Check forward link consistency
                if let Some(next) = node.list_link_ref().next {
                    assert!(
                        next.as_ref().list_link_ref().prev == Some(ptr),
                        "broken next->prev link"
                    );
                }
            }

            last = current;
            current = unsafe { ptr.as_ref().list_link_ref().next };
        }

        // Forward count matches len
        assert!(
            count == self.len,
            "forward: counted {} nodes but len is {}",
            count,
            self.len
        );

        // Last node is tail
        assert!(last == self.tail, "last forward node is not tail");

        // Walk backward and count
        let mut back_count: u32 = 0;
        let mut current = self.tail;
        let mut first: Option<NonNull<T>> = None;

        while let Some(ptr) = current {
            back_count += 1;
            assert!(
                back_count <= self.len,
                "backward: more nodes than len indicates"
            );

            first = current;
            current = unsafe { ptr.as_ref().list_link_ref().prev };
        }

        // Backward count matches forward count
        assert!(
            back_count == count,
            "backward count {} != forward count {}",
            back_count,
            count
        );

        // First node is head
        assert!(first == self.head, "first backward node is not head");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    enum Tag {}

    impl std::fmt::Debug for Tag {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Tag")
        }
    }

    #[derive(Debug)]
    struct Node {
        value: u32,
        link: ListLink<Node, Tag>,
    }

    impl Node {
        fn new(value: u32) -> Self {
            Self {
                value,
                link: ListLink::new(),
            }
        }
    }

    impl ListNode<Tag> for Node {
        fn list_link(&mut self) -> &mut ListLink<Self, Tag> {
            &mut self.link
        }
        fn list_link_ref(&self) -> &ListLink<Self, Tag> {
            &self.link
        }
    }

    #[test]
    fn init_is_empty() {
        let list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();

        assert!(list.is_empty());
        assert!(list.len() == 0);
        assert!(list.peek_front().is_none());
        assert!(list.peek_back().is_none());
    }

    #[test]
    fn push_back_pop_front_single() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(42);

        list.push_back(&mut a);

        assert!(!list.is_empty());
        assert!(list.len() == 1);
        assert!(a.link.is_linked());

        let ptr = list.pop_front().unwrap();
        assert!(unsafe { ptr.as_ref().value } == 42);

        assert!(list.is_empty());
        assert!(!a.link.is_linked());
    }

    #[test]
    fn push_front_pop_back_single() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(42);

        list.push_front(&mut a);

        assert!(!list.is_empty());
        assert!(list.len() == 1);

        let ptr = list.pop_back().unwrap();
        assert!(unsafe { ptr.as_ref().value } == 42);

        assert!(list.is_empty());
    }

    #[test]
    fn fifo_order_push_back_pop_front() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.push_back(&mut a);
        list.push_back(&mut b);
        list.push_back(&mut c);

        assert!(list.len() == 3);

        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 1);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 2);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 3);
        assert!(list.pop_front().is_none());
    }

    #[test]
    fn lifo_order_push_front_pop_front() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.push_front(&mut a);
        list.push_front(&mut b);
        list.push_front(&mut c);

        assert!(list.len() == 3);

        // Stack order: 3, 2, 1
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 3);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 2);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 1);
    }

    #[test]
    fn mixed_push_front_back() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.push_back(&mut a); // [1]
        list.push_front(&mut b); // [2, 1]
        list.push_back(&mut c); // [2, 1, 3]

        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 2);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 1);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 3);
    }

    #[test]
    fn remove_head() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.push_back(&mut a);
        list.push_back(&mut b);
        list.push_back(&mut c);

        unsafe { list.remove(NonNull::from(&mut a)) };

        assert!(list.len() == 2);
        assert!(!a.link.is_linked());

        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 2);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 3);
    }

    #[test]
    fn remove_tail() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.push_back(&mut a);
        list.push_back(&mut b);
        list.push_back(&mut c);

        unsafe { list.remove(NonNull::from(&mut c)) };

        assert!(list.len() == 2);
        assert!(!c.link.is_linked());

        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 1);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 2);
    }

    #[test]
    fn remove_middle() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.push_back(&mut a);
        list.push_back(&mut b);
        list.push_back(&mut c);

        unsafe { list.remove(NonNull::from(&mut b)) };

        assert!(list.len() == 2);
        assert!(!b.link.is_linked());

        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 1);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 3);
    }

    #[test]
    fn remove_only_element() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);

        list.push_back(&mut a);

        unsafe { list.remove(NonNull::from(&mut a)) };

        assert!(list.is_empty());
        assert!(!a.link.is_linked());
    }

    #[test]
    fn contains_finds_nodes() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.push_back(&mut a);
        list.push_back(&mut b);

        assert!(list.contains(&a));
        assert!(list.contains(&b));
        assert!(!list.contains(&c));
    }

    #[test]
    fn node_link_cleared_after_pop() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);

        list.push_back(&mut a);
        list.push_back(&mut b);

        let ptr = list.pop_front().unwrap();

        // Popped node should be unlinked
        assert!(!unsafe { ptr.as_ref().link.is_linked() });
        assert!(unsafe { ptr.as_ref().link.prev.is_none() });
        assert!(unsafe { ptr.as_ref().link.next.is_none() });
    }

    #[test]
    #[cfg(debug_assertions)]
    fn check_invariants_throughout_lifecycle() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.check_invariants(); // Empty

        list.push_back(&mut a);
        list.check_invariants();

        list.push_back(&mut b);
        list.check_invariants();

        list.push_front(&mut c);
        list.check_invariants();

        list.pop_back();
        list.check_invariants();

        list.pop_front();
        list.check_invariants();

        list.pop_front();
        list.check_invariants(); // Empty again
    }

    #[test]
    #[should_panic(expected = "push_back: node already linked")]
    fn push_back_linked_node_panics() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);

        list.push_back(&mut a);
        list.push_back(&mut a); // Should panic
    }

    #[test]
    #[should_panic(expected = "push_front: node already linked")]
    fn push_front_linked_node_panics() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);

        list.push_front(&mut a);
        list.push_front(&mut a); // Should panic
    }

    #[test]
    #[should_panic(expected = "remove: node not linked")]
    fn remove_unlinked_node_panics() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);

        list.push_back(&mut a);

        // Try to remove b which was never added
        unsafe { list.remove(NonNull::from(&mut b)) };
    }

    #[test]
    fn interleaved_operations() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);
        let mut d = Node::new(4);

        list.push_back(&mut a); // [1]
        list.push_back(&mut b); // [1, 2]
        list.pop_front(); // [2], a is now free

        list.push_front(&mut c); // [3, 2]
        list.push_back(&mut a); // [3, 2, 1] (reuse a)
        list.push_back(&mut d); // [3, 2, 1, 4]

        unsafe { list.remove(NonNull::from(&mut b)) }; // [3, 1, 4]

        assert!(list.len() == 3);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 3);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 1);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 4);
    }
}
