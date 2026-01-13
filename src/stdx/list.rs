//! Intrusive doubly-linked list.
//!
//! Unlike `std::collections::LinkedList`, nodes own their link storage via [`ListLink`],
//! enabling O(1) removal without searching. Nodes can belong to multiple lists simultaneously
//! using distinct `Tag` types.
//!
//! # Design
//!
//! Intrusive lists avoid per-node allocation and enable O(1) removal when you have a pointer
//! to the node. The tradeoff: nodes must embed [`ListLink`] and cannot move while linked.
//!
//! # Example
//!
//! ```
//! use consensus::stdx::list::{ListLink, ListNode, DoublyLinkedList};
//!
//! // Tag types to distinguish different lists
//! struct ReadyQueue;
//! struct TimerWheel;
//!
//! struct Task {
//!     id: u32,
//!     ready_link: ListLink<Task, ReadyQueue>,
//!     timer_link: ListLink<Task, TimerWheel>,
//! }
//!
//! impl Task {
//!     fn new(id: u32) -> Self {
//!         Self {
//!             id,
//!             ready_link: ListLink::new(),
//!             timer_link: ListLink::new(),
//!         }
//!     }
//! }
//!
//! impl ListNode<ReadyQueue> for Task {
//!     fn list_link(&mut self) -> &mut ListLink<Self, ReadyQueue> { &mut self.ready_link }
//!     fn list_link_ref(&self) -> &ListLink<Self, ReadyQueue> { &self.ready_link }
//! }
//!
//! impl ListNode<TimerWheel> for Task {
//!     fn list_link(&mut self) -> &mut ListLink<Self, TimerWheel> { &mut self.timer_link }
//!     fn list_link_ref(&self) -> &ListLink<Self, TimerWheel> { &self.timer_link }
//! }
//!
//! // A task can be in both lists simultaneously
//! let mut ready_queue: DoublyLinkedList<Task, ReadyQueue> = DoublyLinkedList::init();
//! let mut timer_wheel: DoublyLinkedList<Task, TimerWheel> = DoublyLinkedList::init();
//!
//! let mut task = Task::new(1);
//! ready_queue.push_back(&mut task);
//! timer_wheel.push_back(&mut task);
//!
//! assert!(task.ready_link.is_linked());
//! assert!(task.timer_link.is_linked());
//! ```

use core::marker::PhantomData;
use core::ptr::NonNull;

// Compile-time validations
const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

/// Embedded link storage for intrusive list membership.
///
/// Each `Tag` type represents a distinct list, allowing a node to be in multiple
/// lists simultaneously by embedding multiple `ListLink` fields.
#[derive(Debug)]
pub struct ListLink<T, Tag> {
    prev: Option<NonNull<T>>,
    next: Option<NonNull<T>>,
    // Tracks membership because head (prev=None) and tail (next=None) are
    // otherwise indistinguishable from unlinked nodes.
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

    #[inline]
    pub fn is_linked(&self) -> bool {
        self.linked
    }

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

/// Implement for types that can be stored in a [`DoublyLinkedList<Self, Tag>`].
///
/// Each `Tag` requires a separate implementation pointing to its corresponding
/// [`ListLink`] field.
pub trait ListNode<Tag>: Sized {
    fn list_link(&mut self) -> &mut ListLink<Self, Tag>;
    fn list_link_ref(&self) -> &ListLink<Self, Tag>;
}

/// Doubly-linked list where nodes own their link storage.
///
/// Nodes must not move while linked. Use `u32` length for cross-platform consistency.
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

    /// # Panics
    /// If `node` is already linked.
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

    /// # Panics
    /// If `node` is already linked.
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
    ///
    /// - `node` must be in **this specific list instance**, not just any list of the same type.
    /// - `node` must remain valid for the duration of this call.
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

    /// O(n) search for `node`. Intended for debugging/assertions only.
    pub fn contains(&self, node: &T) -> bool {
        let target = node as *const T;
        let mut current = self.head;
        let mut visited: u32 = 0;

        while let Some(ptr) = current {
            // Safety check: detect infinite loops
            visited += 1;
            assert!(visited <= self.len, "cycle detected in list");

            if std::ptr::eq(ptr.as_ptr(), target) {
                return true;
            }

            // SAFETY: ptr is valid (part of our list)
            current = unsafe { ptr.as_ref().list_link_ref().next };
        }

        false
    }

    /// Verify structural invariants. Only available in debug builds.
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
        assert!(list.peek_front().is_none());
        assert!(list.peek_back().is_none());
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
    fn contains_present_and_absent() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let c = Node::new(3);

        list.push_back(&mut a);
        list.push_back(&mut b);

        assert!(list.contains(&a));
        assert!(list.contains(&b));
        assert!(!list.contains(&c));
    }

    #[test]
    fn pop_front_clears_link() {
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
    #[should_panic(expected = "push_back: node already linked")]
    fn panic_double_push_back() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);

        list.push_back(&mut a);
        list.push_back(&mut a); // Should panic
    }

    #[test]
    #[should_panic(expected = "push_front: node already linked")]
    fn panic_double_push_front() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);

        list.push_front(&mut a);
        list.push_front(&mut a); // Should panic
    }

    #[test]
    #[should_panic(expected = "remove: node not linked")]
    fn panic_remove_unlinked() {
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

    // ========================================================================
    // API Coverage Tests
    // ========================================================================

    #[test]
    fn peek_operations() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        // Empty list
        assert!(list.peek_front().is_none());
        assert!(list.peek_back().is_none());

        // Single element: head == tail
        list.push_back(&mut a);
        assert!(unsafe { list.peek_front().unwrap().as_ref().value } == 1);
        assert!(unsafe { list.peek_back().unwrap().as_ref().value } == 1);
        assert!(list.peek_front() == list.peek_back());

        // Multiple elements
        list.push_back(&mut b);
        list.push_back(&mut c);
        assert!(unsafe { list.peek_front().unwrap().as_ref().value } == 1);
        assert!(unsafe { list.peek_back().unwrap().as_ref().value } == 3);

        // Verify peek doesn't modify list
        assert!(list.len() == 3);
        assert!(unsafe { list.peek_front().unwrap().as_ref().value } == 1);
    }

    #[test]
    fn default_trait() {
        let list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::default();

        assert!(list.is_empty());
        assert!(list.peek_front().is_none());
        assert!(list.peek_back().is_none());
    }

    #[test]
    fn list_link_default_trait() {
        let link: ListLink<Node, Tag> = ListLink::default();

        assert!(!link.is_linked());
        assert!(link.prev.is_none());
        assert!(link.next.is_none());
    }

    // ========================================================================
    // pop_back Order Verification (Critical Gap)
    // ========================================================================

    #[test]
    fn alternating_pop_front_back() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);
        let mut d = Node::new(4);

        list.push_back(&mut a);
        list.push_back(&mut b);
        list.push_back(&mut c);
        list.push_back(&mut d);
        // [1, 2, 3, 4]

        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 1); // [2, 3, 4]
        assert!(unsafe { list.pop_back().unwrap().as_ref().value } == 4); // [2, 3]
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 2); // [3]
        assert!(unsafe { list.pop_back().unwrap().as_ref().value } == 3); // []

        assert!(list.is_empty());
        assert!(list.pop_front().is_none());
        assert!(list.pop_back().is_none());
    }

    #[test]
    fn push_front_pop_back_order() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.push_front(&mut a); // [1]
        list.push_front(&mut b); // [2, 1]
        list.push_front(&mut c); // [3, 2, 1]

        // Pop from back: should be FIFO for push_front
        assert!(unsafe { list.pop_back().unwrap().as_ref().value } == 1);
        assert!(unsafe { list.pop_back().unwrap().as_ref().value } == 2);
        assert!(unsafe { list.pop_back().unwrap().as_ref().value } == 3);
    }

    // ========================================================================
    // Node State After remove() Tests
    // ========================================================================

    #[test]
    fn remove_clears_link() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.push_back(&mut a);
        list.push_back(&mut b);
        list.push_back(&mut c);

        unsafe { list.remove(NonNull::from(&mut b)) };

        // Removed node should be completely unlinked
        assert!(!b.link.is_linked());
        assert!(b.link.prev.is_none());
        assert!(b.link.next.is_none());

        // Should be able to reuse b immediately
        list.push_back(&mut b);
        assert!(list.len() == 3);
        assert!(b.link.is_linked());
    }

    // ========================================================================
    // Consecutive Remove Tests
    // ========================================================================

    #[test]
    fn remove_consecutive_elements() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);
        let mut d = Node::new(4);
        let mut e = Node::new(5);

        list.push_back(&mut a);
        list.push_back(&mut b);
        list.push_back(&mut c);
        list.push_back(&mut d);
        list.push_back(&mut e);
        // [1, 2, 3, 4, 5]

        // Remove consecutive middle elements
        unsafe { list.remove(NonNull::from(&mut b)) }; // [1, 3, 4, 5]
        unsafe { list.remove(NonNull::from(&mut c)) }; // [1, 4, 5]

        assert!(list.len() == 3);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 1);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 4);
        assert!(unsafe { list.pop_front().unwrap().as_ref().value } == 5);
    }

    #[test]
    fn remove_all_via_remove() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.push_back(&mut a);
        list.push_back(&mut b);
        list.push_back(&mut c);

        // Remove in arbitrary order (middle, head, tail)
        unsafe { list.remove(NonNull::from(&mut b)) };
        assert!(list.len() == 2);

        unsafe { list.remove(NonNull::from(&mut a)) };
        assert!(list.len() == 1);

        unsafe { list.remove(NonNull::from(&mut c)) };
        assert!(list.is_empty());
    }

    #[test]
    fn remove_head_and_tail_consecutively() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);
        let mut c = Node::new(3);

        list.push_back(&mut a);
        list.push_back(&mut b);
        list.push_back(&mut c);
        // [1, 2, 3]

        // Remove head then tail
        unsafe { list.remove(NonNull::from(&mut a)) }; // [2, 3]
        unsafe { list.remove(NonNull::from(&mut c)) }; // [2]

        assert!(list.len() == 1);
        assert!(unsafe { list.peek_front().unwrap().as_ref().value } == 2);
        assert!(list.peek_front() == list.peek_back());
    }

    // ========================================================================
    // Two-Element List Tests
    // ========================================================================

    #[test]
    fn two_element_list_operations() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);

        list.push_back(&mut a);
        list.push_back(&mut b);

        // Verify head/tail pointers are distinct
        assert!(list.peek_front() != list.peek_back());
        assert!(unsafe { list.peek_front().unwrap().as_ref().value } == 1);
        assert!(unsafe { list.peek_back().unwrap().as_ref().value } == 2);

        // Pop front leaves tail as only element
        list.pop_front();
        assert!(list.len() == 1);
        assert!(unsafe { list.peek_front().unwrap().as_ref().value } == 2);
        assert!(list.peek_front() == list.peek_back());
    }

    #[test]
    fn two_element_pop_back_leaves_single() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);

        list.push_back(&mut a);
        list.push_back(&mut b);

        // Pop back leaves head as only element
        list.pop_back();
        assert!(list.len() == 1);
        assert!(unsafe { list.peek_front().unwrap().as_ref().value } == 1);
        assert!(list.peek_front() == list.peek_back());
    }

    #[test]
    fn two_element_remove_first() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);

        list.push_back(&mut a);
        list.push_back(&mut b);

        unsafe { list.remove(NonNull::from(&mut a)) };

        assert!(list.len() == 1);
        assert!(unsafe { list.peek_front().unwrap().as_ref().value } == 2);
        assert!(list.peek_front() == list.peek_back());
    }

    #[test]
    fn two_element_remove_second() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);

        list.push_back(&mut a);
        list.push_back(&mut b);

        unsafe { list.remove(NonNull::from(&mut b)) };

        assert!(list.len() == 1);
        assert!(unsafe { list.peek_front().unwrap().as_ref().value } == 1);
        assert!(list.peek_front() == list.peek_back());
    }

    // ========================================================================
    // Enhanced contains() Tests
    // ========================================================================

    #[test]
    fn contains_empty_list() {
        let list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let a = Node::new(1);

        assert!(!list.contains(&a));
    }

    #[test]
    fn contains_after_removal() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut a = Node::new(1);
        let mut b = Node::new(2);

        list.push_back(&mut a);
        list.push_back(&mut b);

        assert!(list.contains(&a));

        unsafe { list.remove(NonNull::from(&mut a)) };

        assert!(!list.contains(&a));
        assert!(list.contains(&b));
    }

    // ========================================================================
    // Safety Violation Tests
    // ========================================================================

    #[test]
    #[should_panic(expected = "remove from empty list")]
    fn panic_remove_empty() {
        let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
        let mut node = Node::new(1);

        // Force the link to appear linked without actually being in the list
        node.link.linked = true;

        // List is empty but node claims to be linked
        unsafe { list.remove(NonNull::from(&mut node)) };
    }

    #[test]
    fn multi_tag_support() {
        enum Tag1 {}
        enum Tag2 {}

        struct MultiNode {
            #[allow(dead_code)]
            val: u32,
            link1: ListLink<MultiNode, Tag1>,
            link2: ListLink<MultiNode, Tag2>,
        }

        impl MultiNode {
            fn new(val: u32) -> Self {
                Self {
                    val,
                    link1: ListLink::new(),
                    link2: ListLink::new(),
                }
            }
        }

        impl ListNode<Tag1> for MultiNode {
            fn list_link(&mut self) -> &mut ListLink<Self, Tag1> {
                &mut self.link1
            }
            fn list_link_ref(&self) -> &ListLink<Self, Tag1> {
                &self.link1
            }
        }

        impl ListNode<Tag2> for MultiNode {
            fn list_link(&mut self) -> &mut ListLink<Self, Tag2> {
                &mut self.link2
            }
            fn list_link_ref(&self) -> &ListLink<Self, Tag2> {
                &self.link2
            }
        }

        let mut list1: DoublyLinkedList<MultiNode, Tag1> = DoublyLinkedList::init();
        let mut list2: DoublyLinkedList<MultiNode, Tag2> = DoublyLinkedList::init();

        let mut node = MultiNode::new(42);

        list1.push_back(&mut node);
        list2.push_front(&mut node);

        assert!(list1.len() == 1);
        assert!(list2.len() == 1);

        assert!(node.link1.is_linked());
        assert!(node.link2.is_linked());

        list1.pop_back();
        assert!(!node.link1.is_linked());
        assert!(node.link2.is_linked());

        list2.pop_front();
        assert!(!node.link2.is_linked());
    }

    // ========================================================================
    // Property-Based Tests
    // ========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        const PROPTEST_CASES: u32 = 16;

        /// Operations we can perform on the list
        #[derive(Debug, Clone, Copy)]
        enum ListOp {
            PushBack,
            PushFront,
            PopFront,
            PopBack,
        }

        impl ListOp {
            fn from_byte(b: u8) -> Self {
                match b % 4 {
                    0 => ListOp::PushBack,
                    1 => ListOp::PushFront,
                    2 => ListOp::PopFront,
                    _ => ListOp::PopBack,
                }
            }
        }

        proptest! {
        #![proptest_config(ProptestConfig::with_cases(
            crate::test_utils::proptest_cases(PROPTEST_CASES)
        ))]
        /// Property: After any sequence of operations, length matches actual count
        /// Note: We use Box<Node> to ensure stable addresses - Vec reallocation
        /// would move nodes and invalidate pointers (this is expected for intrusive lists).
        #[test]
        fn prop_random_ops_length(ops in prop::collection::vec(any::<u8>(), 0..100)) {
            let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
            // Use Box to ensure stable addresses - Vec can reallocate and move items
            let mut nodes: Vec<Box<Node>> = Vec::new();
            let mut expected_len: usize = 0;

            for (i, &op_byte) in ops.iter().enumerate() {
                match ListOp::from_byte(op_byte) {
                    ListOp::PushBack => {
                        nodes.push(Box::new(Node::new(i as u32)));
                        list.push_back(nodes.last_mut().unwrap().as_mut());
                        expected_len += 1;
                    }
                    ListOp::PushFront => {
                        nodes.push(Box::new(Node::new(i as u32)));
                        list.push_front(nodes.last_mut().unwrap().as_mut());
                        expected_len += 1;
                    }
                    ListOp::PopFront => {
                        if expected_len > 0 {
                            list.pop_front();
                            expected_len -= 1;
                        }
                    }
                    ListOp::PopBack => {
                        if expected_len > 0 {
                            list.pop_back();
                            expected_len -= 1;
                        }
                    }
                }

                // Invariant must hold after every operation
                prop_assert_eq!(list.len() as usize, expected_len);
            }
        }

        /// Property: Invariants always hold after any operation sequence
        #[test]
        #[cfg(debug_assertions)]
        fn prop_random_ops_invariants(ops in prop::collection::vec(any::<u8>(), 0..50)) {
            let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
            // Use Box to ensure stable addresses
            let mut nodes: Vec<Box<Node>> = Vec::new();
            let mut count: usize = 0;

            for (i, &op_byte) in ops.iter().enumerate() {
                match ListOp::from_byte(op_byte) {
                    ListOp::PushBack => {
                        nodes.push(Box::new(Node::new(i as u32)));
                        list.push_back(nodes.last_mut().unwrap().as_mut());
                        count += 1;
                    }
                    ListOp::PushFront => {
                        nodes.push(Box::new(Node::new(i as u32)));
                        list.push_front(nodes.last_mut().unwrap().as_mut());
                        count += 1;
                    }
                    ListOp::PopFront => {
                        if count > 0 {
                            list.pop_front();
                            count -= 1;
                        }
                    }
                    ListOp::PopBack => {
                        if count > 0 {
                            list.pop_back();
                            count -= 1;
                        }
                    }
                }

                // Invariants must hold after every operation
                list.check_invariants();
            }
        }

        /// Property: push_back followed by pop_back returns same element (when list was empty)
        #[test]
        fn push_back_pop_back_roundtrip(value in any::<u32>()) {
            let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
            let mut node = Node::new(value);

            list.push_back(&mut node);
            let ptr = list.pop_back().unwrap();

            prop_assert_eq!(unsafe { ptr.as_ref().value }, value);
            prop_assert!(list.is_empty());
        }

        /// Property: push_front followed by pop_front returns same element (when list was empty)
        #[test]
        fn push_front_pop_front_roundtrip(value in any::<u32>()) {
            let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
            let mut node = Node::new(value);

            list.push_front(&mut node);
            let ptr = list.pop_front().unwrap();

            prop_assert_eq!(unsafe { ptr.as_ref().value }, value);
            prop_assert!(list.is_empty());
        }

        /// Property: push_back preserves FIFO order with pop_front
        /// Pre-allocate all nodes before adding to list to avoid reallocation
        #[test]
        fn fifo_order_preserved(values in prop::collection::vec(any::<u32>(), 1..20)) {
            let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
            // Pre-allocate all nodes before adding to list
            let mut nodes: Vec<Node> = values.iter().map(|&v| Node::new(v)).collect();

            for node in &mut nodes {
                list.push_back(node);
            }

            for &expected in &values {
                let ptr = list.pop_front().unwrap();
                prop_assert_eq!(unsafe { ptr.as_ref().value }, expected);
            }

            prop_assert!(list.is_empty());
        }

        /// Property: push_back preserves LIFO order with pop_back
        #[test]
        fn lifo_order_preserved(values in prop::collection::vec(any::<u32>(), 1..20)) {
            let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
            // Pre-allocate all nodes before adding to list
            let mut nodes: Vec<Node> = values.iter().map(|&v| Node::new(v)).collect();

            for node in &mut nodes {
                list.push_back(node);
            }

            // Pop in reverse order
            for &expected in values.iter().rev() {
                let ptr = list.pop_back().unwrap();
                prop_assert_eq!(unsafe { ptr.as_ref().value }, expected);
            }

            prop_assert!(list.is_empty());
        }

        /// Property: peek_front and peek_back return correct values
        #[test]
        fn prop_peek_endpoints(
            front_values in prop::collection::vec(any::<u32>(), 1..10),
            back_values in prop::collection::vec(any::<u32>(), 0..10),
        ) {
            let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();

            // Pre-allocate all nodes with stable addresses (Box)
            let total = front_values.len() + back_values.len();
            let mut nodes: Vec<Box<Node>> = Vec::with_capacity(total);

            // Create all nodes first
            for &v in &front_values {
                nodes.push(Box::new(Node::new(v)));
            }
            for &v in &back_values {
                nodes.push(Box::new(Node::new(v)));
            }

            // Now add to list
            for node in &mut nodes {
                list.push_back(node.as_mut());
            }

            // First value should be at front
            prop_assert_eq!(
                unsafe { list.peek_front().unwrap().as_ref().value },
                front_values[0]
            );

            // Last value should be at back
            let expected_back = if back_values.is_empty() {
                front_values[front_values.len() - 1]
            } else {
                back_values[back_values.len() - 1]
            };
            prop_assert_eq!(
                unsafe { list.peek_back().unwrap().as_ref().value },
                expected_back
            );
        }

        /// Property: popped nodes are no longer linked
        #[test]
        fn prop_pop_unlinks(count in 1usize..10) {
            let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
            // Pre-allocate all nodes
            let mut nodes: Vec<Node> = (0..count).map(|i| Node::new(i as u32)).collect();

            for node in &mut nodes {
                list.push_back(node);
            }

            // Pop all and verify each becomes unlinked
            for _ in 0..count {
                let ptr = list.pop_front().unwrap();
                let is_linked = unsafe { ptr.as_ref().link.is_linked() };
                prop_assert!(!is_linked);
            }
        }

        /// Property: contains returns true for all nodes in list
        #[test]
        fn prop_contains_all(count in 1usize..15) {
            let mut list: DoublyLinkedList<Node, Tag> = DoublyLinkedList::init();
            // Pre-allocate all nodes
            let mut nodes: Vec<Node> = (0..count).map(|i| Node::new(i as u32)).collect();

            for node in &mut nodes {
                list.push_back(node);
            }

            // All nodes should be found
            for node in &nodes {
                prop_assert!(list.contains(node));
            }
        }
        }
    }
}
