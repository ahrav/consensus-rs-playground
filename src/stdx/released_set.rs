#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReleasedEntry {
    Empty,
    Occupied(usize),
}

#[derive(Debug)]
pub struct ReleasedSet {
    entries: Vec<ReleasedEntry>,
    stack: Vec<usize>,
    mask: usize,
}

impl ReleasedSet {
    pub fn with_capacity(cap: usize) -> Self {
        let slots = released_set_slots(cap);
        Self {
            entries: vec![ReleasedEntry::Empty; slots],
            stack: Vec::with_capacity(cap),
            mask: slots - 1,
        }
    }

    pub fn len(&self) -> usize {
        self.stack.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub fn clear_retaining_capacity(&mut self) {
        self.stack.clear();
        self.entries.fill(ReleasedEntry::Empty);
    }

    pub fn contains(&self, key: usize) -> bool {
        let mut idx = (released_set_hash(key) as usize) & self.mask;

        for _ in 0..self.entries.len() {
            // bound check
            match self.entries[idx] {
                ReleasedEntry::Empty => return false,
                ReleasedEntry::Occupied(k) if k == key => return true,
                _ => idx = (idx + 1) & self.mask,
            }
        }

        false
    }

    pub fn insert(&mut self, key: usize) {
        let mut idx = (released_set_hash(key) as usize) & self.mask;

        for _ in 0..self.entries.len() {
            match self.entries[idx] {
                ReleasedEntry::Empty => {
                    if self.stack.len() >= (self.entries.len() >> 1) {
                        panic!("released set capacity exceeded");
                    }
                    self.entries[idx] = ReleasedEntry::Occupied(key);
                    self.stack.push(key);
                    return;
                }
                ReleasedEntry::Occupied(k) if k == key => return,
                _ => idx = (idx + 1) & self.mask,
            }
        }

        panic!("released set probe exhaustion (table unexpectedly full");
    }

    pub fn pop(&mut self) -> Option<usize> {
        let key = self.stack.pop()?;
        let removed = self.remove(key);
        debug_assert!(
            removed,
            "ReleasedSet invariant broken; stack/table diverged"
        );
        Some(key)
    }

    fn remove(&mut self, key: usize) -> bool {
        let mut idx = (released_set_hash(key) as usize) & self.mask;

        for _ in 0..self.entries.len() {
            match self.entries[idx] {
                ReleasedEntry::Empty => return false,
                ReleasedEntry::Occupied(k) if k == key => {
                    self.backshift_delete(idx);
                    return true;
                }
                _ => idx = (idx + 1) & self.mask,
            }
        }

        false
    }

    fn backshift_delete(&mut self, mut hole: usize) {
        let mut i = (hole + 1) & self.mask;

        loop {
            match self.entries[i] {
                ReleasedEntry::Empty => {
                    self.entries[hole] = ReleasedEntry::Empty;
                    return;
                }
                ReleasedEntry::Occupied(k) => {
                    let home = (released_set_hash(k) as usize) & self.mask;

                    let dist_home_to_i = i.wrapping_sub(home) & self.mask;
                    let dist_hole_to_i = i.wrapping_sub(hole) & self.mask;

                    if dist_home_to_i >= dist_hole_to_i {
                        self.entries[hole] = ReleasedEntry::Occupied(k);
                        hole = i;
                    }
                }
            }
            i = (i + 1) & self.mask;
        }
    }
}

fn released_set_slots(limit: usize) -> usize {
    let min = limit
        .checked_mul(2)
        .expect("released set capacity overflow; limit too large");
    let min = min.max(2);
    min.checked_next_power_of_two()
        .expect("released set capacity overflow; next_power_of_two failed")
}

/// SplitMix64 mixing constants.
///
/// These constants, combined with the specific shift amounts (30, 27, 31),
/// were empirically derived by Sebastiano Vigna to maximize avalanche
/// properties — ensuring small input changes produce ~50% output bit flips.
///
/// Reference: https://prng.di.unimi.it/splitmix64.c
const SPLITMIX64_MUL_1: u64 = 0xbf58476d1ce4e5b9;
const SPLITMIX64_MUL_2: u64 = 0x94d049bb133111eb;

// Shift amounts tuned with the multipliers for optimal avalanche
const SHIFT_1: u32 = 30;
const SHIFT_2: u32 = 27;
const SHIFT_3: u32 = 31;

#[inline]
fn released_set_hash(key: usize) -> u64 {
    let mut x = key as u64;
    x ^= x >> SHIFT_1;
    x = x.wrapping_mul(SPLITMIX64_MUL_1);
    x ^= x >> SHIFT_2;
    x = x.wrapping_mul(SPLITMIX64_MUL_2);
    x ^ (x >> SHIFT_3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    const LIMIT: usize = 16;
    const KEY_MAX: usize = 64;

    #[test]
    fn empty_set_has_no_members() {
        let set = ReleasedSet::with_capacity(LIMIT);
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
        assert!(!set.contains(0));
        assert!(!set.contains(KEY_MAX - 1));
    }

    #[test]
    fn insert_is_idempotent() {
        let mut set = ReleasedSet::with_capacity(LIMIT);
        set.insert(10);
        set.insert(10);
        assert_eq!(set.len(), 1);
        assert!(set.contains(10));
    }

    #[test]
    fn clear_retaining_capacity_resets_len() {
        let mut set = ReleasedSet::with_capacity(LIMIT);
        set.insert(1);
        set.insert(2);
        set.clear_retaining_capacity();
        assert!(set.is_empty());
        assert!(!set.contains(1));
        assert!(!set.contains(2));
    }

    #[test]
    fn pop_drains_all_elements() {
        let mut set = ReleasedSet::with_capacity(LIMIT);
        set.insert(1);
        set.insert(2);
        set.insert(3);

        let mut drained = HashSet::new();
        while let Some(key) = set.pop() {
            assert!(drained.insert(key));
        }

        assert_eq!(drained.len(), 3);
        assert!(set.is_empty());
    }

    #[test]
    fn slots_are_power_of_two_and_min_two() {
        let slots_zero = released_set_slots(0);
        assert!(slots_zero.is_power_of_two());
        assert!(slots_zero >= 2);

        let slots = released_set_slots(LIMIT);
        assert!(slots.is_power_of_two());
        assert!(slots >= LIMIT * 2);
    }

    #[test]
    #[should_panic(expected = "released set capacity exceeded")]
    fn insert_panics_on_overflow() {
        let mut set = ReleasedSet::with_capacity(2);
        set.insert(1);
        set.insert(2);
        set.insert(3);
    }

    proptest! {
        #[test]
        fn prop_insert_contains_len(keys in proptest::collection::vec(0usize..KEY_MAX, 0..128)) {
            let mut set = ReleasedSet::with_capacity(LIMIT);
            let mut model: HashSet<usize> = HashSet::new();

            for key in keys {
                if model.len() == LIMIT && !model.contains(&key) {
                    continue;
                }
                set.insert(key);
                model.insert(key);
            }

            prop_assert_eq!(set.len(), model.len());
            for key in 0..KEY_MAX {
                prop_assert_eq!(set.contains(key), model.contains(&key));
            }
        }
    }

    proptest! {
        #[test]
        fn prop_pop_drains(keys in proptest::collection::hash_set(0usize..KEY_MAX, 0..=LIMIT)) {
            let mut set = ReleasedSet::with_capacity(LIMIT);
            let mut model: HashSet<usize> = HashSet::new();
            for key in &keys {
                set.insert(*key);
                model.insert(*key);
            }

            let mut drained = HashSet::new();
            while let Some(key) = set.pop() {
                prop_assert!(drained.insert(key));
            }

            prop_assert!(set.is_empty());
            prop_assert_eq!(drained, model);
        }
    }

    #[derive(Clone, Debug)]
    enum Op {
        Insert(usize),
        Contains(usize),
        Pop,
        Clear,
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            (0usize..KEY_MAX).prop_map(Op::Insert),
            (0usize..KEY_MAX).prop_map(Op::Contains),
            Just(Op::Pop),
            Just(Op::Clear),
        ]
    }

    proptest! {
        #[test]
        fn prop_operation_sequence(ops in proptest::collection::vec(op_strategy(), 0..256)) {
            let mut set = ReleasedSet::with_capacity(LIMIT);
            let mut model: HashSet<usize> = HashSet::new();

            for op in ops {
                match op {
                    Op::Insert(key) => {
                        if model.len() == LIMIT && !model.contains(&key) {
                            continue;
                        }
                        set.insert(key);
                        model.insert(key);
                    }
                    Op::Contains(key) => {
                        prop_assert_eq!(set.contains(key), model.contains(&key));
                    }
                    Op::Pop => {
                        let got = set.pop();
                        match got {
                            Some(key) => {
                                prop_assert!(model.remove(&key));
                            }
                            None => {
                                prop_assert!(model.is_empty());
                            }
                        }
                    }
                    Op::Clear => {
                        set.clear_retaining_capacity();
                        model.clear();
                    }
                }
            }

            prop_assert_eq!(set.len(), model.len());
            for key in 0..KEY_MAX {
                prop_assert_eq!(set.contains(key), model.contains(&key));
            }
        }
    }
}
