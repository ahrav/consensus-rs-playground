use consensus::stdx::DynamicBitSet;
use consensus::vsr::free_set::{FreeSet, InitOptions, Reservation, SHARD_BITS};
use proptest::prelude::*;

#[derive(Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            0xDEAD_BEEF_DEAD_BEEFu64
        } else {
            seed
        };
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        let denom = (1u64 << 53) as f64;
        (value as f64) / denom
    }

    fn range_inclusive_u64(&mut self, min: u64, max: u64) -> u64 {
        assert!(min <= max);
        let span = max - min + 1;
        min + (self.next_u64() % span)
    }

    fn range_inclusive_usize(&mut self, min: usize, max: usize) -> usize {
        assert!(min <= max);
        let span = max - min + 1;
        min + (self.next_u64() as usize % span)
    }

    fn boolean(&mut self) -> bool {
        (self.next_u64() & 1) != 0
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.range_inclusive_usize(0, i);
            values.swap(i, j);
        }
    }
}

fn random_int_exponential(rng: &mut XorShift64, avg: u64) -> u64 {
    if avg == 0 {
        return 0;
    }
    let u = rng.next_f64().max(f64::MIN_POSITIVE);
    (-u.ln() * avg as f64) as u64
}

#[derive(Clone, Copy, Debug)]
enum FreeSetEvent {
    Reserve { blocks: usize },
    Forfeit,
    Acquire { reservation: usize },
    Release { address: usize },
    Checkpoint,
}

fn weighted_choice(rng: &mut XorShift64, weights: &[(FreeSetEvent, u64)]) -> FreeSetEvent {
    let total: u64 = weights.iter().map(|(_, w)| *w).sum();
    assert!(total > 0);

    let mut roll = rng.range_inclusive_u64(0, total - 1);
    for (event, weight) in weights {
        if *weight == 0 {
            continue;
        }
        if roll < *weight {
            return *event;
        }
        roll -= *weight;
    }
    weights[0].0
}

fn generate_events(
    rng: &mut XorShift64,
    blocks_count: usize,
    events_count: usize,
) -> Vec<FreeSetEvent> {
    let reserve_weight = rng.range_inclusive_u64(1, 100);
    let forfeit_weight = 1;
    let acquire_weight = rng.range_inclusive_u64(1, 1000);
    let release_weight = if rng.boolean() {
        0
    } else {
        rng.range_inclusive_u64(0, 500)
    };
    let checkpoint_weight = random_int_exponential(rng, 10);

    let weights = [
        (FreeSetEvent::Reserve { blocks: 0 }, reserve_weight),
        (FreeSetEvent::Forfeit, forfeit_weight),
        (FreeSetEvent::Acquire { reservation: 0 }, acquire_weight),
        (FreeSetEvent::Release { address: 0 }, release_weight),
        (FreeSetEvent::Checkpoint, checkpoint_weight),
    ];

    let reservation_blocks_mean = rng.range_inclusive_usize(1, blocks_count / 20);
    let mut events = Vec::with_capacity(events_count);

    for _ in 0..events_count {
        let event = match weighted_choice(rng, &weights) {
            FreeSetEvent::Reserve { .. } => FreeSetEvent::Reserve {
                blocks: 1 + random_int_exponential(rng, reservation_blocks_mean as u64) as usize,
            },
            FreeSetEvent::Forfeit => FreeSetEvent::Forfeit,
            FreeSetEvent::Acquire { .. } => FreeSetEvent::Acquire {
                reservation: rng.next_u64() as usize,
            },
            FreeSetEvent::Release { .. } => FreeSetEvent::Release {
                address: rng.next_u64() as usize,
            },
            FreeSetEvent::Checkpoint => FreeSetEvent::Checkpoint,
        };
        events.push(event);
    }

    events
}

fn init_free_set(blocks_count: usize) -> FreeSet {
    assert!(blocks_count.is_multiple_of(SHARD_BITS));
    let grid_size_limit = (blocks_count as u64) * consensus::vsr::free_set::BLOCK_SIZE;
    assert!(grid_size_limit <= usize::MAX as u64);

    FreeSet::init(InitOptions {
        grid_size_limit: grid_size_limit as usize,
        blocks_released_prior_checkpoint_durability_max: 0,
    })
}

fn open_empty_free_set(blocks_count: usize) -> FreeSet {
    let mut set = init_free_set(blocks_count);
    set.open(&[], &[], &[], &[]);
    set.checkpoint_durable = true;
    set
}

struct FreeSetModel {
    blocks_acquired: DynamicBitSet,
    blocks_released: DynamicBitSet,
    blocks_reserved: DynamicBitSet,
    reservation_count: usize,
    reservation_session: usize,
}

impl FreeSetModel {
    fn new(blocks_count: usize) -> Self {
        Self {
            blocks_acquired: DynamicBitSet::empty(blocks_count),
            blocks_released: DynamicBitSet::empty(blocks_count),
            blocks_reserved: DynamicBitSet::empty(blocks_count),
            reservation_count: 0,
            reservation_session: 0,
        }
    }

    fn count_reservations(&self) -> usize {
        self.reservation_count
    }

    fn count_free(&self) -> usize {
        self.blocks_acquired.bit_length() - self.blocks_acquired.count()
    }

    fn count_acquired(&self) -> usize {
        self.blocks_acquired.count()
    }

    fn highest_address_acquired(&self) -> Option<u64> {
        self.blocks_acquired
            .highest_set_bit()
            .map(|bit| bit as u64 + 1)
    }

    fn reserve(&mut self, reserve_count: usize) -> Option<Reservation> {
        assert!(reserve_count > 0);

        let blocks_reserved_count = self.blocks_reserved.count();
        let mut blocks_found_free = 0usize;

        for block in 0..self.blocks_acquired.bit_length() {
            if self.blocks_acquired.is_set(block) {
                continue;
            }

            if block < blocks_reserved_count {
                assert!(self.blocks_reserved.is_set(block));
                continue;
            }

            blocks_found_free += 1;
            if blocks_found_free == reserve_count {
                let block_base = blocks_reserved_count;
                let block_count = block + 1 - block_base;

                for i in 0..block_count {
                    self.blocks_reserved.set(block_base + i);
                }

                self.reservation_count += 1;
                return Some(Reservation {
                    block_base,
                    block_count,
                    session: self.reservation_session,
                });
            }
        }
        None
    }

    fn forfeit(&mut self, reservation: Reservation) {
        self.assert_reservation_active(reservation);
        self.reservation_count -= 1;

        for i in 0..reservation.block_count {
            self.blocks_reserved.unset(reservation.block_base + i);
        }

        if self.reservation_count == 0 {
            self.reservation_session = self.reservation_session.wrapping_add(1);
            assert_eq!(self.blocks_reserved.count(), 0);
        }
    }

    fn acquire(&mut self, reservation: Reservation) -> Option<u64> {
        assert!(reservation.block_count > 0);
        assert!(reservation.block_base < self.blocks_acquired.bit_length());
        assert_eq!(reservation.session, self.reservation_session);
        self.assert_reservation_active(reservation);

        let end = reservation.block_base + reservation.block_count;
        for block in reservation.block_base..end {
            if !self.blocks_acquired.is_set(block) {
                self.blocks_acquired.set(block);
                return Some(block as u64 + 1);
            }
        }
        None
    }

    fn release(&mut self, address: u64) {
        let block = (address - 1) as usize;
        self.blocks_released.set(block);
    }

    fn checkpoint(&mut self) {
        assert_eq!(self.blocks_reserved.count(), 0);

        let released: Vec<usize> = self.blocks_released.iter_set().collect();
        for block in released {
            assert!(self.blocks_released.is_set(block));
            assert!(self.blocks_acquired.is_set(block));

            self.blocks_released.unset(block);
            self.blocks_acquired.unset(block);
        }
        assert_eq!(self.blocks_released.count(), 0);
    }

    fn assert_reservation_active(&self, reservation: Reservation) {
        assert!(self.reservation_count > 0);
        assert_eq!(self.reservation_session, reservation.session);

        for i in 0..reservation.block_count {
            assert!(self.blocks_reserved.is_set(reservation.block_base + i));
        }
    }
}

fn events_max() -> usize {
    std::env::var("FREE_SET_FUZZ_EVENTS_MAX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5_000)
        .max(1)
}

fn run_fuzz(seed: u64, blocks_count: usize) {
    let mut rng = XorShift64::new(seed);
    let events_cap = events_max();
    let events_count = random_int_exponential(&mut rng, blocks_count as u64 * 100) as usize;
    let events_count = events_count.clamp(1, events_cap);

    let events = generate_events(&mut rng, blocks_count, events_count);

    let mut free_set = open_empty_free_set(blocks_count);
    let mut model = FreeSetModel::new(blocks_count);
    let mut active_reservations: Vec<Reservation> = Vec::new();
    let mut active_addresses: Vec<u64> = Vec::new();

    for event in events {
        match event {
            FreeSetEvent::Reserve { blocks } => {
                let reservation_actual = free_set.reserve(blocks);
                let reservation_expect = model.reserve(blocks);
                assert_eq!(reservation_expect, reservation_actual);

                if let Some(reservation) = reservation_actual {
                    active_reservations.push(reservation);
                }
            }
            FreeSetEvent::Forfeit => {
                rng.shuffle(&mut active_reservations);
                for reservation in active_reservations.drain(..) {
                    free_set.forfeit(reservation);
                    model.forfeit(reservation);
                }
            }
            FreeSetEvent::Acquire { reservation } => {
                if active_reservations.is_empty() {
                    continue;
                }
                let reservation = active_reservations[reservation % active_reservations.len()];
                let address_actual = free_set.acquire(reservation);
                let address_expect = model.acquire(reservation);
                assert_eq!(address_expect, address_actual);
                if let Some(address) = address_actual {
                    active_addresses.push(address);
                }
            }
            FreeSetEvent::Release { address } => {
                if active_addresses.is_empty() {
                    continue;
                }
                let address_index = address % active_addresses.len();
                let address = active_addresses.swap_remove(address_index);
                free_set.release(address);
                model.release(address);
            }
            FreeSetEvent::Checkpoint => {
                rng.shuffle(&mut active_reservations);
                for reservation in active_reservations.drain(..) {
                    free_set.forfeit(reservation);
                    model.forfeit(reservation);
                }

                free_set.mark_checkpoint_not_durable();
                free_set.mark_checkpoint_durable();
                model.checkpoint();
            }
        }

        assert_eq!(model.count_reservations(), free_set.count_reservations());
        assert_eq!(model.count_free(), free_set.count_free());
        assert_eq!(model.count_acquired(), free_set.count_acquired());
        assert_eq!(
            model.highest_address_acquired(),
            free_set.highest_address_acquired()
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]
    #[test]
    fn fuzz_free_set_matches_model(seed in any::<u64>(), blocks_multiplier in 1usize..=10) {
        let blocks_count = SHARD_BITS * blocks_multiplier;
        run_fuzz(seed, blocks_count);
    }
}
