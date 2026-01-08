use consensus::ewah::Ewah;
use consensus::stdx::DynamicBitSet;
use consensus::vsr::free_set::{FreeSet, InitOptions, SHARD_BITS, Word};
use std::mem::size_of;

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

fn encode_size_max_bytes(blocks_count: usize) -> usize {
    let word_count = blocks_count / 64;
    Ewah::<Word>::encode_size_max(word_count)
}

fn words_as_bytes(words: &[Word]) -> &[u8] {
    let byte_len = std::mem::size_of_val(words);
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, byte_len) }
}

fn words_as_bytes_mut(words: &mut [Word]) -> &mut [u8] {
    let byte_len = std::mem::size_of_val(words);
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, byte_len) }
}

fn assert_bitset_eq(a: &DynamicBitSet, b: &DynamicBitSet) {
    assert_eq!(a.bit_length(), b.bit_length());
    assert_eq!(a.word_len(), b.word_len());
    assert_eq!(a.words(), b.words());
}

fn assert_free_set_eq(a: &FreeSet, b: &FreeSet) {
    assert_bitset_eq(&a.blocks_acquired, &b.blocks_acquired);
    assert_bitset_eq(&a.blocks_released, &b.blocks_released);
    assert_bitset_eq(&a.index, &b.index);
    assert_eq!(
        a.blocks_released_prior_checkpoint_durability.count(),
        b.blocks_released_prior_checkpoint_durability.count()
    );
}

#[test]
fn free_set_reserve_acquire_matches_source() {
    let blocks_count_total = SHARD_BITS;
    let mut set = open_empty_free_set(blocks_count_total);

    assert_eq!(None, set.reserve(blocks_count_total + 1));
    let r1 = set.reserve(blocks_count_total - 1).unwrap();
    let r2 = set.reserve(1).unwrap();
    assert_eq!(None, set.reserve(1));
    set.forfeit(r1);
    set.forfeit(r2);

    let mut address = 1u64;
    {
        let reservation = set.reserve(2).unwrap();
        assert_eq!(Some(address), set.acquire(reservation));
        assert_eq!(Some(address + 1), set.acquire(reservation));
        assert_eq!(None, set.acquire(reservation));
        set.forfeit(reservation);
    }
    address += 2;

    {
        let reservation_1 = set.reserve(2).unwrap();
        let reservation_2 = set.reserve(2).unwrap();
        assert_eq!(Some(address), set.acquire(reservation_1));
        assert_eq!(Some(address + 2), set.acquire(reservation_2));
        assert_eq!(Some(address + 1), set.acquire(reservation_1));
        assert_eq!(None, set.acquire(reservation_1));
        assert_eq!(Some(address + 3), set.acquire(reservation_2));
        assert_eq!(None, set.acquire(reservation_2));
        set.forfeit(reservation_1);
        set.forfeit(reservation_2);
    }
}

#[test]
fn free_set_acquire_part_way_through_shard_matches_source() {
    let mut set = open_empty_free_set(SHARD_BITS * 3);

    let reservation_a = set.reserve(1).unwrap();
    let reservation_b = set.reserve(2 * SHARD_BITS).unwrap();

    for i in 0..reservation_b.block_count {
        let address = set.acquire(reservation_b).unwrap();
        assert_eq!(address - 1, reservation_a.block_count as u64 + i as u64);
    }
    assert_eq!(None, set.acquire(reservation_b));

    set.forfeit(reservation_b);
    set.forfeit(reservation_a);
}

#[test]
fn free_set_checkpoint_roundtrip_matches_source() {
    let blocks_count = SHARD_BITS;
    let mut set = open_empty_free_set(blocks_count);
    let empty = open_empty_free_set(blocks_count);
    let mut full = open_empty_free_set(blocks_count);

    {
        let reservation = full.reserve(blocks_count).unwrap();
        for i in 0..blocks_count {
            assert_eq!(Some(i as u64 + 1), full.acquire(reservation));
        }
        full.forfeit(reservation);
    }

    {
        let reservation = set.reserve(blocks_count).unwrap();
        for i in 0..blocks_count {
            assert_eq!(Some(i as u64 + 1), set.acquire(reservation));
            set.release(i as u64 + 1);

            assert_eq!(i + 1, set.count_acquired());
            assert_eq!(blocks_count - i - 1, set.count_free());
        }
        assert_eq!(None, set.acquire(reservation));
        set.forfeit(reservation);
    }

    set.mark_checkpoint_not_durable();
    set.mark_checkpoint_durable();

    assert_free_set_eq(&empty, &set);
    assert_eq!(0, set.blocks_released.count());

    {
        let reservation = set.reserve(blocks_count).unwrap();
        for i in 0..blocks_count {
            assert_eq!(Some(i as u64 + 1), set.acquire(reservation));
            set.release(i as u64 + 1);
        }
        set.forfeit(reservation);
    }

    let encoded_size = encode_size_max_bytes(blocks_count);
    assert_eq!(encoded_size % size_of::<Word>(), 0);

    let mut buffer_blocks_acquired = vec![0u64; encoded_size / size_of::<Word>()];
    let mut buffer_blocks_released = vec![0u64; encoded_size / size_of::<Word>()];

    let sizes = {
        let mut acquired_chunk = words_as_bytes_mut(&mut buffer_blocks_acquired);
        let mut released_chunk = words_as_bytes_mut(&mut buffer_blocks_released);
        set.encode_chunks(&mut [&mut acquired_chunk], &mut [&mut released_chunk])
    };

    let encoded_blocks_acquired =
        &words_as_bytes(&buffer_blocks_acquired)[..sizes.encoded_size_blocks_acquired as usize];
    let encoded_blocks_released =
        &words_as_bytes(&buffer_blocks_released)[..sizes.encoded_size_blocks_released as usize];

    let chunks_blocks_acquired: Vec<&[u8]> = if encoded_blocks_acquired.is_empty() {
        Vec::new()
    } else {
        vec![encoded_blocks_acquired]
    };
    let chunks_blocks_released: Vec<&[u8]> = if encoded_blocks_released.is_empty() {
        Vec::new()
    } else {
        vec![encoded_blocks_released]
    };

    let mut set_decoded = init_free_set(blocks_count);
    set_decoded.decode_chunks(&chunks_blocks_acquired, &chunks_blocks_released);
    assert_free_set_eq(&set, &set_decoded);

    let sizes_full = {
        let mut acquired_chunk = words_as_bytes_mut(&mut buffer_blocks_acquired);
        let mut released_chunk = words_as_bytes_mut(&mut buffer_blocks_released);
        full.encode_chunks(&mut [&mut acquired_chunk], &mut [&mut released_chunk])
    };

    let encoded_blocks_acquired_full = &words_as_bytes(&buffer_blocks_acquired)
        [..sizes_full.encoded_size_blocks_acquired as usize];
    let encoded_blocks_released_full = &words_as_bytes(&buffer_blocks_released)
        [..sizes_full.encoded_size_blocks_released as usize];

    let chunks_blocks_acquired_full: Vec<&[u8]> = if encoded_blocks_acquired_full.is_empty() {
        Vec::new()
    } else {
        vec![encoded_blocks_acquired_full]
    };
    let chunks_blocks_released_full: Vec<&[u8]> = if encoded_blocks_released_full.is_empty() {
        Vec::new()
    } else {
        vec![encoded_blocks_released_full]
    };

    let mut full_decoded = init_free_set(blocks_count);
    full_decoded.decode_chunks(&chunks_blocks_acquired_full, &chunks_blocks_released_full);
    assert_free_set_eq(&full, &full_decoded);
}

#[test]
fn free_set_decode_small_bitset_into_large_matches_source() {
    let shard_bits = SHARD_BITS;
    let mut small_set = open_empty_free_set(shard_bits);

    {
        let reservation = small_set.reserve(shard_bits).unwrap();
        for _ in 0..shard_bits {
            assert!(small_set.acquire(reservation).is_some());
        }
        small_set.forfeit(reservation);
    }

    let encoded_size = encode_size_max_bytes(shard_bits);
    assert_eq!(encoded_size % size_of::<Word>(), 0);
    let mut buffer_blocks_acquired = vec![0u64; encoded_size / size_of::<Word>()];
    let mut buffer_blocks_released = vec![0u64; encoded_size / size_of::<Word>()];

    let sizes = {
        let mut acquired_chunk = words_as_bytes_mut(&mut buffer_blocks_acquired);
        let mut released_chunk = words_as_bytes_mut(&mut buffer_blocks_released);
        small_set.encode_chunks(&mut [&mut acquired_chunk], &mut [&mut released_chunk])
    };

    let encoded_blocks_acquired =
        &words_as_bytes(&buffer_blocks_acquired)[..sizes.encoded_size_blocks_acquired as usize];
    let chunks_blocks_acquired: Vec<&[u8]> = if encoded_blocks_acquired.is_empty() {
        Vec::new()
    } else {
        vec![encoded_blocks_acquired]
    };

    let mut big_set = init_free_set(2 * shard_bits);
    big_set.decode_chunks(&chunks_blocks_acquired, &[]);
    big_set.opened = true;

    for block in 0..2 * shard_bits {
        let address = block as u64 + 1;
        assert_eq!(block >= shard_bits, big_set.is_free(address));
    }
}

#[test]
fn free_set_decode_big_bitset_into_small_matches_source() {
    let shard_bits = SHARD_BITS;
    let mut big_set = open_empty_free_set(2 * shard_bits);

    {
        let acquired_block_count = big_set.blocks_acquired.bit_length() / 2;
        let reservation = big_set.reserve(acquired_block_count).unwrap();
        for _ in 0..acquired_block_count {
            assert!(big_set.acquire(reservation).is_some());
        }
        big_set.forfeit(reservation);
    }

    let encoded_size = encode_size_max_bytes(2 * shard_bits);
    assert_eq!(encoded_size % size_of::<Word>(), 0);
    let mut buffer_blocks_acquired = vec![0u64; encoded_size / size_of::<Word>()];
    let mut buffer_blocks_released = vec![0u64; encoded_size / size_of::<Word>()];

    let sizes = {
        let mut acquired_chunk = words_as_bytes_mut(&mut buffer_blocks_acquired);
        let mut released_chunk = words_as_bytes_mut(&mut buffer_blocks_released);
        big_set.encode_chunks(&mut [&mut acquired_chunk], &mut [&mut released_chunk])
    };

    let encoded_blocks_acquired =
        &words_as_bytes(&buffer_blocks_acquired)[..sizes.encoded_size_blocks_acquired as usize];
    let chunks_blocks_acquired: Vec<&[u8]> = if encoded_blocks_acquired.is_empty() {
        Vec::new()
    } else {
        vec![encoded_blocks_acquired]
    };

    let mut small_set = init_free_set(shard_bits);
    small_set.decode_chunks(&chunks_blocks_acquired, &[]);
    small_set.opened = true;

    for block in 0..shard_bits {
        let address = block as u64 + 1;
        assert!(!small_set.is_free(address));
    }
}
