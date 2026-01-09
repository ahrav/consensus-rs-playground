use consensus::lsm::binary_search::{Config, Mode, binary_search_values_upsert_index};
use core::mem::size_of;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{TestRng, TestRunner};
use std::env;
use std::hint::black_box;

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;
const GIB: usize = 1024 * MIB;
const MAX_PAGE_COUNT: usize = 1_048_576;

const DEFAULT_SAMPLE_SIZE: usize = 32;
const DEFAULT_SEARCHES: usize = 20_000;
const DEFAULT_BLOB_SIZE: usize = GIB;

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    values_per_page: usize,
    page_buffer_size: usize,
}

const SCENARIOS: [Scenario; 2] = [
    Scenario {
        name: "in-cache",
        values_per_page: 64,
        page_buffer_size: 256 * KIB,
    },
    Scenario {
        name: "out-of-cache",
        values_per_page: 4_096,
        page_buffer_size: GIB,
    },
];

#[repr(C)]
#[derive(Clone, Copy)]
struct Value128 {
    key: u64,
    body: [u8; 120],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Value32 {
    key: [u8; 32],
}

const _: () = assert!(size_of::<Value128>() == 128);
const _: () = assert!(size_of::<Value32>() == 32);

fn bench_binary_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_search");
    group.sample_size(env_usize("BINARY_SEARCH_SAMPLE_SIZE").unwrap_or(DEFAULT_SAMPLE_SIZE));

    // Env overrides: BINARY_SEARCH_BLOB_SIZE_MIB, BINARY_SEARCH_SEARCHES, BINARY_SEARCH_SAMPLE_SIZE.
    let blob_size = env_usize("BINARY_SEARCH_BLOB_SIZE_MIB")
        .map(|mib| mib.saturating_mul(MIB))
        .unwrap_or(DEFAULT_BLOB_SIZE);
    let search_count = env_usize("BINARY_SEARCH_SEARCHES").unwrap_or(DEFAULT_SEARCHES);

    for scenario in SCENARIOS {
        let page_buffer_size = scenario.page_buffer_size.min(blob_size);

        let mut runner_128 = TestRunner::deterministic();
        let fixture_128 = build_fixture(
            scenario,
            page_buffer_size,
            "K=8B V=128B",
            make_value128,
            &mut runner_128,
        );

        let input_128 = BenchInput {
            scenario,
            search_count,
            layout_name: "K=8B V=128B",
            fixture: &fixture_128,
        };

        bench_custom(
            &mut group,
            input_128,
            CustomOptions {
                config: Config {
                    mode: Mode::LowerBound,
                    prefetch: true,
                },
                label: "custom prefetch-on",
            },
            key_from_value_u64,
            key_from_index_u64,
            checksum_u64,
        );

        bench_custom(
            &mut group,
            input_128,
            CustomOptions {
                config: Config {
                    mode: Mode::LowerBound,
                    prefetch: false,
                },
                label: "custom prefetch-off",
            },
            key_from_value_u64,
            key_from_index_u64,
            checksum_u64,
        );

        bench_std(
            &mut group,
            input_128,
            key_from_value_u64,
            key_from_index_u64,
            checksum_u64,
        );

        bench_partition_point(
            &mut group,
            input_128,
            key_from_value_u64,
            key_from_index_u64,
            checksum_u64,
        );

        drop(fixture_128);

        let mut runner_32 = TestRunner::deterministic();
        let fixture_32 = build_fixture(
            scenario,
            page_buffer_size,
            "K=32B V=32B",
            make_value32,
            &mut runner_32,
        );

        let input_32 = BenchInput {
            scenario,
            search_count,
            layout_name: "K=32B V=32B",
            fixture: &fixture_32,
        };

        bench_custom(
            &mut group,
            input_32,
            CustomOptions {
                config: Config {
                    mode: Mode::LowerBound,
                    prefetch: true,
                },
                label: "custom prefetch-on",
            },
            key_from_value_32,
            key_from_index_32,
            checksum_key32,
        );

        bench_custom(
            &mut group,
            input_32,
            CustomOptions {
                config: Config {
                    mode: Mode::LowerBound,
                    prefetch: false,
                },
                label: "custom prefetch-off",
            },
            key_from_value_32,
            key_from_index_32,
            checksum_key32,
        );

        bench_std(
            &mut group,
            input_32,
            key_from_value_32,
            key_from_index_32,
            checksum_key32,
        );

        bench_partition_point(
            &mut group,
            input_32,
            key_from_value_32,
            key_from_index_32,
            checksum_key32,
        );
    }

    group.finish();
}

struct Fixture<Value> {
    values: Vec<Value>,
    page_picker: Vec<usize>,
    value_picker: Vec<usize>,
    values_per_page: usize,
}

#[derive(Clone, Copy)]
struct BenchInput<'a, Value> {
    scenario: Scenario,
    search_count: usize,
    layout_name: &'a str,
    fixture: &'a Fixture<Value>,
}

#[derive(Clone, Copy)]
struct CustomOptions<'a> {
    config: Config,
    label: &'a str,
}

fn build_fixture<Value, MakeValue>(
    scenario: Scenario,
    page_buffer_size: usize,
    layout_name: &str,
    mut make_value: MakeValue,
    runner: &mut TestRunner,
) -> Fixture<Value>
where
    Value: Copy,
    MakeValue: FnMut(usize, usize, &mut TestRng) -> Value,
{
    let values_per_page = scenario.values_per_page;
    let page_size = values_per_page
        .checked_mul(size_of::<Value>())
        .unwrap_or_else(|| panic!("page_size overflow for {}", layout_name));
    let page_count = page_buffer_size / page_size;

    assert!(page_count > 0, "page_count must be > 0");
    assert!(
        page_count <= MAX_PAGE_COUNT,
        "page_count exceeds safety cap"
    );

    let total_values = page_count
        .checked_mul(values_per_page)
        .unwrap_or_else(|| panic!("total_values overflow for {}", layout_name));

    let page_picker = shuffled_indices(runner, page_count);
    let value_picker = shuffled_indices(runner, values_per_page);

    let rng = runner.rng();
    let mut values = Vec::with_capacity(total_values);
    for page in 0..page_count {
        for index in 0..values_per_page {
            values.push(make_value(page, index, rng));
        }
    }

    Fixture {
        values,
        page_picker,
        value_picker,
        values_per_page,
    }
}

fn bench_custom<Value, Key, KeyFromValue, KeyFromIndex, KeyToChecksum>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    input: BenchInput<'_, Value>,
    options: CustomOptions<'_>,
    key_from_value: KeyFromValue,
    key_from_index: KeyFromIndex,
    key_to_checksum: KeyToChecksum,
) where
    Value: Copy,
    Key: Ord + Copy,
    KeyFromValue: Fn(&Value) -> Key,
    KeyFromIndex: Fn(usize) -> Key,
    KeyToChecksum: Fn(Key) -> u64,
{
    let bench_id = BenchmarkId::new(
        format!(
            "{} {} {}",
            options.label, input.scenario.name, input.layout_name
        ),
        format!("N={}", input.fixture.values_per_page),
    );
    group.throughput(Throughput::Elements(input.search_count as u64));

    group.bench_function(bench_id, |b| {
        b.iter(|| {
            let mut checksum = 0u64;
            for i in 0..input.search_count {
                let target_index = input.fixture.value_picker[i % input.fixture.value_picker.len()];
                let target_key = key_from_index(target_index);
                let page_index = input.fixture.page_picker[i % input.fixture.page_picker.len()];
                let offset = page_index * input.fixture.values_per_page;
                let page = &input.fixture.values[offset..offset + input.fixture.values_per_page];
                let hit_index = binary_search_values_upsert_index(
                    page,
                    target_key,
                    options.config,
                    &key_from_value,
                );
                let hit = unsafe { page.get_unchecked(hit_index as usize) };
                let hit_key = key_from_value(hit);
                debug_assert!(hit_key == target_key);
                checksum = checksum.wrapping_add(key_to_checksum(hit_key));
            }
            black_box(checksum);
        });
    });
}

fn bench_std<Value, Key, KeyFromValue, KeyFromIndex, KeyToChecksum>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    input: BenchInput<'_, Value>,
    key_from_value: KeyFromValue,
    key_from_index: KeyFromIndex,
    key_to_checksum: KeyToChecksum,
) where
    Value: Copy,
    Key: Ord + Copy,
    KeyFromValue: Fn(&Value) -> Key,
    KeyFromIndex: Fn(usize) -> Key,
    KeyToChecksum: Fn(Key) -> u64,
{
    let bench_id = BenchmarkId::new(
        format!(
            "std binary_search {} {}",
            input.scenario.name, input.layout_name
        ),
        format!("N={}", input.fixture.values_per_page),
    );
    group.throughput(Throughput::Elements(input.search_count as u64));

    group.bench_function(bench_id, |b| {
        b.iter(|| {
            let mut checksum = 0u64;
            for i in 0..input.search_count {
                let target_index = input.fixture.value_picker[i % input.fixture.value_picker.len()];
                let target_key = key_from_index(target_index);
                let page_index = input.fixture.page_picker[i % input.fixture.page_picker.len()];
                let offset = page_index * input.fixture.values_per_page;
                let page = &input.fixture.values[offset..offset + input.fixture.values_per_page];
                let hit_index = page
                    .binary_search_by_key(&target_key, &key_from_value)
                    .expect("target key missing");
                let hit_key = key_from_value(unsafe { page.get_unchecked(hit_index) });
                debug_assert!(hit_key == target_key);
                checksum = checksum.wrapping_add(key_to_checksum(hit_key));
            }
            black_box(checksum);
        });
    });
}

fn bench_partition_point<Value, Key, KeyFromValue, KeyFromIndex, KeyToChecksum>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    input: BenchInput<'_, Value>,
    key_from_value: KeyFromValue,
    key_from_index: KeyFromIndex,
    key_to_checksum: KeyToChecksum,
) where
    Value: Copy,
    Key: Ord + Copy,
    KeyFromValue: Fn(&Value) -> Key,
    KeyFromIndex: Fn(usize) -> Key,
    KeyToChecksum: Fn(Key) -> u64,
{
    let bench_id = BenchmarkId::new(
        format!(
            "std partition_point {} {}",
            input.scenario.name, input.layout_name
        ),
        format!("N={}", input.fixture.values_per_page),
    );
    group.throughput(Throughput::Elements(input.search_count as u64));

    group.bench_function(bench_id, |b| {
        b.iter(|| {
            let mut checksum = 0u64;
            for i in 0..input.search_count {
                let target_index = input.fixture.value_picker[i % input.fixture.value_picker.len()];
                let target_key = key_from_index(target_index);
                let page_index = input.fixture.page_picker[i % input.fixture.page_picker.len()];
                let offset = page_index * input.fixture.values_per_page;
                let page = &input.fixture.values[offset..offset + input.fixture.values_per_page];
                let hit_index = page.partition_point(|value| key_from_value(value) < target_key);
                let hit_key = key_from_value(unsafe { page.get_unchecked(hit_index) });
                debug_assert!(hit_key == target_key);
                checksum = checksum.wrapping_add(key_to_checksum(hit_key));
            }
            black_box(checksum);
        });
    });
}

fn shuffled_indices(runner: &mut TestRunner, len: usize) -> Vec<usize> {
    let indices: Vec<usize> = (0..len).collect();
    let strategy = Just(indices).prop_shuffle();
    let tree = strategy.new_tree(runner).expect("shuffle");
    tree.current()
}

fn make_value128(_page: usize, index: usize, rng: &mut TestRng) -> Value128 {
    let mut body = [0u8; 120];
    rng.fill_bytes(&mut body);
    Value128 {
        key: index as u64,
        body,
    }
}

fn make_value32(_page: usize, index: usize, _rng: &mut TestRng) -> Value32 {
    Value32 {
        key: key_from_index_32(index),
    }
}

fn key_from_value_u64(value: &Value128) -> u64 {
    value.key
}

fn key_from_value_32(value: &Value32) -> [u8; 32] {
    value.key
}

fn key_from_index_u64(index: usize) -> u64 {
    index as u64
}

fn key_from_index_32(index: usize) -> [u8; 32] {
    let mut key = [0u8; 32];
    key[24..].copy_from_slice(&(index as u64).to_be_bytes());
    key
}

fn checksum_u64(key: u64) -> u64 {
    key
}

fn checksum_key32(key: [u8; 32]) -> u64 {
    u64::from_be_bytes(key[24..].try_into().expect("key tail"))
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().and_then(|value| value.parse().ok())
}

criterion_group!(benches, bench_binary_search);
criterion_main!(benches);
