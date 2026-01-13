pub fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

pub fn proptest_cases(default: u32) -> u32 {
    env_u32("PROPTEST_CASES").unwrap_or(default)
}

pub fn proptest_fuzz_multiplier(default: u32) -> u32 {
    env_u32("PROPTEST_FUZZ_MULTIPLIER")
        .unwrap_or(default)
        .max(1)
}
