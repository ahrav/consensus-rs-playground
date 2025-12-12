pub mod iops;

// Compile-time validations
const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);
