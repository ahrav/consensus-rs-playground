# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Rust consensus library implementing Viewstamped Replication (VSR) protocol primitives, inspired by TigerBeetle's design philosophy. Uses Rust 2024 edition.

## Build Commands

```bash
cargo build                          # Build with default features
cargo build --all-features           # Build with all features
cargo build --no-default-features    # Minimal build
cargo test --all-features            # Run all tests
cargo test <test_name>               # Run specific test
cargo fmt --all -- --check           # Check formatting
cargo clippy --all-targets --all-features -- -D warnings  # Lint
cargo miri test --all-features       # Memory safety checks (requires nightly)
orb cargo test --all-features --lib io::backend_linux # Run Linux tests (requires OrbStack)
```

## Architecture

### Module Structure

- **`constants`**: Protocol constants (sizes, versions) with compile-time validation. Uses `u32` for portability across 32/64-bit platforms.

- **`vsr::wire`**: Wire protocol primitives
  - `header`: 256-byte fixed-layout `#[repr(C)]` message header with dual checksums
  - `command`: Stable `#[repr(u8)]` discriminants for protocol messages (0-20)
  - `checksum`: Deterministic AEGIS-128L MAC (zero key/nonce for reproducibility)

- **`stdx`**: No-heap data structures
  - `RingBuffer<T, N>`: Stack-allocated ring buffer using `MaybeUninit`
  - `BitSet<N>`: Single `u128`-backed bitset for up to 128 flags

- **`util::zero`**: Byte utilities for reserved field validation

### Key Design Patterns

**Compile-Time Assertions**: Invariants are enforced via `const _: () = assert!(...)` blocks. When adding/modifying constants, add corresponding compile-time assertions.

**Tiger Style Assertions**: Runtime assertions are used liberally for defense-in-depth. Functions include precondition and postcondition assertions.

**Wire Protocol Stability**: Command discriminants and header layout are part of the protocol specification. Never change existing discriminant values.

**Property-Based Testing**: Tests use `proptest` extensively. New functionality should include both unit tests and property-based tests.

**Checksum Order**: When computing message checksums, always set `checksum_body` before `checksum` since the header checksum covers the body checksum field.

### Header Wire Layout

```
Bytes 0-15:   checksum (covers bytes 16-255)
Bytes 16-31:  checksum_padding
Bytes 32-47:  checksum_body (covers message body)
Bytes 48-63:  checksum_body_padding
Bytes 64-79:  nonce_reserved
Bytes 80-95:  cluster (u128)
Bytes 96-99:  size (u32)
...
```

## Dependencies

- `aegis`: AEGIS-128L authenticated encryption for checksums
- `proptest` (dev): Property-based testing framework
