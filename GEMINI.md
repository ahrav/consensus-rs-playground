# GEMINI.md

## Project Overview

**Project Name:** `consensus` (Repository: `consensus-rs-playground`)
**Description:** A Rust consensus library implementing Viewstamped Replication (VSR) protocol primitives. The design philosophy is heavily inspired by TigerBeetle, emphasizing deterministic simulation, memory safety, and performance through zero-allocation patterns and strict compile-time assertions.
**Language:** Rust (2024 Edition)

## Architecture & Key Modules

*   **`src/lib.rs`**: Library entry point, wiring public modules.
*   **`src/constants.rs`**: Protocol-wide constants (e.g., sizes, versions) and alignment helpers. Heavily utilizes compile-time assertions (`const _: () = assert!(...)`) to enforce invariants.
*   **`src/vsr/wire/`**: Defines the wire protocol specifications.
    *   `header.rs`: Fixed-layout 256-byte `#[repr(C)]` message header with dual checksums.
    *   `command.rs`: Protocol message types with stable `#[repr(u8)]` discriminants.
    *   `checksum.rs`: Deterministic AEGIS-128L MAC implementation.
*   **`src/stdx/`**: Custom standard library extensions focused on no-heap, stack-allocated data structures.
    *   `ring_buffer.rs`: Stack-allocated ring buffer (`RingBuffer<T, N>`) using `MaybeUninit`.
    *   `bitset.rs`: `u128`-backed bitset.
*   **`src/util/`**: Small helpers, such as zero-check utilities for reserved fields.

## Building & Running

This project uses standard Cargo workflows, with specific emphasis on strict linting and property-based testing.

### Core Commands

| Action | Command | Description |
| :--- | :--- | :--- |
| **Build** | `cargo build --all-features` | Build the project with all features enabled. |
| **Test** | `cargo test --all-features` | Run all unit and property-based tests. |
| **Lint** | `cargo clippy --all-targets --all-features -- -D warnings` | Run strict lints. All warnings are treated as errors. |
| **Format** | `cargo fmt --all` | Format code using `rustfmt`. |
| **Audit** | `cargo audit` | Check for security vulnerabilities in dependencies (requires `cargo-audit`). |

### Advanced / CI Commands

*   **Miri (Memory Safety):** `cargo miri test --all-features` (Requires Nightly Rust). Used to detect undefined behavior.
*   **Minimal Build:** `cargo build --no-default-features`

## Development Conventions

### "Tiger Style" & Design Patterns

1.  **Compile-Time Safety:** Enforce invariants at compile time whenever possible using `const` assertions.
    *   *Example:* `const _: () = assert!(size_of::<Header>() == 256);`
2.  **Runtime Assertions:** Use liberal `assert!` calls for "defense-in-depth," covering both preconditions and postconditions in functions.
3.  **No-Heap / Zero-Allocation:** Prefer stack allocation. Use fixed-size arrays and custom structures like `RingBuffer` instead of `Vec` where deterministic performance is critical.
4.  **Wire Protocol Stability:**
    *   Use `#[repr(C)]` for wire types to ensure consistent layout.
    *   Command discriminants are stable part of the protocol; do not reorder or change values.
    *   Always validate reserved fields are zeroed.
5.  **Checksumming:** Header checksums cover the body checksum field. Ensure `checksum_body` is calculated *before* the main header `checksum`.

### Testing

*   **Property-Based Testing:** Extensive use of `proptest`. New logic (especially wire formats and data structures) should include property tests verifying round-trip serialization and edge cases.
*   **Colocation:** Tests are located in the same file as the code (inline `mod tests`), not in a separate `tests/` directory (except perhaps for integration tests).
*   **Performance:** Keep property test case counts reasonable (e.g., `Config::with_cases(256)`) to ensure `cargo test` remains fast enough for local development.

### Commit Guidelines

*   **Style:** Imperative, short subject lines (e.g., "Implement VSR Header checksum").
*   **Content:** Commits should be atomic. Pull requests should summarize changes to invariants and confirm tests/lints pass.
