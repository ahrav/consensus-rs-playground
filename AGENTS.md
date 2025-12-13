# Repository Guidelines

## Project Structure & Module Organization

- `src/lib.rs`: crate entry point; wires public modules.
- `src/constants.rs`: protocol constants and compile-time invariants.
- `src/vsr/wire/`: Viewstamped Replication wire primitives (`Header`, `Command`, `Message`, framing, pools).
- `src/stdx/`: no-heap data structures (`RingBuffer`, `BitSet`, intrusive `List`/`Queue`).
- `src/util/`: small helpers (e.g., zero-check utilities).
- `src/io/`: async I/O backends (`io_uring` on Linux, GCD on macOS) guarded by `cfg(target_os = ...)`.
- `.github/workflows/ci.yml`: CI definition (fmt, clippy, build/test, audit, Miri).
- `target/`: Cargo build output; don’t commit.

## Build, Test, and Development Commands

- `cargo build` / `cargo build --all-features`: compile the library.
- `cargo build --no-default-features`: minimal build sanity check (mirrors CI).
- `cargo test --all-features`: run unit + property + doc tests.
- `cargo fmt --all` (CI: `cargo fmt --all -- --check`): format with `rustfmt`.
- `cargo clippy --all-targets --all-features -- -D warnings`: lint; warnings fail CI.
- `cargo audit`: dependency CVE scan (install via `cargo install cargo-audit --locked`).
- Miri (nightly): `cargo +nightly miri setup && cargo +nightly miri test --all-features`.

## Coding Style & Naming Conventions

- Keep code `rustfmt`-clean (4-space indentation; no hand-aligned columns).
- Follow Rust naming: `snake_case` modules/functions, `CamelCase` types, `SCREAMING_SNAKE_CASE` constants.
- Prefer “Tiger style” invariants: `const _: () = assert!(...)` for layout/size checks and runtime `assert!` for pre/postconditions.
- Wire protocol stability matters: don’t change existing `#[repr(u8)]` discriminants or `#[repr(C)]` layouts; add new fields/variants compatibly and extend tests.

## Testing Guidelines

- Tests are colocated with code (`#[cfg(test)] mod tests`); add coverage with the implementation.
- Use `proptest` for round-trips and edge cases; tune locally with `PROPTEST_CASES=32 cargo test`.
- Keep tests deterministic and fast enough for `cargo test` on a laptop.

## Commit & Pull Request Guidelines

- Commit subjects are short and imperative (common prefixes: “Add”, “Fix”, “Refactor”, “Update”, “Remove”); avoid trailing periods.
- Keep commits focused and atomic; include rationale in the PR description.
- PRs should note protocol/safety impact (especially under `src/vsr/wire/` and `src/io/`), list how you tested (commands + platform), and ensure CI is green.
