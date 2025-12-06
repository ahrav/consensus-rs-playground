# Repository Guidelines

## Project Structure & Module Organization
- `src/lib.rs` wires public modules.
- `src/constants.rs` holds VSR protocol constants and alignment helpers with compile-time assertions.
- `src/stdx/` contains foundational utilities (`bitset`, `ring_buffer`) with property and unit tests inline.
- `src/vsr/wire/` defines wire-level types (`header`, `command`, `checksum`) for serialization and validation.
- `src/util/` stores small helpers (e.g., zero checks). Tests live next to code; no separate fixtures directory.

## Build, Test, and Development Commands
- `cargo build` compiles the library in debug; surfaces const assertions and type errors.
- `cargo test` runs all unit and property tests (uses proptest); expect a few seconds per suite.
- `cargo fmt` formats with rustfmt; run before committing.
- `cargo clippy -- -D warnings` lints for correctness; fix or document exceptions.
- Example: `cargo test -p consensus -- --nocapture` to debug property failures with full output.

## Coding Style & Naming Conventions
- Rust 2024 edition; follow `clippy` guidance unless protocol requirements disagree.
- Prefer const invariants (`const _: () = assert!(...)`) and explicit bounds/overflow checks; keep `#[inline]` on hot paths only when justified.
- Public APIs use `///` docs focused on protocol behavior; use `//` for implementation rationale.
- Naming: types `PascalCase`, functions `snake_case`, constants `SCREAMING_SNAKE_CASE`; mirror VSR terminology in names and comments.
- Preserve zeroed reserved fields and alignment assertions; avoid unchecked casts unless proven safe.

## Testing Guidelines
- Place tests alongside the code they cover; mirror existing structure for new modules.
- Keep property tests bounded to maintain runtime (e.g., `Config::with_cases(256)` when adding new ones).
- For wire changes, include serialization round-trips and checksum validations patterned after `Header` tests.
- Cover panic/edge paths (capacity limits, size bounds, zeroed fields) to lock in invariants.

## Commit & Pull Request Guidelines
- Commits use short, imperative subjects (e.g., `Add VSR Wire Protocol Header Implementation`); group related edits together.
- PRs should summarize behavior changes, note invariants touched, and list commands run (`cargo test`, `cargo clippy`, etc.).
- Link issues when present; include repro inputs for failures. Screenshots only when behavior is user-facing.
