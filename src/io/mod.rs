//! Async I/O over platform completion APIs.
//!
//! Targets Linux `io_uring` and macOS Grand Central Dispatch (GCD).
//!
//! The API is split into:
//! - [`IoBackend`]: submit and drain completions
//! - [`Operation`]: read/write/fsync description
//! - [`Completion`]: per-op state and callback
//!
//! # Safety / Ownership
//!
//! `Completion` values must have a stable address (pinned or otherwise immovable) while an
//! operation is in flight. Buffers are caller-owned; this layer stores raw pointers and
//! identifies completions via `user_data` (`Completion*` cast to `u64`).

pub mod iops;

mod io_core;

#[cfg(target_os = "linux")]
mod backend_linux;
#[cfg(target_os = "macos")]
mod backend_macos;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("This I/O layer currently supports only Linux (io_uring) and macOS (GCD).");

#[cfg(target_os = "linux")]
use backend_linux::UringBackend as BackendImpl;
#[cfg(target_os = "macos")]
use backend_macos::GcdBackend as BackendImpl;

const _: () = assert!(
    size_of::<usize>() >= size_of::<u32>(),
    "Platform must have at least 32-bit addressing"
);

// Re-export core types for public API.
pub use self::io_core::{
    Completion, CompletionCallback, CompletionState, IoBackend, IoCore, IoHandler, Operation,
    typed_handler_shim,
};

/// Platform-specific I/O instance. Use this type alias for portable code.
pub type Io = IoCore<BackendImpl>;
