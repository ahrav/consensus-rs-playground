pub mod buffer;
pub mod engine;
pub mod iocb;
pub mod layout;

pub mod constants {
    pub use crate::constants::{
        IO_ENTRIES_DEFAULT, MAX_NEXT_TICK_ITERATIONS, SECTOR_SIZE_DEFAULT, SECTOR_SIZE_MAX,
        SECTOR_SIZE_MIN,
    };
}

// Re-exports for convenient access
pub use crate::vsr::storage::Synchronicity;
pub use buffer::AlignedBuf;
pub use constants::{SECTOR_SIZE_DEFAULT, SECTOR_SIZE_MAX, SECTOR_SIZE_MIN};
pub use engine::{Options, Storage};
pub use iocb::{NextTickQueue, Read, Write};
pub use layout::{Layout, Zone, ZoneSpec};

#[macro_export]
macro_rules! container_of {
    ($ptr:expr, $parent:ty, $field:ident) => {{
        let ptr_u8 = ($ptr as *const _ as *const u8) as usize;

        // Compute field offset without taking references to uninitialized data.
        let offset = {
            let uninit = core::mem::MaybeUninit::<$parent>::uninit();
            let base = uninit.as_ptr();
            let field_ptr = unsafe { core::ptr::addr_of!((*base).$field) };
            (field_ptr as usize) - (base as usize)
        };

        ((ptr_u8 - offset) as *mut $parent)
    }};
}
