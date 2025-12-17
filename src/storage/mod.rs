pub mod buffer;
pub mod constants;
pub mod engine;
pub mod iocb;
pub mod layout;

// Re-exports for convenient access
pub use buffer::AlignedBuf;
pub use constants::{SECTOR_SIZE_DEFAULT, SECTOR_SIZE_MAX, SECTOR_SIZE_MIN};
pub use engine::{Options, Storage};
pub use iocb::{NextTickQueue, Read, Synchronicity, Write};
pub use layout::{Layout, Zone, ZoneSpec};

#[macro_export]
macro_rules! container_of {
    ($ptr:expr, $parent:ty, $field:ident) => {{
        let ptr_u8 = ($ptr as *const _ as *const u8) as usize;

        // Compute field offset without taking references to uninitialized data.
        let offset = unsafe {
            let uninit = core::mem::MaybeUninit::<$parent>::uninit();
            let base = uninit.as_ptr();
            let field_ptr = core::ptr::addr_of!((*base).$field);
            (field_ptr as usize) - (base as usize)
        };

        ((ptr_u8 - offset) as *mut $parent)
    }};
}
