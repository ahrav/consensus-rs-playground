pub mod checksum;
pub mod command;
pub mod framing;
pub mod header;
pub mod message;
pub mod pool;

pub mod constants {
    pub use crate::constants::{
        ClusterId, HEADER_SIZE, HEADER_SIZE_USIZE, MESSAGE_BODY_SIZE_MAX,
        MESSAGE_BODY_SIZE_MAX_USIZE, MESSAGE_SIZE_MAX, MESSAGE_SIZE_MAX_USIZE, VSR_VERSION,
    };
}

#[cfg(target_os = "linux")]
pub use checksum::checksum_aegis_mac;
pub use checksum::{Checksum128, checksum};
pub use command::{Command, InvalidCommand};
pub use constants::{
    ClusterId, HEADER_SIZE, HEADER_SIZE_USIZE, MESSAGE_BODY_SIZE_MAX, MESSAGE_BODY_SIZE_MAX_USIZE,
    MESSAGE_SIZE_MAX, MESSAGE_SIZE_MAX_USIZE, VSR_VERSION,
};
pub use header::Header;
pub use message::Message;
pub use pool::{MessageHandle, MessagePool};
