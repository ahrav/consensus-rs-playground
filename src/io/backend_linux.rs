use io_uring::IoUring;

pub struct UringBackend {
    #[allow(dead_code)]
    ring: IoUring,
}
