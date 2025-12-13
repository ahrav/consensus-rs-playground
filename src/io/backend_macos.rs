use crate::io::{IoBackend, Operation};
use std::io;

pub struct GcdBackend;

impl IoBackend for GcdBackend {
    const ENTRIES_MIN: u32 = 1;
    const ENTRIES_MAX: u32 = 4096;

    fn new(_entries: u32) -> io::Result<Self> {
        Ok(Self)
    }

    unsafe fn try_push(&mut self, _op: &Operation, _user_data: u64) -> Result<(), ()> {
        unimplemented!("macOS backend not implemented")
    }

    fn flush(&mut self, _wait_for_one: bool) -> io::Result<()> {
        Ok(())
    }

    fn drain<F: FnMut(u64, i32)>(&mut self, _f: F) {}
}
