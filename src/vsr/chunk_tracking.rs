#![allow(dead_code)]

use core::cmp::min;
use core::mem::size_of;

use super::journal_primitives::{HEADER_CHUNK_COUNT, HEADER_CHUNK_WORDS};
use crate::constants;
use crate::stdx::BitSet;
use crate::vsr::{Header, HeaderPrepare};

pub type HeaderChunks = BitSet<HEADER_CHUNK_COUNT, HEADER_CHUNK_WORDS>;
