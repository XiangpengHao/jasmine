#![feature(slice_ptr_get)]

mod backoff;
mod cache;
mod error;
mod segment;
mod shard;
mod utils;

/// Jasmine manages memory at the granularity of segments.
pub const SEGMENT_SIZE: usize = 2 * 1024 * 1024; // 2MB
pub const SEGMENT_ALIGN: usize = SEGMENT_SIZE - 1;

pub use cache::ClockCache;
pub use error::JasmineError;
pub use segment::{Segment, SegmentIter};
pub use shard::ShardCache;

#[cfg(test)]
mod tests;
