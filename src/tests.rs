#[cfg(feature = "shuttle")]
use shuttle::{
    sync::{atomic::Ordering, Arc},
    thread,
};

#[cfg(not(feature = "shuttle"))]
use std::{
    sync::{atomic::Ordering, Arc},
    thread,
};

const EFFECTIVE_SEGMENT_SIZE: usize = SEGMENT_SIZE - std::mem::size_of::<Segment>();

use crate::*;

#[derive(Default)]
struct TestEntry {
    val: [u16; 28],
}

impl TestEntry {
    fn init(val: u16) -> Self {
        TestEntry { val: [val; 28] }
    }

    fn sanity_check(&self) {
        let first = self.val.first().unwrap();
        for v in self.val.iter() {
            assert_eq!(*v, *first);
        }
    }
}

impl ClockCache {
    fn get_prob_loc_idx(&self) -> usize {
        let ptr = self.probe_loc.load(Ordering::Relaxed);
        let segment = Segment::from_entry(ptr);
        let first = unsafe { &*segment }.first_entry();

        (ptr as usize - first as usize) / self.entry_size
    }
}

#[cfg(not(feature = "shuttle"))]
#[test]
fn empty_cache() {
    let _cache = ClockCache::new(0, std::mem::size_of::<TestEntry>());
}

#[cfg(not(feature = "shuttle"))]
#[test]
fn basic() {
    let seg_cnt = 2;
    let cache_size = SEGMENT_SIZE * seg_cnt;
    let entry_size = std::mem::size_of::<TestEntry>();

    let entry_per_seg = EFFECTIVE_SEGMENT_SIZE / (entry_size + std::mem::size_of::<EntryMeta>());
    let cache_capacity = entry_per_seg * seg_cnt;
    let cache = ClockCache::new(cache_size, entry_size);

    let mut allocated = vec![];

    for i in 0..cache_capacity {
        let prob_loc = cache.get_prob_loc_idx();
        assert_eq!(prob_loc, i % entry_per_seg);
        let entry = cache.probe_entry().unwrap();
        let mut entry_meta = entry.load_meta(Ordering::Relaxed);
        assert_empty_entry(entry);

        let test_entry = TestEntry::init(i as u16);
        let cached_ptr = entry.data_ptr() as *mut TestEntry;
        unsafe { cached_ptr.write(test_entry) };
        allocated.push(cached_ptr);

        entry_meta.held = false;
        entry_meta.referenced = true;
        entry_meta.occupied = true;
        entry.set_meta(entry_meta, Ordering::Relaxed);
    }

    for ptr in allocated.iter() {
        unsafe { &**ptr }.sanity_check();
    }

    // now the cache is full, probe entry will reset the reference bit
    let mut prob_loc = cache.get_prob_loc_idx();
    for _i in 0..cache_capacity / 16 {
        let entry = cache.probe_entry();
        let new_loc = cache.get_prob_loc_idx();
        assert_eq!(new_loc, (16 + prob_loc) % entry_per_seg);
        prob_loc = new_loc;
        assert!(entry.is_none());
    }

    // visit again, now every probe_entry will return a entry to evict.
    for _i in 0..cache_capacity {
        let entry = cache.probe_entry().unwrap();
        let entry_meta = entry.load_meta(Ordering::Relaxed);
        assert_eq!(entry_meta.referenced, false);
        assert_eq!(entry_meta.held, false);
        assert_eq!(entry_meta.occupied, true);

        let cached_ptr = entry.data_ptr() as *mut TestEntry;
        unsafe { &*cached_ptr }.sanity_check();
    }

    // visit again, now every probe_entry will return a entry to evict.
    let new_entry = TestEntry::init(cache_capacity as u16);
    let byte_stream = unsafe {
        std::slice::from_raw_parts(
            &new_entry as *const TestEntry as *const u8,
            std::mem::size_of_val(&new_entry),
        )
    };
    for _i in 0..cache_capacity {
        cache
            .probe_entry_evict(
                |p: *mut u8| unsafe {
                    std::ptr::copy_nonoverlapping(byte_stream.as_ptr(), p, byte_stream.len());
                    Ok(())
                },
                |p: *mut u8| {
                    let val = unsafe { &*(p as *const TestEntry) };
                    val.sanity_check();
                },
            )
            .unwrap();
    }
    for ptr in allocated.iter() {
        let val = unsafe { &**ptr };
        val.sanity_check();
        assert_eq!(val.val[0], cache_capacity as u16);
    }

    std::mem::drop(cache);
}

fn assert_empty_entry(entry: &EntryMeta) {
    let entry_meta = entry.load_meta(Ordering::Relaxed);
    assert_eq!(entry_meta.held, true);
    assert_eq!(entry_meta.referenced, false);
    assert_eq!(entry_meta.occupied, false);
}

#[test]
fn empty_add_segment() {
    let cache = ClockCache::new(0, std::mem::size_of::<TestEntry>());
    unsafe {
        cache.add_segment(Segment::alloc());
    }
}

#[test]
fn add_remove_segment() {
    let seg_cnt = 1;
    let cache_size = SEGMENT_SIZE * seg_cnt;
    let entry_size = std::mem::size_of::<TestEntry>();

    let entry_per_seg = EFFECTIVE_SEGMENT_SIZE / (entry_size + std::mem::size_of::<EntryMeta>());
    let cache = ClockCache::new(cache_size, entry_size);

    let mut allocated = vec![];

    for i in 1..=entry_per_seg {
        let entry = cache.probe_entry().unwrap();
        let mut entry_meta = entry.load_meta(Ordering::Relaxed);
        assert_empty_entry(entry);

        let test_entry = TestEntry::init(i as u16);
        let cached_ptr = entry.data_ptr() as *mut TestEntry;
        unsafe { cached_ptr.write(test_entry) };
        allocated.push(cached_ptr);

        entry_meta.set_occupied();
        entry.set_meta(entry_meta, Ordering::Relaxed);
    }

    // move the cursor to next segment
    for _i in 0..entry_per_seg / 16 {
        let entry = cache.probe_entry();
        assert!(entry.is_none());
    }

    unsafe { cache.add_segment(Segment::alloc()) };

    for i in 1..=entry_per_seg {
        let entry = cache.probe_entry().unwrap();
        assert_empty_entry(entry);
        let mut entry_meta = entry.load_meta(Ordering::Relaxed);

        let test_entry = TestEntry::init((i + entry_per_seg) as u16);
        let cached_ptr = entry.data_ptr() as *mut TestEntry;
        unsafe { cached_ptr.write(test_entry) };
        allocated.push(cached_ptr);

        entry_meta.set_occupied();
        entry.set_meta(entry_meta, Ordering::Relaxed);
    }

    for ptr in allocated.iter() {
        unsafe { &**ptr }.sanity_check();
    }

    loop {
        let mut evicted = 0;
        let seg_iter = match cache.remove_segment() {
            Some(seg_iter) => seg_iter,
            None => break,
        };
        for _entry in seg_iter.iter() {
            evicted += 1;
        }

        assert_eq!(evicted, entry_per_seg);
    }

    for _i in 0..10 {
        let entry = cache.probe_entry();
        assert!(entry.is_none());
    }
    std::mem::drop(cache);
}

fn thread_probe_entry(cache: &ClockCache, i: usize) {
    let entry = cache.probe_entry();
    match entry {
        Some(e) => {
            let mut meta = e.load_meta(Ordering::Relaxed);
            if meta.held {
                // means the entry is ready to write
                let val = TestEntry::init(i as u16);
                let ptr = e.data_ptr() as *mut TestEntry;
                unsafe {
                    ptr.write(val);
                }
                meta.held = false;
                meta.referenced = true;
                meta.occupied = true;
                e.set_meta(meta, Ordering::Relaxed);
            } else {
                // the entry was occupied, we check its sanity
                let ptr = e.data_ptr() as *const TestEntry;
                unsafe { &*ptr }.sanity_check();
            }
        }
        None => {}
    }
}

#[test]
fn multi_thread_add_remove_segment() {
    let seg_cnt = 1;
    let cache_size = SEGMENT_SIZE * seg_cnt;
    let entry_per_seg = 13;
    let entry_size = EFFECTIVE_SEGMENT_SIZE / 13; // increase entry size to increase the probability of contention

    let cache = ClockCache::new(cache_size, entry_size);
    let cache = Arc::new(cache);

    let probe_thread_cnt = 2;
    let mut thread_handlers = vec![];
    for _i in 1..=probe_thread_cnt {
        let cache = cache.clone();
        let handle = thread::spawn(move || {
            for i in 1..=2 * entry_per_seg {
                thread_probe_entry(&cache, i);
            }
        });
        thread_handlers.push(handle);
    }

    let cache = cache.clone();
    thread_handlers.push(thread::spawn(move || {
        let mut pin = crossbeam::epoch::pin();
        unsafe { cache.add_segment(Segment::alloc()) };

        for i in 1..=entry_per_seg {
            pin.repin();
            thread_probe_entry(&cache, i);
        }

        loop {
            let seg_iter = match cache.remove_segment() {
                Some(seg_iter) => seg_iter,
                None => break,
            };

            for i in 1..=entry_per_seg {
                pin.repin();
                thread_probe_entry(&cache, i);
            }

            pin.repin();

            pin.defer(move || {
                std::mem::drop(seg_iter);
            });
        }
    }));

    for h in thread_handlers {
        h.join().unwrap();
    }
}

/// This test case is intended for shuttle testing
fn multi_thread_basic() {
    let seg_cnt = 2;
    let cache_size = SEGMENT_SIZE * seg_cnt;

    let entry_per_seg = 12;
    let entry_size = EFFECTIVE_SEGMENT_SIZE / entry_per_seg; // increase entry size to increase the probability of contention

    let cache_capacity = entry_per_seg * seg_cnt;
    let cache = ClockCache::new(cache_size, entry_size);
    let cache = Arc::new(cache);

    let thread_cnt = 3;
    let mut thread_handlers = vec![];
    for _i in 1..=thread_cnt {
        let cache = cache.clone();
        let handle = thread::spawn(move || {
            for i in 1..=cache_capacity {
                thread_probe_entry(&cache, i);
            }
        });
        thread_handlers.push(handle);
    }

    for h in thread_handlers {
        h.join().unwrap();
    }
}

#[cfg(not(feature = "shuttle"))]
#[test]
fn multi_thread_basic_test() {
    multi_thread_basic();
}

#[cfg(feature = "shuttle")]
#[test]
fn shuttle_add_remove_segment() {
    let config = shuttle::Config::default();
    let mut runner = shuttle::PortfolioRunner::new(true, config);
    runner.add(shuttle::scheduler::PctScheduler::new(5, 40_000));
    runner.add(shuttle::scheduler::RandomScheduler::new(40_000));
    runner.run(multi_thread_add_remove_segment);
}

#[cfg(feature = "shuttle")]
#[test]
fn shuttle_basic() {
    let config = shuttle::Config::default();
    let mut runner = shuttle::PortfolioRunner::new(true, config);
    runner.add(shuttle::scheduler::PctScheduler::new(5, 40_000));
    runner.add(shuttle::scheduler::RandomScheduler::new(40_000));
    runner.run(multi_thread_basic);
}

#[cfg(feature = "shuttle")]
#[test]
fn shuttle_replay() {
    // shuttle::replay(
    // multi_thread_add_remove_segment,
    // "91028",
    // );
}
