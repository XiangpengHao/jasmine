#[cfg(feature = "shuttle")]
use shuttle::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};

use std::alloc::Layout;
#[cfg(not(feature = "shuttle"))]
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};

const EFFECTIVE_SEGMENT_SIZE: usize = SEGMENT_SIZE - std::mem::size_of::<Segment>();

use crate::*;

#[derive(Default)]
struct TestEntry {
    lock: AtomicBool,
    val: [u16; 28],
}

impl TestEntry {
    fn init(val: u16) -> Self {
        TestEntry {
            lock: AtomicBool::new(false),
            val: [val; 28],
        }
    }

    fn sanity_check(&self) {
        let first = self.val.first().unwrap();
        for v in self.val.iter() {
            assert_eq!(*v, *first);
        }
    }
}

impl ClockCache {
    fn get_prob_loc_idx(&self, entry_align: usize) -> usize {
        let ptr = self.probe_loc.load(Ordering::Relaxed);
        let segment = Segment::from_entry(ptr);
        let first = unsafe { &*segment }.first_entry(entry_align);

        (ptr as usize - first as usize) / self.entry_layout.size()
    }
}

#[cfg(not(feature = "shuttle"))]
#[test]
fn empty_cache() {
    let _cache = ClockCache::new(
        0,
        Layout::from_size_align(std::mem::size_of::<TestEntry>(), 2).unwrap(),
        douhua::MemType::DRAM,
    );
}

#[cfg(not(feature = "shuttle"))]
#[test]
fn basic() {
    let seg_cnt = 2;
    let cache_size = SEGMENT_SIZE * seg_cnt;
    let entry_size = std::mem::size_of::<TestEntry>();

    let cache = ClockCache::new(
        cache_size,
        Layout::from_size_align(entry_size, 2).unwrap(),
        douhua::MemType::DRAM,
    );
    let entry_per_seg = EFFECTIVE_SEGMENT_SIZE / cache.entry_layout.size();
    let cache_capacity = entry_per_seg * seg_cnt;
    let mut allocated = vec![];

    for i in 0..cache_capacity {
        let prob_loc = cache.get_prob_loc_idx(2);
        assert_eq!(prob_loc, i % entry_per_seg);
        let (entry, _evicted) = cache
            .probe_entry_evict(
                |_ptr| {
                    unreachable!("should not evict");
                },
                |ptr| {
                    let test_entry = TestEntry::init(i as u16);
                    let cached_ptr = ptr as *mut TestEntry;
                    unsafe { cached_ptr.write(test_entry) };
                    cached_ptr
                },
            )
            .unwrap();

        allocated.push(entry);
    }

    for ptr in allocated.iter() {
        unsafe { &**ptr }.sanity_check();
    }

    // now the cache is full, probe entry will reset the reference bit
    let mut prob_loc = cache.get_prob_loc_idx(2);
    for _i in 0..cache_capacity / 16 {
        let entry = cache.probe_entry_evict(|_v| unreachable!(), |_v| unreachable!());
        let new_loc = cache.get_prob_loc_idx(2);
        assert_eq!(new_loc, (16 + prob_loc) % entry_per_seg);
        prob_loc = new_loc;
        assert!(entry.is_err());
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
                    Some(())
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

#[test]
fn empty_add_segment() {
    let cache = ClockCache::new(
        0,
        Layout::from_size_align(std::mem::size_of::<TestEntry>(), 2).unwrap(),
        douhua::MemType::DRAM,
    );
    unsafe {
        cache.add_segment(Segment::alloc(douhua::MemType::DRAM));
    }
}

#[test]
fn add_remove_segment() {
    let seg_cnt = 1;
    let cache_size = SEGMENT_SIZE * seg_cnt;
    let entry_size = std::mem::size_of::<TestEntry>();

    let cache = ClockCache::new(
        cache_size,
        Layout::from_size_align(entry_size, 2).unwrap(),
        douhua::MemType::DRAM,
    );
    let entry_per_seg = EFFECTIVE_SEGMENT_SIZE / cache.entry_layout.size();

    let mut allocated = vec![];

    for i in 1..=entry_per_seg {
        let (entry, _evicted) = cache
            .probe_entry_evict(
                |_ptr| unreachable!(),
                |ptr| {
                    let test_entry = TestEntry::init(i as u16);
                    let cached_ptr = ptr as *mut TestEntry;
                    unsafe { cached_ptr.write(test_entry) };
                    cached_ptr
                },
            )
            .unwrap();

        allocated.push(entry);
    }

    // move the cursor to next segment
    for _i in 0..entry_per_seg / 16 {
        let entry = cache.probe_entry_evict(|_ptr| unreachable!(), |_ptr| unreachable!());
        assert!(entry.is_err());
    }

    unsafe { cache.add_segment(Segment::alloc(douhua::MemType::DRAM)) };

    for i in 1..=entry_per_seg {
        let (entry, _evicted) = cache
            .probe_entry_evict(
                |_ptr| unreachable!(),
                |ptr| {
                    let test_entry = TestEntry::init((i + entry_per_seg) as u16);
                    let cached_ptr = ptr as *mut TestEntry;
                    unsafe { cached_ptr.write(test_entry) };
                    cached_ptr
                },
            )
            .unwrap();

        allocated.push(entry);
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
        let entry = cache.probe_entry_evict(|_v| unreachable!(), |_v| unreachable!());
        assert!(entry.is_err());
    }
    std::mem::drop(cache);
}

fn thread_probe_entry(cache: &ClockCache, i: usize) {
    let _= cache.probe_entry_evict(|ptr| {
        // the entry was occupied, we check its sanity
        let val = unsafe { &*(ptr as *const TestEntry) };
        val.lock
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .expect("The lock should be false, because we should be the only one accessing the value.");
        val.sanity_check();
        val.lock
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap();
            Some(())
    }, |ptr| {
        // means the entry is ready to write
        let val = TestEntry::init(i as u16);
        let ptr = ptr as *mut TestEntry;
        unsafe {
            ptr.write(val);
        }
    }) ;
}

#[test]
fn multi_thread_add_remove_segment() {
    let seg_cnt = 1;
    let cache_size = SEGMENT_SIZE * seg_cnt;
    let entry_per_seg = 13;
    let entry_size = EFFECTIVE_SEGMENT_SIZE / 13; // increase entry size to increase the probability of contention

    let cache = ClockCache::new(
        cache_size,
        Layout::from_size_align(entry_size, 2).unwrap(),
        douhua::MemType::DRAM,
    );
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
        unsafe { cache.add_segment(Segment::alloc(douhua::MemType::DRAM)) };

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
    let cache = ClockCache::new(
        cache_size,
        Layout::from_size_align(entry_size, 2).unwrap(),
        douhua::MemType::DRAM,
    );
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
