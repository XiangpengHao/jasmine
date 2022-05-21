#[cfg(all(feature = "shuttle", test))]
use shuttle::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

#[cfg(not(all(feature = "shuttle", test)))]
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

use crate::backoff::Backoff;
use spin::{rwlock::RwLockWriteGuard, RwLockReadGuard};
use std::mem::ManuallyDrop;

const SEGMENT_SIZE: usize = 4 * 1024; // 4KB
const SEGMENT_ALIGN: usize = 0x0fff;

/// The unit of cache allocation
struct Segment {
    next: AtomicPtr<Segment>,
    migration_lock: spin::RwLock<()>,
}

#[cfg(test)]
#[cfg(not(feature = "shuttle"))]
static_assertions::const_assert_eq!(std::mem::size_of::<Segment>(), 16);

impl Segment {
    fn first_entry(&self) -> *mut EntryMeta {
        unsafe {
            (self as *const Segment as *mut u8).add(std::mem::size_of::<Segment>())
                as *mut EntryMeta
        }
    }

    fn from_entry(ptr: *const EntryMeta) -> *const Segment {
        let ptr = (ptr as usize) & !SEGMENT_ALIGN;
        ptr as *mut Segment
    }

    fn alloc() -> *mut u8 {
        let seg_layout = std::alloc::Layout::from_size_align(SEGMENT_SIZE, SEGMENT_SIZE).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(seg_layout) };
        ptr
    }

    unsafe fn dealloc(ptr: *mut u8) {
        std::alloc::dealloc(
            ptr,
            std::alloc::Layout::from_size_align(SEGMENT_SIZE, SEGMENT_SIZE).unwrap(),
        );
    }
}

pub struct SegmentIter<'a> {
    segment: *const Segment,
    cur_entry: Option<*mut EntryMeta>,
    entry_size: usize,
    _lock: ManuallyDrop<RwLockWriteGuard<'a, ()>>, // never drop the lock (marking it as obsolete)
}

unsafe impl Send for SegmentIter<'_> {}
unsafe impl Sync for SegmentIter<'_> {}

impl Drop for SegmentIter<'_> {
    fn drop(&mut self) {
        unsafe {
            Segment::dealloc(self.segment as *mut Segment as *mut u8);
        }
        // don't drop the lock (marking it as obsolete)
    }
}

impl<'a> SegmentIter<'a> {
    fn new(segment: *const Segment, entry_size: usize, lock: RwLockWriteGuard<'a, ()>) -> Self {
        Self {
            segment,
            cur_entry: Some(unsafe { &*segment }.first_entry()),
            entry_size,
            _lock: ManuallyDrop::new(lock),
        }
    }
}

impl<'a> Iterator for SegmentIter<'a> {
    type Item = *mut EntryMeta;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur_entry?;
        self.cur_entry = unsafe { &*cur }.next_entry(self.entry_size);
        Some(cur)
    }
}

#[derive(Clone)]
pub struct EntryMetaUnpacked {
    pub(crate) referenced: bool,
    pub(crate) held: bool,
    pub(crate) occupied: bool,
}

impl EntryMetaUnpacked {
    pub fn set_occupied(&mut self) {
        assert_eq!(self.occupied, false);
        assert_eq!(self.held, true);
        self.held = false;
        self.occupied = true;
        self.referenced = true;
    }
}

impl From<u8> for EntryMetaUnpacked {
    fn from(value: u8) -> Self {
        EntryMetaUnpacked {
            referenced: value & 1 > 0,
            held: value & 0b10 > 0,
            occupied: value & 0b100 > 0,
        }
    }
}

impl From<EntryMetaUnpacked> for u8 {
    fn from(value: EntryMetaUnpacked) -> Self {
        (value.occupied as u8) << 2 | (value.held as u8) << 1 | (value.referenced as u8)
    }
}

pub struct EntryMeta {
    pub(crate) meta: AtomicU8, // ref: 1 bit, reserved: 1 bit
}

impl EntryMeta {
    pub fn data_ptr(&self) -> *mut u8 {
        unsafe {
            (self as *const EntryMeta as *mut u8).add(std::mem::size_of::<EntryMeta>()) as *mut u8
        }
    }

    /// Returns the next entry within the segment.
    /// Returns `None` if the entry is the last one.
    fn next_entry(&self, cache_entry_size: usize) -> Option<*mut EntryMeta> {
        let segment = Segment::from_entry(self);
        let next_entry = unsafe { (self as *const EntryMeta as *const u8).add(cache_entry_size) };
        let next_entry_end = next_entry as usize + cache_entry_size;
        if next_entry_end > (segment as usize + SEGMENT_SIZE) {
            None
        } else {
            Some(next_entry as *mut EntryMeta)
        }
    }
}

/// A cache for storing arbitrary length data, the cache size can grow and shrink with page granularity
///
/// There is only one major API: `probe_entry()`, which either returns an empty entry or an entry that should be evicted (to accommodate the new value).
///
/// Remove/add segment can only be invoked by one thread at a time.
pub struct ClockCache {
    segments: AtomicPtr<Segment>, // a doubly-linked circular buffer
    probe_len: usize,
    entry_size: usize, // the size of the cache entry + entry meta
    probe_loc: AtomicPtr<EntryMeta>,
}

impl Drop for ClockCache {
    fn drop(&mut self) {
        let mut cur = self.segments.load(Ordering::Relaxed);
        let start = cur;
        loop {
            let next = unsafe { &*cur }.next.load(Ordering::Relaxed);
            unsafe {
                Segment::dealloc(cur as *mut u8);
            }
            if next == start {
                break;
            }
            cur = next;
        }
    }
}

impl ClockCache {
    pub fn new(cache_total_size: usize, cache_size: usize) -> Self {
        let seg_cnt = cache_total_size / SEGMENT_SIZE;
        let entry_size = cache_size + std::mem::size_of::<EntryMeta>();

        assert!(seg_cnt > 0);

        let mut first: *mut Segment = std::ptr::null_mut();
        let mut prev: *mut Segment = std::ptr::null_mut();
        for i in 0..seg_cnt {
            let ptr = Segment::alloc() as *mut Segment;
            unsafe {
                (*ptr).next = AtomicPtr::new(std::ptr::null_mut());
            }

            if i == 0 {
                first = ptr;
            }
            if i == (seg_cnt - 1) {
                unsafe {
                    (*ptr).next = AtomicPtr::new(first);
                }
            }
            if !prev.is_null() {
                unsafe {
                    (*prev).next.store(ptr, Ordering::Relaxed);
                }
            }
            prev = ptr;
        }
        let first_entry = unsafe { &*first }.first_entry();
        ClockCache {
            segments: AtomicPtr::new(first),
            probe_loc: AtomicPtr::new(first_entry),
            probe_len: 16,
            entry_size,
        }
    }

    // only one thread at any time should call this function
    pub fn add_segment(&self, ptr: *mut u8) {
        let cur = self.segments.load(Ordering::Relaxed);
        let cur_next = unsafe { &*cur }.next.load(Ordering::Relaxed);

        unsafe {
            std::ptr::write_bytes(ptr, 0, SEGMENT_SIZE);
        }
        let ptr = ptr as *mut Segment;

        let new_segment = unsafe { &*ptr };
        new_segment.next.store(cur_next, Ordering::Relaxed);
        unsafe { &*cur }.next.store(ptr, Ordering::Relaxed);
        self.segments.store(ptr, Ordering::Release);
    }

    /// The caller need to iterate all the entries in the segment and evict them
    /// As long as the lock guard is held, not other thread can promote entry to this segment
    ///
    /// Returns None if the segment is the last one in the cache
    pub fn remove_segment(&self) -> Option<SegmentIter<'static>> {
        // step 1: loop over all segments to find one that we can acquire write lock
        let backoff = Backoff::new();
        let mut prev = self.segments.load(Ordering::Relaxed);
        let mut cur = unsafe { &*prev }.next.load(Ordering::Relaxed);
        let lock_guard = loop {
            match unsafe { &*cur }.migration_lock.try_write() {
                Some(v) => break v,
                None => {
                    let next = unsafe { (*cur).next.load(Ordering::Relaxed) };
                    if next == cur {
                        // meaning we are the last segment
                        return None;
                    }
                    prev = cur;
                    cur = next;
                    backoff.spin();
                }
            }
        };

        let next = unsafe { (*cur).next.load(Ordering::Relaxed) };
        if next == cur {
            return None;
        }

        // step 2: make sure the probe_loc is not pointing to the selected segment
        let prob_loc = self.probe_loc.load(Ordering::Acquire);
        let prob_seg = Segment::from_entry(prob_loc);
        if prob_seg == cur {
            // the prob_loc is in the current segment, we need to update to next segment
            self.probe_loc
                .store(unsafe { &*next }.first_entry(), Ordering::Release);
        }

        unsafe { &*prev }.next.store(next, Ordering::Relaxed);
        self.segments.store(next, Ordering::Relaxed);

        Some(SegmentIter::new(cur, self.entry_size, lock_guard))
    }

    /// Optional current segment lock, this is just a performance optimization
    /// Returns the old entry as well as its segment lock
    fn next_entry<'a>(
        &'a self,
        mut seg_lock: Option<RwLockReadGuard<'a, ()>>,
    ) -> (*mut EntryMeta, RwLockReadGuard<()>) {
        let mut cur_entry = self.probe_loc.load(Ordering::Relaxed);
        let backoff = Backoff::new();
        loop {
            let (entry, lock) = match unsafe { &*cur_entry }.next_entry(self.entry_size) {
                Some(v) => {
                    let seg_lock = match seg_lock {
                        Some(l) => l,
                        None => {
                            let segment = Segment::from_entry(cur_entry);
                            if let Some(seg_lock) = unsafe { &*segment }.migration_lock.try_read() {
                                seg_lock
                            } else {
                                cur_entry = self.probe_loc.load(Ordering::Relaxed);
                                backoff.spin();
                                continue;
                            }
                        }
                    };
                    (v, seg_lock)
                }
                None => {
                    let cur_segment = Segment::from_entry(cur_entry);
                    let segment = unsafe { &*cur_segment }.next.load(Ordering::Relaxed);

                    if let Some(seg_lock) = unsafe { &*segment }.migration_lock.try_read() {
                        let new_entry = unsafe { &*segment }.first_entry();
                        (new_entry, seg_lock)
                    } else {
                        cur_entry = self.probe_loc.load(Ordering::Relaxed);
                        backoff.spin();
                        continue;
                    }
                }
            };

            match self.probe_loc.compare_exchange_weak(
                cur_entry,
                entry,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(old) => {
                    return (old, lock);
                }
                Err(old) => {
                    seg_lock = Some(lock);
                    cur_entry = old;
                }
            }
            backoff.spin();
        }
    }

    /// Returns whether we find a cached location.
    /// If yes, returns the cached location.
    ///
    /// The caller need to check the reserved bit, if not set, this entry need to be evicted to the storage
    ///
    /// Unless the reserved bit is set (the entry is empty), it is the caller's responsibility to serialize the concurrent read/write.
    pub fn probe_entry(&self) -> Option<*mut EntryMeta> {
        let mut seg_lock = None;
        for _p in 0..self.probe_len {
            let (cur_entry, cur_lock) = self.next_entry(seg_lock);
            seg_lock = Some(cur_lock);

            let entry = match unsafe { &*cur_entry }.meta.compare_exchange_weak(
                0,
                0b10,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => Some(cur_entry),
                Err(v) => {
                    let mut meta = EntryMetaUnpacked::from(v);

                    let reserved = meta.held;
                    let referenced = meta.referenced;
                    if meta.referenced {
                        meta.referenced = false;
                        unsafe {
                            (*cur_entry).meta.store(meta.into(), Ordering::Release);
                        }
                    }

                    if reserved || referenced {
                        None
                    } else {
                        Some(cur_entry)
                    }
                }
            };

            if let Some(entry) = entry {
                return Some(entry);
            }
        }

        None
    }
}

#[cfg(test)]
mod test {

    #[cfg(feature = "shuttle")]
    use shuttle::{sync::Arc, thread};

    #[cfg(not(feature = "shuttle"))]
    use std::{sync::Arc, thread};

    const EFFECTIVE_SEGMENT_SIZE: usize = SEGMENT_SIZE - std::mem::size_of::<Segment>();

    use super::*;

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
    fn basic() {
        let seg_cnt = 2;
        let cache_size = SEGMENT_SIZE * seg_cnt;
        let entry_size = std::mem::size_of::<TestEntry>();

        let entry_per_seg =
            EFFECTIVE_SEGMENT_SIZE / (entry_size + std::mem::size_of::<EntryMeta>());
        let cache_capacity = entry_per_seg * seg_cnt;
        let cache = ClockCache::new(cache_size, entry_size);

        let mut allocated = vec![];

        for i in 0..cache_capacity {
            let prob_loc = cache.get_prob_loc_idx();
            assert_eq!(prob_loc, i % entry_per_seg);
            let entry = cache.probe_entry().unwrap();
            let mut entry_meta = assert_empty_entry(entry);

            let test_entry = TestEntry::init(i as u16);
            let cached_ptr = unsafe { &*entry }.data_ptr() as *mut TestEntry;
            unsafe { cached_ptr.write(test_entry) };
            allocated.push(cached_ptr);

            entry_meta.held = false;
            entry_meta.referenced = true;
            entry_meta.occupied = true;
            unsafe { &*entry }
                .meta
                .store(entry_meta.into(), Ordering::Relaxed);
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

        for _i in 0..cache_capacity {
            let entry = cache.probe_entry().unwrap();
            let entry_meta =
                EntryMetaUnpacked::from(unsafe { &*entry }.meta.load(Ordering::Relaxed));
            assert_eq!(entry_meta.referenced, false);
            assert_eq!(entry_meta.held, false);
            assert_eq!(entry_meta.occupied, true);

            let cached_ptr = unsafe { &*entry }.data_ptr() as *mut TestEntry;
            unsafe { &*cached_ptr }.sanity_check();
        }

        std::mem::drop(cache);
    }

    fn assert_empty_entry(entry: *mut EntryMeta) -> EntryMetaUnpacked {
        let entry_meta = EntryMetaUnpacked::from(unsafe { &*entry }.meta.load(Ordering::Relaxed));
        assert_eq!(entry_meta.held, true);
        assert_eq!(entry_meta.referenced, false);
        assert_eq!(entry_meta.occupied, false);
        entry_meta
    }

    #[test]
    fn add_remove_segment() {
        let seg_cnt = 1;
        let cache_size = SEGMENT_SIZE * seg_cnt;
        let entry_size = std::mem::size_of::<TestEntry>();

        let entry_per_seg =
            EFFECTIVE_SEGMENT_SIZE / (entry_size + std::mem::size_of::<EntryMeta>());
        let cache = ClockCache::new(cache_size, entry_size);

        let mut allocated = vec![];

        for i in 1..=entry_per_seg {
            let entry = cache.probe_entry().unwrap();
            let mut entry_meta = assert_empty_entry(entry);

            let test_entry = TestEntry::init(i as u16);
            let cached_ptr = unsafe { &*entry }.data_ptr() as *mut TestEntry;
            unsafe { cached_ptr.write(test_entry) };
            allocated.push(cached_ptr);

            entry_meta.set_occupied();
            unsafe { &*entry }
                .meta
                .store(entry_meta.into(), Ordering::Relaxed);
        }

        // move the cursor to next segment
        for _i in 0..entry_per_seg / 16 {
            let entry = cache.probe_entry();
            assert!(entry.is_none());
        }

        cache.add_segment(Segment::alloc());

        for i in 1..=entry_per_seg {
            let entry = cache.probe_entry().unwrap();
            let mut entry_meta = assert_empty_entry(entry);

            let test_entry = TestEntry::init((i + entry_per_seg) as u16);
            let cached_ptr = unsafe { &*entry }.data_ptr() as *mut TestEntry;
            unsafe { cached_ptr.write(test_entry) };
            allocated.push(cached_ptr);

            entry_meta.set_occupied();
            unsafe { &*entry }
                .meta
                .store(entry_meta.into(), Ordering::Relaxed);
        }

        for ptr in allocated.iter() {
            unsafe { &**ptr }.sanity_check();
        }

        {
            let mut evicted = 0;
            let seg_iter = cache.remove_segment().unwrap();
            let segment = seg_iter.segment;
            for _entry in seg_iter {
                evicted += 1;
            }

            assert_eq!(evicted, entry_per_seg);

            unsafe {
                Segment::dealloc(segment as *mut u8);
            }
        }
        assert!(cache.remove_segment().is_none());
        std::mem::drop(cache);
    }

    fn thread_probe_entry(cache: &ClockCache, i: usize) {
        let entry = cache.probe_entry();
        match entry {
            Some(e) => {
                let meta = unsafe { &*e }.meta.load(Ordering::Relaxed);
                let meta = EntryMetaUnpacked::from(meta);
                if meta.referenced {
                    // means the entry is ready to write
                    let val = TestEntry::init(i as u16);
                    let ptr = unsafe { &*e }.data_ptr() as *mut TestEntry;
                    unsafe {
                        ptr.write(val);
                    }
                    unsafe { &*e }.meta.store(meta.into(), Ordering::Release);
                } else {
                    // the entry was occupied, we check its sanity
                    let ptr = unsafe { &*e }.data_ptr() as *const TestEntry;
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
        let entry_size = std::mem::size_of::<TestEntry>() * 10; // increase entry size to increase the probability of contention

        let entry_per_seg =
            EFFECTIVE_SEGMENT_SIZE / (entry_size + std::mem::size_of::<EntryMeta>());
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
            cache.add_segment(Segment::alloc());

            for i in 1..=entry_per_seg {
                pin.repin();
                thread_probe_entry(&cache, i);
            }

            let seg_iter = cache.remove_segment().unwrap();

            pin.repin();

            pin.defer(move || {
                std::mem::drop(seg_iter);
            });
        }));

        for h in thread_handlers {
            h.join().unwrap();
        }
    }

    /// This test case is intended for shuttle testing
    fn multi_thread_basic() {
        let seg_cnt = 2;
        let cache_size = SEGMENT_SIZE * seg_cnt;
        let entry_size = std::mem::size_of::<TestEntry>() * 10; // increase entry size to increase the probability of contention

        let entry_per_seg =
            EFFECTIVE_SEGMENT_SIZE / (entry_size + std::mem::size_of::<EntryMeta>());
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
        runner.add(shuttle::scheduler::PctScheduler::new(5, 5_000));
        runner.add(shuttle::scheduler::PctScheduler::new(5, 5_000));
        runner.add(shuttle::scheduler::PctScheduler::new(5, 5_000));
        runner.add(shuttle::scheduler::PctScheduler::new(5, 5_000));
        runner.add(shuttle::scheduler::RandomScheduler::new(5_000));
        runner.add(shuttle::scheduler::RandomScheduler::new(5_000));
        runner.add(shuttle::scheduler::RandomScheduler::new(5_000));
        runner.add(shuttle::scheduler::RandomScheduler::new(5_000));
        runner.run(multi_thread_add_remove_segment);
    }

    #[cfg(feature = "shuttle")]
    #[test]
    fn shuttle_basic() {
        let config = shuttle::Config::default();
        let mut runner = shuttle::PortfolioRunner::new(true, config);
        runner.add(shuttle::scheduler::PctScheduler::new(5, 5_000));
        runner.add(shuttle::scheduler::PctScheduler::new(5, 5_000));
        runner.add(shuttle::scheduler::PctScheduler::new(5, 5_000));
        runner.add(shuttle::scheduler::PctScheduler::new(5, 5_000));
        runner.add(shuttle::scheduler::RandomScheduler::new(5_000));
        runner.add(shuttle::scheduler::RandomScheduler::new(5_000));
        runner.add(shuttle::scheduler::RandomScheduler::new(5_000));
        runner.add(shuttle::scheduler::RandomScheduler::new(5_000));
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
}