use douhua::{Allocator, MemType};
#[cfg(all(feature = "shuttle", test))]
use shuttle::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};

#[cfg(not(all(feature = "shuttle", test)))]
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};

use crate::{backoff::Backoff, JasmineError};
use spin::{rwlock::RwLockWriteGuard, RwLockReadGuard};
use std::{alloc::Layout, mem::ManuallyDrop};

/// Jasmine manages memory at the granularity of segments.
pub const SEGMENT_SIZE: usize = 2 * 1024 * 1024; // 2MB
const SEGMENT_ALIGN: usize = SEGMENT_SIZE - 1;

/// The unit of cache allocation
pub struct Segment {
    next: AtomicPtr<Segment>,
    migration_lock: spin::RwLock<()>,
}

#[cfg(test)]
#[cfg(not(feature = "shuttle"))]
static_assertions::const_assert_eq!(std::mem::size_of::<Segment>(), 16);

/// Align the given `size` upwards to alignment `align`.
///
/// Requires that `align` is a power of two.
fn align_up(size: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (size + align - 1) & !(align - 1)
}

impl Segment {
    pub(crate) fn first_entry(&self, entry_align: usize) -> *mut EntryMeta {
        let ptr = unsafe { (self as *const Segment as *mut u8).add(std::mem::size_of::<Segment>()) }
            as usize;
        let ptr = align_up(ptr, entry_align);
        ptr as *mut EntryMeta
    }

    pub(crate) fn from_entry(ptr: *const EntryMeta) -> *const Segment {
        let ptr = (ptr as usize) & !SEGMENT_ALIGN;
        ptr as *mut Segment
    }

    pub fn alloc(mem_type: MemType) -> *mut u8 {
        unsafe {
            let seg_layout =
                std::alloc::Layout::from_size_align(SEGMENT_SIZE, SEGMENT_SIZE).unwrap();

            Allocator::get()
                .alloc_zeroed(seg_layout, mem_type)
                .expect("OOM")
            // std::alloc::alloc_zeroed(seg_layout)
        }
    }

    /// # Safety
    /// Deallocate a segment, the ptr must be allocated from Segment::alloc().
    pub unsafe fn dealloc(ptr: *mut u8, mem_type: MemType) {
        let layout = std::alloc::Layout::from_size_align(SEGMENT_SIZE, SEGMENT_SIZE).unwrap();
        Allocator::get().dealloc(ptr, layout, mem_type);
    }
}

pub struct ObsoleteSegment<'a> {
    ptr: *mut Segment,
    entry_layout: Layout,
    mem_type: MemType,
    _lock: ManuallyDrop<RwLockWriteGuard<'a, ()>>,
}

impl Drop for ObsoleteSegment<'_> {
    fn drop(&mut self) {
        unsafe {
            Segment::dealloc(self.ptr as *mut u8, self.mem_type);
        }
        // don't drop the lock (marking it as obsolete)
    }
}

unsafe impl Send for ObsoleteSegment<'_> {}
unsafe impl Sync for ObsoleteSegment<'_> {}

impl<'a> ObsoleteSegment<'a> {
    fn new(
        ptr: *mut Segment,
        entry_layout: Layout,
        mem_type: MemType,
        lock: RwLockWriteGuard<'a, ()>,
    ) -> Self {
        Self {
            ptr,
            entry_layout,
            mem_type,
            _lock: ManuallyDrop::new(lock),
        }
    }

    pub fn iter<'b>(&'_ self) -> SegmentIter<'b> {
        SegmentIter::new(self.ptr, self.entry_layout)
    }

    /// Consumes the segment and returns the internal raw pointer.
    /// The pointer is size of SEGMENT_SIZE
    pub fn into_raw(self) -> *mut u8 {
        let ptr = self.ptr as *mut u8;
        std::mem::forget(self);
        ptr
    }
}

pub struct SegmentIter<'a> {
    cur_entry: Option<*mut EntryMeta>,
    entry_layout: Layout,
    phantom: std::marker::PhantomData<&'a ()>,
}

unsafe impl Send for SegmentIter<'_> {}
unsafe impl Sync for SegmentIter<'_> {}

impl<'a> SegmentIter<'a> {
    fn new(segment: *const Segment, entry_layout: Layout) -> Self {
        Self {
            cur_entry: Some(unsafe { &*segment }.first_entry(entry_layout.align())),
            entry_layout,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'a> Iterator for SegmentIter<'a> {
    type Item = *mut EntryMeta;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur_entry?;
        self.cur_entry = unsafe { &*cur }.next_entry(self.entry_layout.size());
        Some(cur)
    }
}

#[derive(Clone)]
pub struct EntryMetaUnpacked {
    pub referenced: bool,
    pub locked: bool,
    pub occupied: bool,
}

impl EntryMetaUnpacked {
    pub fn set_occupied(&mut self) {
        self.locked = false;
        self.occupied = true;
        self.referenced = true;
    }
}

impl From<u8> for EntryMetaUnpacked {
    fn from(value: u8) -> Self {
        EntryMetaUnpacked {
            referenced: value & 1 == 1,
            locked: value & 0b10 == 0b10,
            occupied: value & 0b100 == 0b100,
        }
    }
}

impl From<EntryMetaUnpacked> for u8 {
    fn from(value: EntryMetaUnpacked) -> Self {
        (value.occupied as u8) << 2 | (value.locked as u8) << 1 | (value.referenced as u8)
    }
}

pub struct EntryMeta {
    meta: AtomicU8,
}

impl EntryMeta {
    pub fn data_ptr(&self) -> *mut u8 {
        unsafe {
            (self as *const EntryMeta as *mut u8).add(std::mem::size_of::<EntryMeta>()) as *mut u8
        }
    }

    pub fn load_meta(&self, order: Ordering) -> EntryMetaUnpacked {
        self.meta.load(order).into()
    }

    pub fn set_meta(&self, value: EntryMetaUnpacked, order: Ordering) {
        self.meta.store(value.into(), order);
    }

    /// This is the shortcut of loading the meta, set it to referenced, and then store it back.
    pub fn set_referenced(&self) {
        let mut meta = self.load_meta(Ordering::Relaxed);
        meta.referenced = true;
        self.set_meta(meta, Ordering::Relaxed);
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
    pub(crate) entry_layout: Layout, // the layout of each entry, the size is with the metadata (1 byte)
    pub(crate) probe_loc: AtomicPtr<EntryMeta>,
    segment_cnt: AtomicUsize,
    mem_type: MemType,
}

impl Drop for ClockCache {
    fn drop(&mut self) {
        let mut cur = self.segments.load(Ordering::Relaxed);
        if cur.is_null() {
            return;
        }

        let start = cur;
        loop {
            let next = unsafe { &*cur }.next.load(Ordering::Relaxed);
            unsafe {
                Segment::dealloc(cur as *mut u8, self.mem_type);
            }
            if next == start {
                break;
            }

            assert!(
                !cur.is_null(),
                "segments are circularly linked so it can't be null"
            );
            cur = next;
        }
    }
}

impl ClockCache {
    pub fn new(cache_size_byte: usize, entry_layout: Layout, mem_type: douhua::MemType) -> Self {
        let seg_cnt = cache_size_byte / SEGMENT_SIZE;
        let entry_size = entry_layout.size() + std::mem::size_of::<EntryMeta>();
        let entry_size = align_up(entry_size, entry_layout.align());
        let entry_layout = Layout::from_size_align(entry_size, entry_layout.align()).unwrap();

        let mut first: *mut Segment = std::ptr::null_mut();
        let mut prev: *mut Segment = std::ptr::null_mut();
        for i in 0..seg_cnt {
            let ptr = Segment::alloc(mem_type) as *mut Segment;
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

        let first_entry = if seg_cnt > 0 {
            unsafe { &*first }.first_entry(entry_layout.align())
        } else {
            std::ptr::null_mut()
        };

        ClockCache {
            segments: AtomicPtr::new(first),
            probe_loc: AtomicPtr::new(first_entry),
            probe_len: 16,
            entry_layout,
            mem_type,
            segment_cnt: AtomicUsize::new(seg_cnt),
        }
    }

    pub fn cache_size(&self) -> usize {
        self.segment_cnt.load(Ordering::Relaxed) * SEGMENT_SIZE
    }

    /// only one thread at any time should call this function
    ///
    /// # Safety
    /// The ptr must be valid and at least `SEGMENT_SIZE` (default 4096) bytes long.
    ///
    /// The only safe and stable way to get a ptr is to call `Segment::alloc()`.
    pub unsafe fn add_segment(&self, ptr: *mut u8) {
        let cur = self.segments.load(Ordering::Relaxed);

        {
            std::ptr::write_bytes(ptr, 0, SEGMENT_SIZE);
        }
        let ptr = ptr as *mut Segment;
        let new_segment = { &*ptr };

        if cur.is_null() {
            new_segment.next.store(ptr, Ordering::Relaxed);
            self.segments.store(ptr, Ordering::Release);
        } else {
            let cur_next = { &*cur }.next.load(Ordering::Relaxed);

            new_segment.next.store(cur_next, Ordering::Relaxed);
            { &*cur }.next.store(ptr, Ordering::Relaxed);
            self.segments.store(ptr, Ordering::Release);
        }

        self.segment_cnt.fetch_add(1, Ordering::Relaxed);

        self.probe_loc.store(
            new_segment.first_entry(self.entry_layout.align()),
            Ordering::Release,
        );
    }

    /// The caller need to iterate all the entries in the segment and evict them
    /// As long as the lock guard is held, not other thread can promote entry to this segment
    ///
    /// Returns None when all segments in the cache are removed
    pub fn remove_segment(&self) -> Option<ObsoleteSegment<'static>> {
        // step 1: loop over all segments to find one that we can acquire write lock
        let backoff = Backoff::new();
        let mut prev = self.segments.load(Ordering::Relaxed);
        if prev.is_null() {
            return None;
        }

        let mut cur = unsafe { &*prev }.next.load(Ordering::Relaxed);
        let lock_guard = loop {
            if cur.is_null() {
                // All segments are removed
                return None;
            }
            match unsafe { &*cur }.migration_lock.try_write() {
                Some(v) => break v,
                None => {
                    let next = unsafe { (*cur).next.load(Ordering::Relaxed) };
                    prev = cur;
                    cur = next;
                    backoff.spin();
                }
            }
        };

        let next = unsafe { (*cur).next.load(Ordering::Relaxed) };

        // step 2: make sure the probe_loc is not pointing to the selected segment
        let prob_loc = self.probe_loc.load(Ordering::Acquire);
        let prob_seg = Segment::from_entry(prob_loc);
        if prob_seg == cur {
            if next == cur {
                // we are the last one, no thread will probe_entry on us
                // this means that when the cache is empty, no one should call probe_entry, otherwise a segment fault will happen
                self.probe_loc
                    .store(std::ptr::null_mut(), Ordering::Release);
            } else {
                // the prob_loc is in the current segment, we need to update to next segment
                self.probe_loc.store(
                    unsafe { &*next }.first_entry(self.entry_layout.align()),
                    Ordering::Release,
                );
            }
        }

        unsafe { &*prev }.next.store(next, Ordering::Relaxed);

        self.segment_cnt.fetch_sub(1, Ordering::Relaxed);

        if next == cur {
            // this is the last segment, we are done
            self.segments.store(std::ptr::null_mut(), Ordering::Relaxed);
        } else {
            self.segments.store(next, Ordering::Relaxed);
        }

        Some(ObsoleteSegment::new(
            cur,
            self.entry_layout,
            self.mem_type,
            lock_guard,
        ))
    }

    /// Optional current segment lock, this is just a performance optimization
    /// Returns the old entry as well as its segment lock
    ///
    /// Returns None if the cache is empty (due to `remove_segment`)
    fn next_entry<'a>(
        &'a self,
        mut seg_lock: Option<RwLockReadGuard<'a, ()>>,
    ) -> Option<(*mut EntryMeta, RwLockReadGuard<()>)> {
        let mut cur_entry = self.probe_loc.load(Ordering::Relaxed);

        let backoff = Backoff::new();
        loop {
            if cur_entry.is_null() {
                return None;
            }
            let (entry, lock) = match unsafe { &*cur_entry }.next_entry(self.entry_layout.size()) {
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
                        let new_entry = unsafe { &*segment }.first_entry(self.entry_layout.align());
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
                    return Some((old, lock));
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
    /// It is not impossible for two threads to return the same entry, especially when the cache size is small (probing wrapped around is unlikely)
    /// However, in practice, the caller can assume that calling `probe_entry` is serialized.
    ///
    /// There are two cases this function may return None:
    /// 1. The cache is empty (due to `remove_segment`)
    /// 2. The all the entries are referenced within the probe_len (default to 16)
    pub(crate) fn probe_entry(&self) -> Result<&EntryMeta, JasmineError> {
        let mut seg_lock = None;
        for _p in 0..self.probe_len {
            let (cur_entry, cur_lock) =
                self.next_entry(seg_lock).ok_or(JasmineError::CacheEmpty)?;
            seg_lock = Some(cur_lock);

            match unsafe { &*cur_entry }.meta.compare_exchange_weak(
                0,
                0b10,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(unsafe { &*cur_entry }),
                Err(v) => {
                    let mut meta = EntryMetaUnpacked::from(v);

                    if meta.locked {
                        continue;
                    } else if meta.referenced {
                        meta.referenced = false;
                        let _ = unsafe { &*cur_entry }.meta.compare_exchange_weak(
                            v,
                            meta.into(),
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ); // we don't care if the CAS fails, but we need to use CAS so that we won't accidentally set other bits
                        continue;
                    } else {
                        meta.locked = true;
                        meta.referenced = true;
                        meta.occupied = true;
                        if unsafe { &*cur_entry }
                            .meta
                            .compare_exchange_weak(
                                v,
                                meta.into(),
                                Ordering::Release,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            return Ok(unsafe { &*cur_entry });
                        }
                        continue;
                    }
                }
            };
        }

        Err(JasmineError::ProbeLimitExceeded)
    }

    /// Returns the (cache entry address, whether evict happens), if found.
    /// If find, it evicts the old entry (if any) using the `evict_callback`, then fill in the new data using the `fill_data_callback`.
    /// Both the callbacks get the entry pointer as input, note that each entry has exactly one byte of metadata right before the data pointer,
    /// The caller should not change it.
    ///
    /// Unless the reserved bit is set (the entry is empty), it is the caller's responsibility to serialize the concurrent read/write.
    /// It is not impossible for two threads to return the same entry, especially when the cache size is small (probing wrapped around is unlikely)
    /// However, in practice, the caller can assume that calling `probe_entry` is serialized.
    ///
    /// There are two cases this function may return None:
    /// 1. The cache is empty (due to `remove_segment`)
    /// 2. The all the entries are referenced within the probe_len (default to 16)
    ///
    /// Invariants:
    /// 1. Fill callback can never fail, it will only be called once, and after called, the function returns Ok.
    /// 2. Evict callback can fail, if it fails, it returns Err and fill callback will not be called.
    pub fn probe_entry_evict<
        Evict: FnOnce(*mut u8) -> Option<ET>,
        Fill: FnOnce(*mut u8) -> FT,
        ET,
        FT,
    >(
        &self,
        evict_callback: Evict,
        fill_callback: Fill,
    ) -> Result<(Option<ET>, FT), JasmineError> {
        let e = self.probe_entry()?;

        let mut meta = e.load_meta(Ordering::Acquire);

        // assert!(meta.locked);

        if meta.occupied {
            let et = evict_callback(e.data_ptr()).ok_or(JasmineError::EvictFailure)?;

            let filled = fill_callback(e.data_ptr());
            meta.locked = false;
            meta.referenced = true;
            meta.occupied = true;
            e.set_meta(meta, Ordering::Release);
            Ok((Some(et), filled))
        } else {
            // the entry was empty, no eviction needed.
            let ptr = e.data_ptr();

            let filled = fill_callback(ptr);

            meta.locked = false;
            meta.referenced = true;
            meta.occupied = true;
            e.set_meta(meta, Ordering::Release);
            Ok((None, filled))
        }
    }

    /// Mark the entry as referenced so that it won't be evicted too soon.
    ///
    /// # Safety
    /// The caller must ensure the entry ptr is valid: (1) non-null, (2) pointing to the right entry with right offset.
    pub unsafe fn mark_referenced(&self, entry: *mut EntryMeta) {
        let mut meta = unsafe { &*entry }.load_meta(Ordering::Relaxed);
        meta.referenced = true;
        unsafe {
            (*entry).set_meta(meta, Ordering::Release);
        }
    }

    /// Mark the entry as empty.
    ///
    /// # Safety
    /// The caller must ensure the entry ptr is valid: (1) non-null, (2) pointing to the right entry with right offset.
    pub unsafe fn mark_empty(&self, entry: *mut EntryMeta) {
        let mut meta = unsafe { &*entry }.load_meta(Ordering::Relaxed);
        meta.locked = false;
        meta.referenced = false;
        meta.occupied = false;
        unsafe {
            (*entry).set_meta(meta, Ordering::Release);
        }
    }

    /// # Safety
    /// Do not use this.
    pub unsafe fn pin_entry(&self, entry: *mut EntryMeta) -> Result<u8, u8> {
        let meta = unsafe { &*entry }.load_meta(Ordering::Relaxed);
        if meta.locked {
            return Err(meta.into());
        }
        let mut new_meta = meta.clone();
        new_meta.locked = true;

        unsafe { &*entry }.meta.compare_exchange_weak(
            meta.into(),
            new_meta.into(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        )
    }

    /// # Safety
    /// Do not use this.
    pub unsafe fn unpin_entry(&self, entry: *mut EntryMeta) {
        let mut meta = unsafe { &*entry }.load_meta(Ordering::Relaxed);
        assert!(meta.locked);
        meta.locked = false;
        unsafe {
            (*entry).set_meta(meta, Ordering::Release);
        }
    }
}
