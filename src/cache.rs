use std::{alloc::Layout, mem::ManuallyDrop, ops::Deref};

#[cfg(feature = "shuttle")]
use shuttle::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(not(feature = "shuttle"))]
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};

use douhua::{Allocator, MemType};

use crate::{backoff::Backoff, utils::align_up, JasmineError, SEGMENT_ALIGN, SEGMENT_SIZE};

pub struct Segment {
    next: AtomicPtr<Segment>,
    migration_lock: RwLock<()>,
    entry_start_offset: u32,
    entry_phy_size: u32,
}

#[cfg(test)]
#[cfg(not(feature = "shuttle"))]
static_assertions::const_assert_eq!(std::mem::size_of::<Segment>(), 32);

impl Segment {
    pub(crate) fn first_meta(&self) -> *const EntryMeta {
        let ptr =
            unsafe { (self as *const Segment as *const u8).add(std::mem::size_of::<Segment>()) };
        ptr as *const EntryMeta
    }

    #[inline]
    pub(crate) fn from_meta(ptr: *const EntryMeta) -> *const Segment {
        let ptr = (ptr as usize) & !SEGMENT_ALIGN;
        ptr as *mut Segment
    }

    pub(crate) fn from_entry(ptr: *const u8) -> *const Segment {
        let ptr = (ptr as usize) & !SEGMENT_ALIGN;
        ptr as *mut Segment
    }

    pub fn new_from_heap(
        mem_type: MemType,
        entry_phy_size: u32,
        entry_start_offset: u32,
    ) -> *mut Segment {
        unsafe {
            let seg_layout =
                std::alloc::Layout::from_size_align(SEGMENT_SIZE, SEGMENT_SIZE).unwrap();

            let ptr = Allocator::get()
                .alloc_zeroed(seg_layout, mem_type)
                .expect("OOM") as *mut Segment;
            let seg_ref = &mut *ptr;
            seg_ref.next.store(std::ptr::null_mut(), Ordering::Relaxed);
            seg_ref.migration_lock = RwLock::new(());
            seg_ref.entry_phy_size = entry_phy_size;
            seg_ref.entry_start_offset = entry_start_offset;
            ptr
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
    entry_cnt_per_seg: usize,
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
        mem_type: MemType,
        entry_cnt_per_seg: usize,
        lock: RwLockWriteGuard<'a, ()>,
    ) -> Self {
        Self {
            ptr,
            mem_type,
            entry_cnt_per_seg,
            _lock: ManuallyDrop::new(lock),
        }
    }

    pub fn iter<'b>(&'_ self) -> SegmentIter<'b> {
        SegmentIter::new(self.ptr, self.entry_cnt_per_seg)
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
    cur_entry: Option<*const EntryMeta>,
    entry_cnt_per_seg: usize,
    phantom: std::marker::PhantomData<&'a ()>,
}

unsafe impl Send for SegmentIter<'_> {}
unsafe impl Sync for SegmentIter<'_> {}

impl<'a> SegmentIter<'a> {
    fn new(segment: *const Segment, entry_cnt_per_seg: usize) -> Self {
        Self {
            cur_entry: Some(unsafe { &*segment }.first_meta()),
            entry_cnt_per_seg,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'a> Iterator for SegmentIter<'a> {
    type Item = *const EntryMeta;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur_entry?;
        self.cur_entry = unsafe { &*cur }.next_entry(self.entry_cnt_per_seg);
        Some(cur)
    }
}

pub struct EntryMeta {
    pub(crate) meta: AtomicU8,
}

impl EntryMeta {
    /// Returns the next entry within the segment.
    /// Returns `None` if the entry is the last one.
    fn next_entry(&self, max_entry_per_seg: usize) -> Option<*const EntryMeta> {
        let segment = Segment::from_meta(self);
        let cur_entry_idx = ((self as *const EntryMeta as usize - segment as usize)
            - std::mem::size_of::<Segment>())
            / std::mem::size_of::<EntryMeta>();
        if cur_entry_idx + 1 >= max_entry_per_seg {
            None
        } else {
            Some(unsafe { (self as *const EntryMeta).add(1) })
        }
    }
}

#[derive(Clone, Debug)]
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

pub(crate) struct LockedEntry<'a> {
    entry: &'a EntryMeta,
}

impl Deref for LockedEntry<'_> {
    type Target = EntryMeta;

    fn deref(&self) -> &Self::Target {
        self.entry
    }
}

impl Drop for LockedEntry<'_> {
    fn drop(&mut self) {
        let old: EntryMetaUnpacked = self.entry.meta.load(Ordering::Relaxed).into();
        assert!(old.locked, "entry should be locked");
        let mut new = old.clone();
        new.locked = false;

        let rv = self.entry.meta.compare_exchange_weak(
            old.into(),
            new.into(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        assert!(rv.is_ok());
    }
}

pub struct ClockCache {
    segments: AtomicPtr<Segment>,
    probe_len: usize,
    pub(crate) probe_loc: AtomicPtr<EntryMeta>,
    segment_cnt: AtomicUsize,
    pub(crate) entry_phy_size: usize,
    pub(crate) entry_per_seg: usize,
    pub(crate) entry_offset_of_seg: usize,
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
    /// Get entry start offset and count of a segment
    fn entry_offset_and_cnt(entry_layout: &Layout) -> (usize, usize) {
        let effective_size = SEGMENT_SIZE - std::mem::size_of::<Segment>();
        let entry_phy_size = align_up(entry_layout.size(), entry_layout.align());

        let upper_bound = effective_size / (entry_phy_size + std::mem::size_of::<EntryMeta>());
        let entry_start = align_up(
            std::mem::size_of::<Segment>() + upper_bound * std::mem::size_of::<EntryMeta>(),
            entry_layout.align(),
        );
        let effective_entry_cnt = (SEGMENT_SIZE - entry_start) / entry_phy_size;
        (entry_start, std::cmp::min(effective_entry_cnt, upper_bound))
    }

    fn data_ptr(&self, entry: *const EntryMeta) -> *mut u8 {
        let segment = Segment::from_meta(entry);
        let cur_entry_idx = ((entry as *const EntryMeta as usize - segment as usize)
            - std::mem::size_of::<Segment>())
            / std::mem::size_of::<EntryMeta>();
        let entry_offset = self.entry_offset_of_seg + cur_entry_idx * self.entry_phy_size;
        unsafe { (segment as *mut u8).add(entry_offset) }
    }

    pub fn cache_size(&self) -> usize {
        self.segment_cnt.load(Ordering::Relaxed) * SEGMENT_SIZE
    }

    pub fn new(cache_size_byte: usize, entry_layout: Layout, mem_type: douhua::MemType) -> Self {
        let seg_cnt = cache_size_byte / SEGMENT_SIZE;

        let entry_phy_size = align_up(entry_layout.size(), entry_layout.align());

        let (entry_offset, entry_cnt) = Self::entry_offset_and_cnt(&entry_layout);
        let mut first: *mut Segment = std::ptr::null_mut();
        let mut prev: *mut Segment = std::ptr::null_mut();
        for i in 0..seg_cnt {
            let ptr = Segment::new_from_heap(mem_type, entry_phy_size as u32, entry_offset as u32);

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
            unsafe { &*first }.first_meta()
        } else {
            std::ptr::null_mut()
        };

        ClockCache {
            segments: AtomicPtr::new(first),
            probe_loc: AtomicPtr::new(first_entry as *mut EntryMeta),
            probe_len: 16,
            entry_phy_size,
            mem_type,
            entry_per_seg: entry_cnt,
            entry_offset_of_seg: entry_offset,
            segment_cnt: AtomicUsize::new(seg_cnt),
        }
    }

    /// only one thread at any time should call this function
    ///
    /// # Safety
    /// The ptr must be valid and at least `SEGMENT_SIZE` (default 4096) bytes long.
    ///
    /// The only safe and stable way to get a ptr is to call `Segment::alloc()`.
    pub unsafe fn add_segment(&self, ptr: *mut Segment) {
        let cur = self.segments.load(Ordering::Relaxed);
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
            new_segment.first_meta() as *mut EntryMeta,
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
                Ok(v) => break v,
                Err(_) => {
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
        let prob_seg = Segment::from_meta(prob_loc);
        if prob_seg == cur {
            if next == cur {
                // we are the last one, no thread will probe_entry on us
                // this means that when the cache is empty, no one should call probe_entry, otherwise a segment fault will happen
                self.probe_loc
                    .store(std::ptr::null_mut(), Ordering::Release);
            } else {
                // the prob_loc is in the current segment, we need to update to next segment
                self.probe_loc.store(
                    unsafe { &*next }.first_meta() as *mut EntryMeta,
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
            self.mem_type,
            self.entry_per_seg,
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
            let (entry, lock) = match unsafe { &*cur_entry }.next_entry(self.entry_per_seg) {
                Some(v) => {
                    let seg_lock = match seg_lock {
                        Some(l) => l,
                        None => {
                            let segment = Segment::from_meta(cur_entry);
                            if let Ok(seg_lock) = unsafe { &*segment }.migration_lock.try_read() {
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
                    let cur_segment = Segment::from_meta(cur_entry);
                    let segment = unsafe { &*cur_segment }.next.load(Ordering::Relaxed);

                    if let Ok(seg_lock) = unsafe { &*segment }.migration_lock.try_read() {
                        let new_entry = unsafe { &*segment }.first_meta();
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
                entry as *mut EntryMeta,
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
    pub(crate) fn probe_entry(&self) -> Result<LockedEntry, JasmineError> {
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
                Ok(_) => {
                    return Ok(LockedEntry {
                        entry: unsafe { &*cur_entry },
                    })
                }
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
                            return Ok(LockedEntry {
                                entry: unsafe { &*cur_entry },
                            });
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
        Fill: FnOnce(Option<ET>, *mut u8) -> FT,
        ET,
        FT,
    >(
        &self,
        evict_callback: Evict,
        fill_callback: Fill,
    ) -> Result<FT, JasmineError> {
        let e = self.probe_entry()?;

        let mut meta: EntryMetaUnpacked = e.meta.load(Ordering::Acquire).into();

        assert!(meta.locked, "Entry should be locked: {:?}", meta);

        if meta.occupied {
            let et = evict_callback(self.data_ptr(e.entry)).ok_or(JasmineError::EvictFailure)?;

            let filled = fill_callback(Some(et), self.data_ptr(e.entry));
            meta.locked = false;
            meta.referenced = true;
            meta.occupied = true;
            e.meta.store(meta.into(), Ordering::Release);
            std::mem::forget(e);
            Ok(filled)
        } else {
            // the entry was empty, no eviction needed.
            let ptr = self.data_ptr(e.entry);

            let filled = fill_callback(None, ptr);

            meta.locked = false;
            meta.referenced = true;
            meta.occupied = true;
            e.meta.store(meta.into(), Ordering::Release);
            std::mem::forget(e);
            Ok(filled)
        }
    }

    /// Mark the entry as referenced so that it won't be evicted too soon.
    ///
    /// # Safety
    /// The caller must ensure the entry ptr is valid: (1) non-null, (2) pointing to the right entry with right offset.
    pub unsafe fn mark_referenced(&self, entry: *const u8) {
        let segment_ptr = Segment::from_entry(entry);
        let segment = unsafe { &*segment_ptr };
        let entry_offset =
            (entry as usize - segment_ptr as usize - segment.entry_start_offset as usize)
                / segment.entry_phy_size as usize;

        let first_meta = segment.first_meta();
        let target_meta = first_meta.add(entry_offset);
        let old = unsafe { &*target_meta }.meta.load(Ordering::Relaxed);
        let mut meta = EntryMetaUnpacked::from(old);
        if meta.referenced || meta.locked {
            return;
        }
        meta.referenced = true;
        let _ = unsafe { &*target_meta }.meta.compare_exchange_weak(
            old,
            meta.into(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ); // we don't care if it fails, but we need to use CAS to make sure the old value is still valid.
    }

    /// Mark the entry as empty.
    ///
    /// # Safety
    /// The caller must ensure the entry ptr is valid: (1) non-null, (2) pointing to the right entry with right offset.
    pub unsafe fn mark_empty(&self, entry: *const u8) {
        let segment_ptr = Segment::from_entry(entry);
        let segment = unsafe { &*segment_ptr };
        let entry_offset =
            (entry as usize - segment_ptr as usize - segment.entry_start_offset as usize)
                / segment.entry_phy_size as usize;

        let first_meta = segment.first_meta();
        let target_meta = first_meta.add(entry_offset);

        let backoff = Backoff::new();
        loop {
            let old = unsafe { &*target_meta }.meta.load(Ordering::Relaxed);
            let mut meta = EntryMetaUnpacked::from(old);
            if meta.locked {
                // we must wait the lock to be released.
                backoff.spin();
                continue;
            }
            meta.locked = false;
            meta.referenced = false;
            meta.occupied = false;
            match unsafe { &*target_meta }.meta.compare_exchange_weak(
                old,
                meta.into(),
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(_) => {
                    backoff.snooze();
                    continue;
                }
            }
        }
    }
}
