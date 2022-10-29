use std::mem::ManuallyDrop;

use douhua::{Allocator, MemType, TieredAllocator};

#[cfg(not(feature = "shuttle"))]
use std::sync::atomic::{AtomicPtr, Ordering};

#[cfg(feature = "shuttle")]
use shuttle::sync::atomic::{AtomicPtr, Ordering};

use std::sync::{RwLock, RwLockWriteGuard};

use crate::{cache::EntryMeta, SEGMENT_ALIGN, SEGMENT_SIZE};

pub struct Segment {
    pub(crate) next: AtomicPtr<Segment>,
    pub(crate) migration_lock: RwLock<()>,
    pub(crate) entry_start_offset: u32,
    pub(crate) entry_phy_size: u32,
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
                .allocate_zeroed(seg_layout, mem_type)
                .expect("OOM");
            let ptr = ptr.as_non_null_ptr().as_ptr() as *mut Segment;
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
        Allocator::get().deallocate(std::ptr::NonNull::new_unchecked(ptr), layout, mem_type);
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
    pub(crate) fn new(
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
