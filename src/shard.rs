use nanorand::Rng;

use crate::{ClockCache, EntryMeta};

pub struct ShardCache<const N: usize> {
    sub_cache: [ClockCache; N],
}

impl<const N: usize> ShardCache<N> {
    pub fn get_random(&self, rng: &mut impl rand::Rng) -> &ClockCache {
        let index = rng.gen_range(0..N);
        unsafe { self.sub_cache.get_unchecked(index) }
    }

    pub fn get(&self) -> &ClockCache {
        let index = nanorand::tls_rng().generate_range(0..N);
        unsafe { self.sub_cache.get_unchecked(index) }
    }

    pub fn new(cache_size_byte: usize, entry_size: usize, entry_align: usize) -> Self {
        let cache_size = cache_size_byte / N;
        let sub_cache = {
            let mut sub_cache: [std::mem::MaybeUninit<ClockCache>; N] =
                unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            for elem in &mut sub_cache {
                unsafe {
                    std::ptr::write(
                        elem.as_mut_ptr(),
                        ClockCache::new(cache_size, entry_size, entry_align),
                    )
                }
            }

            // can't transmute here, see this: https://github.com/rust-lang/rust/issues/61956
            let ptr = &mut sub_cache as *mut _ as *mut [ClockCache; N];
            let res = unsafe { ptr.read() };
            std::mem::forget(sub_cache);
            res
        };
        Self { sub_cache }
    }

    pub fn cache_size(&self) -> usize {
        let mut seg_cnt = 0;
        for s in self.sub_cache.iter() {
            seg_cnt += s.cache_size();
        }
        seg_cnt
    }

    /// Mark the entry as referenced so that it won't be evicted too soon.
    ///
    /// # Safety
    /// The caller must ensure the entry ptr is valid: (1) non-null, (2) pointing to the right entry with right offset.
    pub unsafe fn mark_referenced(&self, entry: *mut EntryMeta) {
        let mut meta = unsafe { &*entry }.load_meta(std::sync::atomic::Ordering::Relaxed);
        meta.referenced = true;
        unsafe {
            (*entry).set_meta(meta, std::sync::atomic::Ordering::Release);
        }
    }
}