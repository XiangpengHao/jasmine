use crate::ClockCache;

pub struct ShardCache<const N: usize> {
    sub_cache: [ClockCache; N],
}

impl<const N: usize> ShardCache<N> {
    pub fn get_random(&self, rng: &mut impl rand::Rng) -> &ClockCache {
        let index = rng.gen_range(0..N);
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
}
