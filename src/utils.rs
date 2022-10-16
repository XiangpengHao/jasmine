/// Align the given `size` upwards to alignment `align`.
///
/// Requires that `align` is a power of two.
pub(crate) fn align_up(size: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (size + align - 1) & !(align - 1)
}
