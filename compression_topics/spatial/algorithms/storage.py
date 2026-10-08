"""Storage formulas shared by spatial kernels and study planners (bits)."""
import math


def residual_mask_bits(kept, dim, residuals, mask="bitmap"):
    """Positions of ``kept`` residual entries: a ``dim``-bit bitmap, one index each, or the smaller."""
    if not kept:
        return 0
    bitmap, index = residuals * dim, residuals * kept * math.ceil(math.log2(dim))
    if mask == "bitmap":
        return bitmap
    if mask == "index":
        return index
    if mask == "auto":
        return min(bitmap, index)
    raise ValueError("mask must be 'bitmap', 'index' or 'auto'")


def vector_group_bits(kept, dim, group_size, *, include_norms, mask="bitmap"):
    residuals = 1 if group_size == 2 else group_size
    return (16 * (dim + residuals * kept + (group_size if include_norms else 0))
            + residual_mask_bits(kept, dim, residuals, mask))


def metadata_bits(groups, bits_per_group=1):
    return 8 * math.ceil(bits_per_group * groups / 8) + 16
