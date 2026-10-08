"""Storage formulas shared by spatial kernels and study planners (bits)."""
import math


def vector_group_bits(kept, dim, group_size, *, include_norms):
    residuals = 1 if group_size == 2 else group_size
    return 16 * (dim + residuals * kept + (group_size if include_norms else 0)) + (residuals * dim if kept else 0)


def metadata_bits(groups):
    return 8 * math.ceil(groups / 8) + 16
