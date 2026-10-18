"""Storage formats and bit costs of spatial compression, per KV head.

Tokens are grouped inside consecutive eight-token windows. A group of 2, 4 or 8
adjacent tokens stores one bf16 mean direction, one bf16 norm per token and an
optional sparse residual: kept bf16 values plus a ``head_dim``-bit position mask.
A pair stores one residual (its two deviations are opposite); quads and eights
store one per token. Every group (and every dense pair) carries a 3-bit format
tag, and each compressed layer slot one 16-bit header. Dense tokens cost 16 bits
per value.
"""

WINDOW = 8
MENUS = {2: (0, 16, 32), 4: (0, 16), 8: (0, 16)}
VALUE_BITS = 16
TAG_BITS = 3
HEADER_BITS = 16
REFERENCE_DIM = 128  # menu residual counts are per this many values
ACCOUNTING = "spatial_v3_per_head_bf16_norms_bitmap_tag3"


def kept_entries(residual, dim):
    """Residual entries kept for a menu entry: menus are per 128 values, so a
    256-value head (Qwen3.5) keeps twice as many and PR32 is always a 25% residual."""
    return residual * dim // REFERENCE_DIM


def group_bits(residual, dim, size, norms=True):
    """Payload bits of one merged group (excluding its format tag).

    ``norms=False`` drops the per-token bf16 norms (the mean is stored unnormalized).
    """
    if size not in MENUS or residual not in MENUS[size]:
        raise ValueError(f"no {size}-token format with {residual} residual entries")
    residuals = 1 if size == 2 else size
    mask = residuals * dim if residual else 0
    kept = kept_entries(residual, dim)
    return VALUE_BITS * (dim + residuals * kept + (size if norms else 0)) + mask


def dense_bits(tokens, dim):
    return VALUE_BITS * tokens * dim


def head_levels(dim, norms=True):
    """Allowed fixed per-head savings: 0, 1/8, 1/4 and 3/8 of a head's dense bits, and
    the most a head can save with pairs (every pair merged with no residual)."""
    most = 1 - group_bits(0, dim, 2, norms) / dense_bits(2, dim)
    return (0.0, 0.125, 0.25, 0.375, most)


def max_saving(dim, size=2, residual=0):
    """Saving of a slot stored entirely in one format, ignoring the header."""
    return 1 - (group_bits(residual, dim, size) + TAG_BITS) / dense_bits(size, dim)


def sizes(max_group):
    """Group sizes allowed by ``max_group`` (2: pairs; 4: and quads; 8: and eights)."""
    if isinstance(max_group, bool) or max_group not in MENUS:
        raise ValueError("max_group must be 2, 4 or 8")
    return tuple(g for g in MENUS if g <= max_group)
