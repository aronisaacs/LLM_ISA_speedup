"""Storage formats and bit costs of spatial compression, per KV head.

Tokens are grouped inside consecutive eight-token windows. A group of 2, 4 or 8
adjacent tokens stores one bf16 mean direction, one bf16 norm per token and an
optional sparse residual: kept bf16 values plus a ``head_dim``-bit position mask.
A pair stores one residual (its two deviations are opposite); quads and eights
store one per token. Every terminal group carries a 3-bit format tag, and each
compressed layer slot one 16-bit header. Dense tokens cost 16 bits per value.
"""

WINDOW = 8
MENUS = {2: (0, 16, 32), 4: (0, 16), 8: (0, 16)}
VALUE_BITS = 16
TAG_BITS = 3
HEADER_BITS = 16
ACCOUNTING = "spatial_v3_per_head_bf16_norms_bitmap_tag3"


def group_bits(residual, dim, size):
    """Payload bits of one merged group (excluding its format tag)."""
    if size not in MENUS or residual not in MENUS[size]:
        raise ValueError(f"no {size}-token format with {residual} residual entries")
    residuals = 1 if size == 2 else size
    mask = residuals * dim if residual else 0
    return VALUE_BITS * (dim + residuals * residual + size) + mask


def dense_bits(tokens, dim):
    return VALUE_BITS * tokens * dim


def max_saving(dim, size=2, residual=0):
    """Saving of a slot stored entirely in one format, ignoring the header."""
    return 1 - (group_bits(residual, dim, size) + TAG_BITS) / dense_bits(size, dim)


def validate_layout(
    max_group, *, fixed_residual=None, residual_counts=None, merge_sizes=None
):
    """One validation contract for both public configuration and tree construction."""
    if (
        isinstance(max_group, bool)
        or not isinstance(max_group, int)
        or max_group not in MENUS
    ):
        raise ValueError("max_group must be 2, 4 or 8")
    enabled = (
        tuple(g for g in MENUS if g <= max_group)
        if merge_sizes is None
        else tuple(merge_sizes)
    )
    if (
        not enabled
        or len(set(enabled)) != len(enabled)
        or any(
            isinstance(g, bool)
            or not isinstance(g, int)
            or g not in MENUS
            or g > max_group
            for g in enabled
        )
    ):
        raise ValueError("merge_sizes must be a nonempty subset of enabled group sizes")
    if fixed_residual is not None and (
        max_group != 2
        or 2 not in enabled
        or isinstance(fixed_residual, bool)
        or not isinstance(fixed_residual, int)
        or fixed_residual not in MENUS[2]
    ):
        raise ValueError("a fixed residual count requires pairs and 0/16/32 entries")
    if residual_counts is not None:
        if (
            not isinstance(residual_counts, dict)
            or fixed_residual is not None
            or set(residual_counts) != {str(g) for g in enabled}
            or any(
                isinstance(r, bool) or not isinstance(r, int) or r not in MENUS[int(g)]
                for g, r in residual_counts.items()
            )
        ):
            raise ValueError(
                "residual_counts must fix one legal count for each enabled group size"
            )
    return enabled
