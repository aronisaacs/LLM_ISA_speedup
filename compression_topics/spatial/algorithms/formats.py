"""Storage formats and bit costs of spatial pair compression, per KV head.

Adjacent tokens (2i, 2i+1) of one KV head form a pair. A pair is stored dense,
or merged: one bf16 mean direction, one bf16 scale per token and a sparse
residual of 0, 16 or 32 kept bf16 values plus a ``head_dim``-bit position mask
(PR0, PR16, PR32). Every pair carries a 2-bit format tag, and each compressed
layer slot one 16-bit header. Dense tokens cost 16 bits per value.
"""

WINDOW = 16  # tokens compressed together, per KV head
RESIDUALS = (0, 16, 32)  # kept residual entries per 128 values: PR0, PR16, PR32
VALUE_BITS = 16
TAG_BITS = 2  # dense, PR0, PR16 or PR32
HEADER_BITS = 16
REFERENCE_DIM = 128  # residual counts are per this many values
ACCOUNTING = "spatial_v4_per_head_bf16_scales_bitmap_tag2"


def kept_entries(residual, dim):
    """Residual entries kept: counts are per 128 values, so a 256-value head
    (Qwen3.5) keeps twice as many and PR32 is always a 25% residual."""
    return residual * dim // REFERENCE_DIM


def pair_bits(residual, dim):
    """Payload bits of one merged pair (excluding its format tag)."""
    if residual not in RESIDUALS:
        raise ValueError(f"no pair format with {residual} residual entries")
    mask = dim if residual else 0
    return VALUE_BITS * (dim + kept_entries(residual, dim) + 2) + mask


def dense_bits(tokens, dim):
    return VALUE_BITS * tokens * dim


def max_saving(dim):
    """Saving when every pair is merged with no residual, ignoring the header."""
    return 1 - (pair_bits(0, dim) + TAG_BITS) / dense_bits(2, dim)
