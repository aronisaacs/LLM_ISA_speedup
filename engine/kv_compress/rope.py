"""Undo and reapply the Llama RoPE that Cache.update already sees on keys.

``k_embed = k * cos + rotate_half(k) * sin``. The inverse flips the sine sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RopeTables:
    """``rope_theta`` and head dim used to rebuild cos/sin for one model.

    ``inv_freq``, when set, is the table the loaded decoder actually uses
    (including Llama 3 frequency scaling). Otherwise it is built from ``rope_theta``.
    """

    rope_theta: float
    head_dim: int
    inv_freq: tuple[float, ...] | None = None
    attention_scaling: float = 1.0

    def cos_sin(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim != 1:
            raise ValueError("RoPE positions must be a 1-D index")
        if self.inv_freq is None:
            exponents = torch.arange(0, self.head_dim, 2, device=positions.device, dtype=torch.float32) / self.head_dim
            inv_freq = 1.0 / (self.rope_theta ** exponents)
        else:
            inv_freq = torch.tensor(self.inv_freq, device=positions.device, dtype=torch.float32)
        freqs = torch.outer(positions.to(dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        if emb.shape[-1] != self.head_dim:
            raise ValueError(
                f"RoPE head_dim={self.head_dim} did not produce a matching table (got {emb.shape[-1]})"
            )
        return (emb.cos() * self.attention_scaling).to(dtype=dtype), (emb.sin() * self.attention_scaling).to(dtype=dtype)


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first = tensor[..., : tensor.shape[-1] // 2]
    second = tensor[..., tensor.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rope(keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, inverse: bool) -> torch.Tensor:
    """Rotate ``keys`` shaped [batch, heads, sequence, head_dim]."""
    if keys.shape[-1] != cos.shape[-1]:
        raise ValueError(f"key head dim {keys.shape[-1]} does not match RoPE table {cos.shape[-1]}")
    if keys.shape[-2] != cos.shape[-2]:
        raise ValueError(f"key sequence {keys.shape[-2]} does not match RoPE positions {cos.shape[-2]}")
    signed = -sin if inverse else sin
    wide_cos = cos.unsqueeze(0).unsqueeze(0)
    wide_sin = signed.unsqueeze(0).unsqueeze(0)
    return keys * wide_cos + rotate_half(keys) * wide_sin


def rope_from_config(config) -> RopeTables | None:
    """Read ``rope_theta`` and head dim off a decoder config. Missing theta skips RoPE."""
    if config is None:
        return None
    parameters = getattr(config, "rope_parameters", None)
    if parameters is None and isinstance(config, dict):
        parameters = config.get("rope_parameters")
    theta = getattr(config, "rope_theta", None)
    if theta is None and isinstance(config, dict):
        theta = config.get("rope_theta")
    if theta is None and isinstance(parameters, dict):
        theta = parameters.get("rope_theta")
    if not isinstance(theta, (int, float)) or isinstance(theta, bool):
        return None
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and isinstance(config, dict):
        head_dim = config.get("head_dim")
    if not isinstance(head_dim, int) or isinstance(head_dim, bool):
        hidden = getattr(config, "hidden_size", None)
        heads = getattr(config, "num_attention_heads", None)
        if isinstance(config, dict):
            hidden = config.get("hidden_size", hidden)
            heads = config.get("num_attention_heads", heads)
        if not isinstance(hidden, int) or not isinstance(heads, int) or heads <= 0:
            return None
        head_dim = hidden // heads
    return RopeTables(rope_theta=float(theta), head_dim=int(head_dim))
