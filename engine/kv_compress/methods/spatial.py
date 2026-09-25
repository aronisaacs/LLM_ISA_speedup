"""Static chunk rewrites along the sequence axis.

Same-shape output: stock attention still runs. A tail shorter than ``chunk``
stays exact. The mean is the average of every token in the closed chunk.
"""

from __future__ import annotations

import torch

from engine.kv_compress.methods import METHODS


def apply_pool(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    chunk: int = 8,
    **_unused,
) -> torch.Tensor:
    del layer_idx, target, _unused
    return _rewrite(tensor, chunk=chunk, kind="pool")


def apply_top1(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    chunk: int = 8,
    **_unused,
) -> torch.Tensor:
    del layer_idx, target, _unused
    return _rewrite(tensor, chunk=chunk, kind="top1")


def apply_tile(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    chunk: int = 8,
    tile: int = 8,
    **_unused,
) -> torch.Tensor:
    del layer_idx, target, _unused
    return _rewrite(tensor, chunk=chunk, kind="tile", tile=tile)


def apply_feature(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    target: str,
    chunk: int = 8,
    **_unused,
) -> torch.Tensor:
    del layer_idx, target, _unused
    return _rewrite(tensor, chunk=chunk, kind="feature")


def _rewrite(tensor: torch.Tensor, *, chunk: int, kind: str, tile: int = 8) -> torch.Tensor:
    _check_chunk(chunk)
    if tensor.ndim < 2:
        raise ValueError(f"spatial {kind} expects a sequence and a feature dimension")
    sequence = tensor.shape[-2]
    closed = (sequence // chunk) * chunk
    if closed == 0:
        return tensor
    prefix = tensor[..., :closed, :]
    chunks = prefix.reshape(*prefix.shape[:-2], closed // chunk, chunk, prefix.shape[-1])
    mean = chunks.mean(dim=-2, keepdim=True)
    if kind == "pool":
        filled = mean.expand_as(chunks)
    elif kind == "top1":
        filled = _keep_token(chunks, mean)
    elif kind == "feature":
        filled = _keep_features(chunks, mean)
    elif kind == "tile":
        filled = _keep_tiles(chunks, mean, tile)
    else:
        raise ValueError(f"unknown spatial kind {kind!r}")
    out = tensor.clone()
    out[..., :closed, :] = filled.reshape_as(prefix)
    return out


def _keep_token(chunks: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    distance = (chunks - mean).norm(dim=-1)
    winner = distance.argmax(dim=-1)
    index = winner.unsqueeze(-1).unsqueeze(-1).expand(*winner.shape, 1, chunks.shape[-1])
    out = mean.expand_as(chunks).clone()
    out.scatter_(-2, index, chunks.gather(-2, index))
    return out


def _keep_features(chunks: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    winner = (chunks - mean).abs().argmax(dim=-2)
    mask = _position_mask(winner, chunks.shape[-2])
    out = mean.expand_as(chunks).clone()
    return torch.where(mask, chunks, out)


def _keep_tiles(chunks: torch.Tensor, mean: torch.Tensor, tile: int) -> torch.Tensor:
    if not isinstance(tile, int) or isinstance(tile, bool) or tile <= 0:
        raise ValueError("spatial tile must be a positive integer")
    features = chunks.shape[-1]
    if features % tile != 0:
        raise ValueError(
            f"spatial tile requires the last dimension to be divisible by tile={tile}, got {features}"
        )
    n_tiles = features // tile
    tiled = chunks.reshape(*chunks.shape[:-1], n_tiles, tile)
    mean_tiled = mean.reshape(*mean.shape[:-1], n_tiles, tile)
    winner = (tiled - mean_tiled).norm(dim=-1).argmax(dim=-2)
    mask = _position_mask(winner, chunks.shape[-2])
    out = mean_tiled.expand_as(tiled).clone()
    out = torch.where(mask.unsqueeze(-1), tiled, out)
    return out.reshape_as(chunks)


def _position_mask(winner: torch.Tensor, chunk: int) -> torch.Tensor:
    """True where the token index along the chunk matches ``winner``."""
    positions = torch.arange(chunk, device=winner.device)
    view = (1,) * (winner.ndim - 1) + (chunk, 1)
    return winner.unsqueeze(-2) == positions.view(view)


def _check_chunk(chunk: int) -> None:
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise ValueError("spatial chunk must be a positive integer")


METHODS["spatial_pool"] = apply_pool
METHODS["spatial_top1"] = apply_top1
METHODS["spatial_tile"] = apply_tile
METHODS["spatial_feature"] = apply_feature
