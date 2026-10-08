"""Sparse signed residual reconstruction shared by spatial methods."""
import torch


def largest_residual(delta: torch.Tensor, count: int) -> torch.Tensor:
    """The ``count`` largest-|d| entries of each vector, zeros elsewhere."""
    if count <= 0:
        return torch.zeros_like(delta)
    if count >= delta.shape[-1]:
        return delta
    index = delta.abs().topk(count, dim=-1).indices
    return torch.zeros_like(delta).scatter(-1, index, delta.gather(-1, index))

