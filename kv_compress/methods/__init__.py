"""Registry of KV rewrite callables used by the pipeline.

Each method is apply(tensor, *, layer_idx, target, **kwargs) -> tensor and
registers itself on METHODS. get_method looks up a JSON "method" name.
"""

from collections.abc import Callable
from typing import Any

import torch

Method = Callable[..., torch.Tensor]

METHODS: dict[str, Method] = {}


def get_method(name: str) -> Method:
    try:
        return METHODS[name]
    except KeyError as error:
        known = ", ".join(sorted(METHODS)) or "(none registered)"
        raise ValueError(f"Unknown KV compress method {name!r}. Known methods: {known}") from error


from kv_compress.methods import sparsify_nm as _sparsify_nm  # noqa: E402,F401
