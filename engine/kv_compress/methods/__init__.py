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


from engine.kv_compress.methods import checksparse_l1 as _checksparse_l1  # noqa: E402,F401
from engine.kv_compress.methods import dynamic_precision as _dynamic_precision  # noqa: E402,F401
from engine.kv_compress.methods import qjl as _qjl  # noqa: E402,F401
from engine.kv_compress.methods import sparsify_nm as _sparsify_nm  # noqa: E402,F401
from engine.kv_compress.methods import spatial as _spatial  # noqa: E402,F401
from engine.kv_compress.methods import vector_compress as _vector_compress  # noqa: E402,F401
