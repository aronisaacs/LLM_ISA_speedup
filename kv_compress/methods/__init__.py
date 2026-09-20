"""Named KV rewrite functions. Step 1 has none; later methods register here."""

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
