"""Name → apply() table used by the pipeline.

Each algorithm under compression_topics registers itself on METHODS when this
module is imported. get_method looks up a JSON "method" name.
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


import compression_topics as _compression_topics  # noqa: E402,F401
