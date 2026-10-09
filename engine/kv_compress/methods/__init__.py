"""Name → apply() table used by the pipeline.

Each algorithm under compression_topics registers itself on METHODS when this
module is imported. get_method looks up a JSON "method" name.
"""

from collections.abc import Callable

import torch

Method = Callable[..., torch.Tensor]

METHODS: dict[str, Method] = {}
# Optional method-owned callbacks for groups completed across cache updates.
AFTER_APPEND: dict[str, Callable] = {}
OPTION_SIGNATURES: dict[str, Callable] = {}


def validate_options(name, options):
    """Reject misspelled method options, even for kernels accepting **context."""
    import inspect
    parameters = inspect.signature(OPTION_SIGNATURES.get(name, get_method(name))).parameters
    allowed = {key for key, value in parameters.items()
               if value.kind not in (value.VAR_KEYWORD, value.VAR_POSITIONAL)}
    allowed -= {"tensor", "layer_idx", "target", "seq_start", "rope_tables", "token_weights"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown options: {sorted(unknown)}")



def get_method(name: str) -> Method:
    try:
        return METHODS[name]
    except KeyError as error:
        known = ", ".join(sorted(METHODS)) or "(none registered)"
        raise ValueError(f"Unknown KV compress method {name!r}. Known methods: {known}") from error


import compression_topics as _compression_topics  # noqa: E402,F401
