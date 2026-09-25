"""KV-cache compression package used by multi_run.py.

Parse a JSON kv spec (parse_kv_spec) and attach it to a loaded HF model
(install) by wrapping Transformers Cache.update. Methods live in kv_compress.methods.
"""

from engine.kv_compress.install import install
from engine.kv_compress.spec import parse_kv_spec

__all__ = ["install", "parse_kv_spec"]
