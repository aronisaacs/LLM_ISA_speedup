"""Runtime KV-cache compression, installed onto Transformers ``Cache.update``."""

from kv_compress.install import install
from kv_compress.spec import parse_kv_spec

__all__ = ["install", "parse_kv_spec"]
