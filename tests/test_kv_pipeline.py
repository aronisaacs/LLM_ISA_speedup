import unittest

import torch

from kv_compress.cache import patch_cache_update
from kv_compress.install import install
from kv_compress.pipeline import compress_kv
from kv_compress.spec import parse_kv_spec


class ParseKvSpecTests(unittest.TestCase):
    def test_omitted_and_empty_are_identity(self):
        for value in (None, {}, {"pipeline": []}):
            spec = parse_kv_spec(value)
            self.assertTrue(spec.is_identity())
            self.assertEqual(spec.to_dict(), {"pipeline": []})

    def test_unknown_kv_key_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_kv_spec({"backend": "ours", "pipeline": []})

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_kv_spec({"pipeline": [{"method": "sparsify_nm", "k_layers": "all"}]})

    def test_layer_lists_and_all(self):
        from kv_compress import methods as methods_module

        def _identity(tensor, **_kwargs):
            return tensor

        methods_module.METHODS["identity"] = _identity
        try:
            spec = parse_kv_spec(
                {
                    "pipeline": [
                        {"method": "identity", "k_layers": "all", "v_layers": [0, 2]}
                    ]
                }
            )
            step = spec.pipeline[0]
            self.assertEqual(step.k_layers, "all")
            self.assertEqual(step.v_layers, frozenset({0, 2}))
            self.assertEqual(
                spec.to_dict(),
                {
                    "pipeline": [
                        {"method": "identity", "k_layers": "all", "v_layers": [0, 2]}
                    ]
                },
            )
        finally:
            methods_module.METHODS.pop("identity", None)

    def test_negative_layer_is_rejected(self):
        from kv_compress import methods as methods_module

        def _identity(tensor, **_kwargs):
            return tensor

        methods_module.METHODS["identity"] = _identity
        try:
            with self.assertRaises(ValueError):
                parse_kv_spec({"pipeline": [{"method": "identity", "k_layers": [-1]}]})
        finally:
            methods_module.METHODS.pop("identity", None)


class PipelineAndInstallTests(unittest.TestCase):
    def test_identity_pipeline_leaves_tensors_unchanged(self):
        spec = parse_kv_spec({"pipeline": []})
        key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        value = key + 1
        out_key, out_value = compress_kv(key, value, layer_idx=0, spec=spec)
        self.assertIs(out_key, key)
        self.assertIs(out_value, value)

    def test_install_identity_does_not_patch_cache(self):
        from transformers.cache_utils import Cache

        original = Cache.update
        uninstall = install(None, parse_kv_spec({"pipeline": []}))
        try:
            self.assertIs(Cache.update, original)
        finally:
            uninstall()
        self.assertIs(Cache.update, original)

    def test_cache_update_identity_patch_round_trip(self):
        from transformers.cache_utils import Cache, DynamicCache

        spec = parse_kv_spec({"pipeline": []})
        original = Cache.update
        uninstall = patch_cache_update(spec)
        try:
            self.assertIsNot(Cache.update, original)
            cache = DynamicCache()
            key = torch.randn(1, 2, 3, 4)
            value = torch.randn(1, 2, 3, 4)
            out_key, out_value = cache.update(key, value, 0)
            self.assertTrue(torch.equal(out_key, key))
            self.assertTrue(torch.equal(out_value, value))
        finally:
            uninstall()
        self.assertIs(Cache.update, original)


if __name__ == "__main__":
    unittest.main()
