import argparse
import unittest

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashInferCuteDslPrimsArgs(unittest.TestCase):
    def test_cli_parses_sub_backend(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--attention-backend",
                "flashinfer",
                "--flashinfer-prefill-backend",
                "cute-dsl-prims",
                "--kv-cache-dtype",
                "fp8_e4m3",
            ]
        )
        self.assertEqual(args.flashinfer_prefill_backend, "cute-dsl-prims")

    def _args(self, **kwargs):
        defaults = dict(
            model_path="dummy",
            attention_backend="flashinfer",
            flashinfer_prefill_backend="cute-dsl-prims",
            kv_cache_dtype="fp8_e4m3",
        )
        defaults.update(kwargs)
        return ServerArgs(**defaults)

    def test_page_size_defaults_to_64(self):
        args = self._args()
        args._handle_page_size()
        self.assertEqual(resolved_view(args).page_size, 64)

    def test_explicit_supported_page_size_is_preserved(self):
        args = self._args(page_size=32)
        args._handle_page_size()
        self.assertEqual(resolved_view(args).page_size, 32)

    def test_default_flashinfer_behavior_is_unchanged(self):
        args = ServerArgs(model_path="dummy", attention_backend="flashinfer")
        args._handle_page_size()
        self.assertEqual(resolved_view(args).page_size, 1)

    def test_requires_flashinfer_top_level_backend(self):
        args = self._args(attention_backend="triton")
        with self.assertRaisesRegex(ValueError, "resolved prefill.*flashinfer"):
            args._handle_page_size()

    def test_requires_fp8_e4m3(self):
        args = self._args(kv_cache_dtype="bf16")
        with self.assertRaisesRegex(ValueError, "kv-cache-dtype fp8_e4m3"):
            args._handle_page_size()

    def test_rejects_unsupported_page_size(self):
        args = self._args(page_size=8)
        with self.assertRaisesRegex(ValueError, "page-size"):
            args._handle_page_size()

    def test_rejects_speculative_decoding(self):
        args = self._args(speculative_algorithm="NGRAM")
        with self.assertRaisesRegex(ValueError, "speculative verify/draft"):
            args._handle_page_size()

    def test_rejects_multi_item_scoring(self):
        args = self._args(enable_mis=True)
        with self.assertRaisesRegex(ValueError, "multi-item scoring"):
            args._handle_page_size()

    def test_rejects_deterministic_fixed_split(self):
        args = self._args(enable_deterministic_inference=True)
        with self.assertRaisesRegex(ValueError, "deterministic fixed-split"):
            args._handle_page_size()


if __name__ == "__main__":
    unittest.main()
