"""Manual SM120-only page-metadata coverage (no SM120 CI runner exists)."""

import unittest

import torch

from sglang.kernels.ops.attention.utils import (
    create_flashinfer_page_indices_triton,
)
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterPrefill,
)


class _RecordingWrapper:
    def begin_forward(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class TestFlashInferCuteDslPrimsMetadata(unittest.TestCase):
    def test_page_metadata_boundaries(self):
        page_size = 64
        lengths = torch.tensor(
            [0, 1, 63, 64, 65, 129], dtype=torch.int32, device="cuda"
        )
        counts = (lengths + page_size - 1) // page_size
        indptr = torch.zeros(7, dtype=torch.int32, device="cuda")
        indptr[1:] = torch.cumsum(counts, 0)
        req_indices = torch.arange(6, dtype=torch.int32, device="cuda")
        req_to_token = torch.empty((6, 256), dtype=torch.int32, device="cuda")
        for batch in range(6):
            start = (batch + 1) * page_size
            req_to_token[batch] = torch.arange(
                start, start + 256, dtype=torch.int32, device="cuda"
            )
        indices = torch.empty(16, dtype=torch.int32, device="cuda")
        create_flashinfer_page_indices_triton[(6,)](
            req_to_token,
            req_indices,
            lengths,
            indptr,
            None,
            indices,
            req_to_token.shape[1],
            PAGE_SIZE=page_size,
        )
        torch.cuda.synchronize()
        used = int(indptr[-1])
        self.assertEqual(indptr.cpu().tolist(), [0, 0, 1, 2, 3, 5, 8])
        self.assertEqual(indices[:used].cpu().tolist(), [2, 3, 4, 5, 6, 6, 7, 8])
        last_page_len = torch.where(
            lengths > 0, (lengths - 1) % page_size + 1, page_size
        )
        self.assertEqual(last_page_len.cpu().tolist(), [64, 1, 63, 64, 1, 1])

    def test_updater_passes_page_indices_to_wrapper(self):
        page_size = 64
        lengths = torch.tensor([65, 129], dtype=torch.int32, device="cuda")
        req_to_token = torch.stack(
            [
                torch.arange(64, 320, dtype=torch.int32, device="cuda"),
                torch.arange(384, 640, dtype=torch.int32, device="cuda"),
            ]
        )
        backend = type(
            "Backend",
            (),
            {
                "use_prims_prefill": True,
                "page_size": page_size,
                "dq_paged_kernel_lens": None,
            },
        )()
        updater = FlashInferIndicesUpdaterPrefill.__new__(
            FlashInferIndicesUpdaterPrefill
        )
        updater.attn_backend = backend
        updater.req_to_token = req_to_token
        updater.kv_last_page_len = torch.empty(2, dtype=torch.int32, device="cuda")
        updater.num_qo_heads = 4
        updater.num_kv_heads = 2
        updater.head_dim = 32
        updater.q_data_type = torch.float8_e4m3fn
        updater.data_type = torch.float8_e4m3fn
        updater._swa_kv_pool = None

        wrapper = _RecordingWrapper()
        updater.call_begin_forward(
            _RecordingWrapper(),
            wrapper,
            torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            lengths,
            int(lengths.sum()),
            lengths,
            torch.zeros(2, dtype=torch.int32, device="cuda"),
            torch.zeros(2, dtype=torch.int32, device="cuda"),
            torch.zeros(3, dtype=torch.int32, device="cuda"),
            torch.zeros(3, dtype=torch.int32, device="cuda"),
            False,
            None,
            seq_lens_cpu=torch.tensor([65, 129], dtype=torch.int32),
            prefix_lens_cpu=[0, 0],
        )
        torch.cuda.synchronize()
        self.assertEqual(wrapper.args[1].cpu().tolist(), [0, 2, 5])
        self.assertEqual(wrapper.args[2][:5].cpu().tolist(), [1, 2, 6, 7, 8])
        self.assertEqual(wrapper.args[3].cpu().tolist(), [1, 1])
        self.assertEqual(wrapper.args[7], page_size)


if __name__ == "__main__":
    unittest.main()
