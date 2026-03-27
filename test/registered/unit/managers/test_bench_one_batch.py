from types import SimpleNamespace
from unittest.mock import patch

from sglang.bench_one_batch import _maybe_prepare_mlp_sync_batch


def test_maybe_prepare_mlp_sync_batch_passes_dp_attention_parallelism():
    batch = object()
    model_runner = SimpleNamespace(
        tp_rank=5,
        tp_size=8,
        attn_cp_size=2,
        tp_group=object(),
        server_args=SimpleNamespace(
            enable_dp_attention=True,
            dp_size=2,
            disable_cuda_graph=False,
            disable_overlap_schedule=False,
            moe_dense_tp_size=None,
            enable_dp_lm_head=False,
        ),
    )

    with patch("sglang.bench_one_batch.prepare_mlp_sync_batch_raw") as prepare_mock:
        _maybe_prepare_mlp_sync_batch(batch, model_runner)

    prepare_mock.assert_called_once()
    _, kwargs = prepare_mock.call_args
    assert kwargs["attn_tp_size"] == 2
    assert kwargs["attn_cp_size"] == 2
