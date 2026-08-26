"""Tests for TP2 rank-shard merging (issue #37).

Run: python3 kvbridge/test_tp2_merge.py   (no pytest, runs anywhere).

Geometry: 27B NVFP4 TP2 — 4 KV heads total, 2 per rank. Attention dump
[T, 2, 512] per rank (K|V concat, 256 each). Recurrent tail per linear
layer: R (conv) and S (SSM) rows with heads split across ranks.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from tp2_merge import merge_attn_shards, merge_recurrent_shards, merged_dump_dir  # noqa: E402

T, H_RANK, D2 = 10, 2, 512          # tokens, heads per rank, K|V concat
LAYER = "language_model.model.layers.11.self_attn.attn.safetensors"


def _shard(seed):
    torch.manual_seed(seed)
    return torch.randn(T, H_RANK, D2).bfloat16()


def test_merge_attn_shapes():
    merged = merge_attn_shards([_shard(1), _shard(2)])
    assert merged.shape == (T, 2 * H_RANK, D2), merged.shape


def test_merge_attn_rank_order_preserved():
    a, b = _shard(1), _shard(2)
    merged = merge_attn_shards([a, b])
    # rank 0's heads first, then rank 1's — vLLM TP splits kv heads contiguously
    assert torch.equal(merged[:, :H_RANK], a)
    assert torch.equal(merged[:, H_RANK:], b)


def test_merge_recurrent_heads():
    # issue #39: conv is [T, C/tp] (channel-sharded); ssm is [H/tp, V, K]
    r0, r1 = torch.randn(3, 96), torch.randn(3, 96)
    s0, s1 = torch.randn(2, 128, 64), torch.randn(2, 128, 64)
    R, S = merge_recurrent_shards((r0, s0), (r1, s1))
    assert R.shape == (3, 192) and S.shape == (4, 128, 64), (R.shape, S.shape)
    assert torch.equal(R[:, :96], r0) and torch.equal(R[:, 96:], r1)
    assert torch.equal(S[2:], s1)


def test_merged_dump_dir_layout():
    import tempfile
    with tempfile.TemporaryDirectory() as d0, tempfile.TemporaryDirectory() as d1:
        import safetensors.torch as st
        st.save_file({"kv_cache": _shard(1)}, f"{d0}/{LAYER}")
        st.save_file({"kv_cache": _shard(2)}, f"{d1}/{LAYER}")
        out = merged_dump_dir(d0, [d1])
        files = [f for f in os.listdir(out) if f.endswith(".safetensors")]
        assert files == [LAYER], files
        got = st.load_file(f"{out}/{LAYER}")["kv_cache"]
        assert got.shape == (T, 2 * H_RANK, D2), got.shape


def test_single_shard_passthrough():
    # solo mode (world_size 1): merge with one shard must be identity
    a = _shard(1)
    assert torch.equal(merge_attn_shards([a]), a)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests green")
