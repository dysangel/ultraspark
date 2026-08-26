"""Tests for the vLLM→llama.cpp KV converter core (issue #8).

Run: python3 kvbridge/test_convert.py  (no pytest dependency on purpose —
must run anywhere, including the Spark).
"""
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(__file__))
from convert import load_dump, to_canonical  # noqa: E402

N_TOK, N_HEADS, HEAD_DIM = 32, 4, 256


def _fake_dump(tmp="/tmp/uspk_test_dump"):
    os.makedirs(tmp, exist_ok=True)
    import safetensors.torch as st
    torch.manual_seed(7)
    # [tokens, kv_heads, K|V] post-RoPE bf16, layer-name style from vLLM
    kv = torch.randn(N_TOK, N_HEADS, 2 * HEAD_DIM).bfloat16()
    st.save_file({"kv_cache": kv}, f"{tmp}/language_model.model.layers.3.self_attn.attn.safetensors")
    st.save_file({"kv_cache": kv}, f"{tmp}/language_model.model.layers.31.self_attn.attn.safetensors")
    return tmp


def test_load_dump_orders_layers():
    d = load_dump(_fake_dump())
    assert sorted(d.keys()) == [3, 31], d.keys()


def test_canonical_shape_and_split():
    d = load_dump(_fake_dump())
    layers = to_canonical(d, n_layers=32)
    # only full-attention layers carry KV
    assert layers[3].shape == (N_TOK, N_HEADS, 2, HEAD_DIM), layers[3].shape
    assert 0 not in layers and 31 in layers


def test_k_split_matches_source():
    tmp = _fake_dump()
    raw = torch.cat(
        [x.unsqueeze(0) for x in []]) if False else None  # placeholder guard
    import safetensors.torch as st
    src = st.load_file(f"{tmp}/language_model.model.layers.3.self_attn.attn.safetensors")["kv_cache"]
    layers = to_canonical(load_dump(tmp), n_layers=32)
    got = layers[3]
    assert torch.equal(got[:, :, 0, :], src[:, :, :HEAD_DIM])   # K
    assert torch.equal(got[:, :, 1, :], src[:, :, HEAD_DIM:])   # V


def test_dtype_f16():
    layers = to_canonical(load_dump(_fake_dump()), n_layers=32, dtype=torch.float16)
    assert layers[3].dtype == torch.float16


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests green")
