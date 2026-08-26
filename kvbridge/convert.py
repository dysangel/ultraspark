"""vLLM KV dump → canonical representation for llama.cpp state synthesis.

Issue #8. Input layout (established empirically, see issue #2 comments):
    [tokens, kv_heads, K(0:head_dim) | V(head_dim:)]  bf16, post-RoPE,
    one safetensors file per FULL-ATTENTION layer (hybrid: 8 of 32).

Canonical output: dict[layer_idx] -> [tokens, kv_heads, 2(K,V), head_dim]
in the requested dtype (llama.cpp GGUF K-quants use f16 K/V).
Position ids are implicit (0..n-1) — post-RoPE means no rope math here,
just token accounting when coverage is partial (caller's job, v0).
"""
import glob
import os
import re

import torch


def load_dump(dump_dir: str) -> dict[int, torch.Tensor]:
    """Load per-layer safetensors, keyed by layer index, sorted by name."""
    import safetensors.torch as st

    layers: dict[int, torch.Tensor] = {}
    for path in sorted(glob.glob(os.path.join(dump_dir, "*.safetensors"))):
        m = re.search(r"layers\.(\d+)\.", os.path.basename(path))
        if not m:
            raise ValueError(f"unparseable layer name: {path}")
        layers[int(m.group(1))] = st.load_file(path)["kv_cache"]
    return layers


def to_canonical(
    dump: dict[int, torch.Tensor],
    n_layers: int,
    dtype: torch.dtype = torch.float16,
) -> dict[int, torch.Tensor]:
    """[T, H, 2*D] -> [T, H, 2, D] with explicit K/V axis, cast to dtype."""
    out = {}
    for idx, kv in dump.items():
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"layer {idx} outside model ({n_layers} layers)")
        t, h, two_d = kv.shape
        if two_d % 2:
            raise ValueError(f"layer {idx}: odd last dim {two_d}, expected K|V concat")
        d = two_d // 2
        out[idx] = kv.reshape(t, h, 2, d).to(dtype).contiguous()
    return out
