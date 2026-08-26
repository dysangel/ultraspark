"""Merge rank-sharded KV dumps from vLLM TP2 into a solo-equivalent dump (issue #37).

Under TP2 each rank's connector dumps its own shard: attention layers as
[T, H/tp, 2*D] (K|V concat) and the recurrent GDN state with heads split the
same way. vLLM splits KV heads contiguously by rank, so a plain concat along
the head axis in rank order reconstructs the unsharded layout the synthesizer
expects. Solo (single dump dir) is a passthrough — callers never merge when
world_size == 1.
"""
import os
import shutil

import torch


def merge_attn_shards(shards: list[torch.Tensor]) -> torch.Tensor:
    """Concat rank shards along the head axis: [T, H/tp, 2*D] each -> [T, H, 2*D]."""
    if len(shards) == 1:
        return shards[0]
    t, h, d2 = shards[0].shape
    for s in shards[1:]:
        assert s.shape == (t, h, d2), f"shard shape mismatch {s.shape} vs {(t, h, d2)}"
    return torch.cat(shards, dim=1)


def merge_recurrent_shards(
    rank0: tuple[torch.Tensor, torch.Tensor],
    *others: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concat per-rank (R conv, S ssm) shards in rank order.

    conv is [T, C/tp] — TP shards the CHANNEL axis, so concat dim=-1.
    ssm is [H_v/tp, V, K] — TP shards the head axis, so concat dim=0.
    (issue #39: the llama.cpp v-head permutation is applied later, in
    gdn_layout.finalize_states, because llama-order heads interleave
    across ranks.)
    """
    rs = [rank0[0]] + [o[0] for o in others]
    ss = [rank0[1]] + [o[1] for o in others]
    return torch.cat(rs, dim=-1), torch.cat(ss, dim=0)


def merged_dump_dir(local_dir: str, peer_dirs: list[str]) -> str:
    """Materialize a merged dump: copy local dir, then overwrite each attention
    safetensors with the head-concat of local + peers (same file names).
    Non-attention files (metadata, recurrent states) are kept from local;
    recurrent merging is handled by the connector via merge_recurrent_shards.
    """
    if not peer_dirs:
        return local_dir
    import safetensors.torch as st

    out = local_dir.rstrip("/") + ".merged"
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(local_dir, out)
    for name in sorted(os.listdir(local_dir)):
        if not name.endswith(".safetensors"):
            continue
        shards = [st.load_file(os.path.join(d, name))["kv_cache"] for d in [local_dir, *peer_dirs]]
        st.save_file({"kv_cache": merge_attn_shards(shards)}, os.path.join(out, name))
    return out
