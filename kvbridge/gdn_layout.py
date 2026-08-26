"""GDN state-layout fixes for the 27B (issue #39).

Two convention mismatches between vLLM and llama.cpp, both proven by
forensics on live dumps (see issue #39 comments):

1. v-head ordering: vLLM interleaves the GQA value heads (k-head = h //
   (H_v/H_k)); llama.cpp orders them j = (h % ratio)*H_k + h // ratio.
   The permutation applies to BOTH the ssm state's head axis and the conv
   state's v channel segment.
2. ssm within-head layout is IDENTITY: both store [H_v, V, K] with K
   fastest in memory. No transpose.

Conv channel segments [q | k | v] = [H_k*d_k | H_k*d_k | H_v*d_v] are in
the same order in both implementations (verified 0.999 on q/k segments);
only the v segment's internal head order differs.
"""
import torch


def vhead_permutation(n_v_heads: int, n_k_heads: int) -> list[int]:
    """llama.cpp v-head index for each vLLM v-head index.

    vLLM: head h uses k-head h // ratio (interleaved).
    llama.cpp: head j = (h % ratio)*n_k_heads + h // ratio.
    Returns perm where perm[h] = j.
    """
    ratio, rem = divmod(n_v_heads, n_k_heads)
    if rem or ratio < 1:
        raise ValueError(
            f"n_v_heads ({n_v_heads}) must be a positive multiple of "
            f"n_k_heads ({n_k_heads})")
    return [(h % ratio) * n_k_heads + h // ratio for h in range(n_v_heads)]


def permute_conv_channels(
    conv: torch.Tensor, n_k_heads: int, d_k: int, n_v_heads: int, d_v: int
) -> torch.Tensor:
    """Reorder the v segment of a conv state [T, C] to llama.cpp head order.

    Channels: [q (n_k*d_k) | k (n_k*d_k) | v (n_v*d_v)]. q and k segments
    are already identical in both implementations — only v's heads move.
    """
    qk = 2 * n_k_heads * d_k
    v_len = n_v_heads * d_v
    if conv.shape[-1] != qk + v_len:
        raise ValueError(
            f"conv channels {conv.shape[-1]} != {qk + v_len} expected from "
            f"heads/dims — refusing to permute blindly (issue #23 spirit)")
    out = conv.clone()
    flat = conv[..., qk:].reshape(*conv.shape[:-1], n_v_heads, d_v)
    perm = vhead_permutation(n_v_heads, n_k_heads)
    reordered = torch.empty_like(flat)
    reordered[..., perm, :] = flat
    out[..., qk:] = reordered.reshape(*conv.shape[:-1], v_len)
    return out


def permute_ssm_heads(
    ssm: torch.Tensor, n_k_heads: int, n_v_heads: int
) -> torch.Tensor:
    """Reorder the head axis of an ssm state [H_v, V, K] to llama.cpp order.

    Within-head layout is untouched (identity, K fastest in both).
    """
    if ssm.shape[0] != n_v_heads:
        raise ValueError(
            f"ssm heads {ssm.shape[0]} != expected {n_v_heads}")
    perm = vhead_permutation(n_v_heads, n_k_heads)
    out = torch.empty_like(ssm)
    out[perm] = ssm
    return out


def merge_rank_states(recs: list[dict]) -> dict:
    """Merge per-rank rec_states {layer: {conv, ssm}} into global order.

    conv: per-segment concat as above; ssm: head-axis concat, rank order.
    Returns {layer: {conv, ssm}} in vLLM global head order (finalize_states
    applies the llama.cpp permutation afterwards).
    """
    layers = set(recs[0])
    for r in recs[1:]:
        if set(r) != layers:
            raise ValueError(
                f"rank shard layer mismatch: {sorted(set(r) ^ layers)}")
    out = {}
    for L in sorted(layers):
        ssm = torch.cat([r[L]["ssm"] for r in recs], dim=0)
        n_v_r, d_v, d_k = recs[0][L]["ssm"].shape
        convs = [r[L]["conv"] for r in recs]
        c_r = convs[0].shape[-1]
        qk_r = c_r - n_v_r * d_v
        n_k_r, rem = divmod(qk_r, 2 * d_k)
        if rem or any(c.shape[-1] != c_r for c in convs):
            raise ValueError(
                f"layer {L}: implausible rank conv geometry C_r={c_r} "
                f"H_v_r={n_v_r} d_k={d_k} rem={rem}")
        qk = n_k_r * d_k
        conv = torch.cat(
            [torch.cat([c[..., :qk] for c in convs], dim=-1),      # q
             torch.cat([c[..., qk:2 * qk] for c in convs], dim=-1),  # k
             torch.cat([c[..., 2 * qk:] for c in convs], dim=-1)],  # v
            dim=-1)
        out[L] = {"conv": conv, "ssm": ssm}
    return out


def finalize_states(rec_states: dict) -> dict:
    """Apply the llama.cpp head ordering to merged (global) rec states.

    Called in wait_for_save AFTER solo/TP2 merge — geometry is derived from
    the (already rank-concatenated) tensors and must be the full model's:
    ssm [H_v, V, K], conv [T, C] with C = 2*H_k*d_k + H_v*d_v. Fails loud
    on implausible geometry rather than permuting blindly.
    """
    n_v, d_v, d_k = next(iter(rec_states.values()))["ssm"].shape
    conv_c = next(iter(rec_states.values()))["conv"].shape[-1]
    n_k, rem = divmod(conv_c - n_v * d_v, 2 * d_k)
    if rem or n_k < 1 or n_v % n_k:
        raise ValueError(
            f"implausible GDN geometry: conv C={conv_c} ssm H_v={n_v} "
            f"V={d_v} K={d_k} -> H_k={n_k} rem={rem}")
    return {
        L: {"conv": permute_conv_channels(v["conv"], n_k, d_k, n_v, d_v),
            "ssm": permute_ssm_heads(v["ssm"], n_k, n_v)}
        for L, v in rec_states.items()
    }
    """Apply the llama.cpp head ordering to merged (global) rec states.

    Called in wait_for_save AFTER solo/TP2 merge — geometry is derived from
    the (already rank-concatenated) tensors and must be the full model's:
    ssm [H_v, V, K], conv [T, C] with C = 2*H_k*d_k + H_v*d_v. Fails loud
    on implausible geometry rather than permuting blindly.
    """
    n_v, d_v, d_k = next(iter(rec_states.values()))["ssm"].shape
    conv_c = next(iter(rec_states.values()))["conv"].shape[-1]
    n_k, rem = divmod(conv_c - n_v * d_v, 2 * d_k)
    if rem or n_k < 1 or n_v % n_k:
        raise ValueError(
            f"implausible GDN geometry: conv C={conv_c} ssm H_v={n_v} "
            f"V={d_v} K={d_k} -> H_k={n_k} rem={rem}")
    return {
        L: {"conv": permute_conv_channels(v["conv"], n_k, d_k, n_v, d_v),
            "ssm": permute_ssm_heads(v["ssm"], n_k, n_v)}
        for L, v in rec_states.items()
    }
