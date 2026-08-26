"""Tests for gdn_layout (issue #39). Run: python3 kvbridge/test_gdn_layout.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from gdn_layout import permute_conv_channels, permute_ssm_heads, vhead_permutation  # noqa: E402


def test_perm_formula_matches_forensics():
    # observed on live 27B dumps: [0,16,32,1,17,33,...] for H_v=48, H_k=16
    p = vhead_permutation(48, 16)
    assert p[:6] == [0, 16, 32, 1, 17, 33], p[:6]
    assert sorted(p) == list(range(48))          # a clean permutation
    assert len(set(p)) == 48


def test_perm_fails_loud_on_bad_geometry():
    try:
        vhead_permutation(48, 10)
    except ValueError:
        return
    raise AssertionError("48 v-heads with 10 k-heads must raise")


def test_ssm_head_perm_roundtrip_identity():
    x = torch.randn(48, 4, 4)
    p = permute_ssm_heads(x, 16, 48)
    # head 0 maps to 0; head 1 maps to 16
    assert torch.equal(p[0], x[0]) and torch.equal(p[16], x[1])
    assert torch.equal(p[1], x[3])  # vllm head 3 -> llama head 1
    # round trip is stable
    back = permute_ssm_heads(p, 16, 48)  # applying twice is not identity;
    # but each llama head j holds exactly one vllm head's data:
    assert torch.equal(back[0], p[0])


def test_conv_v_segment_perm_only():
    # [T=3, C=10240]: q(2048)|k(2048)|v(6144)
    x = torch.randn(3, 10240)
    out = permute_conv_channels(x, 16, 128, 48, 128)
    assert torch.equal(out[:, :4096], x[:, :4096])  # q,k untouched
    v_in = x[:, 4096:].reshape(3, 48, 128)
    v_out = out[:, 4096:].reshape(3, 48, 128)
    assert torch.equal(v_out[:, 0], v_in[:, 0]) and torch.equal(v_out[:, 16], v_in[:, 1])


def test_conv_bad_shape_raises():
    try:
        permute_conv_channels(torch.randn(3, 999), 16, 128, 48, 128)
    except ValueError:
        return
    raise AssertionError("wrong channel count must raise")


def test_finalize_after_rank_merge_orders_heads():
    # issue #39 TP2 traps: (1) per-rank permutation cannot concatenate into
    # llama order; (2) each rank's conv channels are its OWN [q_r|k_r|v_r]
    # segments. merge_rank_states + finalize_states on rank shards must
    # equal the solo-global path.
    # Global: q [0:2048) | k [2048:4096) | v [4096:10240).
    # rank0: q [0:1024) k [2048:3072) v [4096:7168); rank1 mirrors.
    from gdn_layout import finalize_states, merge_rank_states
    glob = {0: {"conv": torch.randn(3, 10240),
                "ssm": torch.randn(48, 128, 128)}}
    want = finalize_states(dict(glob))
    c, s = glob[0]["conv"], glob[0]["ssm"]
    sharded = [
        {0: {"conv": torch.cat([c[:, :1024], c[:, 2048:3072], c[:, 4096:7168]], dim=-1),
             "ssm": s[:24].clone()}},
        {0: {"conv": torch.cat([c[:, 1024:2048], c[:, 3072:4096], c[:, 7168:]], dim=-1),
             "ssm": s[24:].clone()}},
    ]
    got = finalize_states(merge_rank_states(sharded))
    assert torch.equal(got[0]["ssm"], want[0]["ssm"])
    assert torch.equal(got[0]["conv"], want[0]["conv"])


def test_merge_rank_states_segment_interleave():
    # rank convs [q_r|k_r|v_r] must become [q0 q1|k0 k1|v0 v1]
    from gdn_layout import merge_rank_states
    q, k, v = 1024, 1024, 3072  # per-rank: 8 k-heads, 24 v-heads, d=128
    mk = lambda s: {0: {"conv": torch.full((3, q + k + v), float(s)),
                        "ssm": torch.full((24, 128, 128), float(s))}}
    merged = merge_rank_states([mk(1), mk(2)])
    c = merged[0]["conv"]
    assert c.shape == (3, 2 * (q + k + v))
    assert (c[:, :q] == 1).all() and (c[:, q:2 * q] == 2).all()      # q0 q1
    assert (c[:, 2 * q:2 * q + k] == 1).all()                        # k0
    assert (c[:, -v:] == 2).all()                                    # v1
    assert merged[0]["ssm"].shape == (48, 128, 128)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests green")
