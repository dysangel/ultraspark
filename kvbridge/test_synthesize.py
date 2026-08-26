"""Tests for llama.cpp state-blob synthesis (issue #8 part 2). TDD: run first.

Run: python3 kvbridge/test_synthesize.py
"""
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from synthesize import (
    MAGIC,
    blob_layer_order,
    canonical_from_dump_pair,
    synthesize,
)  # noqa: E402

N_TOK, N_HEADS, HEAD_DIM = 16, 4, 256
ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]


def _pair_dump(tmp="/tmp/uspk_test_pairdump"):
    """Fake dump with the empirically-verified pair layout (issue #8 comments):
    file L%8==3: [K(L) | K(L+4)], file L%8==7: [V(L-4) | V(L)].
    Values encode (layer, "K"/"V") so any mixup is loud.
    """
    import safetensors.torch as st

    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    for la in [3, 11, 19, 27]:
        lb = la + 4
        kfile = torch.empty(N_TOK, N_HEADS, 2 * HEAD_DIM)
        vfile = torch.empty(N_TOK, N_HEADS, 2 * HEAD_DIM)
        for h in range(N_HEADS):
            # small ints: exact in bf16 and f16; V is negative so K/V mixups are loud
            kfile[:, h, :HEAD_DIM] = la * 4 + h
            kfile[:, h, HEAD_DIM:] = lb * 4 + h
            vfile[:, h, :HEAD_DIM] = -(la * 4 + h)
            vfile[:, h, HEAD_DIM:] = -(lb * 4 + h)
        st.save_file({"kv_cache": kfile.bfloat16()},
                     f"{tmp}/language_model.model.layers.{la}.self_attn.attn.safetensors")
        st.save_file({"kv_cache": vfile.bfloat16()},
                     f"{tmp}/language_model.model.layers.{lb}.self_attn.attn.safetensors")
    return tmp


def test_blob_layer_order():
    assert blob_layer_order(ATTN_LAYERS) == [3, 11, 19, 27, 7, 15, 23, 31]


def test_canonical_pair_layout():
    d = canonical_from_dump_pair(_pair_dump(), attn_layers=ATTN_LAYERS)
    assert sorted(d.keys()) == ATTN_LAYERS
    for L in ATTN_LAYERS:
        t = d[L]
        assert t.shape == (N_TOK, N_HEADS, 2, HEAD_DIM), t.shape
        assert t.dtype == torch.float16
        for h in range(N_HEADS):
            assert torch.all(t[:, h, 0, :] == L * 4 + h)
            assert torch.all(t[:, h, 1, :] == -(L * 4 + h))


def _synth_bytes(**kw):
    d = canonical_from_dump_pair(_pair_dump(), attn_layers=ATTN_LAYERS)
    return synthesize(d, attn_layers=ATTN_LAYERS, **kw)


def test_blob_header_and_meta():
    b = _synth_bytes()
    o = 0
    magic, seq = struct.unpack_from("<Ii", b, o); o += 8
    assert magic == MAGIC and seq == 0
    n_stream = struct.unpack_from("<I", b, o)[0]; o += 4
    assert n_stream == 1
    cc = struct.unpack_from("<I", b, o)[0]; o += 4
    assert cc == N_TOK
    for i in range(cc):
        pos, n_sid, x, y = struct.unpack_from("<iIii", b, o); o += 16
        sid = struct.unpack_from("<i", b, o)[0]; o += 4
        assert (pos, n_sid, sid) == (i, 1, 0)
        assert (x, y) == (i, i)  # mrope ext mirrors pos, as in the real blob
    v_trans, n_layer = struct.unpack_from("<II", b, o); o += 8
    assert v_trans == 0 and n_layer == 8
    assert o == 8 + 4 + 4 + N_TOK * 20 + 8


def test_blob_layer_blocks():
    b = _synth_bytes()
    o = 8 + 4 + 4 + N_TOK * 20 + 8
    order = [3, 11, 19, 27, 7, 15, 23, 31]
    for L in order:
        kt, kr = struct.unpack_from("<iQ", b, o); o += 12
        assert kt == 1 and kr == 2048, (kt, kr)
        k = np.frombuffer(b, dtype="<f2", count=N_TOK * 1024, offset=o).reshape(N_TOK, 1024)
        o += N_TOK * 2048
        vt, vr = struct.unpack_from("<iQ", b, o); o += 12
        assert vt == 1 and vr == 2048
        v = np.frombuffer(b, dtype="<f2", count=N_TOK * 1024, offset=o).reshape(N_TOK, 1024)
        o += N_TOK * 2048
        # heads concatenated, K then V per head dim block
        for h in range(N_HEADS):
            assert np.all(k[:, h * 256:(h + 1) * 256] == L * 4 + h)
            assert np.all(v[:, h * 256:(h + 1) * 256] == -(L * 4 + h))
    assert o == len(b), (o, len(b))


def test_pos_offset_and_seq():
    b = _synth_bytes(pos_offset=5, seq_id=3)
    _, seq = struct.unpack_from("<Ii", b, 0)
    assert seq == 3
    pos, n_sid, x, y = struct.unpack_from("<iIii", b, 16)
    assert (pos, n_sid, x, y) == (5, 1, 5, 5)
    sid = struct.unpack_from("<i", b, 32)[0]
    assert sid == 3


def test_roundtrip_canonical_values():
    # exact-value round trip: synthesize from canonical, parse K row back
    b = _synth_bytes()
    base = 8 + 4 + 4 + N_TOK * 20 + 8
    # first blob layer is model layer 3
    kt, kr = struct.unpack_from("<iQ", b, base)
    k = np.frombuffer(b, dtype="<f2", count=N_TOK * 1024, offset=base + 12).reshape(N_TOK, 1024)
    assert np.all(k[:4, 0] == 12.0)  # layer 3, head 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests green")
