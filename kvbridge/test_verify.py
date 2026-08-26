"""Tests for verify.py (issue #8 part 2). Run: python3 kvbridge/test_verify.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import verify  # noqa: E402
from synthesize import canonical_from_dump_pair, synthesize  # noqa: E402
from test_synthesize import ATTN_LAYERS, _pair_dump  # noqa: E402


def _synth():
    d = canonical_from_dump_pair(_pair_dump(), attn_layers=ATTN_LAYERS)
    return synthesize(d, attn_layers=ATTN_LAYERS)


def test_identical_blobs_pass():
    assert verify.main.__call__  # smoke
    open("/tmp/uspk_v_a.bin", "wb").write(_synth())
    open("/tmp/uspk_v_b.bin", "wb").write(_synth())
    assert verify.main("/tmp/uspk_v_a.bin", "/tmp/uspk_v_b.bin") == 0


def test_corrupted_v_fails():
    b = bytearray(_synth())
    # flip some V bytes in the last layer block (near the end of the blob)
    b[-1000:-900] = b"\xff" * 100
    open("/tmp/uspk_v_c.bin", "wb").write(bytes(b))
    open("/tmp/uspk_v_d.bin", "wb").write(_synth())
    assert verify.main("/tmp/uspk_v_c.bin", "/tmp/uspk_v_d.bin") == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests green")
