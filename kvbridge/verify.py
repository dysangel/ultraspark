"""Verify a synthesized attention blob against a real llama.cpp blob.

Usage: python3 kvbridge/verify.py <synth.bin> <real.bin> [n_tokens]

Checks: magic/header, cell meta agreement (pos, mrope ext, seq ids),
v_trans/n_layer, per-layer type/row_bytes, and K/V cosine similarity per
layer over the first n tokens (default: all overlapping tokens).
Exits non-zero if any layer's median cosine < 0.99.
"""
import struct
import sys

import numpy as np


def parse_attn(blob: bytes):
    """-> (header dict, cells list, [(k_type, k_row, K bytes, v_type, v_row, V bytes)])"""
    o = 0

    def rd(fmt):
        nonlocal o
        v = struct.unpack_from(fmt, blob, o)
        o += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    if rd("<I") != 0xAF143CD8:
        raise ValueError("bad magic")
    header = {"seq_id": rd("<i"), "n_stream": rd("<I")}
    layers = []
    for _ in range(header["n_stream"]):
        cc = rd("<I")
        cells = []
        for _ in range(cc):
            pos, n_sid, x, y = struct.unpack_from("<iIii", blob, o); o += 16
            sids = [struct.unpack_from("<i", blob, o + 4 * j)[0] for j in range(n_sid)]
            o += 4 * n_sid
            cells.append((pos, x, y, tuple(sids)))
        v_trans = rd("<I")
        header["v_trans"], header["n_layer"] = v_trans, rd("<I")
        for _ in range(header["n_layer"]):
            kt, kr = rd("<iQ")
            kb = blob[o:o + cc * kr]; o += cc * kr
            vt, vr = rd("<iQ")
            vb = blob[o:o + cc * vr]; o += cc * vr
            layers.append((kt, kr, kb, vt, vr, vb, cells))
        header["cells"] = cells
    return header, layers, o


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.median(num / den))


def main(synth_path: str, real_path: str, n_tok: int | None = None):
    synth = open(synth_path, "rb").read()
    real = open(real_path, "rb").read()
    hs, ls, end_s = parse_attn(synth)
    hr, lr, end_r = parse_attn(real)

    ok = True
    assert hs["n_layer"] == hr["n_layer"], "layer count mismatch"
    if hs["cells"] != hr["cells"][:len(hs["cells"])]:
        print("WARN: cell meta differs (synth may cover a prefix) — first mismatch:")
        for a, b in zip(hs["cells"], hr["cells"]):
            if a != b:
                print("  synth", a, "real", b); break
    n = min(len(hs["cells"]), len(hr["cells"]))
    if n_tok:
        n = min(n, n_tok)
    print(f"synth attn section: {end_s} bytes; real total {len(real)} bytes "
          f"(tail {len(real) - end_r} bytes is the recurrent section, not synthesized)")
    print(f"comparing first {n} tokens; v_trans synth={hs['v_trans']} real={hr['v_trans']}")
    if hs["v_trans"] != hr["v_trans"]:
        print("FAIL: v_trans mismatch"); return 1
    for i, ((kt, kr, kb, vt, vr, vb, _), (Kt, Kr, KB, Vt, Vr, VB, _)) in enumerate(zip(ls, lr)):
        ck = cos(np.frombuffer(kb, "<f2").reshape(-1, kr // 2)[:n],
                 np.frombuffer(KB, "<f2").reshape(-1, Kr // 2)[:n])
        cv = cos(np.frombuffer(vb, "<f2").reshape(-1, vr // 2)[:n],
                 np.frombuffer(VB, "<f2").reshape(-1, Vr // 2)[:n])
        good = ck > 0.99 and cv > 0.99 and (kt, kr, vt, vr) == (Kt, Kr, Vt, Vr)
        ok &= good
        print(f"blob layer {i}: types/rows {'ok' if (kt,kr,vt,vr)==(Kt,Kr,Vt,Vr) else 'MISMATCH'} "
              f"cosK={ck:.4f} cosV={cv:.4f} {'PASS' if good else 'FAIL'}")
    print("VERIFY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None))
