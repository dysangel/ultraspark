"""Parse llama.cpp seq-state blob → per-layer K rows; correlate vs vLLM dump.

Issue #8: empirical layout verification. K rows are [n_embd_k_gqa] f16 per
token, heads concatenated, post-RoPE. vLLM canonical K = [heads, 256].
"""
import struct
import sys

import numpy as np
import torch


def parse(blob: bytes):
    o = 0
    def rd(fmt):
        nonlocal o
        v = struct.unpack_from(fmt, blob, o)
        o += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    magic = rd("<I"); seq = rd("<i")
    assert magic == 0xAF143CD8, hex(magic)

    # --- attention (kv cache) ---
    n_stream = rd("<I")
    attn = []
    for _ in range(n_stream):
        cell_count = rd("<I")
        if cell_count == 0:
            continue
        meta = []
        for _ in range(cell_count):
            pos = rd("<i"); n_sid = rd("<I")
            _ = rd("<ii")  # llama_kv_cell_ext (mrope x,y), present on this arch
            _sids = [rd("<i") for _ in range(n_sid)]
            meta.append(pos)
        v_trans = rd("<I"); n_layer = rd("<I")
        for _ in range(n_layer):
            k_type = rd("<i"); k_row = rd("<Q")
            kbytes = blob[o:o + cell_count * k_row]; o += cell_count * k_row
            v_type = rd("<i"); v_row = rd("<Q")
            vbytes = blob[o:o + cell_count * v_row]; o += cell_count * v_row
            attn.append({"pos": meta, "k": kbytes, "v": vbytes,
                         "v_trans": v_trans, "k_row": k_row})
    return attn


if __name__ == "__main__":
    blob = open(sys.argv[1], "rb").read()
    dump_dir = sys.argv[2]  # vLLM dump folder
    attn = parse(blob)
    print(f"layers: {len(attn)}, tokens: {len(attn[0]['pos'])}, "
          f"pos[0..3]={attn[0]['pos'][:3]}, k_row={attn[0]['k_row']}")

    import glob
    import safetensors.torch as st
    f = sorted(glob.glob(f"{dump_dir}/*.safetensors"))[0]  # layer 3 = first attn
    vllm = st.load_file(f)["kv_cache"].float()  # [T, H, K|V]

    lc_k = np.frombuffer(attn[0]["k"], dtype=np.float16).reshape(-1, 1024)
    lc_k = torch.from_numpy(lc_k.copy()).float()

    n = min(len(lc_k), vllm.shape[0])
    cos = torch.nn.functional.cosine_similarity
    # candidate mappings: vLLM head-major concat vs dim1-flip
    vllm_k = vllm[:n, :, :256].reshape(n, 1024)
    c1 = cos(lc_k[:n], vllm_k).median()
    vllm_k2 = vllm[:n, :, 256:].reshape(n, 1024)
    c2 = cos(lc_k[:n], vllm_k2).median()
    print(f"cos(lcK, vllm[:, :, :256] heads-concat): {c1:.4f}")
    print(f"cos(lcK, vllm[:, :, 256:] as K):        {c2:.4f}")
