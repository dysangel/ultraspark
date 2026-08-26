"""Synthesize an (attention-only) llama.cpp seq-state blob from the vLLM dump.

Issue #8 part 2. Everything below was verified empirically against
/tmp/lc_state.bin (2001-token decode of the fox prompt on qwen35-4b-bf16)
vs the Spark vLLM dump (issue #8 comments):

  * blob header: u32 magic 0xaf143cd8, i32 seq_id, u32 n_stream(1)
  * per stream: u32 cell_count; per cell: i32 pos, u32 n_seq_id,
    2x i32 mrope ext (mirrors pos on this arch), n_seq_id x i32 seq ids
  * u32 v_trans (0!), u32 n_layer
  * per layer: i32 type (1 = GGML_F16), u64 row_bytes, then cell_count rows;
    each row = kv heads concatenated (1024 f16 = 2048 bytes). v_trans=0 means
    V rows are per-cell exactly like K.
  * blob layer ORDER is NOT model order: llama.cpp emits
    [3,11,19,27,7,15,23,31] for this model's attn layers [3,7,...,31].
  * The vLLM dump files are PAIRS, not [K|V] per layer:
      file L%8==3: [K(L) | K(L+4)]   file L%8==7: [V(L-4) | V(L)]
    (per head: first 256 dims = first layer of the pair, last 256 = second).
  * position alignment: lc cell i <-> vllm token i (identity — vLLM had the
    BOS and llama.cpp's cell 0 is that BOS; confirmed 0.99+ cosine at offset 0).

The synthesized blob is ATTENTION-ONLY: the real blob's recurrent/delta
section is not synthesized (v1: Mac recalculates delta states). Restoration
must splice the attention section into a blob whose tail is intact, or
recompute the tail — see verify.py / state_restore.cpp.
"""
import glob
import os
import re
import struct

import torch

MAGIC = 0xAF143CD8
GGML_F16 = 1


def blob_layer_order(attn_layers: list[int]) -> list[int]:
    """llama.cpp emits attn layers as first-of-pairs then second-of-pairs.

    Empirical for qwen3.5 hybrid (pairs are (3,7),(11,15),(19,23),(27,31)):
    order [3,11,19,27,7,15,23,31]. Raises on non-contiguous-pair models.
    """
    pairs = sorted(zip(attn_layers[0::2], attn_layers[1::2]))
    order = [a for a, _ in pairs] + [b for _, b in pairs]
    if sorted(order) != sorted(attn_layers):
        raise ValueError(f"unexpected attn layer set: {attn_layers}")
    return order


def canonical_from_dump_pair(
    dump_dir: str,
    attn_layers: list[int],
    dtype: torch.dtype = torch.float16,
) -> dict[int, torch.Tensor]:
    """Pair-layout dump -> canonical {layer: [T, H, 2, D]} (K at index 0)."""
    import safetensors.torch as st

    files = {}
    for path in sorted(glob.glob(os.path.join(dump_dir, "*.safetensors"))):
        m = re.search(r"layers\.(\d+)\.", os.path.basename(path))
        if not m:
            raise ValueError(f"unparseable layer name: {path}")
        files[int(m.group(1))] = st.load_file(path)["kv_cache"]

    out: dict[int, torch.Tensor] = {}
    for la, lb in zip(attn_layers[0::2], attn_layers[1::2]):
        # tolerate either labelling: kfile at la, vfile at lb
        if la in files and lb in files:
            kfile, vfile = files[la], files[lb]
        else:
            raise ValueError(f"missing dump pair for layers {la}/{lb}: have {sorted(files)}")
        t, h, two_d = kfile.shape
        if two_d % 2 or vfile.shape != kfile.shape:
            raise ValueError(f"pair shape mismatch at {la}: {kfile.shape} vs {vfile.shape}")
        d = two_d // 2
        for L, half in ((la, slice(0, d)), (lb, slice(d, two_d))):
            kv = torch.empty(t, h, 2, d, dtype=dtype)
            kv[:, :, 0, :] = kfile[:, :, half].to(dtype)
            kv[:, :, 1, :] = vfile[:, :, half].to(dtype)
            out[L] = kv.contiguous()
    return out


GGML_F32 = 0


def recurrent_tail(
    conv: dict[int, "torch.Tensor"],
    ssm: dict[int, "torch.Tensor"],
    n_layer: int,
    last_pos: int,
    cell_count: int = 1,
) -> bytes:
    """Serialize the recurrent R/S section (llama-memory-recurrent.cpp
    state_write order) to append after the attention section.

    conv: {linear_layer: [d_conv-1, conv_channels] f32} — llama.cpp stores a
          row channel-major (time fastest), so we transpose before flattening
          (verified: 0.75 cos vs real tail on the shared prompt suffix).
    ssm:  {linear_layer: [n_v_heads, v_dim, k_dim] f32} — issue #39: both
          vLLM and llama.cpp store K fastest in memory (verified cos 0.9989
          against a state_dump reference); identity flatten, NO transpose.
          Head ordering to llama.cpp convention is applied at capture time
          (gdn_layout.permute_ssm_heads).
    Both are single-cell (one recurrent state per sequence, regardless of
    token count). Only layers present in the dicts get headers, like the
    C++ writer skips null r_l/s_l layers.
    """
    out = bytearray()
    out += struct.pack("<I", cell_count)
    for i in range(cell_count):
        out += struct.pack("<iI", last_pos, 0)   # pos, n_seq_id=0 (per-seq write)
    out += struct.pack("<II", 0, n_layer)        # s_trans, n_layer
    for table in (conv, ssm):
        for L in sorted(table):
            row = table[L]
            if table is conv:
                flat = row.transpose(0, 1).reshape(-1)   # -> channel-major
            else:
                flat = row.reshape(-1)  # k-fastest identity (issue #39)
            out += struct.pack("<iQ", GGML_F32, flat.numel() * 4)
            out += flat.numpy().astype("<f4").tobytes()
    return bytes(out)

def synthesize(
    canonical: dict[int, torch.Tensor],
    attn_layers: list[int],
    seq_id: int = 0,
    pos_offset: int = 0,
) -> bytes:
    """Canonical {layer: [T,H,2,D] f16} -> attention-only blob bytes."""
    order = blob_layer_order(attn_layers)
    t = canonical[order[0]].shape[0]
    out = bytearray()
    out += struct.pack("<Ii", MAGIC, seq_id)
    out += struct.pack("<I", 1)                      # n_stream
    out += struct.pack("<I", t)                      # cell_count
    for i in range(t):
        pos = i + pos_offset
        out += struct.pack("<iIii", pos, 1, pos, pos)  # pos, n_seq, mrope x,y
        out += struct.pack("<i", seq_id)
    out += struct.pack("<II", 0, len(order))         # v_trans=0, n_layer
    for L in order:
        kv = canonical[L]
        if kv.shape[0] != t:
            raise ValueError(f"layer {L}: {kv.shape[0]} tokens != {t}")
        _, h, _, d = kv.shape
        row = h * d                                  # f16 elements per row
        k = kv[:, :, 0, :].reshape(t, row)           # heads concatenated
        v = kv[:, :, 1, :].reshape(t, row)
        for name, m in (("K", k), ("V", v)):
            out += struct.pack("<iQ", GGML_F16, row * 2)
            out += m.numpy().astype("<f2").tobytes()
    return bytes(out)
