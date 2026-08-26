# USPK KV wire format v1 (issues #11, #20)

Pull-mode transfer of a synthesized llama.cpp hybrid KV blob from the
Spark (vLLM host, `StreamConnector`) to the Mac (`kvbridge/pull_blob.py`).

## Transport

Plain TCP. Server (Spark, `stream_connector.BlobServer`) listens on
`stream_port` (default **52901**) for the whole vLLM process lifetime and
runs an **accept loop**: each client connection may issue one or more
requests until it closes; the socket keeps accepting afterwards (v0's
single-accept broke the second pull in the e2e run). Blobs are **keyed by
prompt hash** — exactly the dump-dir name vLLM's connector derives from the
prompt token ids — so every successive `generate()` stays pullable from the
same process. Offsets in the protocol make any frame boundary a valid resume
point across reconnects.

## Framing

`kvbridge/framing.py`. Every message is a frame:

    u64 big-endian payload_length   (max 16 MiB)
    payload_length bytes

Frames are self-delimiting and never over-read, so any frame boundary is a
valid resume point. `recv` loops internally on partial reads; if the
connection drops inside a frame, `IncompleteFrameError` carries the partial
payload and the declared length.

## Protocol

1. Client sends a JSON request frame:

   ```json
   {"op": "get", "key": "<prompt-hash-hex>", "offset": 0}
   ```

   `key` selects the blob (prompt hash; may be omitted when the server holds
   exactly one blob). `offset` is a byte offset into the blob (0 for a fresh
   pull; the offset of the last completed frame to resume).

   There is also a discovery op:

   ```json
   {"op": "list"}
   ```

   answered with one JSON-array frame of
   `[{"key": "…", "blob_size": N, "prompt_token_ids": [...], "model_id": "…"}, …]`.

2. Server replies to `get` with a JSON header frame:

   ```json
   {"key": "…", "blob_size": 12345678, "offset": 0, "sha256": "…",
    "prompt_token_ids": [9707, 525, …], "model_id": "qwen35-4b-fp8"}
   ```

   `sha256` covers the *entire* blob (not just the requested suffix), so the
   client can verify a resumed transfer end-to-end. `prompt_token_ids` and
   `model_id` (issue #20) let the Mac verify blob identity *before* restore;
   clients that don't know these fields ignore them (backwards-compatible).
   On a bad op, unknown/ambiguous key, or out-of-range offset the server
   replies `{"error": "…"}` instead.

3. Server streams blob bytes as consecutive frames of at most 4 MiB each,
   starting at `offset` and ending at `blob_size`.

4. Server sends one empty frame (`length 0`) as the end-of-stream sentinel.

The client (`pull_blob.py`) accumulates frames, re-issues `get` with the same
`key` and the last completed frame's end-offset on connection drop, and checks
`sha256` at the sentinel. `pull_blob.list_blobs(host, port)` wraps the
discovery op; the CLI is
`python3 pull_blob.py <host> <port> <out.bin> [key|-l]`.

## Fail-loud coverage (issue #20)

The server refuses to serve a blob it cannot vouch for: missing recurrent
states (attention-only blobs are not restorable on this hybrid, issue #16),
partial linear-layer coverage, a missing mamba slot, or unrecorded prompt
token ids all raise on the Spark side instead of warn-and-continue.

## Payload

The blob is a FULL hybrid llama.cpp seq-state produced by
`kvbridge/synthesize.py`:

1. Attention section (`canonical_from_dump_pair` + `synthesize`) — magic
   0xAF143CD8, cell records, v_trans=0, per-layer K/V rows; see the
   synthesizer docstring for the exact layout. Sourced from the
   `HMADumpConnector` safetensors dump pairs.
2. Recurrent R/S tail (`synthesize.recurrent_tail`, issue #16 finding:
   llama.cpp never rebuilds delta/recurrent states, so an attention-only blob
   is rejected / diverges) — single cell, per linear layer (24 of 32 on
   qwen35-4b) an R row (conv state, 98304 B) and an S row (SSM state,
   2097152 B), both GGML_F32. Layout conversions from vLLM's cache: conv is
   stored channel-major (time fastest), SSM per head v-fastest (vLLM stores
   k-fastest). Captured in `StreamConnector.save_kv_layer` at the last
   attention layer's save hook, when all linear layers have already run and
   their pool slots hold the final state. ~52.7 MiB flat regardless of token
   count.

Known caveat (issue #16): vLLM and llama.cpp tokenize the fox prompt
differently (1584 vs 2001 tokens), so the tail cannot be cosine-verified
against a llama.cpp-produced blob for the same text; conv layout is
empirically confirmed via the shared prompt suffix (cos 0.75 on the last
linear layer), SSM layout is derived from source (delta-net-base.cpp asserts
`s->ne[0] == S_v`). End-to-end logits validation should run through the
issue #16 USPK_BRIDGE_FILE harness.

## Deployment (Spark)

`stream_connector.py`, `synthesize.py`, `framing.py` are scp'd to
`~/kvbridge` on the Spark and loaded with `PYTHONPATH=$HOME/kvbridge` and
`kv_connector=StreamConnector` (`kv_connector_module_path=stream_connector`).
The blob is synthesized from the safetensors the parent `HMADumpConnector`
already writes, then served.

## 27B (Qwen3.8-27B, issue #23)

Same pipeline, different geometry — the connector now derives layer indices
from `kv_cache_groups` instead of the hardcoded 4B values:

* layer pattern: `full_attention` every 4th layer, indices
  `[3, 7, ..., 63]` — 16 attn + 48 linear of 64 (same every-4th scheme as
  the 4B, just more of them; see `config.json` `layer_types`).
* KV per token: 16 layers × 4 KV-heads × 256 head_dim × 2 (K+V) × f16 =
  **32768 B/token** (4B was 8 layers, same heads → 16384 B/token).
* recurrent tail: 48 linear layers × (R 73728 B + S 3145728 B) f32 =
  **154,533,888 B ≈ 147.4 MiB flat** (4B: 52.7 MiB / 24 layers).
* blob sizes seen: 259.7 MB (1568-token dump of a 2000-token fox prompt),
  2213 MB (31360-token dump of a 32000-token prompt). vLLM truncates the
  dump to block boundaries (block 16), so blobs cover a prefix — the Mac
  restore hook prefills the remainder (this is why `_pull_blob` prefix-
  matches instead of requiring equality).
* vLLM flags that mattered: `--max-model-len 32768` (default 262144 makes
  vLLM reject `max_num_batched_tokens`), `--no-enable-chunked-prefill`,
  `--max-num-batched-tokens 32768`. See `~/kvbridge/serve_kv_27b.sh` on the
  Spark; Mac GGUF `~/models/qwen38/Qwen3.8-27B-UD-Q4_K_XL-dysangel.gguf`
  (text path only — the mmproj/vision tower is never bridged).
* gotchas found: decode-step save rounds have no mamba slot (connector now
  skips non-store rounds instead of crashing the engine); verify.py
  cosines vs a Q4_K_XL-decode reference blob run 0.86–0.999 on deep layers
  (fp8-prefill vs Q4-decode numeric drift) — the binding check is the
  greedy-continuation gate, which passes IDENTICAL at 2k and 16k.
* Bench (Mac M3 Ultra decode, 27B Q4_K_XL; TTFT incl. prefill/orchestration):
  2k prompt local 5412 ms → cold 1770 ms / warm 114 ms;
  16k prompt local 98134 ms → cold 5958 ms / warm 185 ms (16.5x cold).

## Note on the "no Python in the KV byte path" rule

That rule concerns the Mac's runtime decode path. Python here is
control-plane/serving-side only (Spark, at request-finish); the Mac-side
runtime consumer is llama.cpp C++ and never touches Python for KV bytes.
