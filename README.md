# UltraSpark

**Disaggregated LLM inference: vLLM prefill on GPU nodes → KV/state blob
→ llama.cpp decode on a second machine.**

Prefill is compute-bound and loves big GPUs. Decode is bandwidth-bound and
loves cheap memory. Split them: a long prompt is prefilled once on the
fast node, the resulting KV cache is shipped over the wire as a blob, and
the decode node restores it into llama.cpp's state cache and continues
generation — no re-prefill, no re-reading gigabytes of prompt.

```
        prefill (vLLM, stock)                decode (llama.cpp fork)
   ┌──────────────────────────┐        ┌──────────────────────────────┐
   │  vllm serve …            │        │  llama-server (this fork)    │
   │   --kv-transfer-config   │  blob  │   USPK_BRIDGE_DIR=…          │
   │   StreamConnector /      │ ─────► │   · prompt fingerprint check │
   │   HmaDumpConnector       │  file/ │   · KV restore into slot     │
   │                          │  TCP   │   · logits gate, then decode │
   └──────────────────────────┘        └──────────────────────────────┘
                 optional front door: uspk-manager :8899
                 (OpenAI + Anthropic APIs, SSE relay, routing)
```

## Repositories

| Repo | What |
|---|---|
| **this repo** | The vLLM-side connectors, the kvbridge toolkit (dump / convert / merge / verify / restore), e2e demos and tests |
| [`dysangel/llama.cpp`](https://github.com/dysangel/llama.cpp) (branch **`ultraspark2`**) | The llama.cpp **fork**: server-side snapshot/restore hooks + `uspk-manager` — one commit on top of upstream |

No vLLM fork is needed — the connectors are plain Python modules loaded
by stock vLLM through `--kv-transfer-config` (the `KVConnectorBase_V1`
hook).

## Quickstart

1. **Build the decode side** — clone the forked llama.cpp repo
   ([dysangel/llama.cpp](https://github.com/dysangel/llama.cpp), branch
   `ultraspark2` — one commit on top of upstream):

   ```sh
   git clone -b ultraspark2 https://github.com/dysangel/llama.cpp
   cd llama.cpp   # then follow ULTRASPARK.md
   ```

   Full walkthrough:
   [ULTRASPARK.md](https://github.com/dysangel/llama.cpp/blob/ultraspark2/ULTRASPARK.md).
2. **Run the prefill side** — stock vLLM plus one of the connectors:

   ```sh
   PYTHONPATH=$PWD/kvbridge vllm serve <model> \
     --port 8081 \
     --kv-transfer-config '{"kv_connector":"StreamConnector","kv_role":"kv_both","kv_connector_module_path":"stream_connector","kv_connector_extra_config":{"shared_storage_path":"'$HOME'/kvbridge/dumps","stream_port":52901}}'
   ```

   (`kvbridge/serve_kv_27b.sh` is a known-good launcher; the dump-only
   variant is `hma_dump_connector.py`.)
3. **Point the decode node at the blobs** (`USPK_BRIDGE_DIR`), optionally
   front everything with `uspk-manager`, and talk to it over OpenAI or
   Anthropic APIs. Full walkthrough: ULTRASPARK.md link above.

## kvbridge contents

| File | What |
|---|---|
| `stream_connector.py` | vLLM connector: streams KV over TCP as it prefills |
| `hma_dump_connector.py` | vLLM connector: dumps KV per-layer safetensors (hybrid-attention layouts) |
| `convert.py` | vLLM dump → canonical `[T,H,KV,D]` for llama.cpp state synthesis |
| `synthesize.py` / `parse_state.py` | Blob synthesis + llama.cpp state parsing |
| `tp2_merge.py` | Merge two TP-shard dumps into one full-attention blob |
| `state_dump.cpp` / `state_restore.cpp` | llama.cpp→llama.cpp blob export/import |
| `verify.py` | Cross-engine KV verification (bit-exactness discipline) |
| `e2e_demo.sh`, `e2e_service.sh`, `e2e_ttft.py` | End-to-end demos + TTFT measurement |
| `WIRE.md` | The blob/stream wire format spec |
| `test_*.py` | Unit tests for all of the above |

## Design rules

- **Fail loudly**: when something can't be verified, it errors instead of
  guessing. Every restore is checked against a content fingerprint of the
  prompt — blobs that don't match are rejected, not applied hopefully.
- **Bit-exactness gates**: the first token after a cross-engine restore is
  checked against reference logits before decode continues.
- **Stock vLLM, additive llama.cpp**: the fork keeps every change inside
  `tools/server/` + `kvbridge/` so rebasing on upstream main stays trivial.

## Status

Working end-to-end on the author's fleet (DGX Spark prefill nodes, Apple
silicon decode). Expect sharp edges — issues welcome.
