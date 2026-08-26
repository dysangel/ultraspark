# E2E demo report — bridged KV vs local prefill (issue #10)

Run: 2026-08-20, Mac (M3 Ultra), `qwen35-4b-bf16.gguf`, llama-server
`build-metal` @ `feat/kv-import` (USPK_BRIDGE_FILE hook, issue #16),
`llama-server -m … --port 5566 -ngl 99 -c 8192`, greedy (temp 0, top_k 1),
TTFT = request start → first generated token byte over a streamed response
(includes full prompt processing — exactly what the bridge eliminates).
Harness: `kvbridge/e2e_demo.sh` + `kvbridge/e2e_ttft.py`; artifacts in
`/tmp/uspk_e2e/`.

**Spark side is stubbed** (issue #11 not merged): blobs were generated locally
with `kvbridge/state_dump` — llama.cpp's own full-hybrid seq-state export. The
Mac serving path (USPK_BRIDGE_FILE) is byte-identical for a Spark-synthesized
blob, so the TTFT/sanity numbers below are the real deliverable; only the blob
origin differs.

## Numbers

| case | tokens | TTFT local | TTFT bridged | speedup | continuation | top-5 match |
|------|--------|-----------|--------------|---------|--------------|-------------|
| fox200 | 2000 | 381 ms | 74 ms | 5.1x | IDENTICAL (64 tok) | 5/5, p equal to ~6 decimals |
| fox400 | 4000 | 1267 ms | 88 ms | 14.4x | IDENTICAL (64 tok) | 5/5, p equal to ~6 decimals |

Blobs: fox200 = 112.8 MiB, fox400 = 175.4 MiB (full-hybrid: attention cells +
52.7 MiB flat recurrent tail). Bridged TTFT is flat in prompt length (74→88 ms
while local 381→1267 ms): the restore is O(blob bytes) + ≥1 token prefill, so
the win grows linearly with prompt size. Decode throughput after TTFT is
unchanged (total ~3.2 s local vs ~2.3 s bridged for 64 tokens; the delta is
just the saved prefill).

Sanity verdict: **bit-faithful**. Greedy continuations identical for 64 tokens
on both cases, and the first-token top-5 distributions agree to the printed
precision (e.g. fox400: ` The` p=0.995647 both paths). This is expected for a
same-engine blob (f16 K/V round-trip is exact); a Spark/vLLM-synthesized blob
goes through f16 quantization of vLLM bf16 K/V, where PR #14's gate already
showed 0.99+ cosines — expect identical argmax usually, occasional near-tie
flips. The harness logs top-5 overlap automatically when texts diverge.

## The pain

1. **Chunked/partial blob coverage.** `state_dump` decoded the 4000-token
   prompt in one `llama_decode` call and asserted (`n_tokens_all <= n_batch`)
   because default `n_batch`=2048. Fixed: `cp.n_batch = n_ubatch = n_ctx`.
   Same class of bug bit the vLLM dump earlier (1584 of 2000 tokens covered).
   **Rule: a blob must cover the ENTIRE prompt.** Emitting a silent partial
   blob is worse than failing — the restore would succeed and produce wrong
   text. The bridge hook should refuse a blob whose cell_count < prompt-1
   (currently it only warns once and serves anyway).
2. **Empty-array `set -u`** in the harness (`"${extra[@]}"` with no bridge
   env) — trivial but it silently skipped a measurement path on first run.
3. **`nprobs` is ignored** by the native `/completion` endpoint in this
   build; top-5 had to come from OAI `/v1/completions` `logprobs`.
4. **Metal teardown abort at exit** (issue #15) fires on every `state_dump`
   run — after the blob is written. The harness tolerates the nonzero exit
   and verifies by file size; it still pollutes logs and masks real failures.

## What needs hardening

- **Fingerprint check at restore time.** The bridge hook restores the blob
  into whatever prompt arrives first if it's long enough; nothing verifies the
  prompt's token prefix matches the blob's cells. A token-hash (or even first/
  last-token + length check) in the blob header would prevent cross-prompt
  garbage. Worst current failure mode: silently plausible, wrong continuation.
- **Coverage enforcement** (pain #1): fail loud, not warn-once, when
  `cell_count < prompt_tokens - 1`.
- **Streaming the blob** instead of file-drop: once #11's WIRE transport is
  merged, the Mac should pull directly into the restore path (no temp file),
  and ideally restore before the server even reports ready, so TTFT doesn't
  include a cold 70 ms file read. 175 MiB over 10GbE ≈ 20 ms — negligible.
- **Blob reuse across requests**: `bridge_loaded_` caches the blob, but
  llama-server's own slot cache makes the *second* identical request fast on
  both paths anyway; the honest comparison (fresh server per measurement) is
  what the harness does. A real deployment wants prefix-hash → blob selection,
  not a single hardcoded file.
- **Spark integration**: when #11 lands, run `e2e_demo.sh --mode spark`; the
  script already pulls via `pull_blob.py` (sha256-verified) and falls back to
  local blobs if the connector isn't up. Numbers to collect then: blob
  synthesis + transfer wall time vs saved prefill — at 4000 tokens the Mac
  saves ~1.2 s per request; synthesis on Spark must stay well under that.
