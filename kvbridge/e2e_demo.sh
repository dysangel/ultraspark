#!/usr/bin/env bash
# e2e_demo.sh — UltraSpark2 end-to-end demo harness (issue #10).
#
# Measures TTFT for a bridged-KV request vs a plain local-prefill request on
# the SAME llama-server config, plus greedy-continuation sanity between the
# two paths.
#
#   MODE=local  (default, works today): the full-hybrid blob is generated
#               LOCALLY by kvbridge/state_dump (llama.cpp's own state export,
#               PR #14). This stands in for the Spark blob; byte format and
#               the Mac serving path (USPK_BRIDGE_FILE, issue #16) are
#               identical.
#   MODE=spark  (issue #11 not merged yet): pull the synthesized blob from the
#               Spark StreamConnector via kvbridge/pull_blob.py (WIRE.md).
#               Currently a documented stub — see STUB NOTES at the bottom.
#
# Usage:
#   kvbridge/e2e_demo.sh [--mode local|spark] [--out DIR]
#
# Outputs into the out dir (default /tmp/uspk_e2e): prompt files, blobs,
# per-path JSON results, and a summary table on stdout.

set -euo pipefail

MODE=local
OUT=/tmp/uspk_e2e
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --out)  OUT="$2";  shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- fleet topology --------------------------------------------------------
MAC_GGUF="$HOME/models/qwen35-4b-bf16.gguf"
LC="$HOME/Projects/llama.cpp-ultraspark2"
SERVER="$LC/build-metal/bin/llama-server"
STATE_DUMP="$HOME/Projects/ultraspark2/kvbridge/state_dump"
SPARK_HOST=<prefill-host>
SPARK_STREAM_PORT=52901          # WIRE.md default
PORT=5566

[[ -f "$MAC_GGUF" ]]  || { echo "missing model: $MAC_GGUF" >&2; exit 1; }
[[ -x "$SERVER" ]]    || { echo "missing server: $SERVER" >&2; exit 1; }
[[ -x "$STATE_DUMP" ]] || { echo "missing tool: $STATE_DUMP (see kvbridge/state_dump.cpp)" >&2; exit 1; }

mkdir -p "$OUT"
cd "$(dirname "$0")"             # kvbridge/, for python helpers

# ---- prompts ---------------------------------------------------------------
# fox200: the established 2001-token prompt (~200 fox-sentence repeats, BOS+eos
# accounting per issue #8). fox400: ~4001 tokens, the "real-world" size.
FOX_SENT="The quick brown fox jumps over the lazy dog."
python3 - "$OUT" "$FOX_SENT" <<'EOF'
import sys
out, sent = sys.argv[1], sys.argv[2]
for n, name in ((200, "fox200"), (400, "fox400")):
    with open(f"{out}/{name}.txt", "w") as f:
        f.write(" ".join([sent] * n))
EOF

# ---- blob acquisition ------------------------------------------------------
acquire_blob () {  # $1 = prompt file, $2 = out blob
  if [[ "$MODE" == "spark" ]]; then
    # issue #11: StreamConnector on the Spark serves the synthesized blob per
    # WIRE.md. Requires: vLLM on the Spark started with the connector +
    # max_num_batched_tokens >= 8192 (chunked prefill truncated the 2000-token
    # dump to 1584 tokens otherwise).
    if python3 pull_blob.py "$SPARK_HOST" "$SPARK_STREAM_PORT" "$2"; then
      return 0
    fi
    echo "STUB: pull_blob.py failed (issue #11 not merged / connector not up)." >&2
    echo "      falling back to a locally generated full-hybrid blob." >&2
  fi
  # NB: state_dump aborts in Metal teardown at exit (issue #15, known noise)
  # AFTER writing the blob — tolerate the nonzero status, verify by file size.
  DYLD_LIBRARY_PATH="$LC/build-metal/bin" "$STATE_DUMP" "$MAC_GGUF" "$(cat "$1")" "$2" \
    || echo "(state_dump exit=$? — issue #15 teardown noise, ignored)"
  [[ -s "$2" ]] || { echo "blob not produced: $2" >&2; exit 1; }
}

# ---- server lifecycle ------------------------------------------------------
SRV_PID=0
start_server () {  # $1 = port, $2 = blob or "" for plain
  local extra=()
  if [[ -n "$2" ]]; then
    # USPK_BRIDGE_FILE is read once at startup (issue #16 hook)
    extra=(env "USPK_BRIDGE_FILE=$2")
  fi
  "${extra[@]+${extra[@]}}" "$SERVER" -m "$MAC_GGUF" --port "$1" -ngl 99 -c 8192 \
      > "$OUT/server_$1.log" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "server on :$1 failed to start (see $OUT/server_$1.log)" >&2
  return 1
}
stop_server () {
  [[ "$SRV_PID" -ne 0 ]] && kill "$SRV_PID" 2>/dev/null || true
  wait "$SRV_PID" 2>/dev/null || true   # Metal teardown assert at exit = issue #15 noise
  SRV_PID=0
}
trap stop_server EXIT

# warmup with an UNRELATED prompt so neither path benefits from llama-server's
# own prompt cache for the measured request.
warmup () {  # $1 = port
  curl -sf "http://127.0.0.1:$1/completion" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"warmup hello","n_predict":1,"temperature":0}' >/dev/null
}

# ---- run one prompt through both paths -------------------------------------
run_case () {  # $1 = case name (fox200|fox400)
  local name="$1" prompt="$OUT/$1.txt"
  echo "=== $name ($(wc -c < "$prompt") bytes) ==="

  acquire_blob "$prompt" "$OUT/$name.blob"
  ls -l "$OUT/$name.blob"

  echo "--- local prefill (plain server) ---"
  start_server $PORT ""
  warmup $PORT
  python3 e2e_ttft.py "http://127.0.0.1:$PORT" "$prompt" > "$OUT/$name.local.json"
  stop_server

  echo "--- bridged (USPK_BRIDGE_FILE) ---"
  start_server $PORT "$OUT/$name.blob"
  warmup $PORT
  python3 e2e_ttft.py "http://127.0.0.1:$PORT" "$prompt" > "$OUT/$name.bridged.json"
  stop_server

  python3 - "$OUT" "$name" <<'EOF'
import json, sys
out, name = sys.argv[1:3]
loc = json.load(open(f"{out}/{name}.local.json"))
brg = json.load(open(f"{out}/{name}.bridged.json"))
speedup = loc["ttft_ms"] / brg["ttft_ms"] if brg["ttft_ms"] else float("inf")
print(f"TTFT local={loc['ttft_ms']:.0f}ms bridged={brg['ttft_ms']:.0f}ms speedup={speedup:.1f}x")
if loc["text"] == brg["text"]:
    print("SANITY: greedy continuations IDENTICAL")
else:
    lt = {e["tok"] for e in (loc["top5"] or [])}
    bt = {e["tok"] for e in (brg["top5"] or [])}
    print(f"SANITY: MISMATCH. top5 overlap={len(lt & bt)}/5: {sorted(lt & bt)}")
    print(f"  local:   {loc['text'][:120]!r}")
    print(f"  bridged: {brg['text'][:120]!r}")
EOF
  echo
}

echo "mode=$MODE out=$OUT"
run_case fox200
run_case fox400

echo "results in $OUT ({fox200,fox400}.{local,bridged}.json); report: kvbridge/e2e_report.md"

# ---- STUB NOTES (issue #11) ------------------------------------------------
# The Spark side is stubbed behind MODE=spark. When PR for issue #11 merges:
#   1. On the Spark (ssh <user>@<prefill-host>): start vLLM on qwen35-4b with the
#      StreamConnector attached and --max-num-batched-tokens 8192 (avoid chunked
#      prefill so the dump covers the ENTIRE prompt), stream port 52901.
#   2. Prime it with each prompt (the connector snapshots on prefill).
#   3. Run: kvbridge/e2e_demo.sh --mode spark  -> acquire_blob pulls per WIRE.md,
#      verifies sha256, and the Mac side proceeds unchanged.
# Everything else in this script already treats blobs as opaque full-hybrid
# seq-state files, so no Mac-side changes are needed.
