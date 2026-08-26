#!/usr/bin/env bash
# e2e_service.sh — issue #19 end-to-end orchestration integration test.
#
# One curl to the manager :8899 with a cold prompt must produce:
#   Mac /tokenize -> Spark vLLM prefill (StreamConnector blob) -> pull_blob
#   -> USPK_BRIDGE_DIR -> llama-server restore -> Mac decode.
#
# Asserts output sanity vs the local-prefill path and reports TTFT cold/warm.
#
# Topology:
#   Spark (<prefill-host>): vLLM serve :8081 + BlobServer :52901 (serve_kv.sh)
#   Mac: llama-server :5567 (USPK_BRIDGE_DIR) + uspk-manager :8899
#
# Usage: kvbridge/e2e_service.sh [--out DIR]   (env MODEL=4b|27b, default 4b)
#
# MODEL=27b (issue #23): Qwen3.8-27B everywhere (Spark FP8 prefill, Mac
# Q4_K_XL GGUF decode), adds a ~16k-token prompt case (16x fox200) and needs
# USPK_MAX_PROMPT_TOKENS>8000 exported for the manager.
set -euo pipefail

OUT=/tmp/uspk_e2e_service
MODEL=${MODEL:-4b}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SPARK_HOST="${SPARK_HOST:-<prefill-host>}"  # env-overridable
VLLM_URL="http://$SPARK_HOST:8081"
DECODE_PORT=5567
MGR_PORT=8899
BRIDGE_DIR="$OUT/bridge"
CTX=8192
SERVE_SCRIPT=serve_kv.sh
export USPK_MAX_PROMPT_TOKENS=8000
case "$MODEL" in
  4b)  MAC_GGUF="$HOME/models/qwen35-4b-bf16.gguf" ;;
  27b) MAC_GGUF="$HOME/models/qwen38/Qwen3.8-27B-UD-Q4_K_XL-dysangel.gguf"
       CTX=32768; SERVE_SCRIPT=serve_kv_27b.sh
       export USPK_MAX_PROMPT_TOKENS=32768 ;;
  *) echo "unknown MODEL: $MODEL" >&2; exit 2 ;;
esac
SERVER="$HOME/Projects/llama.cpp-ultraspark2/build-metal/bin/llama-server"
REPO="$HOME/Projects/ultraspark2"
FOX_SENT="The quick brown fox jumps over the lazy dog."

mkdir -p "$OUT" "$BRIDGE_DIR"
cd "$REPO/kvbridge"

fail() { echo "FAIL: $*" >&2; exit 1; }

[[ -f "$MAC_GGUF" ]] || fail "missing model"
[[ -x "$SERVER" ]]   || fail "missing llama-server build"

python3 - "$OUT" "$FOX_SENT" "$MODEL" <<'EOF'
import sys
out, sent, model = sys.argv[1], sys.argv[2], sys.argv[3]
# fox200: the established ~2001-token prompt
open(f"{out}/fox200.txt", "w").write(" ".join([sent.strip()] * 200))
# follow-up: same prompt with a tail appended (fresh blob, prefix reuse)
open(f"{out}/followup.txt", "w").write(" ".join([sent.strip()] * 200)
                                       + " Now tell me about the weather today.")
if model == "27b":
    # ~16k-token prompt: 16x fox200 concatenated (issue #23 bench size)
    open(f"{out}/fox3200.txt", "w").write(" ".join([sent.strip()] * 3200))
EOF

# ---- Spark prefill node ----------------------------------------------------
if ! curl -sf -m 5 "$VLLM_URL/v1/models" >/dev/null 2>&1; then
  echo "Spark vLLM down - starting serve_kv.sh"
  ssh "$SPARK_HOST" "rm -rf ~/kvbridge/dumps; mkdir -p ~/kvbridge/dumps;
                     (setsid nohup ~/kvbridge/$SERVE_SCRIPT > ~/kvbridge/serve_kv.log 2>&1 &)"
  for _ in $(seq 1 240); do
    curl -sf -m 5 "$VLLM_URL/v1/models" >/dev/null 2>&1 && break
    sleep 2
  done
fi
curl -sf -m 5 "$VLLM_URL/v1/models" >/dev/null || fail "Spark vLLM unreachable"

# ---- Mac decode node -------------------------------------------------------
SRV_PID=0
start_decode() {
  rm -rf "$BRIDGE_DIR"; mkdir -p "$BRIDGE_DIR"   # cold bridge dir
  USPK_BRIDGE_DIR="$BRIDGE_DIR" "$SERVER" -m "$MAC_GGUF" \
      --port "$DECODE_PORT" -ngl 99 -c "$CTX" > "$OUT/decode.log" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$DECODE_PORT/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  fail "decode server failed to start (see $OUT/decode.log)"
}
stop_decode() {
  [[ "$SRV_PID" -ne 0 ]] && { kill "$SRV_PID" 2>/dev/null || true;
                              wait "$SRV_PID" 2>/dev/null || true; }   # issue #15 teardown noise
  SRV_PID=0
}
trap stop_decode EXIT

MGR_PID=0
start_manager() {
  USPK_DECODE_URL="http://127.0.0.1:$DECODE_PORT" \
  USPK_PREFILL_URL="$VLLM_URL" \
  USPK_BRIDGE_DIR="$BRIDGE_DIR" \
  USPK_STREAM_HOST="$SPARK_HOST" USPK_STREAM_PORT=52901 \
    python3 "$REPO/manager/uspk_manager.py" > "$OUT/manager.log" 2>&1 &
  MGR_PID=$!
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$MGR_PORT/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  fail "manager failed to start (see $OUT/manager.log)"
}
stop_manager() {
  [[ "$MGR_PID" -ne 0 ]] && { kill "$MGR_PID" 2>/dev/null || true;
                              wait "$MGR_PID" 2>/dev/null || true; }
  MGR_PID=0
}
trap 'stop_manager; stop_decode' EXIT

echo "=== phase 1: baseline local prefill (no blob) ==="
start_decode
# warm with an UNRELATED prompt so llama-server's own cache is hot but empty
# for the measured prompt
curl -sf "http://127.0.0.1:$DECODE_PORT/completion" -H 'Content-Type: application/json' \
     -d '{"prompt":"warmup hello","n_predict":1,"temperature":0}' >/dev/null
python3 e2e_ttft.py "http://127.0.0.1:$DECODE_PORT" "$OUT/fox200.txt" > "$OUT/fox200.local.json"
stop_decode

echo "=== phase 2: cold orchestration via :$MGR_PORT (one curl) ==="
start_decode
start_manager
python3 e2e_ttft.py "http://127.0.0.1:$MGR_PORT" "$OUT/fox200.txt" > "$OUT/fox200.cold.json"

echo "=== phase 3: warm (blob already local) ==="
python3 e2e_ttft.py "http://127.0.0.1:$MGR_PORT" "$OUT/fox200.txt" > "$OUT/fox200.warm.json"

echo "=== phase 4: follow-up prompt (fresh blob, fire-and-forget pattern) ==="
python3 e2e_ttft.py "http://127.0.0.1:$MGR_PORT" "$OUT/followup.txt" > "$OUT/followup.cold.json"

echo "=== phase 4b: real-client smoke (curl must GET bytes + a status line) ==="
# issue #19 regression: the manager can complete requests server-side while
# the client receives nothing. This is checked from the CLIENT side only —
# curl's own http_code + size_download — with a realistic timeout, plus a
# total-latency bound that fails if orchestration stalls the relay.
SMOKE_PROMPT="A fresh smoke-test prompt about zebras and telescopes."
SMOKE=$(curl -sS -m 45 -o "$OUT/smoke.body" -w '%{http_code} %{size_download} %{time_total}' \
    "http://127.0.0.1:$MGR_PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$SMOKE_PROMPT"), \"max_tokens\": 8, \"temperature\": 0}") \
  || fail "curl (client) timed out / connection died with no response"
read -r SMOKE_CODE SMOKE_BYTES SMOKE_T <<< "$SMOKE"
echo "smoke: http=$SMOKE_CODE bytes=$SMOKE_BYTES time=${SMOKE_T}s"
[[ "$SMOKE_CODE" == "200" ]] || fail "client saw HTTP $SMOKE_CODE, not 200"
[[ "${SMOKE_BYTES%%.*}" -gt 0 ]] || fail "client received 0 bytes (issue #19)"
python3 - "$OUT/smoke.body" <<'EOF' || fail "client body has no completion text"
import json, sys
obj = json.load(open(sys.argv[1]))
assert obj.get("choices") and obj["choices"][0].get("text", "").strip(), obj
EOF
awk -v t="$SMOKE_T" 'BEGIN { exit !(t < 30.0) }' \
  || fail "smoke took ${SMOKE_T}s — orchestration is stalling the relay"

if [[ "$MODEL" == "27b" ]]; then
  echo "=== phase 5 (27B): ~16k prompt — local baseline vs cold/warm bridge ==="
  stop_decode; start_decode     # drop llama-server prompt cache for cold local run
  curl -sf "http://127.0.0.1:$DECODE_PORT/completion" -H 'Content-Type: application/json' \
       -d '{"prompt":"warmup hello","n_predict":1,"temperature":0}' >/dev/null
  python3 e2e_ttft.py "http://127.0.0.1:$DECODE_PORT" "$OUT/fox3200.txt" > "$OUT/fox3200.local.json"
  stop_decode; start_decode     # cold bridge dir + fresh ctx for the bridged run
  stop_manager; start_manager
  python3 e2e_ttft.py "http://127.0.0.1:$MGR_PORT" "$OUT/fox3200.txt" > "$OUT/fox3200.cold.json"
  python3 e2e_ttft.py "http://127.0.0.1:$MGR_PORT" "$OUT/fox3200.txt" > "$OUT/fox3200.warm.json"
fi

echo "=== decode log: restore evidence ==="
grep -E "uspk: bridge" "$OUT/decode.log" | tail -5 || true
echo "=== manager log: orchestration evidence ==="
grep -E "orch:" "$OUT/manager.log" | tail -5 || true

python3 - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
loc = json.load(open(f"{out}/fox200.local.json"))
cold = json.load(open(f"{out}/fox200.cold.json"))
warm = json.load(open(f"{out}/fox200.warm.json"))
fu = json.load(open(f"{out}/followup.cold.json"))

print(f"TTFT local-prefill : {loc['ttft_ms']:8.0f} ms")
print(f"TTFT cold (1 curl) : {cold['ttft_ms']:8.0f} ms  (incl. Spark prefill + blob pull)")
print(f"TTFT warm          : {warm['ttft_ms']:8.0f} ms")
print(f"TTFT follow-up     : {fu['ttft_ms']:8.0f} ms")

assert cold["text"].strip(), "FAIL: cold orchestration produced no output"
assert warm["text"].strip(), "FAIL: warm request produced no output"
assert fu["text"].strip(),  "FAIL: follow-up request produced no output"

n = min(len(loc["text"]), len(cold["text"]))
same = loc["text"][:n] == cold["text"][:n]
print(f"SANITY bridged-vs-local: first {n} chars "
      f"{'IDENTICAL' if same else 'MISMATCH'}")
if not same:
    print(f"  local  : {loc['text'][:100]!r}")
    print(f"  bridged: {cold['text'][:100]!r}")
    # fp8-vs-bf16 source drift can flip near-ties; require top-5 agreement
    lt = {e["tok"] for e in (loc["top5"] or [])}
    ct = {e["tok"] for e in (cold["top5"] or [])}
    print(f"  top5 overlap: {len(lt & ct)}/5")
    sys.exit(0 if lt & ct else 1)
EOF

if [[ "$MODEL" == "27b" ]]; then
python3 - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
loc = json.load(open(f"{out}/fox3200.local.json"))
cold = json.load(open(f"{out}/fox3200.cold.json"))
warm = json.load(open(f"{out}/fox3200.warm.json"))
print(f"TTFT 16k local-prefill : {loc['ttft_ms']:8.0f} ms")
print(f"TTFT 16k cold (1 curl) : {cold['ttft_ms']:8.0f} ms  (incl. Spark prefill + blob pull)")
print(f"TTFT 16k warm          : {warm['ttft_ms']:8.0f} ms")
n = min(len(loc["text"]), len(cold["text"]))
same = loc["text"][:n] == cold["text"][:n]
print(f"SANITY 16k bridged-vs-local: first {n} chars "
      f"{'IDENTICAL' if same else 'MISMATCH'}")
if not same:
    print(f"  local  : {loc['text'][:100]!r}")
    print(f"  bridged: {cold['text'][:100]!r}")
    lt = {e["tok"] for e in (loc["top5"] or [])}
    ct = {e["tok"] for e in (cold["top5"] or [])}
    print(f"  top5 overlap: {len(lt & ct)}/5")
    sys.exit(0 if lt & ct else 1)  # binding, same rule as the 2k gate
EOF
fi

echo
echo "PASS - results in $OUT"
