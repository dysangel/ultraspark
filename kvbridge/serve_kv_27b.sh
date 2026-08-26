#!/usr/bin/env bash
# vLLM OpenAI server with StreamConnector — Qwen3.8-27B FP8 (issue #23).
# Deployed on the Spark as ~/kvbridge/serve_kv_27b.sh.
export PATH=$HOME/vllm-spike/bin:$PATH
export PYTHONPATH=$HOME/kvbridge
exec vllm serve $HOME/models/qwen38-27b-fp8 \
  --port 8081 --gpu-memory-utilization 0.85 --enforce-eager \
  --no-enable-chunked-prefill \
  --max-model-len 32768 --max-num-batched-tokens 32768 \
  --kv-transfer-config "{\"kv_connector\":\"StreamConnector\",\"kv_role\":\"kv_both\",\"kv_connector_module_path\":\"stream_connector\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"$HOME/kvbridge/dumps\",\"stream_port\":52901}}"
