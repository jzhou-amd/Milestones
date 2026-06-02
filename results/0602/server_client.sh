#!/bin/bash

# Simplified online profiling setup
# Based on run_docker_vllm.sh but simplified for quick profiling iterations

set -e

CACHE_DIR="/data/jenzhou/vllm_cache"
PROJECT_DIR="${HOME}/vLLM_profiling"
MODEL="Qwen/Qwen2-7B-Instruct"
MODEL_NAME="Qwen2-7B-Instruct"

echo "=== Simplified Online vLLM Profiling Setup ==="
echo "Model: $MODEL"
echo "Instrumented files from: ${PROJECT_DIR}/vllm-source"
echo ""

# Cleanup existing containers
echo "Cleaning up existing containers..."
docker container rm -f docker-vllm-server 2>/dev/null || true
docker container rm -f docker-vllm-client 2>/dev/null || true

# -----------------------------
# Start SERVER
# -----------------------------
echo ""
echo "=== Starting vLLM Server ==="
# need hugginface token 

docker run --name docker-vllm-server \
  --entrypoint /bin/bash \
  --gpus all \
  --network host \
#   -e HF_TOKEN="$HF_TOKEN" \
  -e PYTHONUNBUFFERED=1 \
  -v ${CACHE_DIR}:/models/hfcache \
  -v ${PROJECT_DIR}/output:/vllm-workspace/output \
  -v ${PROJECT_DIR}/vllm-source:/usr/local/lib/python3.12/dist-packages/vllm \
  vllm/vllm-openai:latest \
  -lc "
    vllm serve ${MODEL} \
      --host 0.0.0.0 \
      --port 8001 \
      --gpu-memory-utilization 0.20 \
      --enable-prefix-caching \
      --kv-offloading-size 64 \
      --kv-offloading-backend native \
      --disable-hybrid-kv-cache-manager \
      --max-num-seqs 1 \
      | tee /vllm-workspace/output/server_${MODEL_NAME}.log
  " &

SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"

# -----------------------------
# Wait for server readiness
# -----------------------------
echo ""
echo "Waiting for server to become ready..."
SERVER_URL="http://localhost:8001"

until curl -s "$SERVER_URL/v1/models" | grep -q "id"; do
  printf "."
  sleep 2
done

echo ""
echo "✓ Server ready!"

# -----------------------------
# Run CLIENT
# -----------------------------
echo ""
echo "=== Running Client Requests ==="


docker run --name docker-vllm-client \
  --entrypoint bash \
  --network host \
  -v ${PROJECT_DIR}/docker_scripts:/vllm-workspace/scripts \
  vllm/vllm-openai:latest \
  -c "mkdir -p /vllm-workspace/scripts && /vllm-workspace/scripts/online_client_sweep.sh"

echo ""
echo "=== Profiling Complete ==="
echo ""
echo "Check output files:"
echo "  - Server logs: ${PROJECT_DIR}/output/server_${MODEL_NAME}.log"
echo "  - Profiler output: ${PROJECT_DIR}/output/request_timeline.json"
echo ""
echo "Look for [PROFILER] lines in server logs!"
echo ""
echo "To stop server: docker stop docker-vllm-server"
echo "To cleanup: docker rm -f docker-vllm-server docker-vllm-client"
