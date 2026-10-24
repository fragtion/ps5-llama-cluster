#!/bin/sh
# Build llama.cpp with the Vulkan and RPC backends in /root/llm/master. One build serves standalone,
# pooled and worker mode; build the same commit on both consoles so the RPC protocol matches.
# REF defaults to the upstream commit this guide was tested on. For ternary (PQ2_0/PTQ1_0) models use
#   REPO=https://github.com/fragtion/llama.cpp REF=ternary-pq2-ptq build-llama.sh
# SCHED_MAX_COPIES=2 halves the pipeline-parallel compute buffers (default 4) at no measured speed cost.
set -e
REPO=${REPO:-https://github.com/ggml-org/llama.cpp}
REF=${REF:-5f5acfe2504b490386cd03020202f59e36861add}
mkdir -p /root/llm/master /root/logs
cd /root/llm/master
[ -d .git ] || git init -q
git fetch -q --depth 1 "$REPO" "$REF"
git checkout -q --detach FETCH_HEAD
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_RPC=ON -DGGML_SCHED_MAX_COPIES=2 \
  > /root/logs/build.log 2>&1
cmake --build build --config Release -j "$(nproc)" >> /root/logs/build.log 2>&1
ls -l build/bin/llama-server build/bin/ggml-rpc-server
