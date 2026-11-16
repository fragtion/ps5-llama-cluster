#!/bin/sh
# Worker mode: lend this PS5's GPU to the coordinator over llama.cpp RPC (port 50052).
# -c keeps the received weights in /root/.cache/llama.cpp/rpc, so later loads skip the network.
. /etc/default/llm-pool
cd /root/llm/master
LD_LIBRARY_PATH=/root/llm/master/build/bin
export LD_LIBRARY_PATH
# Cap llama.cpp's f16 K/V scratch for flash attention (pool/patches/vulkan-fa-dequant-kv-cap.patch).
# Uncapped it grows by 4 KiB per context token and never shrinks: ~670 MB at 164K ran the consoles
# out of RAM. Up to ~32K of context prompt reads keep the faster path.
GGML_VK_FA_DEQUANT_KV_MAX_MB=128
export GGML_VK_FA_DEQUANT_KV_MAX_MB
exec ./build/bin/ggml-rpc-server -H "$WORKER" -p 50052 -c
