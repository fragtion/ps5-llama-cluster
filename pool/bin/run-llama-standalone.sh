#!/bin/sh
# Standalone mode: this PS5 alone serves the smaller 27B quant (no vision; the projector doesn't fit beside it).
# Loaded at start, under a fixed name, so the gateway can route to it without a load step.
#  --cache-ram 0, --ctx-checkpoints 1   the prompt cache and checkpoints live in host RAM, which is the scarce part.
cd /root/llm/master
LD_LIBRARY_PATH=/root/llm/master/build/bin
export LD_LIBRARY_PATH
# Cap llama.cpp's f16 K/V scratch for flash attention (pool/patches/vulkan-fa-dequant-kv-cap.patch).
# Uncapped it grows by 4 KiB per context token and never shrinks: ~670 MB at 164K ran the consoles
# out of RAM. Up to ~32K of context prompt reads keep the faster path.
GGML_VK_FA_DEQUANT_KV_MAX_MB=128
export GGML_VK_FA_DEQUANT_KV_MAX_MB
exec ./build/bin/llama-server \
  -m /root/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS_mtp.gguf --alias Qwen3.8-27B-GSQ-RCO-IQ3_XXS_mtp \
  -ngl 99 \
  -np 1 -fa on -c 60000 -ctk q4_0 -ctv q4_0 -b 1024 -ub 512 --cache-ram 0 --ctx-checkpoints 1 \
  --host 0.0.0.0 --port 1234 \
  --temp 1 --top-k 20 --min-p 0 --top-p 0.95 \
  --spec-type draft-mtp --spec-draft-n-max 3
