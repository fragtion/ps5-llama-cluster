#!/bin/sh
# Standalone mode: this PS5 alone serves the smaller 27B quant (no vision; the projector doesn't fit beside it).
# Loaded at start, under a fixed name, so the gateway can route to it without a load step.
#  --cache-ram 0, --ctx-checkpoints 2   the prompt cache and checkpoints live in host RAM, which is the scarce part.
cd /root/llm/master
LD_LIBRARY_PATH=/root/llm/master/build/bin
export LD_LIBRARY_PATH
exec ./build/bin/llama-server \
  -m /root/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS_mtp.gguf --alias Qwen3.8-27B-GSQ-RCO-IQ3_XXS_mtp \
  -ngl 99 \
  -np 1 -fa on -c 60000 -ctk q4_0 -ctv q4_0 -b 1024 -ub 512 --cache-ram 0 --ctx-checkpoints 2 \
  --host 0.0.0.0 --port 1234 \
  --temp 1 --top-k 20 --min-p 0 --top-p 0.95 \
  --spec-type draft-mtp --spec-draft-n-max 3
