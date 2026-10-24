#!/bin/sh
# Pooled mode: this PS5 coordinates and runs the first layers on its own GPU; the worker PS5
# (ggml-rpc-server on its link address from /etc/llm-cluster.yaml) holds the rest. Same build as standalone mode.
# Memory per console is ~12.5-13 GiB, so:
#  -ts 60,40           index 0 is the worker (RPC0 takes the first layers); the coordinator keeps the head,
#                      the MTP draft layer and vision, so it gets the smaller share.
#  --mmproj-device     keep vision on this console (it otherwise follows the first device, the worker).
#  --cache-ram 0, --ctx-checkpoints 2   the prompt cache and checkpoints live in host RAM (~1 GB free here);
#                      the defaults (8 GiB, 32 x 150 MiB) got llama-server OOM-killed at ~37K tokens.
#  -np 2 -kvu          two slots sharing one 262K KV pool: a side request (a title, a subagent) runs in
#                      the second slot instead of evicting the main conversation's context.
#  -sps 0.5            a slot is reused only when half the new prompt matches it; at the default 0.10 a
#                      short side request matched the main slot on template tokens alone and wiped it.
#  _mtp, --alias       the web UI hides files ending in "-mtp" as MTP draft sidecars, so the HF file
#                      (...IQ3_S-mtp.gguf) is renamed; the alias names it without the /root/models path.
cd /root/llm/master
LD_LIBRARY_PATH=/root/llm/master/build/bin
export LD_LIBRARY_PATH
exec ./build/bin/llama-server \
  -m /root/models/Qwen3.8-27B-GSQ-RCO-IQ3_S_mtp.gguf --alias Qwen3.8-27B-GSQ-RCO-IQ3_S_mtp \
  --mmproj /root/models/mmproj-Qwen3.8-27B-BF16.gguf --mmproj-device Vulkan0 \
  --rpc "$(/root/bin/llm-cluster get peer.link):50052" \
  -ngl 99 -ts 60,40 --load-mode none \
  -np 2 -kvu -fa on -c 262144 -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 --cache-ram 0 --ctx-checkpoints 2 -sps 0.5 \
  --host 0.0.0.0 --port 1234 \
  --temp 1 --top-k 20 --min-p 0 --top-p 0.95 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 128
