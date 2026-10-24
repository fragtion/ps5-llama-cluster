#!/bin/sh
# Worker mode: lend this PS5's GPU to the coordinator over llama.cpp RPC (port 50052).
# -c keeps the received weights in /root/.cache/llama.cpp/rpc, so later loads skip the network.
. /etc/default/llm-pool
cd /root/llm/master
LD_LIBRARY_PATH=/root/llm/master/build/bin
export LD_LIBRARY_PATH
exec ./build/bin/ggml-rpc-server -H "$WORKER" -p 50052 -c
