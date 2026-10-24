#!/bin/sh
# Worker mode: lend this PS5's GPU to the coordinator over llama.cpp RPC, on the ethernet link only
# (/etc/llm-cluster.yaml; ggml-rpc.service.d/link.conf lets only the coordinator connect).
# -c keeps the received weights in /root/.cache/llama.cpp/rpc, so later loads skip the network.
cd /root/llm/master
LD_LIBRARY_PATH=/root/llm/master/build/bin
export LD_LIBRARY_PATH
exec ./build/bin/ggml-rpc-server -H "$(/root/bin/llm-cluster get self.link)" -p 50052 -c
