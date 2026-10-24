#!/bin/sh
# Worker mode: lend this PS5's GPU to the coordinator over llama.cpp RPC, on the ethernet link only
# (/etc/llm-cluster.yaml; ggml-rpc.service.d/link.conf lets only the coordinator connect).
# Weight cache, so later pool starts read the received weights from a local drive instead of the
# network. Checked at every start, first match wins:
#   1. a drive labelled LLAMACACHE (mounted at /mnt/llamacache) on a USB 3 link: move it to
#      whichever console is the worker; the coordinator never uses it. On USB 2 (~20-40 MB/s) it
#      is slower than the network, so it is skipped.
#   2. /root/.cache/llama.cpp/rpc on this console's own drive, if that folder exists
#   3. no cache: the coordinator sends the weights over the link
# The choice is written to /run/llm-cache-choice for `llm-mode status`.
cd /root/llm/master
LD_LIBRARY_PATH=/root/llm/master/build/bin
export LD_LIBRARY_PATH
[ -e /dev/disk/by-label/LLAMACACHE ] && ! mountpoint -q /mnt/llamacache && mount /mnt/llamacache
speed=0
if mountpoint -q /mnt/llamacache; then
  p=$(readlink -f /sys/class/block/"$(basename "$(findmnt -no SOURCE /mnt/llamacache)")")
  while [ "$p" != /sys ] && [ ! -f "$p/speed" ]; do p=${p%/*}; done   # up to the USB device
  [ -f "$p/speed" ] && speed=$(cat "$p/speed")
fi
cache=-c
if [ "${speed%%.*}" -ge 5000 ]; then
  LLAMA_CACHE=/mnt/llamacache   # the server adds /rpc
  export LLAMA_CACHE
  choice="LLAMACACHE drive (/mnt/llamacache/rpc, ${speed} Mbit/s USB)"
elif [ -d /root/.cache/llama.cpp/rpc ]; then
  choice="/root/.cache/llama.cpp/rpc"
else
  cache=
  choice="none (the weights come over the link)"
fi
mountpoint -q /mnt/llamacache && [ -z "$LLAMA_CACHE" ] && choice="$choice; LLAMACACHE skipped: ${speed} Mbit/s USB is slower than the network"
echo "weight cache: $choice" | tee /run/llm-cache-choice
exec ./build/bin/ggml-rpc-server -H "$(/root/bin/llm-cluster get self.link)" -p 50052 $cache
