#!/bin/sh
# Download the models into /root/models and check them against Hugging Face's SHA-256.
#   fetch-models.sh pool        the coordinator: IQ3_S-mtp (12.1 GB) + the vision projector (0.9 GB)
#   fetch-models.sh standalone  both consoles: IQ3_XXS-mtp (10.4 GB), for when one runs alone
# The worker needs no copy of the pooled model: the coordinator sends it its layers over RPC.
# The web UI hides files ending in "-mtp" (it takes them for MTP draft sidecars), so they're
# renamed to "_mtp" afterwards.
set -e
B=https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main
mkdir -p /root/models && cd /root/models
get() {
  wget -c -nv "$B/$1"
  echo "$2  $1" | sha256sum -c
}
case "$1" in
  pool)
    get Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf 58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f
    get mmproj-Qwen3.8-27B-BF16.gguf 13cb7bebccbd04afc8f4090cb949ecf8937cdf7377c5799b1a0c594e7c0d3e16
    mv Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf Qwen3.8-27B-GSQ-RCO-IQ3_S_mtp.gguf ;;
  standalone)
    get Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf 63f29a2189a6b4cc31f81e093d3856ad293a5583114439f41ae1ed7af4093262
    mv Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf Qwen3.8-27B-GSQ-RCO-IQ3_XXS_mtp.gguf ;;
  *) echo "usage: fetch-models.sh pool|standalone"; exit 2 ;;
esac
