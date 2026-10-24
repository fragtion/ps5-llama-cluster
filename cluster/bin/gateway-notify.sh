#!/bin/sh
# keepalived calls this as: <type> <name> <state> <priority>. The console holding the gateway address
# runs LiteLLM; the other stops it, which leaves its few hundred MB to llama-server.
echo "$(date '+%F %T') gateway $3" >> /root/logs/gateway.log
case "$3" in
  MASTER) systemctl start --no-block litellm.service ;;
  *)      systemctl stop --no-block litellm.service ;;
esac
