#!/usr/bin/env python3
"""This node's view of /etc/llm-cluster.yaml, for the PS5 scripts.

  llm-cluster get <key> [default]
                          print a value. "self." and "peer." name this node and the other one,
                          e.g. self.role, peer.link, link.interface, gateway.vip, models.sync
  llm-cluster ensure      apply this node's part of the file: the link address (NetworkManager
                          profile and live), /etc/hosts, the RPC server's firewall, keepalived.conf,
                          and with models.sync on, removing the RPC weight cache
"""
import ipaddress
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import yaml

PATH = "/etc/llm-cluster.yaml"
RPC_DROPIN = Path("/etc/systemd/system/ggml-rpc.service.d/link.conf")
KEEPALIVED = Path("/etc/keepalived/keepalived.conf")
HOSTS = Path("/etc/hosts")
RPC_CACHE = Path("/root/.cache/llama.cpp/rpc")


def load():
    with open(PATH) as f:
        cfg = yaml.safe_load(f)
    me = socket.gethostname()
    if me not in cfg["nodes"]:
        sys.exit(f"{me} is not a node in {PATH}")
    peer = next(n for n in cfg["nodes"] if n != me)
    cfg["self"] = dict(cfg["nodes"][me], name=me)
    cfg["peer"] = dict(cfg["nodes"][peer], name=peer)
    return cfg


def get(cfg, key, default=None):
    value = cfg
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            if default is None:
                sys.exit(f"{key} is not set in {PATH}")
            return default
        value = value[part]
    return str(value).lower() if isinstance(value, bool) else value


def coordinator(cfg):
    return next(n for n in cfg["nodes"].values() if n["role"] == "coordinator")


def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def write_if_changed(path, text):
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def ensure_link(cfg):
    iface = cfg["link"]["interface"]
    cidr = f"{cfg['self']['link']}/{ipaddress.ip_network(cfg['link']['subnet']).prefixlen}"
    conn = run("nmcli", "-g", "GENERAL.CONNECTION", "device", "show", iface)
    # in the profile, so NetworkManager keeps it across reconnects (DHCP still provides the LAN address)
    if conn and run("nmcli", "-g", "ipv4.addresses", "connection", "show", conn) != cidr:
        run("nmcli", "connection", "modify", conn, "ipv4.addresses", cidr)
    if cidr not in run("ip", "-4", "-o", "addr", "show", "dev", iface):
        run("ip", "addr", "replace", cidr, "dev", iface)


def ensure_hosts(cfg):
    # the node names resolve to their link addresses (ssh ps5b, llm-sync); 127.0.1.1 stays as is
    lines = [l for l in HOSTS.read_text().splitlines()
             if l.startswith("127.") or not set(l.split()[1:]) & set(cfg["nodes"])]
    at = next((i + 1 for i, l in enumerate(lines) if l.startswith("127.0.1.1")), len(lines))
    lines[at:at] = [f"{n['link']}\t{name}" for name, n in cfg["nodes"].items()]
    write_if_changed(HOSTS, "\n".join(lines) + "\n")


def ensure_rpc_firewall(cfg):
    text = ("# Written by `llm-cluster ensure` from /etc/llm-cluster.yaml; edit the YAML instead.\n"
            "# RPC has no authentication: only the coordinator may connect.\n"
            "[Service]\n"
            "IPAddressAllow=\n"
            "IPAddressDeny=any\n"
            f"IPAddressAllow={coordinator(cfg)['link']} localhost\n")
    if write_if_changed(RPC_DROPIN, text):
        run("systemctl", "daemon-reload")


def ensure_keepalived(cfg):
    gw = cfg["gateway"]
    text = f"""# Written by `llm-cluster ensure` from /etc/llm-cluster.yaml; edit the YAML instead.
# One console holds the gateway address {gw['vip']}; gateway-notify.sh runs LiteLLM on that one.
global_defs {{
    router_id {cfg['self']['name']}
    enable_script_security
    script_user root
}}

vrrp_instance gateway {{
    state BACKUP
    nopreempt                  # whoever holds the address keeps it; no needless LiteLLM restarts
    interface {gw['interface']}
    virtual_router_id {gw['vrid']}
    priority {150 if cfg['self']['role'] == 'worker' else 100}   # higher on the worker: the coordinator's RAM is tighter
    advert_int 1
    unicast_src_ip {cfg['self']['link']}
    unicast_peer {{
        {cfg['peer']['link']}
    }}
    virtual_ipaddress {{
        {gw['vip']} dev {gw['interface']}
    }}
    notify /root/bin/gateway-notify.sh
}}
"""
    if write_if_changed(KEEPALIVED, text) and run("systemctl", "is-active", "keepalived") == "active":
        run("systemctl", "reload", "keepalived")


def ensure_model_sync(cfg):
    # models.sync replaces the RPC weight cache: free its space before the models arrive
    if get(cfg, "models.sync", "false") == "true" and RPC_CACHE.exists():
        size = sum(f.stat().st_size for f in RPC_CACHE.iterdir()) / 1e9
        shutil.rmtree(RPC_CACHE)
        print(f"models.sync is on: deleted the RPC weight cache {RPC_CACHE} ({size:.1f} GB)")


def ensure(cfg):
    ensure_link(cfg)
    ensure_hosts(cfg)
    ensure_rpc_firewall(cfg)
    ensure_keepalived(cfg)
    ensure_model_sync(cfg)


def main(argv):
    cfg = load()
    if len(argv) in (3, 4) and argv[1] == "get":
        print(get(cfg, *argv[2:]))
    elif len(argv) == 2 and argv[1] == "ensure":
        ensure(cfg)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
