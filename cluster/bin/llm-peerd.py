#!/usr/bin/env python3
"""Run this PS5's llama.cpp pooled while the other PS5 is up, standalone otherwise.

The coordinator (ps5a) pools with llama-pool.service once the worker reports its RPC server
ready; the worker (ps5b) lends its GPU with ggml-rpc.service while the coordinator's daemon
answers. Either node falls back to llama-qwen.service (IQ3_XXS, standalone) when the other
is gone. Each node serves its own state as JSON on :50060/status for the other to read.

Role and peer come from /etc/llm-cluster.yaml (see llm_cluster.py). The nodes talk over the
ethernet link addresses listed there, which this daemon keeps on the port.
`llm-mode <mode>` stops this daemon to take manual control; `llm-mode auto` starts it again.
"""
import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import llm_cluster

CLUSTER = llm_cluster.load()
ROLE = CLUSTER["self"]["role"]
PEER = CLUSTER["peer"]["link"]
PORT = int(os.environ.get("PORT", "50060"))

INTERVAL = 10       # seconds between peer checks
UP_CHECKS = 2       # peer seen this many times in a row before pooling
DOWN_CHECKS = 3     # peer missed this many times in a row before going standalone
BOOT_WAIT = 90      # at boot, wait this long for the peer before starting standalone
POOL_BACKOFF = 1800 # the pool failed while the peer was up: stay standalone this long

STANDALONE = "llama-qwen.service"
POOLED = "llama-pool.service" if ROLE == "coordinator" else "ggml-rpc.service"
POOLED_MODE = "pool" if ROLE == "coordinator" else "worker"

status = {"node": socket.gethostname(), "role": ROLE, "mode": "starting",
          "ready": False, "pool_ok": True}


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def unit_state(unit):
    return subprocess.run(["systemctl", "is-active", unit],
                          capture_output=True, text=True).stdout.strip()


def start(unit):
    log(f"starting {unit}")
    subprocess.run(["systemctl", "reset-failed", unit], capture_output=True)
    # the units conflict with each other, so starting one stops the other
    subprocess.run(["systemctl", "start", "--no-block", unit])


def current_mode():
    if unit_state(POOLED) in ("active", "activating"):
        return POOLED_MODE
    if unit_state(STANDALONE) in ("active", "activating"):
        return "standalone"
    return "off"


def is_ready(mode):
    if mode == "worker":
        out = subprocess.run(["ss", "-Hltn", "sport = :50052"], capture_output=True, text=True).stdout
        return bool(out.strip())
    if mode in ("pool", "standalone"):
        try:
            with urllib.request.urlopen("http://127.0.0.1:1234/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False
    return False


def read_peer():
    try:
        with urllib.request.urlopen(f"http://{PEER}:{PORT}/status", timeout=3) as r:
            return json.load(r)
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/status":
            self.send_error(404)
            return
        body = json.dumps(status).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def peer_wants_us(peer):
    """True when the peer's state means this node should be in its pooled mode."""
    if peer is None:
        return False
    if ROLE == "worker":
        # lend the GPU while the coordinator runs its daemon and isn't backing off from the pool
        return peer.get("pool_ok", True)
    return peer.get("mode") == "worker" and peer.get("ready")


def main():
    llm_cluster.ensure(CLUSTER)
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"{status['node']} up as {ROLE}, peer {PEER}")

    up = down = 0
    # monotonic: the PS5 boots with the clock from its last shutdown and jumps when NTP syncs
    boot_deadline = time.monotonic() + BOOT_WAIT
    backoff_until = 0.0
    while True:
        llm_cluster.ensure_link(CLUSTER)  # NetworkManager drops it if the profile is edited and reapplied
        mode = current_mode()
        peer = read_peer()
        if peer_wants_us(peer):
            up, down = up + 1, 0
        else:
            up, down = 0, down + 1

        pool_ok = time.monotonic() >= backoff_until
        if ROLE == "coordinator" and unit_state(POOLED) == "failed":
            subprocess.run(["systemctl", "reset-failed", POOLED], capture_output=True)
            if mode == "off" and peer is not None:
                # the pool gave up (StartLimit) with the worker still there: don't keep resending the weights
                log(f"pool failed with the peer up; standalone for {POOL_BACKOFF // 60} min")
                backoff_until = time.monotonic() + POOL_BACKOFF
                pool_ok = False

        if mode == POOLED_MODE:
            want = "standalone" if down >= DOWN_CHECKS or not pool_ok else POOLED_MODE
        elif up >= UP_CHECKS and pool_ok:
            want = POOLED_MODE
        elif mode == "off" and time.monotonic() < boot_deadline:
            want = "off"  # still giving the peer a chance to show up
        else:
            want = "standalone"

        if want != mode and want != "off":
            log(f"{mode} -> {want} (peer {'up' if peer else 'down'}: {peer and peer.get('mode')})")
            start(POOLED if want == POOLED_MODE else STANDALONE)
            mode = want

        status.update(mode=mode, ready=is_ready(mode), pool_ok=pool_ok,
                      peer=peer and peer.get("mode"), checked=int(time.time()))
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
