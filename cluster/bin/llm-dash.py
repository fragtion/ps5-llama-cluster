#!/usr/bin/env python3
"""A small web dashboard for the PS5 pair: http://<either console>:50080/

Each console samples itself every 5 s (temperature, CPU, memory, GPU memory, disk and network
throughput, busy llama.cpp slots) and keeps 6 h of history in memory. The page asks the console it
was opened on, which fetches the other console's data over the link, so either address shows both.

Reading is open to the LAN. The buttons (fan, llama.cpp mode, model sync) need the gateway's
LITELLM_MASTER_KEY from /etc/litellm/env; an action for the other console is passed on to it.
Nothing here can reboot a console.
"""
import collections
import hmac
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import llm_cluster

CLUSTER = llm_cluster.load()
NODE = CLUSTER["self"]["name"]
ROLE = CLUSTER["self"]["role"]
PEER = CLUSTER["peer"]
LINK_IF = CLUSTER["link"]["interface"]
PORT = int(os.environ.get("PORT", "50080"))
HERE = Path(__file__).resolve().parent
GPU = Path("/sys/class/drm/card0/device")
FAN = Path("/sys/devices/platform/ps5-fan/target_temp")

INTERVAL = 5
HISTORY = collections.deque(maxlen=6 * 3600 // INTERVAL)
KEYS = ("t", "temp", "cpu", "load", "mem", "swap", "vram", "gtt",
        "disk_r", "disk_w", "link_rx", "link_tx", "wifi_rx", "wifi_tx", "busy")
POOLED_UNIT = "llama-pool" if ROLE == "coordinator" else "ggml-rpc"
POOLED_MODE = "pool" if ROLE == "coordinator" else "worker"
UNITS = ("llm-peer", POOLED_UNIT, "llama-qwen", "litellm", "keepalived", "llm-config-watch",
         "ps5fan", "ps5boost")


def read(path, default=""):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def chip_temp():
    for h in Path("/sys/class/hwmon").iterdir():
        if read(h / "name") == "k10temp":
            return int(read(h / "temp1_input", "0")) / 1000
    return None


def meminfo():
    m = {}
    for line in read("/proc/meminfo").splitlines():
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0]) * 1024
    return m


def wifi_if():
    return next((p.name for p in Path("/sys/class/net").iterdir() if (p / "wireless").exists()), None)


WIFI_IF = wifi_if()


def counters():
    """Monotonic counters for the rates: CPU jiffies, sectors and bytes."""
    cpu = [int(x) for x in read("/proc/stat").splitlines()[0].split()[1:]]
    disk = (0, 0)
    for line in read("/proc/diskstats").splitlines():
        f = line.split()
        if f[2] == "sda":
            disk = (int(f[5]) * 512, int(f[9]) * 512)
    net = {}
    for line in read("/proc/net/dev").splitlines()[2:]:
        name, rest = line.split(":", 1)
        f = rest.split()
        net[name.strip()] = (int(f[0]), int(f[8]))
    return time.monotonic(), cpu, disk, net


def slots():
    try:
        with urllib.request.urlopen("http://127.0.0.1:1234/slots", timeout=2) as r:
            return json.load(r)
    except Exception:
        return None


def sample(prev, now):
    dt = now[0] - prev[0]
    busy_cpu = [a - b for a, b in zip(now[1], prev[1])]
    idle = busy_cpu[3] + busy_cpu[4]  # idle + iowait
    total = sum(busy_cpu) or 1
    rate = lambda a, b: round(max(a - b, 0) / dt / 1e6, 2)  # MB/s
    net = lambda i, k: rate(now[3].get(i, (0, 0))[k], prev[3].get(i, (0, 0))[k])
    m = meminfo()
    s = slots()
    row = {
        "t": int(time.time()),
        "temp": chip_temp(),
        "cpu": round(100 * (total - idle) / total, 1),
        "load": float(read("/proc/loadavg", "0").split()[0]),
        "mem": round(100 * (1 - m["MemAvailable"] / m["MemTotal"]), 1),
        "swap": round((m["SwapTotal"] - m["SwapFree"]) / 2**20),
        "vram": round(int(read(GPU / "mem_info_vram_used", "0")) / 2**20),
        "gtt": round(int(read(GPU / "mem_info_gtt_used", "0")) / 2**20),
        "disk_r": rate(now[2][0], prev[2][0]),
        "disk_w": rate(now[2][1], prev[2][1]),
        "link_rx": net(LINK_IF, 0), "link_tx": net(LINK_IF, 1),
        "wifi_rx": net(WIFI_IF, 0), "wifi_tx": net(WIFI_IF, 1),
        "busy": sum(1 for x in s if x.get("is_processing")) if s else None,
    }
    return [row[k] for k in KEYS]


def collector():
    prev = counters()
    while True:
        time.sleep(INTERVAL)
        now = counters()
        try:
            HISTORY.append(sample(prev, now))
        except Exception as e:  # one bad read mustn't stop the history
            print(time.strftime("%F %T"), "sample failed:", e, flush=True)
        prev = now


def unit(u):
    return subprocess.run(["systemctl", "is-active", f"{u}.service"],
                          capture_output=True, text=True).stdout.strip()


_fan = {"at": 0.0, "target": None}


def fan():
    # each read is a query to the console's controller: at most every 15 s
    if time.monotonic() - _fan["at"] > 15:
        _fan.update(at=time.monotonic(), target=int(read(FAN, "0")) or None)
    return {"target": _fan["target"],
            "mode": read("/run/ps5-fan/mode", "auto"),
            "boot_target": int(read("/run/ps5-fan/boot-target", "0")) or None}


def peerd():
    try:
        with urllib.request.urlopen("http://127.0.0.1:50060/status", timeout=2) as r:
            return json.load(r)
    except Exception:
        return None


def disks():
    out = []
    for path in ("/", "/mnt/llamacache"):
        if path == "/" or os.path.ismount(path):
            u = shutil.disk_usage(path)
            out.append({"path": path, "total": u.total, "used": u.used, "free": u.free})
    return out


def models():
    d = Path("/root/models")
    files = sorted(d.glob("*.gguf")) if d.is_dir() else []
    return [{"name": f.name, "size": f.stat().st_size} for f in files]


def rpc_cache():
    d = Path("/root/.cache/llama.cpp/rpc")
    if not d.is_dir():
        return None
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file())


def info():
    cfg = llm_cluster.load()  # the YAML may have changed since start
    vip = str(cfg["gateway"]["vip"]).split("/")[0]
    addrs = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
    s = slots()
    m = meminfo()
    return {
        "node": NODE, "role": ROLE, "pooled_mode": POOLED_MODE,
        "uptime": int(float(read("/proc/uptime", "0").split()[0])),
        "units": {u: unit(u) for u in UNITS},
        "peerd": peerd(),
        "gateway": {"vip": vip, "here": f" {vip}/" in addrs},
        "fan": fan(),
        "models_sync": llm_cluster.get(cfg, "models.sync", "false") == "true",
        "cache_choice": read("/run/llm-cache-choice") or None,
        "rpc_cache": rpc_cache(),
        "disks": disks(),
        "models": models(),
        "mem_total": m["MemTotal"], "swap_total": m["SwapTotal"],
        "vram_total": int(read(GPU / "mem_info_vram_total", "0")),
        "gtt_total": int(read(GPU / "mem_info_gtt_total", "0")),
        "slots": [{"n_ctx": x.get("n_ctx"), "busy": x.get("is_processing")} for x in s] if s else None,
    }


def snapshot(since):
    return {"info": info(), "keys": KEYS, "rows": [r for r in list(HISTORY) if r[0] > since]}


def from_peer(path, body=None, headers=None, timeout=3):
    req = urllib.request.Request(f"http://{PEER['link']}:{PORT}{path}", data=body,
                                 headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def master_key():
    m = re.search(r"^LITELLM_MASTER_KEY=(.*)$", read("/etc/litellm/env"), re.M)
    return m.group(1).strip().strip("'\"") if m else None


def run_action(action, value):
    """The commands behind the buttons; each is also a command you can type."""
    if action == "fan":
        if value in ("auto", "max"):
            cmd = ["/root/bin/ps5-fan", value]
        elif str(value).isdigit() and 40 <= int(value) <= 95:
            cmd = ["/root/bin/ps5-fan", "target", str(value)]
        else:
            return 400, "fan: auto, max, or a target of 40-95 °C"
        _fan["at"] = 0.0
    elif action == "mode":
        if value not in ("auto", "standalone", "off", POOLED_MODE):
            return 400, f"mode: auto, {POOLED_MODE}, standalone or off"
        cmd = ["/root/bin/llm-mode", value]
    elif action == "models.sync":
        if value not in ("true", "false"):
            return 400, "models.sync: true or false"
        cmd = ["/root/bin/llm-cluster", "set", "models.sync", value]
    else:
        return 400, f"unknown action {action!r}"
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    print(time.strftime("%F %T"), "action:", " ".join(cmd), "->", p.returncode, flush=True)
    return (200 if p.returncode == 0 else 500), (p.stdout + p.stderr).strip()


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        since = int(dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("since", 0) or 0)
        if path == "/":
            self.reply(200, (HERE / "llm-dash.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/self":
            self.reply(200, snapshot(since))
        elif path == "/api/nodes":
            nodes = {NODE: snapshot(since)}
            try:
                nodes[PEER["name"]] = from_peer(f"/api/self?since={since}")
            except Exception as e:
                nodes[PEER["name"]] = {"error": f"unreachable ({type(e).__name__})"}
            self.reply(200, {"self": NODE, "nodes": nodes})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/action":
            return self.reply(404, {"error": "not found"})
        key = master_key()
        given = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not key or not hmac.compare_digest(given.encode(), key.encode()):
            return self.reply(403, {"error": "wrong key"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except ValueError:
            return self.reply(400, {"error": "bad JSON"})
        if req.get("node", NODE) != NODE:
            try:
                return self.reply(200, from_peer("/api/action", json.dumps(req | {"node": PEER["name"]}).encode(),
                                                 {"Authorization": f"Bearer {given}",
                                                  "Content-Type": "application/json"}, timeout=70))
            except urllib.error.HTTPError as e:  # the peer answered: pass its answer on
                return self.reply(e.code, e.read())
            except Exception as e:
                return self.reply(502, {"error": f"{PEER['name']} unreachable: {e}"})
        code, out = run_action(req.get("action"), req.get("value"))
        self.reply(code, {"ok": code == 200, "output": out})

    def log_message(self, *args):
        pass


def main():
    threading.Thread(target=collector, daemon=True).start()
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.daemon_threads = True
    print(time.strftime("%F %T"), f"{NODE}: dashboard on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
