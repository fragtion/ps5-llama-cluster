#!/usr/bin/env python3
"""A small web dashboard for one PS5 or the pair: http://<console>:50080/

Each console samples itself every 5 s (temperature, CPU, memory, GPU memory, disk and network
throughput, busy llama.cpp slots) and keeps 6 h of history in memory. In a pair (a two-node
/etc/llm-cluster.yaml), the page asks the console it was opened on, which fetches the other
console's data over the link, so either address shows both. Without that file the console is shown
on its own.

The llama.cpp tab follows llama-server's logs (llama-pool.log, llama-server.log) for each request's
prefill and decode timings, MTP draft acceptance and context size, and asks its /slots for what is
running now. Nothing has to be restarted for it.

Reading is open to the LAN. The buttons (fan, llama.cpp mode, model sync) need a key: LLM_DASH_KEY
from /etc/default/llm-dash if it is set, otherwise the gateway's LITELLM_MASTER_KEY from
/etc/litellm/env. An action for the other console is passed on to it. Nothing here can reboot a
console.
"""
import bisect
import collections
import hmac
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import llm_cluster
except ImportError:  # a console on its own needs neither it nor PyYAML
    llm_cluster = None


def cluster():
    """The pair from /etc/llm-cluster.yaml, or None for a console on its own (no file, one node)."""
    if llm_cluster is None:
        return None
    try:
        return llm_cluster.load()
    except (OSError, StopIteration, SystemExit):
        return None


def default_if():
    """The interface of the default route: the one the LAN traffic goes through."""
    m = re.search(r"\bdev (\S+)", subprocess.run(["ip", "route", "show", "default"],
                                                  capture_output=True, text=True).stdout)
    return m[1] if m else next((p.name for p in sorted(Path("/sys/class/net").iterdir()) if p.name != "lo"), "lo")


CLUSTER = cluster()
SINGLE = CLUSTER is None
NODE = socket.gethostname() if SINGLE else CLUSTER["self"]["name"]
ROLE = "standalone" if SINGLE else CLUSTER["self"]["role"]
PEER = None if SINGLE else CLUSTER["peer"]
# the network chart: the console-to-console link in a pair, the LAN otherwise
LINK_IF = default_if() if SINGLE else CLUSTER["link"]["interface"]
PORT = int(os.environ.get("PORT", "50080"))  # also the port the two dashboards talk to each other on
# more ports to answer on, e.g. 80 for http://<console>/ ("" for none)
ALSO_PORTS = [int(p) for p in os.environ.get("ALSO_PORTS", "80").split(",") if p.strip()]
HERE = Path(__file__).resolve().parent
GPU = Path("/sys/class/drm/card0/device")
FAN = Path("/sys/devices/platform/ps5-fan/target_temp")

INTERVAL = 5
HISTORY = collections.deque(maxlen=6 * 3600 // INTERVAL)
KEYS = ("t", "temp", "cpu", "load", "mem", "swap", "vram", "gtt",
        "disk_r", "disk_w", "link_rx", "link_tx", "wifi_rx", "wifi_tx", "busy")
POOLED_UNIT = None if SINGLE else "llama-pool" if ROLE == "coordinator" else "ggml-rpc"
POOLED_MODE = None if SINGLE else "pool" if ROLE == "coordinator" else "worker"
UNITS = (("llama-qwen", "litellm", "ps5fan", "ps5boost") if SINGLE else
         ("llm-peer", POOLED_UNIT, "llama-qwen", "litellm", "keepalived", "llm-config-watch", "ps5fan", "ps5boost"))


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
    track(s)
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


# llama-server log lines: "<min>.<sec>.<ms>.<us> I slot <fn>: id  0 | task 12 | <text>", the time
# counted from the server's start, sometimes after a "[pid] " prefix
LOG_LINE = re.compile(r"^(?:\[\d+\] )?(\d+)\.(\d+)\.(\d+)\.(\d+) \w (.*)$")
TASK = re.compile(r"slot +\w+: id +(\d+) \| task (\d+) \| (.*)$")
NUM = r"([\d.]+)"
PATTERNS = {
    "launch": re.compile(r"processing task, is_child = (\d)"),
    "progress": re.compile(rf"prompt processing, n_tokens = +(\d+), progress = {NUM}, t = +{NUM} s / +{NUM} tokens per second"),
    "prefill": re.compile(rf"prompt eval time = +{NUM} ms / +(\d+) tokens"),
    "decode": re.compile(rf"^ *eval time = +{NUM} ms / +(\d+) tokens"),
    "draft": re.compile(rf"draft acceptance = {NUM} \( *(\d+) accepted / +(\d+) generated\), mean len = +{NUM}"),
    "release": re.compile(r"stop processing: n_tokens = (\d+), truncated = (\d)"),
}
MODEL = re.compile(r"loading model '([^']+)'")
LISTENING = re.compile(r"listening on http")
# a fatal error, e.g. "/root/llm/.../ggml-rpc.cpp:650: Failed to connect to 10.13.37.2:50052" (no time prefix)
FATAL = re.compile(r"\.(?:cpp|c|h):\d+: (.+)$")


class LlamaLog:
    """Follows one llama-server log and turns it into finished requests.

    A server run's times count from its start, and a new run starts again from zero. Lines read as
    they arrive are dated by the clock; when catching up at start, the newest run is dated by the
    file's last write (its last line). Older runs in the file can't be dated and are skipped.
    """

    def __init__(self, path, mode):
        self.path, self.mode = Path(path), mode
        self.offset, self.rest, self.last_rel = 0, "", 0.0
        self.base = None    # wall time of this run's start
        self.model = None
        self.open = {}      # task -> request in progress
        self.done = collections.deque(maxlen=200)
        # the start: is this run serving yet, how long the last good start took, why the last one failed
        self.loaded, self.load_s, self.error, self.failed = True, None, None, None

    def catch_up(self):
        try:
            st = self.path.stat()
        except OSError:
            return
        start = max(st.st_size - 8 * 2**20, 0)  # the last 8 MB is plenty
        with open(self.path, "rb") as f:
            f.seek(start)
            lines = f.read(st.st_size - start).decode("utf-8", "replace").split("\n")
        self.offset, self.rest = st.st_size, lines.pop()
        rels = [(i, rel) for i, l in enumerate(lines) if (rel := self.rel(l)) is not None]
        if not rels:
            return
        first = 0  # the newest run begins after the last time the clock went back
        for (_, prev), (i, rel) in zip(rels, rels[1:]):
            if rel + 1 < prev:
                first = i
        self.base = st.st_mtime - rels[-1][1]
        for line in lines[first:]:
            self.line(line)

    def poll(self):
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.offset:  # truncated or replaced
            self.offset, self.rest, self.last_rel = 0, "", 0.0
        if size == self.offset:
            return
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        self.offset = size
        lines = (self.rest + data.decode("utf-8", "replace")).split("\n")
        self.rest = lines.pop()
        for line in lines:
            self.line(line, live=True)

    @staticmethod
    def rel(line):
        m = LOG_LINE.match(line)
        return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 1e3 + int(m[4]) / 1e6 if m else None

    def line(self, line, live=False):
        m = LOG_LINE.match(line)
        if not m:
            if fm := FATAL.search(line):
                self.error = fm[1].strip()[:200]
            return
        rel = self.rel(line)
        if rel + 1 < self.last_rel or self.base is None:  # a new run
            self.open.clear()
            self.model = None if self.base is not None else self.model
            self.base = time.time() - rel
        elif live and abs(self.base + rel - time.time()) > 5:
            self.base = time.time() - rel  # keep up with clock jumps (NTP after boot)
        self.last_rel = rel
        body = m[5]
        if mm := MODEL.search(body):  # a run starts loading
            self.model = Path(mm[1]).name.removesuffix(".gguf")
            if not self.loaded and self.error:
                self.failed = self.error
            self.loaded, self.error = False, None
            return
        if LISTENING.search(body):
            self.loaded, self.load_s, self.failed = True, round(rel), None
            return
        tm = TASK.search(body)
        if not tm:
            return
        slot, task, text = int(tm[1]), int(tm[2]), tm[3]
        at = self.base + rel
        if pm := PATTERNS["launch"].search(text):
            self.open[task] = {"task": task, "slot": slot, "start": at, "child": pm[1] == "1",
                               "mode": self.mode, "model": self.model}
            return
        req = self.open.get(task)
        if req is None:
            return
        if pm := PATTERNS["progress"].search(text):
            req["progress"] = {"tokens": int(pm[1]), "fraction": float(pm[2]), "tps": float(pm[4])}
        elif pm := PATTERNS["prefill"].search(text):
            req.update(prefill_ms=float(pm[1]), prefill_tokens=int(pm[2]))
        elif pm := PATTERNS["decode"].search(text):
            req.update(decode_ms=float(pm[1]), decode_tokens=int(pm[2]))
        elif pm := PATTERNS["draft"].search(text):
            req.update(draft_rate=float(pm[1]), draft_accepted=int(pm[2]), draft_generated=int(pm[3]),
                       draft_len=float(pm[4]))
        elif pm := PATTERNS["release"].search(text):
            del self.open[task]
            req.update(end=at, ctx=int(pm[1]), truncated=pm[2] == "1")
            req.pop("progress", None)
            if "prefill_ms" not in req:  # stopped before it finished: the log has no timings
                req["cancelled"] = True
                self.done.append(req)
                return
            new, out = req.get("prefill_tokens", 0), req.get("decode_tokens", 0)
            req["prompt"] = max(req["ctx"] - out, new)
            req["reused"] = req["prompt"] - new
            if req.get("prefill_ms"):
                req["prefill_tps"] = round(new / req["prefill_ms"] * 1000, 1)
            if req.get("decode_ms"):
                req["decode_tps"] = round(out / req["decode_ms"] * 1000, 2)
            self.done.append(req)


# the units' logs; the run scripts hold the model and how it is split
LOGS = {"llama-pool": LlamaLog("/root/logs/llama-pool.log", "pool"),
        "llama-qwen": LlamaLog("/root/logs/llama-server.log", "standalone")}
SCRIPTS = {"llama-pool": "/root/bin/run-llama-pool.sh", "llama-qwen": "/root/bin/run-llama-standalone.sh"}
_live = {}  # slot id -> (monotonic time, tokens decoded, task, decode t/s since the read before)


def track(s):
    """Each slot's decode speed, from two /slots reads in a row. The sampler reads every 5 s too,
    so a page's first request already gets one."""
    now = time.monotonic()
    for x in s or []:
        decoded, task = (x.get("next_token") or [{}])[0].get("n_decoded", 0), x.get("id_task")
        prev = _live.get(x["id"])
        if prev and prev[2] == task and now - prev[0] < 0.5:
            continue  # too soon after the last read: keep its rate
        same = prev and prev[2] == task and x.get("is_processing") and decoded
        tps = round((decoded - prev[1]) / (now - prev[0]), 1) if same else None
        _live[x["id"]] = (now, decoded, task, tps)


def log_follower():
    for log in LOGS.values():
        log.catch_up()
    while True:
        for log in LOGS.values():
            try:
                log.poll()
            except Exception as e:
                print(time.strftime("%F %T"), f"{log.path}: {e}", flush=True)
        time.sleep(1)


_gguf = {}


def gguf_layers(path):
    """Bytes per block, the input and output tensors and the MTP layer count, from a GGUF header."""
    st = os.stat(path)
    if (path, st.st_mtime) in _gguf:
        return _gguf[(path, st.st_mtime)]
    with open(path, "rb") as f:
        rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        rstr = lambda: f.read(rd("Q")).decode("utf-8", "replace")
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def value(t):
            if t == 8:
                return rstr()
            if t == 9:  # arrays aren't needed: skip them
                et, n = rd("I"), rd("Q")
                if et in scalar:
                    f.seek(n * struct.calcsize(scalar[et]), 1)
                else:
                    for _ in range(n):
                        value(et)
                return None
            return rd(scalar[t])

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        rd("I")
        n_tensors, n_kv = rd("Q"), rd("Q")
        meta = {}
        for _ in range(n_kv):
            k = rstr()
            meta[k] = value(rd("I"))
        tensors = []
        for _ in range(n_tensors):
            name = rstr()
            f.seek(8 * rd("I"), 1)  # dimensions
            rd("I")                 # type
            tensors.append((rd("Q"), name))
        align = meta.get("general.alignment") or 32
        data = (f.tell() + align - 1) // align * align
    # each tensor runs to the next one's start (padding included), the last one to the end of the file
    tensors.sort()
    ends = [o for o, _ in tensors[1:]] + [st.st_size - data]
    blocks, inp, out = collections.Counter(), 0, 0
    for (off, name), end in zip(tensors, ends):
        if m := re.match(r"blk\.(\d+)\.", name):
            blocks[int(m[1])] += end - off
        elif name.startswith("token_embd"):
            inp += end - off
        else:
            out += end - off
    nextn = next((v for k, v in meta.items() if k.endswith(".nextn_predict_layers")), 0) or 0
    result = {"blocks": [blocks[i] for i in range(max(blocks) + 1)], "input": inp, "output": out,
              "nextn": nextn}
    _gguf[(path, st.st_mtime)] = result
    return result


def layout(service):
    """Which device holds each layer, worked out the way llama.cpp's loader does it
    (llama-model.cpp): RPC devices come first, then this console's GPU; with -ts the layers are cut
    at the cumulative split points over n_layer + 1 (the +1 is the output layer); the input
    embedding always stays in host RAM."""
    text = read(SCRIPTS[service])
    model = re.search(r"-m (\S+)", text)
    if not model:
        return None
    path = model[1]
    rpc = re.search(r"--rpc (\S+)", text)
    worker = PEER["name"] if PEER else rpc[1] if rpc and re.fullmatch(r"[\w.-]+:\d+", rpc[1]) else "RPC worker"
    devices = ([worker] if rpc else []) + [NODE]
    ts = re.search(r"-ts ([\d.,]+)", text)
    weights = [float(x) for x in ts[1].split(",")][:len(devices)] if ts else [1.0] * len(devices)
    total = sum(weights) or 1
    splits = [sum(weights[:i + 1]) / total for i in range(len(devices))]
    g = gguf_layers(path)
    n = len(g["blocks"])
    ngl = int(m[1]) if (m := re.search(r"-ngl (\d+)", text)) else 0
    gpu_start, gpu_layers = max(n + 1 - ngl, 0), min(ngl, n + 1)

    def device(il):
        if il < gpu_start or il - gpu_start >= gpu_layers:
            return f"{NODE} RAM"
        return devices[min(bisect.bisect_right(splits, (il - gpu_start) / gpu_layers), len(devices) - 1)]

    layers = [{"dev": device(il), "bytes": b, "mtp": il >= n - g["nextn"]} for il, b in enumerate(g["blocks"])]
    parts = {"input": {"dev": f"{NODE} RAM", "bytes": g["input"]},
             "output": {"dev": device(n), "bytes": g["output"]}}
    if mm := re.search(r"--mmproj (\S+)", text):
        parts["vision"] = {"dev": NODE, "bytes": os.path.getsize(mm[1]) if os.path.exists(mm[1]) else 0}
    per_device = collections.Counter()
    for x in layers + list(parts.values()):
        per_device[x["dev"]] += x["bytes"]
    return {"service": service, "model": Path(path).name.removesuffix(".gguf"), "devices": devices,
            "split": weights if ts else None, "layers": layers, "parts": parts,
            "per_device": dict(per_device)}


def llama():
    """Requests from the logs, totals, what each slot is doing now, and the layer map."""
    reqs = sorted((r for log in LOGS.values() for r in log.done), key=lambda r: r["end"])[-100:]
    keys = ("prefill_tokens", "prefill_ms", "decode_tokens", "decode_ms", "draft_accepted",
            "draft_generated", "reused")
    totals = {}  # per mode: pooled and standalone speeds aren't comparable
    for r in reqs:
        t = totals.setdefault(r["mode"], dict.fromkeys(keys, 0) | {"requests": 0})
        t["requests"] += 1
        for k in keys:
            t[k] += r.get(k) or 0
    active = next((svc for svc in LOGS if unit(svc) in ("active", "activating")), None)
    live = []
    s = slots()
    track(s)
    for x in s or []:
        # n_prompt_tokens is what the slot's cache holds so far (it grows a batch at a time while the
        # prompt is read, then with each token written), not the prompt's length
        row = {"slot": x["id"], "n_ctx": x.get("n_ctx"), "busy": x.get("is_processing"),
               "task": x.get("id_task"), "tokens": x.get("n_prompt_tokens"),
               "decoded": (x.get("next_token") or [{}])[0].get("n_decoded", 0)}
        if row["busy"] and x["id"] in _live:
            row["decode_tps"] = _live[x["id"]][3]
        if row["busy"] and active:
            # the log has the prompt's length (tokens / fraction) and the prefill speed so far
            req = LOGS[active].open.get(row["task"])
            if req:
                row["log_progress"] = req.get("progress")
        live.append(row)
    try:
        lay = layout(active or ("llama-pool" if ROLE == "coordinator" else "llama-qwen"))
    except Exception as e:
        lay = {"error": f"{type(e).__name__}: {e}"}
    if lay and "error" not in lay:
        lay["running"] = active is not None and lay["service"] == active
    return {"node": NODE, "role": ROLE, "single": SINGLE, "server": s is not None, "active": active,
            "model": LOGS[active].model if active else None, "live": live,
            "requests": reqs, "totals": totals, "layout": lay,
            "worker": ROLE == "worker" and unit(POOLED_UNIT) == "active"}


def unit(u):
    return subprocess.run(["systemctl", "is-active", f"{u}.service"],
                          capture_output=True, text=True).stdout.strip()


def units():
    """The state of each of UNITS that is installed here."""
    out = subprocess.run(["systemctl", "show", "-p", "Id,LoadState,ActiveState", *(f"{u}.service" for u in UNITS)],
                         capture_output=True, text=True).stdout
    found = {}
    for block in out.split("\n\n"):
        p = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if p.get("LoadState") == "loaded":
            found[p["Id"].removesuffix(".service")] = p["ActiveState"]
    return {u: found[u] for u in UNITS if u in found}


def startup():
    """The llama-server that is starting here, if one is: how long it has been at it, how long the last
    good start took, and the error the last attempt (or this one) ended with."""
    out = subprocess.run(["systemctl", "show", "-p", "Id,ActiveState,ExecMainStartTimestampMonotonic",
                          *(f"{u}.service" for u in LOGS)], capture_output=True, text=True).stdout
    for block in out.split("\n\n"):
        p = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        u, mono = p.get("Id", "").removesuffix(".service"), int(p.get("ExecMainStartTimestampMonotonic") or 0)
        if u not in LOGS or p.get("ActiveState") not in ("active", "activating") or not mono:
            continue
        log, since = LOGS[u], time.monotonic() - mono / 1e6
        # the log may not have this run's first line yet: then it still describes the run before
        fresh = log.base is not None and log.base >= time.time() - since - 5
        if fresh and log.loaded:
            return None
        return {"unit": u, "mode": log.mode, "since": round(since), "last_s": log.load_s,
                "error": (log.error or log.failed) if fresh else log.error if not log.loaded else None}
    return None


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
    cfg = None if SINGLE else cluster()  # the YAML may have changed since start
    gateway = None
    if cfg:
        vip = str(cfg["gateway"]["vip"]).split("/")[0]
        addrs = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
        gateway = {"vip": vip, "here": f" {vip}/" in addrs}
    s = slots()
    m = meminfo()
    return {
        "node": NODE, "role": ROLE, "single": SINGLE, "pooled_mode": POOLED_MODE, "net_if": LINK_IF,
        "uptime": int(float(read("/proc/uptime", "0").split()[0])),
        "units": units(),
        "peerd": None if SINGLE else peerd(),
        "gateway": gateway,
        "fan": fan(),
        "models_sync": llm_cluster.get(cfg, "models.sync", "false") == "true" if cfg else None,
        "cache_choice": read("/run/llm-cache-choice") or None,
        "rpc_cache": rpc_cache(),
        "disks": disks(),
        "models": models(),
        "mem_total": m["MemTotal"], "swap_total": m["SwapTotal"],
        "vram_total": int(read(GPU / "mem_info_vram_total", "0")),
        "gtt_total": int(read(GPU / "mem_info_gtt_total", "0")),
        "slots": [{"n_ctx": x.get("n_ctx"), "busy": x.get("is_processing")} for x in s] if s else None,
        "startup": startup(),
    }


def snapshot(since):
    return {"info": info(), "keys": KEYS, "rows": [r for r in list(HISTORY) if r[0] > since]}


def from_peer(path, body=None, headers=None, timeout=3):
    req = urllib.request.Request(f"http://{PEER['link']}:{PORT}{path}", data=body,
                                 headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def action_key():
    """LLM_DASH_KEY (set in /etc/default/llm-dash), or else the gateway's master key."""
    if key := os.environ.get("LLM_DASH_KEY", "").strip():
        return key
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
        modes = ("standalone", "off") if SINGLE else ("auto", POOLED_MODE, "standalone", "off")
        if value not in modes:
            return 400, "mode: " + ", ".join(modes)
        cmd = ["/root/bin/llm-mode", value]
    elif action == "models.sync" and not SINGLE:
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

    @staticmethod
    def both(mine, peer_path):
        """This console's data, and in a pair the other one's."""
        nodes = {NODE: mine}
        if PEER:
            try:
                nodes[PEER["name"]] = from_peer(peer_path)
            except Exception as e:
                nodes[PEER["name"]] = {"error": f"unreachable ({type(e).__name__})"}
        return {"self": NODE, "single": SINGLE, "nodes": nodes}

    def do_GET(self):
        path, _, query = self.path.partition("?")
        since = int(dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("since", 0) or 0)
        if path == "/":
            self.reply(200, (HERE / "llm-dash.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/self":
            self.reply(200, snapshot(since))
        elif path == "/api/nodes":
            self.reply(200, self.both(snapshot(since), f"/api/self?since={since}"))
        elif path == "/api/llama":
            self.reply(200, llama())
        elif path == "/api/llama-nodes":
            self.reply(200, self.both(llama(), "/api/llama"))
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/action":
            return self.reply(404, {"error": "not found"})
        key = action_key()
        given = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not key or not hmac.compare_digest(given.encode(), key.encode()):
            return self.reply(403, {"error": "wrong key"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except ValueError:
            return self.reply(400, {"error": "bad JSON"})
        if req.get("node", NODE) != NODE:
            if not PEER:
                return self.reply(400, {"error": f"no console {req['node']!r} here"})
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
    threading.Thread(target=log_follower, daemon=True).start()
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.daemon_threads = True
    serving = [PORT]
    for port in ALSO_PORTS:
        try:
            threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever, daemon=True).start()
            serving.append(port)
        except OSError as e:  # taken by something else: the main port still serves
            print(time.strftime("%F %T"), f"port {port} unavailable: {e}", flush=True)
    print(time.strftime("%F %T"), f"{NODE}: dashboard on " + ", ".join(f":{p}" for p in serving) + ", "
          + ("one console" if SINGLE else f"paired with {PEER['name']}"), flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
