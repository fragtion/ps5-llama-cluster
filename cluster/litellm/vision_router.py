"""Pick the PS5 that serves, and route requests with images to a model that can see them.

- `ps5` goes to the console whose peer daemon (:50060/status) reports a ready llama-server:
  ps5a first (it runs the pool), else ps5b. Asking the daemons up front means a console that is
  down costs nothing; a connection to it would hang until the timeout.
- `ps5` with images: answered natively by the pool (vision). Standalone mode is text only, so the
  request goes to `vision-fallback` instead.
- Models listed in TEXT_ONLY_MODELS (comma-separated, /etc/litellm/env) can't see images: each
  image is replaced by a description from whichever vision model is up (the pool first, then the
  fallback). Descriptions are cached on disk by the image's hash, so the same image always reads
  the same and backend prompt caches keep hitting.
"""
import asyncio
import hashlib
import json
import os
import threading
import time

import httpx
import yaml
from litellm.integrations.custom_logger import CustomLogger

TEXT_ONLY = {m.strip() for m in os.environ.get("TEXT_ONLY_MODELS", "").split(",") if m.strip()}
# config.yaml reads the same variables; the model carries LiteLLM's "openai/" provider prefix
FALLBACK_URL = os.environ.get("VISION_FALLBACK_API_BASE", "").removesuffix("/").removesuffix("/v1")
FALLBACK_MODEL = os.environ.get("VISION_FALLBACK_MODEL", "").removeprefix("openai/")
CACHE_PATH = "/var/lib/litellm/image-captions.json"
PROMPT = ("Describe this image for someone who cannot see it. Transcribe all visible text exactly, "
          "including code, numbers and error messages. Then describe layout, UI elements and anything unusual. "
          "Be factual; do not guess.")

# (name, link address) per console, the coordinator first; the model names in config.yaml match
with open("/etc/llm-cluster.yaml") as f:
    _nodes = yaml.safe_load(f)["nodes"]
NODES = sorted(((name, n["link"]) for name, n in _nodes.items()),
               key=lambda n: _nodes[n[0]]["role"] != "coordinator")

_lock = threading.Lock()
try:
    with open(CACHE_PATH) as f:
        _cache = json.load(f)
except (OSError, ValueError):
    _cache = {}

_serving = {"value": None, "at": 0.0}


async def _status(client, link):
    try:
        r = await client.get(f"http://{link}:50060/status")
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


async def serving_ps5():
    """(name, url, has_vision) of the console that should answer `ps5`, or None. Cached 5 s."""
    if time.time() - _serving["at"] < 5:
        return _serving["value"]
    async with httpx.AsyncClient(timeout=2) as client:
        states = await asyncio.gather(*(_status(client, link) for _, link in NODES))
    value = None
    for (name, link), s in zip(NODES, states):
        if s and s.get("mode") in ("pool", "standalone") and s.get("ready"):
            value = (name, f"http://{link}:1234", s["mode"] == "pool")
            break
    _serving.update(value=value, at=time.time())
    return value


def _images(data):
    for msg in data.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for i, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    yield content, i, part


async def _describe(url: str) -> str:
    key = hashlib.sha256(url.encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    node = await serving_ps5()
    if node and node[2]:
        base, model = node[1], "ps5"
    else:
        base, model = FALLBACK_URL, FALLBACK_MODEL
    body = {"model": model, "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": url}}]}]}
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{base}/v1/chat/completions", json=body)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:  # no vision model up: say so, and don't cache it
        return f"(image could not be described: {type(e).__name__}; no vision model is available)"
    with _lock:
        _cache[key] = text
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_cache, f)
        os.replace(tmp, CACHE_PATH)
    return text


class VisionRouter(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        model = data.get("model")
        images = list(_images(data))
        if model == "ps5":
            node = await serving_ps5()
            if images and not (node and node[2]):
                data["model"] = "vision-fallback"
            elif node:
                data["model"] = node[0]
            # else: no console reports ready; the `ps5` group's order tries ps5a, then ps5b
            return data
        if images and model in TEXT_ONLY:
            for n, (content, i, part) in enumerate(images, 1):
                image = part["image_url"]
                url = image["url"] if isinstance(image, dict) else image
                desc = await _describe(url)
                content[i] = {"type": "text",
                              "text": f"[Image {n}, described by a vision model]\n{desc}\n[End of image {n}]"}
        return data


proxy_handler_instance = VisionRouter()
