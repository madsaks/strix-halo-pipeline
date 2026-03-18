#!/usr/bin/env python3
"""
strix_halo_pipeline.py — NPU+GPU LLM Inference for AMD Strix Halo on Linux
===========================================================================

First-of-its-kind: Runs LLMs across XDNA2 NPU and RDNA 3.5 iGPU simultaneously.

Usage:
    python3 strix_halo_pipeline.py --gpu-model ./models/Qwen3-8B-Q4_K_M.gguf
    python3 strix_halo_pipeline.py --gpu-model ./models/Qwen3-8B-Q4_K_M.gguf --gpu-backend rocm
    python3 strix_halo_pipeline.py --gpu-model ./models/Qwen3.5-27B-Q4_K_M.gguf --draft qwen3:4b --pmode turbo -v

Connect Open WebUI (Ollama mode) to: http://<your-ip>:11435
Connect OpenAI clients to: http://<your-ip>:11435/v1

Author:  Mike Alani / Claude collaboration
License: MIT
"""

__version__ = "1.3.0"

import argparse
import asyncio
import glob
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Optional

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "aiohttp", "--break-system-packages", "-q"])
    import aiohttp
    from aiohttp import web


@dataclass
class PipelineConfig:
    draft_model: str = "qwen3:1.7b"
    draft_pmode: str = "performance"
    draft_port: int = 52625
    gpu_model: str = ""
    gpu_backend: str = "rocm"  # rocm or vulkan
    gpu_port: int = 9999
    gpu_ctx: int = 4096
    gpu_layers: int = 99
    gpu_threads: int = 8
    port: int = 11435
    host: str = "0.0.0.0"
    draft_tokens: int = 128
    temperature: float = 0.6
    top_p: float = 0.95
    mode: str = "pipeline"
    verbose: bool = False


# ═════════════════════════════════════════════════════════════════════════════
# Environment
# ═════════════════════════════════════════════════════════════════════════════

class Environment:
    def __init__(self, log):
        self.log = log

    def fix_vulkan(self):
        correct_icd = "/usr/share/vulkan/icd.d/radeon_icd.json"
        current = os.environ.get("VK_ICD_FILENAMES", "")
        if "latest-vulkan" in current or not current:
            if os.path.exists(correct_icd):
                os.environ["VK_ICD_FILENAMES"] = correct_icd
                self.log.info(f"Fixed VK_ICD_FILENAMES -> {correct_icd}")

    def fix_rocm_libs(self):
        """Ensure ROCm 7.11 HSA runtime is found first (fixes gfx1151 segfault)."""
        rocm711 = "/opt/rocm/core-7.11/lib"
        rocm71 = "/opt/rocm-7.1.1/lib"
        current = os.environ.get("LD_LIBRARY_PATH", "")
        paths = current.split(":") if current else []

        # Ensure 7.11 is first
        needs_fix = False
        if rocm711 not in paths:
            needs_fix = True
        elif paths.index(rocm711) > 0:
            # 7.11 exists but isn't first — check if 7.1.1 comes before it
            if rocm71 in paths and paths.index(rocm71) < paths.index(rocm711):
                needs_fix = True

        if needs_fix and os.path.exists(rocm711):
            # Remove existing entries and re-add in correct order
            paths = [p for p in paths if p not in [rocm711, rocm71]]
            new_paths = [rocm711]
            if os.path.exists(rocm71):
                new_paths.append(rocm71)
            new_paths.extend(paths)
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths)
            self.log.info(f"Fixed LD_LIBRARY_PATH: core-7.11 first (HSA segfault fix)")

    def fix_xrt_symlinks(self):
        target = "/opt/xilinx/xrt/lib/x86_64-linux-gnu"
        source = "/opt/xilinx/xrt/lib"
        if os.path.exists(source) and not os.path.exists(os.path.join(target, "libxrt_core.so.2")):
            try:
                os.makedirs(target, exist_ok=True)
                for f in glob.glob(os.path.join(source, "*.so*")):
                    link = os.path.join(target, os.path.basename(f))
                    if not os.path.exists(link):
                        os.symlink(f, link)
                self.log.info("Created XRT symlinks")
            except PermissionError:
                self.log.warning("Cannot create XRT symlinks - run with sudo once")

    def set_gpu_performance(self):
        for card in glob.glob("/sys/class/drm/card*/device/power_dpm_force_performance_level"):
            try:
                with open(card, "w") as f:
                    f.write("performance")
                self.log.info("GPU performance mode set")
            except PermissionError:
                pass

    def find_flm(self):
        for path in [shutil.which("flm"), "/opt/fastflowlm/bin/flm"]:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def find_llama_server(self, backend="rocm"):
        """Find llama-server binary for given backend."""
        if backend == "rocm":
            candidates = [
                os.path.expanduser("~/llama.cpp/build-rocm/bin/llama-server"),
                "/usr/local/share/lemonade-server/llama/rocm/llama-server",
            ]
        else:  # vulkan
            candidates = [
                os.path.expanduser("~/llama.cpp/build-vulkan/bin/llama-server"),
                "/usr/local/share/lemonade-server/llama/vulkan/build/bin/llama-server",
            ]
        # Also check generic builds
        candidates.extend([
            os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
            shutil.which("llama-server"),
        ])
        for path in candidates:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return path, os.path.dirname(path)
        return None, None

    def find_gpu_model(self, explicit=""):
        if explicit and os.path.isfile(explicit):
            return os.path.abspath(explicit)
        for d in ["~/vitias/models", "~/models", ".", "./models",
                   "/mnt/raid0/vitias/models"]:
            d = os.path.expanduser(d)
            for f in glob.glob(os.path.join(d, "*.gguf")):
                return os.path.abspath(f)
        return None

    def check_npu(self):
        return os.path.exists("/dev/accel/accel0")

    def setup_all(self):
        self.fix_vulkan()
        self.fix_rocm_libs()
        self.fix_xrt_symlinks()
        self.set_gpu_performance()


# ═════════════════════════════════════════════════════════════════════════════
# Process Manager
# ═════════════════════════════════════════════════════════════════════════════

class ProcessManager:
    def __init__(self, config, log):
        self.config = config
        self.log = log
        self.env = Environment(log)
        self.procs = []

    def _wait_for_health(self, url, timeout=120, check_text=""):
        import urllib.request
        for _ in range(timeout):
            try:
                with urllib.request.urlopen(urllib.request.Request(url), timeout=2) as r:
                    if check_text:
                        if check_text in r.read().decode():
                            return True
                    elif r.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def start_flm(self):
        flm_bin = self.env.find_flm()
        if not flm_bin:
            self.log.error("FLM not found")
            return False
        self.log.info(f"Starting NPU: {self.config.draft_model} "
                      f"(pmode={self.config.draft_pmode}) on :{self.config.draft_port}")
        proc = subprocess.Popen(
            [flm_bin, "serve", self.config.draft_model,
             "--pmode", self.config.draft_pmode,
             "--port", str(self.config.draft_port)],
            stdout=subprocess.DEVNULL if not self.config.verbose else None,
            stderr=subprocess.DEVNULL if not self.config.verbose else None)
        self.procs.append(proc)
        if self._wait_for_health(
                f"http://127.0.0.1:{self.config.draft_port}/v1/models", timeout=60):
            self.log.info(f"NPU ready (PID: {proc.pid})")
            return True
        self.log.error("FLM failed to start")
        return False

    def start_llama(self):
        backend = self.config.gpu_backend
        llama_bin, llama_lib = self.env.find_llama_server(backend)
        if not llama_bin:
            self.log.error(f"llama-server ({backend}) not found. "
                           f"Build llama.cpp with -DGGML_{'HIP' if backend == 'rocm' else 'VULKAN'}=ON")
            return False
        model = self.env.find_gpu_model(self.config.gpu_model)
        if not model:
            self.log.error("No GGUF model found")
            return False
        self.config.gpu_model = model
        self.log.info(f"Starting GPU ({backend}): {os.path.basename(model)} "
                      f"on :{self.config.gpu_port}")

        # Build LD_LIBRARY_PATH for the subprocess
        env = os.environ.copy()
        lib_paths = []
        if llama_lib:
            lib_paths.append(llama_lib)
        if backend == "rocm":
            # CRITICAL: core-7.11 first to avoid HSA segfault on gfx1151
            lib_paths.insert(0, "/opt/rocm/core-7.11/lib")
            lib_paths.append("/opt/rocm-7.1.1/lib")
        lib_paths.append("/opt/xilinx/xrt/lib")
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (":" + existing if existing else "")

        proc = subprocess.Popen(
            [llama_bin, "--model", model,
             "--port", str(self.config.gpu_port),
             "--ctx-size", str(self.config.gpu_ctx),
             "--n-gpu-layers", str(self.config.gpu_layers),
             "--threads", str(self.config.gpu_threads),
             "--host", "0.0.0.0"],
            env=env,
            stdout=subprocess.DEVNULL if not self.config.verbose else None,
            stderr=subprocess.DEVNULL if not self.config.verbose else None)
        self.procs.append(proc)
        if self._wait_for_health(
                f"http://127.0.0.1:{self.config.gpu_port}/health",
                timeout=120, check_text="ok"):
            self.log.info(f"GPU ({backend}) ready (PID: {proc.pid})")
            return True
        self.log.error(f"llama-server ({backend}) failed to start")
        # Suggest fallback
        if backend == "rocm":
            self.log.info("Try --gpu-backend vulkan if ROCm segfaults")
        return False

    def stop_all(self):
        for proc in self.procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        self.procs.clear()


# ═════════════════════════════════════════════════════════════════════════════
# Inference Engine
# ═════════════════════════════════════════════════════════════════════════════

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def clean_messages(messages):
    result = []
    for m in messages:
        result.append({"role": m.get("role", "user"),
                       "content": extract_text(m.get("content", ""))})
    return result


class InferenceEngine:
    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger("engine")
        self._session = None

    @property
    def draft_url(self):
        return f"http://127.0.0.1:{self.config.draft_port}"

    @property
    def gpu_url(self):
        return f"http://127.0.0.1:{self.config.gpu_port}"

    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600, sock_read=300))

    async def stop(self):
        if self._session:
            await self._session.close()

    def _patch_messages(self, messages):
        patched = []
        has_sys = False
        for m in messages:
            if m["role"] == "system":
                has_sys = True
                patched.append({"role": "system",
                                "content": m["content"] + " /no_think"})
            else:
                patched.append(m)
        if not has_sys:
            patched.insert(0, {"role": "system",
                               "content": "You are a helpful assistant. /no_think"})
        return patched

    async def _npu_stream(self, messages, max_tokens):
        payload = {
            "model": self.config.draft_model,
            "messages": self._patch_messages(messages),
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }
        async with self._session.post(
                f"{self.draft_url}/v1/chat/completions", json=payload) as resp:
            in_think = False
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    delta = json.loads(data_str)["choices"][0]["delta"].get(
                        "content", "")
                    if not delta:
                        continue
                    if "<think>" in delta:
                        in_think = True
                    if in_think:
                        if "</think>" in delta:
                            in_think = False
                            after = delta.split("</think>", 1)[-1]
                            if after:
                                yield after
                        continue
                    yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def _build_prompt(self, messages, prefix=""):
        prompt = ""
        has_sys = False
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                has_sys = True
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        if not has_sys:
            prompt = ("<|im_start|>system\nYou are a helpful assistant."
                      "<|im_end|>\n" + prompt)
        prompt += f"<|im_start|>assistant\n{prefix}"
        return prompt

    async def _gpu_stream(self, messages, prefix, max_tokens):
        payload = {
            "prompt": self._build_prompt(messages, prefix),
            "n_predict": max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "cache_prompt": True,
            "stream": True,
        }
        async with self._session.post(
                f"{self.gpu_url}/completion", json=payload) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    token = json.loads(data_str).get("content", "")
                    if not token:
                        continue
                    for eos in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                        if eos in token:
                            before = token.split(eos)[0]
                            if before:
                                yield before
                            return
                    yield token
                except (json.JSONDecodeError, KeyError):
                    continue

    async def _gpu_warmup(self, messages):
        try:
            payload = {
                "prompt": self._build_prompt(messages),
                "n_predict": 1,
                "temperature": 0.0,
                "cache_prompt": True,
                "stream": False,
            }
            async with self._session.post(
                    f"{self.gpu_url}/completion", json=payload) as resp:
                await resp.json()
        except Exception:
            pass

    async def pipeline_stream(self, messages, max_tokens):
        cfg = self.config
        npu_tokens = min(max_tokens // 2, cfg.draft_tokens)
        gpu_tokens = max_tokens - npu_tokens
        t0 = time.perf_counter()
        warmup_task = asyncio.create_task(self._gpu_warmup(messages))
        npu_text = ""
        async for chunk in self._npu_stream(messages, npu_tokens):
            npu_text += chunk
            yield chunk
        npu_ms = (time.perf_counter() - t0) * 1000
        if cfg.verbose:
            npu_words = len(npu_text.split())
            rate = npu_words / (npu_ms / 1000) if npu_ms > 0 else 0
            self.log.info(f"[NPU] {npu_words} words in {npu_ms:.0f}ms "
                          f"({rate:.1f} w/s)")
        await warmup_task
        t1 = time.perf_counter()
        async for chunk in self._gpu_stream(messages, npu_text, gpu_tokens):
            yield chunk
        if cfg.verbose:
            gpu_ms = (time.perf_counter() - t1) * 1000
            total_ms = (time.perf_counter() - t0) * 1000
            self.log.info(f"[GPU] Continued in {gpu_ms:.0f}ms")
            self.log.info(f"[TOTAL] {total_ms:.0f}ms "
                          f"(NPU {npu_ms:.0f}ms + GPU {gpu_ms:.0f}ms)")

    async def generate(self, messages, max_tokens, mode=None):
        mode = mode or self.config.mode
        if mode == "npu":
            async for c in self._npu_stream(messages, max_tokens):
                yield c
        elif mode == "gpu":
            async for c in self._gpu_stream(messages, "", max_tokens):
                yield c
        elif mode == "auto":
            user_text = " ".join(
                m["content"] for m in messages if m["role"] == "user")
            if len(user_text.split()) < 20 and max_tokens < 128:
                async for c in self._npu_stream(messages, max_tokens):
                    yield c
            else:
                async for c in self.pipeline_stream(messages, max_tokens):
                    yield c
        else:
            async for c in self.pipeline_stream(messages, max_tokens):
                yield c


# ═════════════════════════════════════════════════════════════════════════════
# API Server — OpenAI + Ollama compatible
# ═════════════════════════════════════════════════════════════════════════════

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Max-Age": "3600",
}

MODE_MAP = {
    "strix-speculative": "pipeline",
    "npu": "npu",
    "gpu": "gpu",
    "auto": "auto",
}


class APIServer:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        self.app = web.Application()
        r = self.app.router
        r.add_route("OPTIONS", "/{path:.*}", self._options)
        r.add_get("/", self._root)
        r.add_get("/health", self._health)
        r.add_get("/v1/health", self._health)
        r.add_get("/v1/models", self._models)
        r.add_post("/v1/chat/completions", self._openai_chat)
        # Ollama endpoints
        r.add_post("/api/chat", self._ollama_chat)
        r.add_get("/api/tags", self._ollama_tags)
        r.add_get("/api/ps", self._ollama_ps)
        r.add_get("/v1/api/tags", self._ollama_tags)
        r.add_get("/v1/api/ps", self._ollama_ps)
        r.add_get("/api/version", self._ollama_version)

    def _resolve_mode(self, model_name):
        clean = model_name.replace(":latest", "").strip()
        return MODE_MAP.get(clean, "pipeline")

    async def _options(self, req):
        return web.Response(status=200, headers=CORS)

    async def _root(self, req):
        return web.json_response({
            "name": "Strix Halo NPU+GPU Pipeline",
            "version": __version__,
            "gpu_backend": self.config.gpu_backend,
        }, headers=CORS)

    async def _health(self, req):
        return web.json_response({"status": "ok"}, headers=CORS)

    # ── OpenAI format ─────────────────────────────────────────────────

    async def _models(self, req):
        return web.json_response({"object": "list", "data": [
            {"id": "strix-speculative", "object": "model",
             "owned_by": "local"},
            {"id": "npu", "object": "model", "owned_by": "local"},
            {"id": "gpu", "object": "model", "owned_by": "local"},
            {"id": "auto", "object": "model", "owned_by": "local"},
        ]}, headers=CORS)

    async def _openai_chat(self, req):
        body = await req.json()
        messages = clean_messages(body.get("messages", []))
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", 2048)
        model = body.get("model", "strix-speculative")
        mode = self._resolve_mode(model)

        if body.get("temperature") is not None:
            self.engine.config.temperature = body["temperature"]

        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        ts = int(time.time())

        if stream:
            resp = web.StreamResponse(headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive", **CORS})
            await resp.prepare(req)

            async def heartbeat():
                while True:
                    await asyncio.sleep(5)
                    try:
                        await resp.write(b": heartbeat\n\n")
                    except Exception:
                        break

            hb = asyncio.create_task(heartbeat())
            try:
                async for chunk in self.engine.generate(
                        messages, max_tokens, mode):
                    sse = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": ts,
                        "model": model,
                        "choices": [{"index": 0,
                                     "delta": {"content": chunk},
                                     "finish_reason": None}],
                    }
                    await resp.write(
                        f"data: {json.dumps(sse)}\n\n".encode())
            finally:
                hb.cancel()

            final = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": ts,
                "model": model,
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": "stop"}],
            }
            await resp.write(f"data: {json.dumps(final)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            return resp
        else:
            full = ""
            t0 = time.perf_counter()
            async for chunk in self.engine.generate(
                    messages, max_tokens, mode):
                full += chunk
            ms = (time.perf_counter() - t0) * 1000
            return web.json_response({
                "id": rid, "object": "chat.completion",
                "created": ts, "model": model,
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": full},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0,
                          "completion_tokens": len(full.split()),
                          "total_tokens": len(full.split())},
                "timings": {"total_ms": round(ms, 1)},
            }, headers=CORS)

    # ── Ollama format (Open WebUI) ────────────────────────────────────

    async def _ollama_chat(self, req):
        body = await req.json()
        messages = clean_messages(body.get("messages", []))
        stream = body.get("stream", True)
        model = body.get("model", "strix-speculative")
        mode = self._resolve_mode(model)
        opts = body.get("options", {})
        max_tokens = opts.get("num_predict", 2048)

        if opts.get("temperature") is not None:
            self.engine.config.temperature = opts["temperature"]

        if stream:
            resp = web.StreamResponse(headers={
                "Content-Type": "application/x-ndjson",
                "Cache-Control": "no-cache", **CORS})
            await resp.prepare(req)

            full_text = ""
            async for chunk in self.engine.generate(
                    messages, max_tokens, mode):
                full_text += chunk
                msg = {
                    "model": model,
                    "created_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": chunk},
                    "done": False,
                }
                await resp.write(json.dumps(msg).encode() + b"\n")

            done_msg = {
                "model": model,
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "total_duration": 0,
                "eval_count": len(full_text.split()),
            }
            await resp.write(json.dumps(done_msg).encode() + b"\n")
            return resp
        else:
            full_text = ""
            async for chunk in self.engine.generate(
                    messages, max_tokens, mode):
                full_text += chunk
            return web.json_response({
                "model": model,
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": full_text},
                "done": True,
                "eval_count": len(full_text.split()),
            }, headers=CORS)

    async def _ollama_tags(self, req):
        models = []
        for name in ["strix-speculative", "npu", "gpu", "auto"]:
            models.append({
                "name": name, "model": name, "size": 0, "digest": "",
                "modified_at": "2026-03-18T00:00:00Z",
                "details": {"family": "qwen", "format": "hybrid",
                            "parameter_size": "8B",
                            "quantization_level": "Q4_K_M"},
            })
        return web.json_response({"models": models}, headers=CORS)

    async def _ollama_ps(self, req):
        return web.json_response({"models": [{
            "name": "strix-speculative",
            "model": "strix-speculative",
            "size": 0, "digest": "",
            "expires_at": "2099-01-01T00:00:00Z",
        }]}, headers=CORS)

    async def _ollama_version(self, req):
        return web.json_response({"version": __version__}, headers=CORS)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

BANNER = """
\033[1;32m
  ======================================================
   STRIX HALO NPU+GPU LLM PIPELINE  v{version}
  ======================================================
   NPU:   {npu:<20s} :{npu_port}  (XDNA2, pmode={pmode})
   GPU:   {gpu:<20s} :{gpu_port}  ({backend}, RDNA 3.5)

   API:   http://{ip}:{api_port}/v1     (OpenAI)
          http://{ip}:{api_port}        (Ollama)

   Models: strix-speculative | npu | gpu | auto
   Ctrl+C to stop
  ======================================================
\033[0m"""


async def run_server(config):
    engine = InferenceEngine(config)
    await engine.start()
    server = APIServer(engine, config)
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, config.host, config.port)
    await site.start()

    try:
        import socket
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "127.0.0.1"

    print(BANNER.format(
        version=__version__,
        npu=config.draft_model,
        npu_port=config.draft_port,
        gpu=os.path.basename(config.gpu_model)[:20],
        gpu_port=config.gpu_port,
        backend=config.gpu_backend.upper(),
        pmode=config.draft_pmode,
        ip=ip,
        api_port=config.port,
    ))

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await engine.stop()
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Strix Halo NPU+GPU LLM Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended configurations:
  Fast:       --draft qwen3:4b  --gpu-model Qwen3-8B    --draft-tokens 64
  Quality:    --draft qwen3:1.7b --gpu-model Qwen3.5-27B --draft-tokens 32
  ROCm:       --gpu-backend rocm  (default, fastest, requires 7.11 HSA fix)
  Vulkan:     --gpu-backend vulkan (fallback, always works)
        """)
    parser.add_argument("--gpu-model", required=True,
                        help="GGUF model for GPU")
    parser.add_argument("--gpu-backend", default="rocm",
                        choices=["rocm", "vulkan"],
                        help="GPU backend (default: rocm)")
    parser.add_argument("--draft", default="qwen3:1.7b",
                        help="FLM draft model")
    parser.add_argument("--pmode", default="performance",
                        choices=["powersaver", "balanced",
                                 "performance", "turbo"],
                        help="NPU power mode (default: performance)")
    parser.add_argument("--draft-tokens", type=int, default=128,
                        help="NPU draft length")
    parser.add_argument("--draft-port", type=int, default=52625)
    parser.add_argument("--gpu-port", type=int, default=9999)
    parser.add_argument("--gpu-ctx", type=int, default=4096,
                        help="GPU context size")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--port", type=int, default=11435,
                        help="API port")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--mode",
                        choices=["pipeline", "npu", "gpu", "auto"],
                        default="pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-launch", action="store_true",
                        help="Don't start servers")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("main")

    config = PipelineConfig(
        draft_model=args.draft,
        draft_pmode=args.pmode,
        draft_port=args.draft_port,
        gpu_model=args.gpu_model,
        gpu_backend=args.gpu_backend,
        gpu_port=args.gpu_port,
        gpu_ctx=args.gpu_ctx,
        gpu_layers=args.gpu_layers,
        port=args.port,
        host=args.host,
        draft_tokens=args.draft_tokens,
        temperature=args.temperature,
        mode=args.mode,
        verbose=args.verbose,
    )

    env = Environment(log)
    env.setup_all()

    proc_mgr = None
    if not args.no_launch:
        proc_mgr = ProcessManager(config, log)
        if not proc_mgr.start_flm():
            proc_mgr.stop_all()
            sys.exit(1)
        if not proc_mgr.start_llama():
            proc_mgr.stop_all()
            sys.exit(1)

    def shutdown(sig=None, frame=None):
        log.info("Shutting down...")
        if proc_mgr:
            proc_mgr.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        pass
    finally:
        if proc_mgr:
            proc_mgr.stop_all()


if __name__ == "__main__":
    main()
