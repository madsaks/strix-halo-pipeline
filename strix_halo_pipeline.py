#!/usr/bin/env python3
"""
strix_halo_pipeline.py — NPU+GPU LLM Inference for AMD Strix Halo on Linux
===========================================================================

First-of-its-kind: Runs LLMs across XDNA2 NPU and RDNA 3.5 iGPU simultaneously.

NPU (FastFlowLM) drafts tokens instantly → GPU (llama.cpp Vulkan) continues with
a larger model. Both accelerators share 128GB unified LPDDR5X via GPUVM.

Usage:
    python3 strix_halo_pipeline.py --gpu-model ./models/Qwen3-8B-Q4_K_M.gguf
    python3 strix_halo_pipeline.py --gpu-model ./models/Qwen3-14B-Q4_K_M.gguf --draft qwen3:4b
    python3 strix_halo_pipeline.py --gpu-model ./models/Qwen3-32B-Q4_K_M.gguf --draft qwen3:4b -v

Connect any OpenAI-compatible client to: http://<your-ip>:11435/v1

Author:  Mike Alani / Claude collaboration
License: MIT
"""

__version__ = "1.0.0"

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
from dataclasses import dataclass, field
from pathlib import Path
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

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    # Draft model (NPU via FastFlowLM)
    draft_model: str = "qwen3:1.7b"
    draft_port: int = 52625

    # Verifier model (GPU via llama.cpp)
    gpu_model: str = ""
    gpu_port: int = 9999
    gpu_ctx: int = 4096
    gpu_layers: int = 99
    gpu_threads: int = 8

    # Pipeline engine
    port: int = 11435
    host: str = "0.0.0.0"
    draft_tokens: int = 128
    temperature: float = 0.6
    top_p: float = 0.95
    mode: str = "pipeline"
    verbose: bool = False


# ═════════════════════════════════════════════════════════════════════════════
# Environment Setup
# ═════════════════════════════════════════════════════════════════════════════

class Environment:
    """Detects and fixes Strix Halo environment issues."""

    def __init__(self, log: logging.Logger):
        self.log = log

    def fix_vulkan(self):
        """Fix common Vulkan ICD path issue on Strix Halo."""
        correct_icd = "/usr/share/vulkan/icd.d/radeon_icd.json"
        current = os.environ.get("VK_ICD_FILENAMES", "")
        if "latest-vulkan" in current or not current:
            if os.path.exists(correct_icd):
                os.environ["VK_ICD_FILENAMES"] = correct_icd
                self.log.info(f"Fixed VK_ICD_FILENAMES → {correct_icd}")

    def fix_xrt_symlinks(self):
        """Create XRT lib symlinks that FLM expects."""
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
                self.log.warning("Cannot create XRT symlinks — run with sudo once, or:")
                self.log.warning(f"  sudo mkdir -p {target}")
                self.log.warning(f"  sudo bash -c 'for f in {source}/*.so*; "
                                 f"do ln -sf \"$f\" {target}/$(basename \"$f\"); done'")

    def set_gpu_performance(self):
        """Attempt to set GPU to performance mode."""
        for card in glob.glob("/sys/class/drm/card*/device/power_dpm_force_performance_level"):
            try:
                with open(card, "w") as f:
                    f.write("performance")
                self.log.info(f"GPU performance mode set")
            except PermissionError:
                pass

    def find_flm(self) -> Optional[str]:
        """Find FLM binary."""
        for path in [shutil.which("flm"), "/opt/fastflowlm/bin/flm"]:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def find_llama_server(self) -> Optional[tuple]:
        """Find llama-server binary with Vulkan support. Returns (binary, lib_dir)."""
        candidates = [
            os.path.expanduser("~/llama.cpp/build-vulkan/bin/llama-server"),
            os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
            "/usr/local/share/lemonade-server/llama/vulkan/build/bin/llama-server",
            shutil.which("llama-server"),
        ]
        for path in candidates:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return path, os.path.dirname(path)
        return None, None

    def find_gpu_model(self, explicit: str = "") -> Optional[str]:
        """Find a GGUF model file."""
        if explicit and os.path.isfile(explicit):
            return os.path.abspath(explicit)
        search_dirs = [
            os.path.expanduser("~/vitias/models"),
            os.path.expanduser("~/models"),
            ".",
            "./models",
        ]
        for d in search_dirs:
            for f in glob.glob(os.path.join(d, "*.gguf")):
                return os.path.abspath(f)
        return None

    def check_npu(self) -> bool:
        return os.path.exists("/dev/accel/accel0")

    def setup_all(self):
        self.fix_vulkan()
        self.fix_xrt_symlinks()
        self.set_gpu_performance()


# ═════════════════════════════════════════════════════════════════════════════
# Process Manager
# ═════════════════════════════════════════════════════════════════════════════

class ProcessManager:
    """Manages FLM and llama.cpp server processes."""

    def __init__(self, config: PipelineConfig, log: logging.Logger):
        self.config = config
        self.log = log
        self.procs: list[subprocess.Popen] = []
        self.env = Environment(log)

    def _wait_for_health(self, url: str, timeout: int = 120, check_text: str = ""):
        """Poll a URL until it responds."""
        import urllib.request
        for i in range(timeout):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=2) as r:
                    if check_text:
                        if check_text in r.read().decode():
                            return True
                    else:
                        if r.status == 200:
                            return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def start_flm(self) -> bool:
        """Start FastFlowLM NPU server."""
        flm_bin = self.env.find_flm()
        if not flm_bin:
            self.log.error("FLM not found. Install FastFlowLM: https://github.com/FastFlowLM/FastFlowLM")
            return False

        if not self.env.check_npu():
            self.log.warning("NPU device /dev/accel/accel0 not found")

        self.log.info(f"Starting NPU server: {self.config.draft_model} on :{self.config.draft_port}")

        proc = subprocess.Popen(
            [flm_bin, "serve", self.config.draft_model, "--port", str(self.config.draft_port)],
            stdout=subprocess.DEVNULL if not self.config.verbose else None,
            stderr=subprocess.DEVNULL if not self.config.verbose else None,
        )
        self.procs.append(proc)

        if self._wait_for_health(
            f"http://127.0.0.1:{self.config.draft_port}/v1/models", timeout=60
        ):
            self.log.info(f"NPU ready ✓ (PID: {proc.pid})")
            return True

        self.log.error("FLM failed to start — check NPU driver and model")
        return False

    def start_llama(self) -> bool:
        """Start llama.cpp Vulkan GPU server."""
        llama_bin, llama_lib = self.env.find_llama_server()
        if not llama_bin:
            self.log.error("llama-server not found. Build llama.cpp with -DGGML_VULKAN=ON")
            return False

        model = self.env.find_gpu_model(self.config.gpu_model)
        if not model:
            self.log.error(f"No GGUF model found. Use --gpu-model <path>")
            return False
        self.config.gpu_model = model

        self.log.info(f"Starting GPU server: {os.path.basename(model)} on :{self.config.gpu_port}")

        env = os.environ.copy()
        if llama_lib:
            env["LD_LIBRARY_PATH"] = f"{llama_lib}:/opt/xilinx/xrt/lib:" + env.get("LD_LIBRARY_PATH", "")

        proc = subprocess.Popen(
            [llama_bin,
             "--model", model,
             "--port", str(self.config.gpu_port),
             "--ctx-size", str(self.config.gpu_ctx),
             "--n-gpu-layers", str(self.config.gpu_layers),
             "--threads", str(self.config.gpu_threads),
             "--host", "0.0.0.0"],
            env=env,
            stdout=subprocess.DEVNULL if not self.config.verbose else None,
            stderr=subprocess.DEVNULL if not self.config.verbose else None,
        )
        self.procs.append(proc)

        if self._wait_for_health(
            f"http://127.0.0.1:{self.config.gpu_port}/health", timeout=120, check_text="ok"
        ):
            self.log.info(f"GPU ready ✓ (PID: {proc.pid})")
            return True

        self.log.error("llama-server failed to start — check Vulkan and GGUF model")
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

class InferenceEngine:
    """
    Pipelined NPU+GPU inference engine.

    Pipeline mode:
      1. GPU starts warming KV cache in background
      2. NPU streams draft tokens to user (fast TTFT)
      3. GPU continues from NPU output (cache pre-warmed)

    Both accelerators contribute to every response.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.log = logging.getLogger("engine")
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def draft_url(self):
        return f"http://127.0.0.1:{self.config.draft_port}"

    @property
    def gpu_url(self):
        return f"http://127.0.0.1:{self.config.gpu_port}"

    async def start(self):
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180))

    async def stop(self):
        if self._session:
            await self._session.close()

    # ── NPU (FastFlowLM) ─────────────────────────────────────────────────

    def _patch_messages(self, messages):
        patched = []
        has_sys = False
        for m in messages:
            if m["role"] == "system":
                has_sys = True
                patched.append({"role": "system", "content": m["content"] + " /no_think"})
            else:
                patched.append(m)
        if not has_sys:
            patched.insert(0, {"role": "system", "content": "You are a helpful assistant. /no_think"})
        return patched

    async def _npu_stream(self, messages, max_tokens) -> AsyncIterator[str]:
        payload = {"model": self.config.draft_model, "messages": self._patch_messages(messages),
                   "max_tokens": max_tokens, "temperature": self.config.temperature, "stream": True}
        async with self._session.post(f"{self.draft_url}/v1/chat/completions", json=payload) as resp:
            in_think = False
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
                    if line[6:].strip() == "[DONE]" if line.startswith("data: ") else False:
                        break
                    continue
                try:
                    delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
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

    async def _npu_collect(self, messages, max_tokens) -> str:
        payload = {"model": self.config.draft_model, "messages": self._patch_messages(messages),
                   "max_tokens": max_tokens, "temperature": self.config.temperature, "stream": False}
        async with self._session.post(f"{self.draft_url}/v1/chat/completions", json=payload) as resp:
            data = await resp.json()
        text = data["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # ── GPU (llama.cpp) ──────────────────────────────────────────────────

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
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + prompt
        prompt += f"<|im_start|>assistant\n{prefix}"
        return prompt

    async def _gpu_stream(self, messages, prefix, max_tokens) -> AsyncIterator[str]:
        payload = {"prompt": self._build_prompt(messages, prefix), "n_predict": max_tokens,
                   "temperature": self.config.temperature, "top_p": self.config.top_p,
                   "cache_prompt": True, "stream": True}
        async with self._session.post(f"{self.gpu_url}/completion", json=payload) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
                    continue
                try:
                    token = json.loads(line[6:]).get("content", "")
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
        """Pre-warm GPU KV cache with the prompt (generates 1 token)."""
        try:
            payload = {"prompt": self._build_prompt(messages), "n_predict": 1,
                       "temperature": 0.0, "cache_prompt": True, "stream": False}
            async with self._session.post(f"{self.gpu_url}/completion", json=payload) as resp:
                await resp.json()
        except Exception:
            pass

    # ── Pipeline: NPU streams + GPU continues ────────────────────────────

    async def pipeline_stream(self, messages, max_tokens) -> AsyncIterator[str]:
        cfg = self.config
        npu_tokens = min(max_tokens // 2, cfg.draft_tokens)
        gpu_tokens = max_tokens - npu_tokens

        t0 = time.perf_counter()

        # Fire GPU warmup in background while NPU streams
        warmup_task = asyncio.create_task(self._gpu_warmup(messages))

        # NPU streams to user — instant TTFT
        npu_text = ""
        async for chunk in self._npu_stream(messages, npu_tokens):
            npu_text += chunk
            yield chunk

        npu_ms = (time.perf_counter() - t0) * 1000
        if cfg.verbose:
            self.log.info(f"[NPU] {len(npu_text.split())} words in {npu_ms:.0f}ms "
                          f"({len(npu_text.split()) / (npu_ms / 1000):.1f} w/s)")

        # Ensure GPU warmup is done
        await warmup_task

        # GPU continues from NPU output — cache is warm
        t1 = time.perf_counter()
        async for chunk in self._gpu_stream(messages, npu_text, gpu_tokens):
            yield chunk

        if cfg.verbose:
            gpu_ms = (time.perf_counter() - t1) * 1000
            total_ms = (time.perf_counter() - t0) * 1000
            self.log.info(f"[GPU] Continued in {gpu_ms:.0f}ms")
            self.log.info(f"[TOTAL] {total_ms:.0f}ms (NPU {npu_ms:.0f}ms + GPU {gpu_ms:.0f}ms)")

    # ── Mode router ──────────────────────────────────────────────────────

    async def generate(self, messages, max_tokens, mode=None) -> AsyncIterator[str]:
        mode = mode or self.config.mode
        if mode == "npu":
            async for c in self._npu_stream(messages, max_tokens):
                yield c
        elif mode == "gpu":
            async for c in self._gpu_stream(messages, "", max_tokens):
                yield c
        elif mode == "auto":
            user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
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
# OpenAI-Compatible API Server
# ═════════════════════════════════════════════════════════════════════════════

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Max-Age": "3600",
}

class APIServer:
    def __init__(self, engine: InferenceEngine, config: PipelineConfig):
        self.engine = engine
        self.config = config
        self.log = logging.getLogger("api")
        self.app = web.Application()
        r = self.app.router
        r.add_route("OPTIONS", "/{path:.*}", self._options)
        r.add_get("/", self._root)
        r.add_get("/health", self._health)
        r.add_get("/v1/health", self._health)
        r.add_get("/v1/models", self._models)
        r.add_post("/v1/chat/completions", self._chat)

    async def _options(self, req):
        return web.Response(status=200, headers=CORS)

    async def _root(self, req):
        return web.json_response({
            "name": "Strix Halo NPU+GPU Pipeline",
            "version": __version__,
            "accelerators": {"npu": self.config.draft_model, "gpu": os.path.basename(self.config.gpu_model)},
            "api": f"http://{self.config.host}:{self.config.port}/v1",
        }, headers=CORS)

    async def _health(self, req):
        return web.json_response({"status": "ok"}, headers=CORS)

    async def _models(self, req):
        return web.json_response({"object": "list", "data": [
            {"id": "strix-speculative", "object": "model", "owned_by": "local"},
            {"id": "npu", "object": "model", "owned_by": "local"},
            {"id": "gpu", "object": "model", "owned_by": "local"},
            {"id": "auto", "object": "model", "owned_by": "local"},
        ]}, headers=CORS)

    async def _chat(self, req):
        body = await req.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", 512)
        model = body.get("model", "strix-speculative")

        if body.get("temperature") is not None:
            self.engine.config.temperature = body["temperature"]

        mode = {"strix-speculative": "pipeline", "npu": "npu",
                "gpu": "gpu", "auto": "auto"}.get(model, "pipeline")

        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        ts = int(time.time())

        if stream:
            resp = web.StreamResponse(headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive", **CORS})
            await resp.prepare(req)
            async for chunk in self.engine.generate(messages, max_tokens, mode):
                sse = {"id": rid, "object": "chat.completion.chunk", "created": ts,
                       "model": model, "choices": [{"index": 0,
                       "delta": {"content": chunk}, "finish_reason": None}]}
                await resp.write(f"data: {json.dumps(sse)}\n\n".encode())
            await resp.write(f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': ts, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            return resp
        else:
            full = ""
            t0 = time.perf_counter()
            async for chunk in self.engine.generate(messages, max_tokens, mode):
                full += chunk
            ms = (time.perf_counter() - t0) * 1000
            return web.json_response({
                "id": rid, "object": "chat.completion", "created": ts, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": full},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": len(full.split()),
                          "total_tokens": len(full.split())},
                "timings": {"total_ms": round(ms, 1)},
            }, headers=CORS)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

BANNER = """
\033[1;32m
  ╔═══════════════════════════════════════════════════════════╗
  ║      STRIX HALO NPU+GPU LLM PIPELINE — v{version}          ║
  ╠═══════════════════════════════════════════════════════════╣
  ║                                                           ║
  ║  NPU:  {npu:<20s} :{npu_port}  (XDNA2, FLM)       ║
  ║  GPU:  {gpu:<20s} :{gpu_port}  (RDNA 3.5, Vulkan)  ║
  ║                                                           ║
  ║  API:  http://{ip}:{api_port}/v1                     ║
  ║                                                           ║
  ║  Models:                                                  ║
  ║    strix-speculative → NPU streams + GPU continues        ║
  ║    npu               → NPU only (fast, low power)         ║
  ║    gpu               → GPU only (high quality)            ║
  ║    auto              → Smart routing                      ║
  ║                                                           ║
  ║  Open WebUI: Settings → Connections → Add                 ║
  ║    URL: http://{ip}:{api_port}/v1                    ║
  ║    Key: anything                                          ║
  ║                                                           ║
  ║  Ctrl+C to stop                                           ║
  ╚═══════════════════════════════════════════════════════════╝
\033[0m"""


async def run_server(config: PipelineConfig):
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
Examples:
  %(prog)s --gpu-model ./models/Qwen3-8B-Q4_K_M.gguf
  %(prog)s --gpu-model ./models/Qwen3-14B-Q4_K_M.gguf --draft qwen3:4b
  %(prog)s --gpu-model ./models/Qwen3-32B-Q4_K_M.gguf --draft qwen3:4b --draft-tokens 64 -v

Then connect Open WebUI to: http://<your-ip>:11435/v1
        """,
    )
    parser.add_argument("--gpu-model", required=True, help="Path to GGUF model for GPU")
    parser.add_argument("--draft", default="qwen3:1.7b", help="FLM draft model (default: qwen3:1.7b)")
    parser.add_argument("--draft-tokens", type=int, default=128, help="NPU draft length (default: 128)")
    parser.add_argument("--draft-port", type=int, default=52625, help="FLM server port")
    parser.add_argument("--gpu-port", type=int, default=9999, help="llama.cpp server port")
    parser.add_argument("--gpu-ctx", type=int, default=4096, help="GPU context size")
    parser.add_argument("--gpu-layers", type=int, default=99, help="GPU layers to offload")
    parser.add_argument("--port", type=int, default=11435, help="API server port (default: 11435)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--mode", choices=["pipeline", "npu", "gpu", "auto"], default="pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show timing stats")
    parser.add_argument("--no-launch", action="store_true",
                        help="Don't start FLM/llama.cpp (assume already running)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("main")

    config = PipelineConfig(
        draft_model=args.draft,
        draft_port=args.draft_port,
        gpu_model=args.gpu_model,
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

    # Setup environment
    env = Environment(log)
    env.setup_all()

    proc_mgr = None

    if not args.no_launch:
        # Kill any leftover processes
        for pattern in ["flm serve", f"llama-server.*{config.gpu_port}", "strix_halo_pipeline"]:
            subprocess.run(["pkill", "-f", pattern], capture_output=True)
        time.sleep(1)

        proc_mgr = ProcessManager(config, log)

        # Start FLM (NPU)
        if not proc_mgr.start_flm():
            log.error("Failed to start NPU server")
            proc_mgr.stop_all()
            sys.exit(1)

        # Start llama.cpp (GPU)
        if not proc_mgr.start_llama():
            log.error("Failed to start GPU server")
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
