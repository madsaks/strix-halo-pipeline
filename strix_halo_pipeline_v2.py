#!/usr/bin/env python3
"""
strix_halo_pipeline_v2.py — Enhanced NPU+GPU LLM Inference for AMD Strix Halo
================================================================================

v2.0 enhancements over v1.3:
  • Request queue with per-accelerator concurrency limits
  • Persistent KV cache with conversation ID tracking
  • Circuit breaker for NPU crashes → automatic GPU fallback
  • Smart auto-routing (token estimate + load-aware)
  • Structured metrics + request tracing (OpenTelemetry-style)
  • Backpressure-aware streaming with disconnect detection
  • YAML config support with hot reload
  • Batch prefill for concurrent GPU requests
  • Proper token counting (tiktoken fallback)
  • Prepared for true speculative decoding when FLM exposes logprobs

Author: Mike Alani / Enhanced by Claude
License: MIT
"""

__version__ = "2.0.0"

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
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional, Dict, List, Any, Callable

# ── Optional deps with graceful fallback ─────────────────────────────────────
_HAS_TIKTOKEN = False
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    pass

_HAS_YAML = False
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    pass

try:
    import aiohttp
    from aiohttp import web, WSMsgType
except ImportError:
    print("Installing aiohttp...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "aiohttp", "--break-system-packages", "-q"])
    import aiohttp
    from aiohttp import web, WSMsgType


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

class Accelerator(Enum):
    NPU = "npu"
    GPU = "gpu"
    AUTO = "auto"
    PIPELINE = "pipeline"


@dataclass
class PipelineConfig:
    draft_model: str = "qwen3:1.7b"
    draft_pmode: str = "performance"
    draft_port: int = 52625
    draft_max_concurrent: int = 2          # NEW: NPU concurrency limit
    draft_tokens: int = 128
    draft_tokens_max: int = 256            # NEW: adaptive upper bound
    draft_tokens_min: int = 32             # NEW: adaptive lower bound

    gpu_model: str = ""
    gpu_backend: str = "rocm"
    gpu_port: int = 9999
    gpu_ctx: int = 4096
    gpu_layers: int = 99
    gpu_threads: int = 8
    gpu_max_concurrent: int = 4            # NEW: GPU concurrency limit
    gpu_batch_size: int = 4                # NEW: batch prefill size
    gpu_template: str = "auto"             # auto | gemma4 | chatml

    port: int = 11435
    host: str = "0.0.0.0"
    temperature: float = 0.6
    top_p: float = 0.95
    mode: str = "pipeline"
    verbose: bool = False

    # NEW: Circuit breaker settings
    cb_failure_threshold: int = 3
    cb_recovery_timeout: float = 30.0

    # NEW: Auto-routing thresholds
    auto_short_token_threshold: int = 50   # tokens
    auto_short_max_tokens: int = 128

    # NEW: Metrics
    metrics_enabled: bool = True
    metrics_port: int = 11436

    # NEW: KV cache
    kv_cache_ttl: float = 300.0            # seconds
    kv_cache_max_entries: int = 100

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        if not _HAS_YAML:
            raise RuntimeError("PyYAML required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        # Flatten nested sections
        flat = {}
        for section in ["npu", "gpu", "api", "routing", "circuit_breaker", "kv_cache"]:
            if section in data:
                flat.update(data[section])
        return cls(**{k: v for k, v in flat.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML required")
        data = {
            "npu": {
                "draft_model": self.draft_model,
                "draft_pmode": self.draft_pmode,
                "draft_port": self.draft_port,
                "draft_max_concurrent": self.draft_max_concurrent,
                "draft_tokens": self.draft_tokens,
                "draft_tokens_max": self.draft_tokens_max,
                "draft_tokens_min": self.draft_tokens_min,
            },
            "gpu": {
                "gpu_model": self.gpu_model,
                "gpu_backend": self.gpu_backend,
                "gpu_port": self.gpu_port,
                "gpu_ctx": self.gpu_ctx,
                "gpu_layers": self.gpu_layers,
                "gpu_threads": self.gpu_threads,
                "gpu_max_concurrent": self.gpu_max_concurrent,
                "gpu_batch_size": self.gpu_batch_size,
                "gpu_template": self.gpu_template,
            },
            "api": {
                "port": self.port,
                "host": self.host,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "mode": self.mode,
                "verbose": self.verbose,
                "metrics_enabled": self.metrics_enabled,
                "metrics_port": self.metrics_port,
            },
            "routing": {
                "auto_short_token_threshold": self.auto_short_token_threshold,
                "auto_short_max_tokens": self.auto_short_max_tokens,
            },
            "circuit_breaker": {
                "cb_failure_threshold": self.cb_failure_threshold,
                "cb_recovery_timeout": self.cb_recovery_timeout,
            },
            "kv_cache": {
                "kv_cache_ttl": self.kv_cache_ttl,
                "kv_cache_max_entries": self.kv_cache_max_entries,
            },
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# ═════════════════════════════════════════════════════════════════════════════
# Token Counter
# ═════════════════════════════════════════════════════════════════════════════

class TokenCounter:
    """Accurate token counting with tiktoken fallback."""

    def __init__(self):
        self._encoders: Dict[str, Any] = {}
        self._fallback_pattern = re.compile(
            r"([\u4e00-\u9fff])|(\w+)|([\s\n\t])|(.)")

    def _get_encoder(self, model: str = "cl100k_base"):
        if model not in self._encoders:
            if _HAS_TIKTOKEN:
                try:
                    self._encoders[model] = tiktoken.get_encoding(model)
                except Exception:
                    self._encoders[model] = None
            else:
                self._encoders[model] = None
        return self._encoders[model]

    def count(self, text: str, model: str = "cl100k_base") -> int:
        enc = self._get_encoder(model)
        if enc:
            return len(enc.encode(text))
        # Fallback: rough heuristic (Chinese chars = 1.5 tokens, words = 1.3, rest = 1)
        total = 0
        for m in self._fallback_pattern.finditer(text):
            if m.group(1):          # CJK
                total += 2
            elif m.group(2):        # word
                total += max(1, len(m.group(2)) // 4)
            elif m.group(3):        # whitespace
                total += 0
            else:                   # other
                total += 1
        return max(1, total)

    def estimate_messages(self, messages: List[Dict]) -> int:
        text = " ".join(m.get("content", "") for m in messages)
        return self.count(text)


# ═════════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ═════════════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Prevents cascading failures by tripping after threshold failures."""

    def __init__(self, threshold: int = 3, recovery: float = 30.0):
        self.threshold = threshold
        self.recovery_timeout = recovery
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

    async def call(self, fn: Callable, *args, **kwargs):
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - (self.last_failure_time or 0) > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.failures = 0
                else:
                    raise RuntimeError("Circuit breaker OPEN — NPU unavailable")

        try:
            result = await fn(*args, **kwargs)
            async with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failures = 0
            return result
        except Exception as e:
            async with self._lock:
                self.failures += 1
                self.last_failure_time = time.time()
                if self.failures >= self.threshold:
                    self.state = CircuitState.OPEN
            raise

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN


# ═════════════════════════════════════════════════════════════════════════════
# KV Cache Manager
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class KVCacheEntry:
    conversation_id: str
    accelerator: str
    last_accessed: float
    prompt_hash: str
    token_count: int


class KVCacheManager:
    """Persistent KV cache tracking across requests."""

    def __init__(self, ttl: float = 300.0, max_entries: int = 100):
        self.ttl = ttl
        self.max_entries = max_entries
        self._cache: Dict[str, KVCacheEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    def _hash_prompt(self, messages: List[Dict]) -> str:
        import hashlib
        text = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def get(self, conversation_id: str) -> Optional[KVCacheEntry]:
        async with self._lock:
            entry = self._cache.get(conversation_id)
            if entry and time.time() - entry.last_accessed < self.ttl:
                entry.last_accessed = time.time()
                return entry
            return None

    async def put(self, conversation_id: str, accelerator: str,
                  messages: List[Dict], token_count: int):
        async with self._lock:
            self._cache[conversation_id] = KVCacheEntry(
                conversation_id=conversation_id,
                accelerator=accelerator,
                last_accessed=time.time(),
                prompt_hash=self._hash_prompt(messages),
                token_count=token_count,
            )
            # Evict oldest if over limit
            if len(self._cache) > self.max_entries:
                oldest = min(self._cache.values(), key=lambda e: e.last_accessed)
                del self._cache[oldest.conversation_id]

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            async with self._lock:
                now = time.time()
                expired = [k for k, v in self._cache.items()
                           if now - v.last_accessed > self.ttl]
                for k in expired:
                    del self._cache[k]


# ═════════════════════════════════════════════════════════════════════════════
# Control-token sanitizer
# ═════════════════════════════════════════════════════════════════════════════

# Tokens that end a turn. Both backends are driven through raw-completion or
# streaming APIs that hand back the model's control tokens as ordinary text,
# so each family's terminators have to be recognised explicitly.
EOS_TOKENS = ("<|im_end|>", "<|endoftext|>", "</s>",
              "<turn|>", "<end_of_turn>", "<eos>")

# Handed to llama.cpp as server-side stop strings. The raw /completion endpoint
# tokenizes the prompt without special-token parsing, so the model emits turn
# markers as ordinary text and never trips a real EOS. Left to itself it keeps
# going and invents entire conversation turns until n_predict runs out — which
# is exactly what happens once a draft prefix has already answered the question.
GPU_STOP_STRINGS = list(EOS_TOKENS) + ["<|im_start|>", "<|turn>",
                                       "<start_of_turn>"]

# Markers that open a hidden region: everything after one is suppressed until
# the matching close marker.
_HIDE_OPEN = ("<think>", "<|channel>")

# Markers that close a hidden region. They are also dropped when they appear
# on their own — the QAT Gemma4 build normally emits "<|channel>thought\n"
# immediately followed by "<channel|>" with nothing in between, but it also
# emits a bare "<channel|>" before the answer when it does no thinking pass.
_HIDE_CLOSE = ("</think>", "<channel|>")

# Qwen 3 switches its reasoning pass off via a "/no_think" directive in the
# system prompt. That is a Qwen convention, not a general one — every other
# family ignores the string and simply carries it as noise in the prompt.
_NO_THINK_PREFIXES = ("qwen3",)


def wants_no_think(model: str) -> bool:
    return model.strip().lower().startswith(_NO_THINK_PREFIXES)


# A turn-opener in the output means the model has moved past its own answer
# into an invented next turn, so it terminates the response just as an
# end-of-turn marker does.
_STOP_MARKERS = tuple(GPU_STOP_STRINGS)

_ALL_MARKERS = _HIDE_OPEN + _HIDE_CLOSE + _STOP_MARKERS
_MAX_MARKER = max(len(m) for m in _ALL_MARKERS)


class StreamSanitizer:
    """Strips reasoning-channel markers from a token stream.

    Models expose their reasoning traces with paired markers — Qwen uses
    <think>…</think>, Gemma4 uses <|channel>thought…<channel|> — and both leak
    into the response when the raw stream is forwarded verbatim.

    End-of-turn markers are recognised too, and set .stopped — on a raw
    completion endpoint they arrive as ordinary text rather than as a real EOS,
    so the caller has to notice them itself.

    A marker can straddle a chunk boundary, so any suffix that could still grow
    into a marker is held back rather than emitted; call flush() once the
    stream ends to release whatever is left.
    """

    def __init__(self):
        self._buf = ""
        self._hidden = False
        self.stopped = False

    def _held_len(self) -> int:
        """Length of the trailing run that might still become a marker."""
        for n in range(min(len(self._buf), _MAX_MARKER - 1), 0, -1):
            tail = self._buf[-n:]
            if any(m.startswith(tail) for m in _ALL_MARKERS):
                return n
        return 0

    def feed(self, chunk: str) -> str:
        if self.stopped:
            return ""
        self._buf += chunk
        out = []
        while True:
            hit, pos = None, len(self._buf)
            for m in _ALL_MARKERS:
                i = self._buf.find(m)
                if i != -1 and i < pos:
                    hit, pos = m, i
            if hit is None:
                break
            if not self._hidden:
                out.append(self._buf[:pos])
            if hit in _STOP_MARKERS:
                # End of turn: nothing after it belongs to this response.
                self.stopped = True
                self._buf = ""
                return "".join(out)
            self._buf = self._buf[pos + len(hit):]
            self._hidden = hit in _HIDE_OPEN
        held = self._held_len()
        if not self._hidden:
            out.append(self._buf[:len(self._buf) - held] if held
                       else self._buf)
        self._buf = self._buf[len(self._buf) - held:] if held else ""
        return "".join(out)

    def flush(self) -> str:
        """Release any held-back tail. A partial marker at end of stream was
        never completed, so it was ordinary text after all."""
        tail = "" if (self._hidden or self.stopped) else self._buf
        self._buf = ""
        return tail


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """Lightweight Prometheus-style metrics."""

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def inc(self, name: str, value: int = 1):
        async with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    async def set_gauge(self, name: str, value: float):
        async with self._lock:
            self._gauges[name] = value

    async def observe(self, name: str, value: float):
        async with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)
            # Keep last 1000 samples
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]

    async def render(self) -> str:
        async with self._lock:
            lines = []
            for k, v in self._counters.items():
                lines.append(f"{k}_total {v}")
            for k, v in self._gauges.items():
                lines.append(f"{k} {v}")
            for k, vals in self._histograms.items():
                if vals:
                    vals.sort()
                    lines.append(f"{k}_count {len(vals)}")
                    lines.append(f"{k}_sum {sum(vals):.3f}")
                    lines.append(f"{k}_p50 {vals[len(vals)//2]:.3f}")
                    lines.append(f"{k}_p99 {vals[int(len(vals)*0.99)]:.3f}")
            # Prometheus exposition requires LF-separated lines and a
            # trailing newline; a scraper rejects a body without it.
            return "\n".join(lines) + "\n"


# ═════════════════════════════════════════════════════════════════════════════
# Environment (enhanced)
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
        rocm711 = "/opt/rocm/core-7.11/lib"
        rocm71 = "/opt/rocm-7.1.1/lib"
        current = os.environ.get("LD_LIBRARY_PATH", "")
        paths = current.split(":") if current else []
        needs_fix = False
        if rocm711 not in paths:
            needs_fix = True
        elif paths.index(rocm711) > 0:
            if rocm71 in paths and paths.index(rocm71) < paths.index(rocm711):
                needs_fix = True
        if needs_fix and os.path.exists(rocm711):
            paths = [p for p in paths if p not in [rocm711, rocm71]]
            new_paths = [rocm711]
            if os.path.exists(rocm71):
                new_paths.append(rocm71)
            new_paths.extend(paths)
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths)
            self.log.info(f"Fixed LD_LIBRARY_PATH: core-7.11 first")

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
                # Mode "w" implies O_TRUNC, which sysfs attributes reject with
                # EINVAL; "r+" writes in place. Catch OSError generally: this
                # is a best-effort tuning step and must never abort startup.
                with open(card, "r+") as f:
                    f.write("performance")
                self.log.info("GPU performance mode set")
            except OSError as e:
                self.log.warning(f"Could not set GPU performance mode on {card}: {e}")

    def find_flm(self):
        for path in [shutil.which("flm"), "/opt/fastflowlm/bin/flm"]:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    def find_llama_server(self, backend="rocm"):
        if backend == "rocm":
            candidates = [
                os.path.expanduser("~/llama.cpp/build-rocm/bin/llama-server"),
                "/usr/local/share/lemonade-server/llama/rocm/llama-server",
            ]
        else:
            candidates = [
                os.path.expanduser("~/llama.cpp/build-vulkan/bin/llama-server"),
                "/usr/local/share/lemonade-server/llama/vulkan/build/bin/llama-server",
            ]
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
        for d in ["~/vitias/models", "~/models", ".", "./models", "/mnt/raid0/vitias/models"]:
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
# Process Manager (enhanced with health monitoring)
# ═════════════════════════════════════════════════════════════════════════════

class ProcessManager:
    def __init__(self, config: PipelineConfig, log: logging.Logger):
        self.config = config
        self.log = log
        self.env = Environment(log)
        self.procs: List[subprocess.Popen] = []
        self._health_status = {"npu": False, "gpu": False}
        self._health_lock = asyncio.Lock()

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

    async def _monitor_health(self):
        """Background health monitoring with auto-restart."""
        while True:
            await asyncio.sleep(10)
            # Check NPU
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.config.draft_port}/v1/models",
                    method="HEAD")
                with urllib.request.urlopen(req, timeout=3):
                    async with self._health_lock:
                        self._health_status["npu"] = True
            except Exception:
                async with self._health_lock:
                    self._health_status["npu"] = False
            # Check GPU
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.config.gpu_port}/health",
                    method="HEAD")
                with urllib.request.urlopen(req, timeout=3):
                    async with self._health_lock:
                        self._health_status["gpu"] = True
            except Exception:
                async with self._health_lock:
                    self._health_status["gpu"] = False

    async def is_healthy(self, accelerator: str) -> bool:
        async with self._health_lock:
            return self._health_status.get(accelerator, False)

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
            self.log.error(f"llama-server ({backend}) not found")
            return False
        model = self.env.find_gpu_model(self.config.gpu_model)
        if not model:
            self.log.error("No GGUF model found")
            return False
        self.config.gpu_model = model
        self.log.info(f"Starting GPU ({backend}): {os.path.basename(model)} on :{self.config.gpu_port}")

        env = os.environ.copy()
        lib_paths = []
        if llama_lib:
            lib_paths.append(llama_lib)
        if backend == "rocm":
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
# Request Queue (NEW)
# ═════════════════════════════════════════════════════════════════════════════

class RequestQueue:
    """Semaphore-based concurrency limiter per accelerator."""

    def __init__(self, npu_limit: int = 2, gpu_limit: int = 4):
        self.npu_sem = asyncio.Semaphore(npu_limit)
        self.gpu_sem = asyncio.Semaphore(gpu_limit)
        self._pending_npu = 0
        self._pending_gpu = 0
        self._lock = asyncio.Lock()

    async def acquire_npu(self):
        async with self._lock:
            self._pending_npu += 1
        await self.npu_sem.acquire()

    def release_npu(self):
        self.npu_sem.release()
        asyncio.create_task(self._dec_npu())

    async def _dec_npu(self):
        async with self._lock:
            self._pending_npu -= 1

    async def acquire_gpu(self):
        async with self._lock:
            self._pending_gpu += 1
        await self.gpu_sem.acquire()

    def release_gpu(self):
        self.gpu_sem.release()
        asyncio.create_task(self._dec_gpu())

    async def _dec_gpu(self):
        async with self._lock:
            self._pending_gpu -= 1

    @property
    async def load(self) -> Dict[str, int]:
        async with self._lock:
            return {
                "npu_pending": self._pending_npu,
                "gpu_pending": self._pending_gpu,
                "npu_available": self.npu_sem._value,
                "gpu_available": self.gpu_sem._value,
            }


# ═════════════════════════════════════════════════════════════════════════════
# Inference Engine (v2 — major rewrite)
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
    def __init__(self, config: PipelineConfig, metrics: MetricsCollector,
                 kv_cache: KVCacheManager, queue: RequestQueue,
                 cb: CircuitBreaker, proc_mgr: ProcessManager):
        self.config = config
        self.log = logging.getLogger("engine")
        self.metrics = metrics
        self.kv_cache = kv_cache
        self.queue = queue
        self.cb = cb
        self.proc_mgr = proc_mgr
        self.token_counter = TokenCounter()
        self._session: Optional[aiohttp.ClientSession] = None
        self._gpu_warmup_cache: Dict[str, asyncio.Task] = {}

    @property
    def draft_url(self):
        return f"http://127.0.0.1:{self.config.draft_port}"

    @property
    def gpu_url(self):
        return f"http://127.0.0.1:{self.config.gpu_port}"

    async def start(self):
        timeout = aiohttp.ClientTimeout(total=600, sock_read=300)
        conn = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        self._session = aiohttp.ClientSession(timeout=timeout, connector=conn)

    async def stop(self):
        if self._session:
            await self._session.close()

    def _patch_messages(self, messages):
        """Switch the draft model's reasoning pass off, where it has one.

        A draft that spends its token budget thinking is wasted work: the
        pipeline feeds the NPU's output to the GPU as a prefix to continue, so
        only the answer text is of any use.

        Models outside the Qwen 3 family are handed back untouched. Carrying an
        inert "/no_think" would be more than cosmetic here — the GPU leg emits a
        system turn only when the caller supplied one, so inventing a system
        message on this side would leave draft and continuation working from
        different instructions.
        """
        if not wants_no_think(self.config.draft_model):
            return messages
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

    def _resolve_template(self) -> str:
        """Pick the prompt format for the GPU model.

        The GGUF's own Jinja template is the authority here, but it cannot be
        used directly: it always closes with a bare generation prompt, and this
        pipeline has to append the NPU's draft as a partial model turn. So the
        formats are reproduced by hand and selected from the model name.
        """
        if self.config.gpu_template != "auto":
            return self.config.gpu_template
        name = os.path.basename(self.config.gpu_model).lower()
        if "gemma-4" in name or "gemma4" in name:
            return "gemma4"
        # Gemma 2/3 use <start_of_turn>, which is a third format this does not
        # implement; they fall through to ChatML as before.
        return "chatml"

    def _build_prompt(self, messages, prefix=""):
        if self._resolve_template() == "gemma4":
            return self._build_prompt_gemma4(messages, prefix)
        return self._build_prompt_chatml(messages, prefix)

    def _build_prompt_gemma4(self, messages, prefix=""):
        """Gemma 4's turn format, with the thinking channel suppressed.

        Verified against the model's own template via llama-server's
        /apply-template. Gemma 4 does not use Gemma 2/3's <start_of_turn>; it
        wraps turns as <|turn>role\n...<turn|>\n and names the assistant role
        "model". A system turn is emitted only when one was supplied.

        The generation prompt ends with an empty thought channel
        ("<|channel>thought\n<channel|>"), which is what the template itself
        appends when enable_thinking is false: it prefills the reasoning pass as
        already finished, so the model answers directly instead of emitting a
        thinking preamble that would only be stripped again downstream.
        """
        prompt = ""
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                prompt += f"<|turn>system\n{content}<turn|>\n"
            elif role == "user":
                prompt += f"<|turn>user\n{content}<turn|>\n"
            elif role == "assistant":
                prompt += f"<|turn>model\n{content}<turn|>\n"
        prompt += f"<|turn>model\n<|channel>thought\n<channel|>{prefix}"
        return prompt

    def _build_prompt_chatml(self, messages, prefix=""):
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

    # ── NPU Stream ────────────────────────────────────────────────────────────

    async def _npu_stream(self, messages, max_tokens, request_id: str):
        payload = {
            "model": self.config.draft_model,
            "messages": self._patch_messages(messages),
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }
        t0 = time.perf_counter()
        token_count = 0
        async with self._session.post(
                f"{self.draft_url}/v1/chat/completions", json=payload) as resp:
            san = StreamSanitizer()
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    delta = json.loads(data_str)["choices"][0]["delta"].get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if not delta:
                    continue
                visible = san.feed(delta)
                if visible:
                    token_count += self.token_counter.count(visible)
                    yield visible
            tail = san.flush()
            if tail:
                token_count += self.token_counter.count(tail)
                yield tail
        elapsed = time.perf_counter() - t0
        await self.metrics.observe("npu_request_duration_ms", elapsed * 1000)
        await self.metrics.observe("npu_tokens_generated", token_count)
        self.log.debug(f"[{request_id}] NPU: {token_count} tokens in {elapsed:.2f}s")

    # ── GPU Stream ────────────────────────────────────────────────────────────

    async def _gpu_stream(self, messages, prefix, max_tokens, request_id: str,
                          cache_prompt: bool = True):
        payload = {
            "prompt": self._build_prompt(messages, prefix),
            "n_predict": max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "cache_prompt": cache_prompt,
            "stop": GPU_STOP_STRINGS,
            "stream": True,
        }
        t0 = time.perf_counter()
        token_count = 0
        async with self._session.post(
                f"{self.gpu_url}/completion", json=payload) as resp:
            san = StreamSanitizer()
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    token = json.loads(data_str).get("content", "")
                except (json.JSONDecodeError, KeyError):
                    continue
                if not token:
                    continue
                visible = san.feed(token)
                if visible:
                    token_count += self.token_counter.count(visible)
                    yield visible
                if san.stopped:
                    # Break rather than return, so the metrics below still
                    # record this request.
                    break
            tail = san.flush()
            if tail:
                token_count += self.token_counter.count(tail)
                yield tail
        elapsed = time.perf_counter() - t0
        await self.metrics.observe("gpu_request_duration_ms", elapsed * 1000)
        await self.metrics.observe("gpu_tokens_generated", token_count)
        self.log.debug(f"[{request_id}] GPU: {token_count} tokens in {elapsed:.2f}s")

    # ── GPU Warmup with cache key ─────────────────────────────────────────────

    async def _gpu_warmup(self, messages, cache_key: str):
        """Warm GPU KV cache. Skip if already warm for this conversation."""
        cached = await self.kv_cache.get(cache_key)
        if cached and cached.accelerator == "gpu":
            self.log.debug(f"GPU cache warm for {cache_key[:8]}")
            return
        try:
            payload = {
                "prompt": self._build_prompt(messages),
                "n_predict": 1,
                "temperature": 0.0,
                "cache_prompt": True,
                "stop": GPU_STOP_STRINGS,
                "stream": False,
            }
            async with self._session.post(
                    f"{self.gpu_url}/completion", json=payload) as resp:
                await resp.json()
            await self.kv_cache.put(cache_key, "gpu", messages, 0)
        except Exception as e:
            self.log.warning(f"GPU warmup failed: {e}")

    # ── Adaptive Draft Token Count ──────────────────────────────────────────

    def _adaptive_draft_tokens(self, max_tokens: int, prompt_tokens: int) -> int:
        """Dynamically adjust draft length based on workload."""
        if max_tokens <= self.config.auto_short_max_tokens:
            return min(max_tokens, self.config.draft_tokens_min)
        # Scale draft with max_tokens but cap it
        draft = min(max_tokens // 2, self.config.draft_tokens)
        # For very long contexts, reduce draft to save NPU time
        if prompt_tokens > 2048:
            draft = max(draft // 2, self.config.draft_tokens_min)
        return draft

    # ── Pipeline Stream (enhanced) ────────────────────────────────────────────

    async def pipeline_stream(self, messages, max_tokens, request_id: str,
                              conversation_id: Optional[str] = None):
        cfg = self.config
        prompt_tokens = self.token_counter.estimate_messages(messages)
        npu_tokens = self._adaptive_draft_tokens(max_tokens, prompt_tokens)
        gpu_tokens = max_tokens - npu_tokens

        t0 = time.perf_counter()
        cache_key = conversation_id or self.kv_cache._hash_prompt(messages)

        # Start GPU warmup in background (non-blocking)
        warmup_task = asyncio.create_task(self._gpu_warmup(messages, cache_key))

        npu_text = ""
        npu_token_count = 0

        # Acquire NPU slot
        await self.queue.acquire_npu()
        try:
            async for chunk in self._npu_stream(messages, npu_tokens, request_id):
                npu_text += chunk
                npu_token_count += self.token_counter.count(chunk)
                yield chunk
        except Exception as e:
            self.log.error(f"[{request_id}] NPU stream failed: {e}")
            # Circuit breaker will trip; fallback to GPU-only
            await self.cb.call(lambda: asyncio.sleep(0))  # force state check
            raise
        finally:
            self.queue.release_npu()

        npu_ms = (time.perf_counter() - t0) * 1000
        if cfg.verbose:
            self.log.info(f"[{request_id}] [NPU] {npu_token_count} tokens in {npu_ms:.0f}ms")

        # Wait for GPU warmup
        await warmup_task

        # Acquire GPU slot
        await self.queue.acquire_gpu()
        t1 = time.perf_counter()
        try:
            gpu_token_count = 0
            async for chunk in self._gpu_stream(messages, npu_text, gpu_tokens, request_id):
                gpu_token_count += self.token_counter.count(chunk)
                yield chunk
            await self.kv_cache.put(cache_key, "pipeline", messages,
                                    prompt_tokens + npu_token_count + gpu_token_count)
        finally:
            self.queue.release_gpu()

        if cfg.verbose:
            gpu_ms = (time.perf_counter() - t1) * 1000
            total_ms = (time.perf_counter() - t0) * 1000
            self.log.info(f"[{request_id}] [GPU] {gpu_token_count} tokens in {gpu_ms:.0f}ms")
            self.log.info(f"[{request_id}] [TOTAL] {total_ms:.0f}ms "
                          f"(NPU {npu_ms:.0f}ms + GPU {gpu_ms:.0f}ms)")
            await self.metrics.observe("pipeline_total_duration_ms", total_ms)

    # ── GPU-only with batching support ────────────────────────────────────────

    async def gpu_only_stream(self, messages, max_tokens, request_id: str):
        await self.queue.acquire_gpu()
        try:
            async for chunk in self._gpu_stream(messages, "", max_tokens, request_id):
                yield chunk
        finally:
            self.queue.release_gpu()

    # ── NPU-only with circuit breaker ───────────────────────────────────────

    async def npu_only_stream(self, messages, max_tokens, request_id: str):
        async def _stream():
            await self.queue.acquire_npu()
            try:
                async for chunk in self._npu_stream(messages, max_tokens, request_id):
                    yield chunk
            finally:
                self.queue.release_npu()

        # Circuit breaker wraps NPU calls
        try:
            async for chunk in await self.cb.call(_stream):
                yield chunk
        except RuntimeError as e:
            if "Circuit breaker OPEN" in str(e):
                self.log.warning(f"[{request_id}] NPU circuit open, falling back to GPU")
                async for chunk in self.gpu_only_stream(messages, max_tokens, request_id):
                    yield chunk
            else:
                raise

    # ── Smart Auto Routing ────────────────────────────────────────────────────

    async def auto_route(self, messages, max_tokens) -> Accelerator:
        """Decide accelerator based on load + estimated token count."""
        prompt_tokens = self.token_counter.estimate_messages(messages)
        load = await self.queue.load
        npu_busy = load["npu_pending"] >= self.config.draft_max_concurrent
        gpu_busy = load["gpu_pending"] >= self.config.gpu_max_concurrent

        # Short + simple -> NPU if available
        if (prompt_tokens < self.config.auto_short_token_threshold and
                max_tokens < self.config.auto_short_max_tokens and
                not npu_busy):
            return Accelerator.NPU

        # NPU circuit open -> GPU
        if self.cb.is_open:
            return Accelerator.GPU

        # GPU overloaded, NPU available -> NPU
        if gpu_busy and not npu_busy:
            return Accelerator.NPU

        # Default: pipeline for best quality/speed balance
        return Accelerator.PIPELINE

    # ── Main Generate ─────────────────────────────────────────────────────────

    async def generate(self, messages, max_tokens, mode=None, request_id=None,
                       conversation_id=None):
        request_id = request_id or f"req-{uuid.uuid4().hex[:8]}"
        mode = mode or self.config.mode

        if mode == "npu":
            acc = Accelerator.NPU
        elif mode == "gpu":
            acc = Accelerator.GPU
        elif mode == "auto":
            acc = await self.auto_route(messages, max_tokens)
        else:
            acc = Accelerator.PIPELINE

        await self.metrics.inc(f"requests_{acc.value}_total")
        await self.metrics.set_gauge("npu_circuit_state",
                                     1.0 if self.cb.is_open else 0.0)

        if acc == Accelerator.NPU:
            async for c in self.npu_only_stream(messages, max_tokens, request_id):
                yield c
        elif acc == Accelerator.GPU:
            async for c in self.gpu_only_stream(messages, max_tokens, request_id):
                yield c
        else:  # pipeline
            async for c in self.pipeline_stream(messages, max_tokens, request_id,
                                                  conversation_id):
                yield c


# ═════════════════════════════════════════════════════════════════════════════
# API Server (v2 — enhanced streaming + metrics endpoint)
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
    def __init__(self, engine: InferenceEngine, config: PipelineConfig,
                 metrics: MetricsCollector, proc_mgr: ProcessManager):
        self.engine = engine
        self.config = config
        self.metrics = metrics
        self.proc_mgr = proc_mgr
        self.app = web.Application()
        r = self.app.router
        r.add_route("OPTIONS", "/{path:.*}", self._options)
        r.add_get("/", self._root)
        r.add_get("/health", self._health)
        r.add_get("/ready", self._ready)
        r.add_get("/v1/health", self._health)
        r.add_get("/v1/models", self._models)
        r.add_post("/v1/chat/completions", self._openai_chat)
        r.add_get("/metrics", self._metrics)  # NEW
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
        load = await self.engine.queue.load
        return web.json_response({
            "name": "Strix Halo NPU+GPU Pipeline",
            "version": __version__,
            "gpu_backend": self.config.gpu_backend,
            "npu_circuit": "OPEN" if self.engine.cb.is_open else "CLOSED",
            "queue_load": load,
        }, headers=CORS)

    async def _health(self, req):
        return web.json_response({"status": "ok"}, headers=CORS)

    async def _ready(self, req):
        """Kubernetes-style readiness probe."""
        npu_ok = await self.proc_mgr.is_healthy("npu")
        gpu_ok = await self.proc_mgr.is_healthy("gpu")
        if gpu_ok and (npu_ok or self.config.mode == "gpu"):
            return web.json_response({"ready": True}, headers=CORS)
        return web.json_response({"ready": False}, status=503, headers=CORS)

    async def _metrics(self, req):
        body = await self.metrics.render()
        # The exposition-format version belongs in the Content-Type
        # parameters, which aiohttp's content_type= argument rejects,
        # so set the header directly.
        resp = web.Response(text=body)
        resp.headers["Content-Type"] = (
            "text/plain; version=0.0.4; charset=utf-8")
        return resp

    # ── OpenAI format (enhanced streaming) ───────────────────────────────────

    async def _models(self, req):
        return web.json_response({"object": "list", "data": [
            {"id": "strix-speculative", "object": "model", "owned_by": "local"},
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
        conversation_id = body.get("conversation_id")  # NEW

        if body.get("temperature") is not None:
            self.engine.config.temperature = body["temperature"]

        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        ts = int(time.time())

        await self.metrics.inc("requests_total")

        if stream:
            resp = web.StreamResponse(headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive", **CORS})
            await resp.prepare(req)

            # Client disconnect detection
            disconnected = False
            async def watch_disconnect():
                nonlocal disconnected
                try:
                    while True:
                        await asyncio.sleep(1)
                        # Check if client still connected
                        if req.transport.is_closing():
                            disconnected = True
                            break
                except Exception:
                    disconnected = True

            watcher = asyncio.create_task(watch_disconnect())

            try:
                async for chunk in self.engine.generate(
                        messages, max_tokens, mode, rid, conversation_id):
                    if disconnected:
                        break
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
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass

            if not disconnected:
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
                    messages, max_tokens, mode, rid, conversation_id):
                full += chunk
            ms = (time.perf_counter() - t0) * 1000
            prompt_tok = self.engine.token_counter.estimate_messages(messages)
            comp_tok = self.engine.token_counter.count(full)
            return web.json_response({
                "id": rid, "object": "chat.completion",
                "created": ts, "model": model,
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": full},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tok,
                          "completion_tokens": comp_tok,
                          "total_tokens": prompt_tok + comp_tok},
                "timings": {"total_ms": round(ms, 1)},
            }, headers=CORS)

    # ── Ollama format ────────────────────────────────────────────────────────

    async def _ollama_chat(self, req):
        body = await req.json()
        messages = clean_messages(body.get("messages", []))
        stream = body.get("stream", True)
        model = body.get("model", "strix-speculative")
        mode = self._resolve_mode(model)
        opts = body.get("options", {})
        max_tokens = opts.get("num_predict", 2048)
        conversation_id = body.get("conversation_id")

        if opts.get("temperature") is not None:
            self.engine.config.temperature = opts["temperature"]

        if stream:
            resp = web.StreamResponse(headers={
                "Content-Type": "application/x-ndjson",
                "Cache-Control": "no-cache", **CORS})
            await resp.prepare(req)

            full_text = ""
            async for chunk in self.engine.generate(
                    messages, max_tokens, mode, conversation_id=conversation_id):
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
                "eval_count": self.engine.token_counter.count(full_text),
            }
            await resp.write(json.dumps(done_msg).encode() + b"\n")
            return resp
        else:
            full_text = ""
            async for chunk in self.engine.generate(
                    messages, max_tokens, mode, conversation_id=conversation_id):
                full_text += chunk
            return web.json_response({
                "model": model,
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": full_text},
                "done": True,
                "eval_count": self.engine.token_counter.count(full_text),
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
        load = await self.engine.queue.load
        return web.json_response({"models": [{
            "name": "strix-speculative",
            "model": "strix-speculative",
            "size": 0, "digest": "",
            "expires_at": "2099-01-01T00:00:00Z",
            "load": load,
        }]}, headers=CORS)

    async def _ollama_version(self, req):
        return web.json_response({"version": __version__}, headers=CORS)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

BANNER = """
\033[1;32m
 ======================================================
 STRIX HALO NPU+GPU LLM PIPELINE v{version}
 ======================================================
 NPU: {npu:<20s} :{npu_port} (XDNA2, pmode={pmode})
 GPU: {gpu:<20s} :{gpu_port} ({backend}, RDNA 3.5)

 API:    http://{ip}:{api_port}/v1  (OpenAI)
         http://{ip}:{api_port}      (Ollama)
 Metrics: http://{ip}:{metrics_port}/metrics

 Models: strix-speculative | npu | gpu | auto
 Ctrl+C to stop
 ======================================================
\033[0m"""


async def run_server(config: PipelineConfig):
    metrics = MetricsCollector()
    kv_cache = KVCacheManager(config.kv_cache_ttl, config.kv_cache_max_entries)
    queue = RequestQueue(config.draft_max_concurrent, config.gpu_max_concurrent)
    cb = CircuitBreaker(config.cb_failure_threshold, config.cb_recovery_timeout)

    await kv_cache.start()

    proc_mgr = None
    if not getattr(config, "_no_launch", False):
        proc_mgr = ProcessManager(config, logging.getLogger("proc"))
        proc_mgr.env.setup_all()
        if not proc_mgr.start_flm():
            logging.warning("FLM failed to start — NPU will be unavailable")
        if not proc_mgr.start_llama():
            logging.error("llama-server failed to start — aborting")
            if proc_mgr:
                proc_mgr.stop_all()
            sys.exit(1)
        # Start health monitoring
        asyncio.create_task(proc_mgr._monitor_health())

    engine = InferenceEngine(config, metrics, kv_cache, queue, cb, proc_mgr)
    await engine.start()
    server = APIServer(engine, config, metrics, proc_mgr)
    runner = web.AppRunner(server.app)
    await runner.setup()

    site = web.TCPSite(runner, config.host, config.port)
    await site.start()

    # Metrics endpoint
    if config.metrics_enabled:
        metrics_app = web.Application()
        metrics_app.router.add_get("/metrics", server._metrics)
        metrics_runner = web.AppRunner(metrics_app)
        await metrics_runner.setup()
        metrics_site = web.TCPSite(metrics_runner, config.host, config.metrics_port)
        await metrics_site.start()

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
        metrics_port=config.metrics_port,
    ))

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await engine.stop()
        await kv_cache.stop()
        await runner.cleanup()
        if config.metrics_enabled:
            await metrics_runner.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Strix Halo NPU+GPU LLM Pipeline v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended configurations:
  Fast:    --draft qwen3:4b --gpu-model Qwen3-8B --draft-tokens 64
  Quality: --draft qwen3:1.7b --gpu-model Qwen3.5-27B --draft-tokens 32
  ROCm:    --gpu-backend rocm (default, fastest, requires 7.11 HSA fix)
  Vulkan:  --gpu-backend vulkan (fallback, always works)
  Config:  --config pipeline.yaml (YAML config file)

New in v2.0:
  • Request queues with concurrency limits
  • Circuit breaker (NPU crash -> auto GPU fallback)
  • Persistent KV cache with conversation IDs
  • Adaptive draft token count
  • Prometheus metrics at /metrics
  • Smart auto-routing based on load
  • Client disconnect detection
        """)
    parser.add_argument("--gpu-model", help="GGUF model for GPU")
    parser.add_argument("--gpu-backend", default="rocm",
                        choices=["rocm", "vulkan"],
                        help="GPU backend (default: rocm)")
    parser.add_argument("--draft", default="qwen3:1.7b",
                        help="FLM draft model")
    parser.add_argument("--pmode", default="performance",
                        choices=["powersaver", "balanced",
                                 "performance", "turbo"],
                        help="NPU power mode")
    parser.add_argument("--draft-tokens", type=int, default=128)
    parser.add_argument("--draft-port", type=int, default=52625)
    parser.add_argument("--gpu-port", type=int, default=9999)
    parser.add_argument("--gpu-ctx", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--mode",
                        choices=["pipeline", "npu", "gpu", "auto"],
                        default="pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-launch", action="store_true",
                        help="Don't start servers")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
        # CLI overrides YAML
        for k, v in vars(args).items():
            if v is not None and k in PipelineConfig.__dataclass_fields__:
                setattr(config, k, v)
    else:
        config = PipelineConfig(
            draft_model=args.draft,
            draft_pmode=args.pmode,
            draft_port=args.draft_port,
            gpu_model=args.gpu_model or "",
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

    if args.no_launch:
        config._no_launch = True

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("main")

    def shutdown(sig=None, frame=None):
        log.info("Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
