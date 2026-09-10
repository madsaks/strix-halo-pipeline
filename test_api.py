#!/usr/bin/env python3
"""API-level tests for the pipeline's HTTP surface.

Run with the system interpreter, not whatever venv is first on PATH:
    /usr/bin/python3 test_api.py

test_stream_sanitizer.py covers string transformations. These cover the wire
protocols, which is where the escaped-newline corruption actually did its
damage: `data: {...}\\n\\n` written with literal backslash-n is not SSE, and no
amount of string-level testing would have caught it.

FastFlowLM and llama-server are replaced by mock aiohttp servers on ephemeral
ports, so nothing here needs the NPU, the GPU, or a model.
"""
import asyncio
import importlib.util
import json
import os
import sys

import aiohttp
from aiohttp import web

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "shp", os.path.join(_HERE, "strix_halo_pipeline_v2.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


# ── Mock backends ────────────────────────────────────────────────────────────

class MockBackends:
    """Stands in for FastFlowLM (:draft_port) and llama-server (:gpu_port).

    npu_chunks/gpu_chunks are the raw content strings each backend streams, so a
    test can inject control tokens exactly where it wants them.
    """

    def __init__(self, npu_chunks, gpu_chunks):
        self.npu_chunks = npu_chunks
        self.gpu_chunks = gpu_chunks
        self.gpu_payloads = []
        self.npu_payloads = []

    async def _npu_chat(self, req):
        self.npu_payloads.append(await req.json())
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        await resp.prepare(req)
        for c in self.npu_chunks:
            body = {"choices": [{"index": 0, "delta": {"content": c}}]}
            await resp.write(f"data: {json.dumps(body)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def _gpu_completion(self, req):
        payload = await req.json()
        self.gpu_payloads.append(payload)
        if not payload.get("stream"):
            # warmup path
            return web.json_response({"content": ""})
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        await resp.prepare(req)
        for c in self.gpu_chunks:
            await resp.write(f"data: {json.dumps({'content': c})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def start(self):
        npu = web.Application()
        npu.router.add_post("/v1/chat/completions", self._npu_chat)
        gpu = web.Application()
        gpu.router.add_post("/completion", self._gpu_completion)
        self._runners = []
        ports = []
        for app in (npu, gpu):
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            ports.append(site._server.sockets[0].getsockname()[1])
            self._runners.append(runner)
        self.npu_port, self.gpu_port = ports

    async def stop(self):
        for r in self._runners:
            await r.cleanup()


class Harness:
    """The real APIServer and InferenceEngine, pointed at the mocks."""

    def __init__(self, backends, **cfg_kwargs):
        self.backends = backends
        self.cfg_kwargs = cfg_kwargs

    async def __aenter__(self):
        await self.backends.start()
        cfg = m.PipelineConfig(
            draft_model="gemma3:1b",
            gpu_model="/models/gemma-4-31B-it-qat.gguf",
            draft_port=self.backends.npu_port,
            gpu_port=self.backends.gpu_port,
            metrics_enabled=False,
            **self.cfg_kwargs)
        self.metrics = m.MetricsCollector()
        kv = m.KVCacheManager(cfg.kv_cache_ttl, cfg.kv_cache_max_entries)
        queue = m.RequestQueue(cfg.draft_max_concurrent, cfg.gpu_max_concurrent)
        cb = m.CircuitBreaker(cfg.cb_failure_threshold, cfg.cb_recovery_timeout)
        self.engine = m.InferenceEngine(cfg, self.metrics, kv, queue, cb, None)
        await self.engine.start()
        server = m.APIServer(self.engine, cfg, self.metrics, None)
        self._runner = web.AppRunner(server.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        return self

    async def __aexit__(self, *exc):
        await self.engine.stop()
        await self._runner.cleanup()
        await self.backends.stop()


async def post_json(url, body):
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=body) as r:
            return r.status, await r.json()


async def post_raw(url, body):
    """Returns the response bytes verbatim — the point is the framing."""
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=body) as r:
            return r.status, await r.read()


# ── Tests ────────────────────────────────────────────────────────────────────

async def test_openai_non_stream():
    b = MockBackends(npu_chunks=["The answer ", "is "],
                     gpu_chunks=["<channel|>", "42."])
    async with Harness(b) as h:
        status, body = await post_json(
            h.url + "/v1/chat/completions",
            {"model": "pipeline", "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 64})
    content = body["choices"][0]["message"]["content"]
    check("openai non-stream: 200 + shape",
          status == 200 and body["object"] == "chat.completion"
          and "usage" in body, f"status={status}")
    check("openai non-stream: channel marker stripped",
          content == "The answer is 42.", repr(content))


async def test_sse_framing():
    b = MockBackends(npu_chunks=["a", "b"], gpu_chunks=["c"])
    async with Harness(b) as h:
        status, raw = await post_raw(
            h.url + "/v1/chat/completions",
            {"model": "pipeline", "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 64, "stream": True})
    text = raw.decode()
    # The corruption this guards against: literal backslash-n instead of LF.
    check("sse: no literal backslash-n in body", "\\n" not in text)
    check("sse: events separated by a blank line", "\n\n" in text)
    check("sse: terminates with [DONE]", text.endswith("data: [DONE]\n\n"),
          repr(text[-24:]))
    events = [e for e in text.split("\n\n") if e.strip()]
    parsed, bad = [], None
    for e in events:
        if not e.startswith("data: "):
            bad = e
            break
        payload = e[6:]
        if payload == "[DONE]":
            continue
        try:
            parsed.append(json.loads(payload))
        except json.JSONDecodeError:
            bad = e
            break
    check("sse: every event is 'data: ' + valid JSON", bad is None, repr(bad))
    content = "".join(p["choices"][0]["delta"].get("content", "") for p in parsed)
    check("sse: reassembled content", content == "abc", repr(content))


async def test_ollama_ndjson_framing():
    b = MockBackends(npu_chunks=["x"], gpu_chunks=["y"])
    async with Harness(b) as h:
        status, raw = await post_raw(
            h.url + "/api/chat",
            {"model": "pipeline", "messages": [{"role": "user", "content": "hi"}],
             "stream": True})
    text = raw.decode()
    check("ndjson: no literal backslash-n in body", "\\n" not in text)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    ok, objs = True, []
    for ln in lines:
        try:
            objs.append(json.loads(ln))
        except json.JSONDecodeError:
            ok = False
            break
    check("ndjson: every line is a JSON object", ok, repr(lines[:1]))
    check("ndjson: final line marks done",
          bool(objs) and objs[-1].get("done") is True)
    content = "".join(o.get("message", {}).get("content", "") for o in objs)
    check("ndjson: reassembled content", content == "xy", repr(content))


async def test_inline_eos_records_metrics():
    """The GPU EOS path used `return`, which skipped metrics.observe()."""
    b = MockBackends(npu_chunks=["draft "], gpu_chunks=["tail", "<turn|>", "junk"])
    async with Harness(b) as h:
        _, body = await post_json(
            h.url + "/v1/chat/completions",
            {"model": "pipeline", "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 64})
        rendered = await h.metrics.render()
    content = body["choices"][0]["message"]["content"]
    check("inline eos: output truncated at the marker",
          content == "draft tail", repr(content))
    check("inline eos: gpu metrics still recorded",
          "gpu_request_duration_ms_count" in rendered
          and "gpu_tokens_generated_count" in rendered)


async def test_stop_marker_of_other_family_survives():
    """Marker profiles are per-format: a Gemma 4 answer may quote ChatML."""
    b = MockBackends(npu_chunks=[""],
                     gpu_chunks=["Use ", "<|im_start|>", "user for ChatML."])
    async with Harness(b) as h:
        _, body = await post_json(
            h.url + "/v1/chat/completions",
            {"model": "pipeline", "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 64})
    content = body["choices"][0]["message"]["content"]
    check("profiles: ChatML marker survives a gemma4 response",
          content == "Use <|im_start|>user for ChatML.", repr(content))


async def test_gpu_payload_carries_stop_strings():
    b = MockBackends(npu_chunks=["d"], gpu_chunks=["g"])
    async with Harness(b) as h:
        await post_json(
            h.url + "/v1/chat/completions",
            {"model": "pipeline", "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 64})
    streaming = [p for p in b.gpu_payloads if p.get("stream")]
    stop = streaming[0].get("stop") if streaming else None
    check("gpu payload: server-side stop strings sent",
          stop == ["<turn|>", "<|turn>", "<eos>", "</s>"], repr(stop))
    check("gpu payload: prompt uses the gemma4 turn format",
          streaming and streaming[0]["prompt"].startswith("<|turn>user\n"),
          repr(streaming[0]["prompt"][:40]) if streaming else "none")


async def test_metrics_endpoint_format():
    b = MockBackends(npu_chunks=["a"], gpu_chunks=["b"])
    async with Harness(b) as h:
        await post_json(
            h.url + "/v1/chat/completions",
            {"model": "pipeline", "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 64})
        async with aiohttp.ClientSession() as s:
            async with s.get(h.url + "/metrics") as r:
                ctype = r.headers.get("Content-Type", "")
                text = await r.text()
    check("metrics: no literal backslash-n", "\\n" not in text)
    check("metrics: trailing newline present", text.endswith("\n"), repr(text[-12:]))
    check("metrics: versioned content type",
          "version=0.0.4" in ctype, ctype)
    ok = all(len(ln.split()) == 2 for ln in text.strip().split("\n"))
    check("metrics: every line is 'name value'", ok)


async def test_unterminated_region_not_swallowed():
    """A reasoning region that never closes must not eat the whole response."""
    b = MockBackends(npu_chunks=[""], gpu_chunks=["<|channel>thinking hard"])
    async with Harness(b) as h:
        _, body = await post_json(
            h.url + "/v1/chat/completions",
            {"model": "pipeline", "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 64})
    content = body["choices"][0]["message"]["content"]
    check("unterminated region: response is not empty",
          content == "thinking hard", repr(content))


async def main():
    for t in (test_openai_non_stream,
              test_sse_framing,
              test_ollama_ndjson_framing,
              test_inline_eos_records_metrics,
              test_stop_marker_of_other_family_survives,
              test_gpu_payload_carries_stop_strings,
              test_metrics_endpoint_format,
              test_unterminated_region_not_swallowed):
        try:
            await t()
        except Exception as e:
            check(t.__name__ + " (raised)", False, f"{type(e).__name__}: {e}")
    total = 17
    print("\n%d/%d passed" % (total - len(FAILS), total))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
