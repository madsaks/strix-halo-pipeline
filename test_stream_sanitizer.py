#!/usr/bin/env python3
"""Tests for StreamSanitizer.

Run with the system interpreter, not whatever venv is first on PATH:
    /usr/bin/python3 test_stream_sanitizer.py

Covers the two control-token families the pipeline sees (Qwen's <think> and
Gemma4's <|channel>...<channel|>), end-of-turn detection, and markers split
across streaming chunk boundaries down to one character at a time.
"""
import importlib.util, sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "shp", os.path.join(_HERE, "strix_halo_pipeline_v2.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
S = m.StreamSanitizer

def run(chunks):
    s = S()
    return "".join(s.feed(c) for c in chunks) + s.flush()

cases = [
    (["<channel|>There are 3 r's."], "There are 3 r's."),
    (["<|channel>thought\n", "<channel|>", "2 + 2 = 4"], "2 + 2 = 4"),
    (["<|channel>thought\nlet me count...<channel|>Answer: 3"], "Answer: 3"),
    (["<think>", "hidden", "</think>", "visible"], "visible"),
    (["Hello ", "world"], "Hello world"),
    (["<chan", "nel|>", "hi"], "hi"),
    (["<|chan", "nel>thought\n", "<chann", "el|>", "done"], "done"),
    (["a<th", "ink>x</th", "ink>b"], "ab"),
    (["answer <chan"], "answer <chan"),
    (["<think>never closed"], "never closed"),   # released, not swallowed
    (["...inference.<channel|>"], "...inference."),
    (["if a < b and c > d"], "if a < b and c > d"),
    ([c for c in "<|channel>thought\n<channel|>The sky is blue."],
     "The sky is blue."),
]
# EOS handling: everything from the marker on is dropped, .stopped is set
eos_cases = [
    (["2 + 2 = 4", "<|im_end|>", "\n<|im_start|>user\nWhat is 5+5?"], "2 + 2 = 4"),
    (["answer<end", "_of_turn>", "junk"], "answer"),
    (["done</s>more"], "done"),
    (["<channel|>hi<|im_end|>trailing"], "hi"),
]
fails = 0
for chunks, want in eos_cases:
    san = S()
    got = "".join(san.feed(c) for c in chunks) + san.flush()
    ok = got == want and san.stopped
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + repr(chunks)[:60] + " -> " + repr(got)
          + " stopped=" + str(san.stopped) + ("" if ok else "  WANT " + repr(want)))

for chunks, want in cases:
    got = run(chunks)
    ok = got == want
    fails += 0 if ok else 1
    label = repr(chunks) if len(repr(chunks)) < 60 else repr(chunks)[:57] + "...]"
    print(("PASS " if ok else "FAIL ") + label + " -> " + repr(got)
          + ("" if ok else "  WANT " + repr(want)))

# an unterminated region is flagged as well as released
san = S()
san.feed("<think>cut off mid-thought")
released = san.flush()
ok = released == "cut off mid-thought" and san.unterminated
fails += 0 if ok else 1
print(("PASS " if ok else "FAIL ") + "unterminated region released and flagged"
      + " -> " + repr(released) + " unterminated=" + str(san.unterminated))

# a closed region leaves the flag clear
san = S()
out = san.feed("<think>x</think>answer") + san.flush()
ok = out == "answer" and not san.unterminated
fails += 0 if ok else 1
print(("PASS " if ok else "FAIL ") + "closed region leaves flag clear -> "
      + repr(out) + " unterminated=" + str(san.unterminated))

# ── Marker profiles ──────────────────────────────────────────────────────────
# A narrow profile must not fire on another family's markers: quoting ChatML in
# a Gemma 4 answer is legitimate text, not the end of the turn.
prof_cases = [
    ("gemma4", "Use <|im_start|>user for ChatML.", "Use <|im_start|>user for ChatML.", False),
    ("gemma4", "Answer<turn|>junk", "Answer", True),
    ("chatml", "Gemma 4 opens with <|turn>model.", "Gemma 4 opens with <|turn>model.", False),
    ("chatml", "Answer<|im_end|>junk", "Answer", True),
    ("gemma", "No channel here <channel|> stays", "No channel here <channel|> stays", False),
    ("gemma", "Answer<end_of_turn>junk", "Answer", True),
    # the catch-all still recognises everything
    ("generic", "Answer<turn|>junk", "Answer", True),
    ("generic", "Answer<|im_end|>junk", "Answer", True),
]
for prof, text, want, want_stop in prof_cases:
    san = S(m.MARKER_PROFILES[prof])
    got = san.feed(text) + san.flush()
    ok = got == want and san.stopped == want_stop
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + "profile %-8s %r -> %r stopped=%s"
          % (prof, text[:34], got[:40], san.stopped)
          + ("" if ok else "  WANT " + repr(want)))

# ── Prompt templates ─────────────────────────────────────────────────────────
# Expected strings are what llama-server's /apply-template produced from the
# GGUF's own Jinja template with enable_thinking=false, plus the draft prefix.

class _Cfg:
    def __init__(self, model, template="auto"):
        self.gpu_model = model
        self.gpu_template = template

GEMMA = "/models/gemma-4-31B-it-qat-GGUF-UD-Q4_K_XL.gguf"

def engine(model, template="auto"):
    e = object.__new__(m.InferenceEngine)
    e.config = _Cfg(model, template)
    return e

tpl_cases = [
    ([{"role": "user", "content": "Hi"}], "",
     "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
    ([{"role": "system", "content": "Be terse."},
      {"role": "user", "content": "Hi"}], "",
     "<|turn>system\nBe terse.<turn|>\n<|turn>user\nHi<turn|>\n"
     "<|turn>model\n<|channel>thought\n<channel|>"),
    ([{"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "Hello"},
      {"role": "user", "content": "Bye"}], "",
     "<|turn>user\nHi<turn|>\n<|turn>model\nHello<turn|>\n"
     "<|turn>user\nBye<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
    # the NPU draft lands inside the answer channel, not before it
    ([{"role": "user", "content": "Hi"}], "There are",
     "<|turn>user\nHi<turn|>\n<|turn>model\n"
     "<|channel>thought\n<channel|>There are"),
]
for msgs, prefix, want in tpl_cases:
    got = engine(GEMMA)._build_prompt(msgs, prefix)
    ok = got == want
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + "gemma4 template "
          + repr([x["role"] for x in msgs]) + (" +prefix" if prefix else "")
          + ("" if ok else "\n   GOT  " + repr(got) + "\n   WANT " + repr(want)))

dispatch = [
    (GEMMA, "auto", "gemma4"),
    ("/models/gemma4-31b.gguf", "auto", "gemma4"),
    ("/models/qwen3-8b-Q4.gguf", "auto", "chatml"),
    ("/models/gemma-3-4b.gguf", "auto", "chatml"),   # gemma 2/3 not implemented
    ("/models/gemma-4b-it-qat-Q4_K_M.gguf", "auto", "chatml"),  # 4B, not Gemma 4
    ("/models/gemma4b.gguf", "auto", "chatml"),
    ("/models/gemma-4-27b.gguf", "auto", "gemma4"),
    ("/models/gemma_4_it.gguf", "auto", "gemma4"),
    (GEMMA, "chatml", "chatml"),                     # explicit override wins
]
for model, tpl, want in dispatch:
    got = engine(model, tpl)._resolve_template()
    ok = got == want
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + "resolve %s (%s) -> %s"
          % (model.rsplit("/", 1)[-1], tpl, got)
          + ("" if ok else "  WANT " + want))

# ChatML path must be untouched for non-Gemma models
got = engine("/models/qwen3-8b.gguf")._build_prompt(
    [{"role": "user", "content": "Hi"}], "")
want = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
ok = got == want
fails += 0 if ok else 1
print(("PASS " if ok else "FAIL ") + "chatml template unchanged"
      + ("" if ok else "\n   GOT  " + repr(got) + "\n   WANT " + repr(want)))

# ── Draft-model /no_think directive ──────────────────────────────────────────

class _DraftCfg:
    def __init__(self, draft_model):
        self.draft_model = draft_model

def draft_engine(model):
    e = object.__new__(m.InferenceEngine)
    e.config = _DraftCfg(model)
    return e

MSGS = [{"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hi"}]
NO_SYS = [{"role": "user", "content": "Hi"}]

nothink_cases = [
    # Qwen 3 keeps upstream behaviour: directive appended, system injected
    ("qwen3:1.7b", MSGS, "Be terse. /no_think", 2),
    ("qwen3:1.7b", NO_SYS, "You are a helpful assistant. /no_think", 2),
    ("qwen3.5:2b", NO_SYS, "You are a helpful assistant. /no_think", 2),
    # everything else is handed back untouched — no directive, no injection
    ("gemma3:1b", MSGS, "Be terse.", 2),
    ("gemma3:1b", NO_SYS, None, 1),
    ("llama3.2:1b", NO_SYS, None, 1),
    ("phi4-mini-it:4b", MSGS, "Be terse.", 2),
]
for model, msgs, want_sys, want_len in nothink_cases:
    got = draft_engine(model)._patch_messages(msgs)
    sys_msgs = [x for x in got if x["role"] == "system"]
    got_sys = sys_msgs[0]["content"] if sys_msgs else None
    ok = got_sys == want_sys and len(got) == want_len
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + "no_think %-16s -> system=%s"
          % (model, repr(got_sys))
          + ("" if ok else "  WANT " + repr(want_sys) + " len=" + str(want_len)))

# untouched must mean the very same object, not a rebuilt copy
same = draft_engine("gemma3:1b")._patch_messages(NO_SYS) is NO_SYS
fails += 0 if same else 1
print(("PASS " if same else "FAIL ") + "non-Qwen draft leaves messages unmodified")

# ── Draft-model profile selection ────────────────────────────────────────────
draft_cases = [
    ("qwen3:1.7b", "chatml"),
    ("gemma3:1b", "gemma"),
    ("gemma4-it:e2b", "gemma4"),
    ("llama3.2:1b", "generic"),
    ("phi4-mini-it:4b", "generic"),
]
for model, want in draft_cases:
    got = m.draft_profile_name(model)
    ok = got == want
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + "draft profile %-16s -> %s" % (model, got)
          + ("" if ok else "  WANT " + want))

# ── Config validation ────────────────────────────────────────────────────────
try:
    m.PipelineConfig(gpu_template="chatlm")
    print("FAIL rejects unknown gpu_template (no error raised)")
    fails += 1
except ValueError:
    print("PASS rejects unknown gpu_template")
ok_cfg = True
try:
    for good in ("auto", "gemma4", "chatml"):
        m.PipelineConfig(gpu_template=good)
except ValueError:
    ok_cfg = False
fails += 0 if ok_cfg else 1
print(("PASS " if ok_cfg else "FAIL ") + "accepts auto/gemma4/chatml")

total = (len(cases) + len(eos_cases) + 2 + len(prof_cases) + len(tpl_cases)
         + len(dispatch) + 1 + len(nothink_cases) + 1 + len(draft_cases) + 2)
print("\n%d/%d passed" % (total - fails, total))
sys.exit(1 if fails else 0)
