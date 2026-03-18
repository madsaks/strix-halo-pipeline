# Strix Halo Pipeline — NPU+GPU LLM Inference on Linux

> **First-of-its-kind**: Run LLMs across both the XDNA2 NPU and RDNA 3.5 iGPU simultaneously on AMD Ryzen AI Max (Strix Halo) under Linux.

One Python script. One command. Both accelerators working together.

```
python3 strix_halo_pipeline.py --gpu-model ./models/Qwen3-8B-Q4_K_M.gguf
```

Then connect **any** OpenAI-compatible client (Open WebUI, n8n, curl, etc.) to `http://<your-ip>:11435/v1`.

---

## Why This Exists

AMD Strix Halo has 3 compute engines: CPU, RDNA 3.5 iGPU, and XDNA2 NPU — plus 128GB of unified memory. But as of March 2026:

- **AMD's hybrid LLM flow is Windows-only** (DirectML + OGA, no Linux support)
- **FastFlowLM runs only on NPU** — GPU sits idle
- **llama.cpp runs only on GPU** — NPU sits idle
- **Nobody had both running simultaneously on a single LLM task on Linux**

This project solves that.

## How It Works

```
                    128GB Unified LPDDR5X (GPUVM)
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   ┌─────────────┐        ┌─────────────────┐    │
    │   │  XDNA2 NPU  │        │  RDNA 3.5 iGPU  │    │
    │   │  FastFlowLM  │        │  llama.cpp       │    │
    │   │  Qwen3-1.7B  │        │  Qwen3-8B        │    │
    │   │  (draft)     │        │  (verifier)      │    │
    │   └──────┬───────┘        └────────┬─────────┘    │
    │          │                         │              │
    └──────────┼─────────────────────────┼──────────────┘
               │     ┌───────────┐       │
               └─────┤ Pipeline  ├───────┘
                     │  Engine   │
                     │ :11435/v1 │
                     └─────┬─────┘
                           │
                    OpenAI-compatible API
                    (Open WebUI, n8n, curl)
```

**Pipeline flow for each request:**

1. **GPU starts warming** its KV cache in the background (`asyncio.create_task`)
2. **NPU streams tokens** to the user immediately (~750 tok/s prefill, ~34 tok/s decode)
3. User sees text within **~700ms** (TTFT)
4. When NPU finishes its portion, **GPU continues** with the larger model
5. GPU's cache is pre-warmed → **276ms to prefill NPU output** (917 tok/s)
6. GPU generates continuation at ~41 tok/s with 8B quality

## Performance

Measured on Ryzen AI Max+ 395 (Strix Halo), Ubuntu 26.10, kernel 7.0-rc3:

| Mode | Speed | TTFT | Quality | Power |
|------|-------|------|---------|-------|
| NPU only (1.7B) | 26 w/s | ~500ms | 1.7B | ~2W |
| GPU only (8B) | 34 w/s | ~2s | 8B | ~25W |
| **NPU+GPU pipeline** | **25 w/s** | **~700ms** | **8B** | Combined |

The pipeline gives you **NPU-like TTFT** with **GPU-quality output**. Scale up the GPU model (14B, 32B) and the NPU's instant streaming becomes even more valuable while the larger model generates the quality continuation.

## Quick Start

### Prerequisites

- AMD Ryzen AI Max / Max+ (Strix Halo) with XDNA2 NPU
- Ubuntu 24.04+ (tested on 26.10 with kernel 7.0-rc3)
- [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) installed (`flm` in PATH)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) built with Vulkan (`-DGGML_VULKAN=ON`)
- Python 3.10+

### Install

```bash
git clone https://github.com/mikealanni/strix-halo-pipeline.git
cd strix-halo-pipeline

# Install Python dependencies
pip install aiohttp --break-system-packages

# Download a GPU model (5GB)
huggingface-cli download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir ./models

# Pull NPU draft model
flm pull qwen3:1.7b
```

### Run

```bash
python3 strix_halo_pipeline.py --gpu-model ./models/Qwen3-8B-Q4_K_M.gguf
```

That's it. The script:
1. Fixes Vulkan ICD paths (common Strix Halo issue)
2. Creates XRT symlinks for FLM
3. Starts FLM on the NPU
4. Starts llama.cpp on the GPU with Vulkan
5. Launches the pipeline engine with OpenAI-compatible API
6. Binds to `0.0.0.0` for remote access

### Connect a Client

**Open WebUI:**
- Settings → Connections → Add Connection
- URL: `http://<your-ip>:11435/v1`
- Key: `anything`

**curl:**
```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "strix-speculative",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**Python:**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11435/v1", api_key="any")
response = client.chat.completions.create(
    model="strix-speculative",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Models

Select different accelerator modes by changing the model name in your API request:

| Model ID | Behavior | Best For |
|----------|----------|----------|
| `strix-speculative` | NPU drafts → GPU continues | Long responses, best overall |
| `npu` | NPU only (FastFlowLM) | Quick answers, low power |
| `gpu` | GPU only (llama.cpp) | Maximum quality |
| `auto` | Smart routing | Mixed workloads |

## Scaling Up

With 128GB unified memory, you can run much larger GPU models:

```bash
# 14B — great quality jump (8.3GB)
huggingface-cli download Qwen/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf --local-dir ./models
python3 strix_halo_pipeline.py --gpu-model ./models/Qwen3-14B-Q4_K_M.gguf --draft qwen3:4b

# 32B — serious quality (19GB)
huggingface-cli download Qwen/Qwen3-32B-GGUF Qwen3-32B-Q4_K_M.gguf --local-dir ./models
python3 strix_halo_pipeline.py --gpu-model ./models/Qwen3-32B-Q4_K_M.gguf --draft qwen3:4b

# 30B-A3B MoE — fast AND smart (17GB)
huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir ./models
python3 strix_halo_pipeline.py --gpu-model ./models/Qwen3-30B-A3B-Q4_K_M.gguf --draft qwen3:4b
```

The bigger the GPU model, the more valuable the NPU draft becomes — users see instant streaming while the large model handles the heavy continuation.

## Options

```
python3 strix_halo_pipeline.py --help

  --gpu-model PATH    GGUF model for GPU (required)
  --draft MODEL       FLM draft model (default: qwen3:1.7b)
  --draft-tokens N    NPU tokens per round (default: 128)
  --gpu-ctx N         GPU context size (default: 4096)
  --gpu-layers N      Layers on GPU (default: 99 = all)
  --port PORT         API port (default: 11435)
  --temperature T     Temperature (default: 0.6)
  --mode MODE         pipeline|npu|gpu|auto (default: pipeline)
  --no-launch         Don't start servers (assume already running)
  -v, --verbose       Show per-request timing stats
```

## Troubleshooting

**FLM crashes with `libxrt_core.so` error:**
```bash
sudo mkdir -p /opt/xilinx/xrt/lib/x86_64-linux-gnu
sudo bash -c 'for f in /opt/xilinx/xrt/lib/*.so*; do ln -sf "$f" /opt/xilinx/xrt/lib/x86_64-linux-gnu/$(basename "$f"); done'
```

**Vulkan not found / GPU fallback to CPU:**
```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
# Make permanent:
sed -i 's|export VK_ICD_FILENAMES=.*|export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json|' ~/.bashrc
```

**llama.cpp segfaults with ROCm:**
Use the Vulkan build instead. Build with `-DGGML_VULKAN=ON` targeting gfx1151.

**NPU not detected:**
Ensure `amd_iommu=off` is NOT in your kernel cmdline (it disables NPU). Check with:
```bash
flm validate
```

## Architecture Deep Dive

### Why Not True Speculative Decode?

Traditional speculative decoding requires per-token logprob comparison between draft and verifier. FLM's OpenAI API doesn't expose logprobs, and llama.cpp's prompt_logprobs support is still being built. This project uses **implicit verification** via prefill instead:

1. NPU generates draft tokens as complete text
2. GPU processes those tokens during PREFILL (parallel, one matrix multiply)
3. GPU's KV cache now reflects the draft, and it continues from there

This gives similar benefits without requiring internal model access.

### What "Parallel" Means Here

For a single request, execution is **overlapped but sequential** — NPU generates first, GPU continues. True simultaneous same-token generation would require shared KV cache between FLM and llama.cpp, which doesn't exist.

What IS parallel:
- GPU cache warmup runs **during** NPU streaming (`asyncio.create_task`)
- Multiple concurrent requests use **different accelerators** simultaneously
- The user sees **continuous streaming** with no gap between NPU and GPU phases

### Future Work

- **True rejection sampling** when llama.cpp adds `prompt_logprobs` to the API
- **ggml backend for XDNA2** using FLM's `libmha.so`, `libgemm.so`, `libdequant.so`
- **Fork FLM** to add direct GPU decode path via HIP/Vulkan

## Credits

Built by [Mike Alani](https://github.com/mikealanni) with Claude assistance.
First demonstrated on Ryzen AI Max+ 395 (Strix Halo), March 17, 2026.

## License

MIT
