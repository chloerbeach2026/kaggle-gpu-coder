# Kaggle GPU Coder Server

Run a strong open-source **coding LLM for free** on Kaggle's 2x Tesla T4 GPUs (30 GPU-hours/week) — or on any other GPU box (Colab, RunPod, vast.ai, your own PC) — and use it from your own machine in Cline, Continue, aider, or any OpenAI-compatible tool.

```
GPU machine (Kaggle 2x T4 / Colab L4 / RTX 3090 / ...)
  └── vLLM  ── OpenAI-compatible API on :8000
        └── Cloudflare quick tunnel ── https://xxxx.trycloudflare.com
              └── your laptop: Cline / Continue / aider / curl
```

## Model choice — automatic

The notebook **detects your GPU** (architecture + total VRAM) and picks the best model that will actually run fast on it. Override anytime by setting `FORCE_MODEL` in the CONFIG cell.

| Your GPU | Auto-selected model | Expected single-stream speed |
|---|---|---|
| Ampere+ with ≥22GB total (A100, L4, RTX 3090/4090, L40S, H100, 2x A5000…) | **Qwen3-Coder-30B-A3B-Instruct-AWQ** (MoE — best quality) | ~40–100 tok/s |
| Ampere+ with 14–22GB (RTX 3080 16GB, A4000…) | Qwen2.5-Coder-14B-Instruct-AWQ | ~30–50 tok/s |
| Ampere+ with <14GB (RTX 3060 12GB…) | Qwen2.5-Coder-7B-Instruct-AWQ | ~40–60 tok/s |
| Turing with ≥24GB total (**Kaggle 2x T4**, RTX 2080Ti x2…) | Qwen2.5-Coder-14B-Instruct-AWQ | ~15–30 tok/s |
| Turing with 12–24GB (1x T4, RTX 2080…) | Qwen2.5-Coder-7B-Instruct-AWQ | ~20–35 tok/s |
| Turing with <12GB | Qwen2.5-Coder-3B-Instruct | ~30+ tok/s |
| Pascal/Volta (Kaggle P100…) — no AWQ support | Qwen2.5-Coder-7B-Instruct (fp16) | ~10–20 tok/s |

> **Why the MoE 30B only on Ampere+?** Tested it on Kaggle's 2x T4: ~1 tok/s. Quantized MoE models need Marlin kernels (SM80+); T4 is SM75 and falls back to an unusably slow path. On Turing/Pascal, dense models only — the auto-detect enforces this.
>
> **About "100 tok/s":** on a single request, T4s (2018 hardware, 320 GB/s memory bandwidth, no FlashAttention-2, no bf16) top out well below that for any model worth coding with. vLLM's continuous batching *does* push **total** throughput past 100 tok/s when several requests run in parallel (e.g. agentic tools firing multiple calls).

## Setup (once)

1. Create a Kaggle account, verify phone number (required for GPU + internet access).
2. Kaggle → **Create → Notebook** → File → Import Notebook → upload `kaggle-coder-server.ipynb`.
3. In the notebook's right sidebar: **Accelerator = GPU T4 x2**, **Internet = On**.
4. Run all cells. First run downloads the model (~17GB, 5–15 min).
5. Copy the printed `https://....trycloudflare.com/v1` URL.

Leave the last (keep-alive) cell running. Kaggle kills sessions after ~9–12h; just re-run — you'll get a new URL.

## Connect from your machine

Everything below uses:
- **Base URL:** `https://<your-tunnel>.trycloudflare.com/v1`
- **API key:** `kaggle-local-key`
- **Model:** `coder`

### aider
```bash
export OPENAI_API_BASE=https://<your-tunnel>.trycloudflare.com/v1
export OPENAI_API_KEY=kaggle-local-key
aider --model openai/coder
```

### Continue (VS Code) — `~/.continue/config.yaml`
```yaml
models:
  - name: kaggle-coder
    provider: openai
    model: coder
    apiBase: https://<your-tunnel>.trycloudflare.com/v1
    apiKey: kaggle-local-key
```

### Cline (VS Code)
Settings → API Provider: **OpenAI Compatible** → Base URL, API key, and Model ID as above.

### Python
```python
from openai import OpenAI
client = OpenAI(base_url="https://<your-tunnel>.trycloudflare.com/v1",
                api_key="kaggle-local-key")
r = client.chat.completions.create(model="coder",
    messages=[{"role": "user", "content": "hello"}])
print(r.choices[0].message.content)
```

### Why not Ollama?
Ollama runs models **locally** — it cannot attach to a remote GPU backend. vLLM's OpenAI-compatible API is the standard way to do remote serving, and every serious coding tool supports it directly.

## Troubleshooting

- **vLLM dies on startup / OOM** → lower `MAX_MODEL_LEN`, or set `FORCE_MODEL` to a smaller model in the CONFIG cell.
- **Not on Kaggle?** The notebook works anywhere with CUDA + Jupyter (Colab, RunPod, vast.ai, local). On single-GPU machines it automatically uses TP=1.
- **Slow first response** → CUDA graph warmup + prefix cache cold; second request is faster.
- **Tunnel URL not printed** → re-run the tunnel cell (quick tunnels occasionally fail to allocate).
- **Kaggle rules note:** keep usage to interactive/dev workloads. Don't run it as a 24/7 public service — that's against Kaggle ToS and will get the account flagged.

## Limits to know

- 30 GPU-hours/week quota; session max ~12h; URL changes every session.
- T4 = compute capability 7.5: no bf16, no FlashAttention-2 (the notebook already handles this with `--dtype half`).
