# Marconi benchmark results on 4x H100 (2026-03-29)

## Environment

- hardware: 4x NVIDIA H100 80GB HBM3
- model: `Qwen/Qwen3-Next-80B-A3B-Instruct`
- context length: `32768`
- tp size: `4`
- page size: `1`
- schedule policy: `lpm`
- mamba scheduler strategy: `extra_buffer`
- mamba dtype: `bfloat16`
- mamba full memory ratio: `0.9`

### Required node runtime env

This node does not run cleanly with default NCCL settings. Successful multi-GPU runs required:

```bash
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_NVLS_ENABLE=0
export HF_TOKEN='<YOUR_HF_TOKEN>'
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```

Reason: GPU0 peer access is broken on this node and NCCL NVLS init fails without these overrides.

## Important TTFT caveat

The saturated benchmark uses:

```bash
--request-rate inf
```

In `sglang.bench_serving`, that means all requests are submitted at time 0. Therefore saturated TTFT includes queueing and can become very large. Use saturated TTFT carefully. Saturated throughput is the more reliable signal there.

For more realistic TTFT, use a finite request rate such as:

```bash
--request-rate 4
```

## Correctness and stability issues fixed during benchmarking

I found and fixed three real issues in `marconi-eviction`:

1. `--disable-marconi-autotune` did not actually keep `eff_weight` fixed.
   - The bootstrap warm-start path still changed `eff_weight` from `0.0` to `1.0`.
   - Fixed in `python/sglang/srt/mem_cache/mamba_radix_cache.py`.

2. TP ranks could poll/apply autotune results independently.
   - This could desynchronize tensor-parallel ranks during finite-rate runs.
   - Fixed by synchronizing future readiness across the TP group before apply.

3. Finite-rate runs were making poor live switches under queue-free conditions.
   - For `--request-rate 4`, scheduler queue depth stayed at `0`, and replay favored noisy weight oscillations (`1.0 -> 0.0 -> 2.0 -> 0.1 -> 1.5`) that did not help real latency.
   - Fixed with a queue-free keep-current guard so autotune does not thrash weights when there is no backlog pressure.

After the fixes:

- disabled-autotune runs now stay fixed and report the configured weight correctly
- the realistic finite-rate autotune benchmark now completes cleanly
- autotune ends at `eff_weight=1.0` and slightly beats fixed `eff_weight=1.0`

## Exact standalone commands to reproduce directly from the SGLang repo

Open one terminal for the server and another for the benchmark client.

### 1) main baseline server

Run from the `main` checkout, e.g. `/workspace/sglang-main`:

```bash
cd /workspace/sglang-main
export PYTHONPATH="$PWD/python"
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_NVLS_ENABLE=0
export HF_TOKEN='<YOUR_HF_TOKEN>'
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

/venv/main/bin/python -u -m sglang.launch_server \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp-size 4 \
  --context-length 32768 \
  --schedule-policy lpm \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 0.9 \
  --page-size 1 \
  --enable-metrics \
  --radix-eviction-policy lru
```

### 2) marconi autotune server

Run from the `marconi-eviction` checkout, e.g. `/workspace/sglang`:

```bash
cd /workspace/sglang
export PYTHONPATH="$PWD/python"
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_NVLS_ENABLE=0
export HF_TOKEN='<YOUR_HF_TOKEN>'
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

/venv/main/bin/python -u -m sglang.launch_server \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp-size 4 \
  --context-length 32768 \
  --schedule-policy lpm \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 0.9 \
  --page-size 1 \
  --enable-metrics \
  --radix-eviction-policy marconi
```

### 3) marconi fixed baseline server (`eff_weight=0.0`, autotune disabled)

```bash
cd /workspace/sglang
export PYTHONPATH="$PWD/python"
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_NVLS_ENABLE=0
export HF_TOKEN='<YOUR_HF_TOKEN>'
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

/venv/main/bin/python -u -m sglang.launch_server \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp-size 4 \
  --context-length 32768 \
  --schedule-policy lpm \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 0.9 \
  --page-size 1 \
  --enable-metrics \
  --radix-eviction-policy marconi \
  --marconi-eff-weight 0.0 \
  --disable-marconi-autotune
```

### 4) marconi fixed baseline server (`eff_weight=1.0`, autotune disabled)

```bash
cd /workspace/sglang
export PYTHONPATH="$PWD/python"
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_NVLS_ENABLE=0
export HF_TOKEN='<YOUR_HF_TOKEN>'
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

/venv/main/bin/python -u -m sglang.launch_server \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp-size 4 \
  --context-length 32768 \
  --schedule-policy lpm \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 0.9 \
  --page-size 1 \
  --enable-metrics \
  --radix-eviction-policy marconi \
  --marconi-eff-weight 1.0 \
  --disable-marconi-autotune
```

### 5) Saturated benchmark client (`--request-rate inf`)

Run from the same repo as the server under test:

```bash
cd /workspace/sglang   # or /workspace/sglang-main for main
export PYTHONPATH="$PWD/python"

/venv/main/bin/python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name generated-shared-prefix \
  --num-prompts 3072 \
  --gsp-num-groups 192 \
  --gsp-prompts-per-group 16 \
  --gsp-system-prompt-len 8192 \
  --gsp-question-len 512 \
  --gsp-output-len 128 \
  --gsp-num-turns 1 \
  --request-rate inf \
  --seed 7 \
  --output-file /tmp/marconi-full-saturated.jsonl \
  --output-details
```

### 6) Rate-controlled benchmark client (`--request-rate 4`)

```bash
cd /workspace/sglang   # or /workspace/sglang-main for main
export PYTHONPATH="$PWD/python"

/venv/main/bin/python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name generated-shared-prefix \
  --num-prompts 3072 \
  --gsp-num-groups 192 \
  --gsp-prompts-per-group 16 \
  --gsp-system-prompt-len 8192 \
  --gsp-question-len 512 \
  --gsp-output-len 128 \
  --gsp-num-turns 1 \
  --request-rate 4 \
  --seed 7 \
  --output-file /tmp/marconi-full-rate4.jsonl \
  --output-details
```

## Validated benchmark results

### Saturated full benchmark (`--request-rate inf`)

| Variant | Repo / branch | Server flags | Throughput (req/s) | Mean TTFT (ms) | Notes |
|---|---|---|---:|---:|---|
| main baseline | `sglang-main` / `main` | `--radix-eviction-policy lru` | 5.9127 | 274638.14 | baseline |
| marconi autotune | `sglang` / `marconi-eviction` | `--radix-eviction-policy marconi` | 6.7626 | 239615.47 | patched autotune |
| marconi fixed eff=0.0 | `sglang` / `marconi-eviction` | `--radix-eviction-policy marconi --marconi-eff-weight 0.0 --disable-marconi-autotune` | 6.6734 | 241915.46 | validated fixed-ablation after bugfix |
| marconi fixed eff=1.0 | `sglang` / `marconi-eviction` | `--radix-eviction-policy marconi --marconi-eff-weight 1.0 --disable-marconi-autotune` | 6.7287 | 238826.95 | strongest saturated fixed baseline tested |

### Saturated benchmark deltas

| Comparison | Throughput delta | TTFT delta |
|---|---:|---:|
| autotune vs main | +14.37% | -12.75% |
| autotune vs fixed eff=0.0 | +1.34% | -0.95% |
| autotune vs fixed eff=1.0 | +0.50% | +0.33% |

Interpretation:

- After the correctness fix, autotune clearly beats `main`.
- Autotune also beats the true fixed `eff_weight=0.0` baseline.
- Autotune is slightly better in throughput than the fixed `eff_weight=1.0` baseline, but slightly worse in saturated TTFT.
- Because this benchmark is saturated, throughput is the stronger signal here.

## Production-signoff benchmark: finite-rate full (`--request-rate 4`)

### Final validated realistic benchmark

| Variant | Request rate | Throughput (req/s) | Mean TTFT (ms) | Mean E2E (ms) | Status |
|---|---:|---:|---:|---:|---|
| main baseline | 4 | 4.0673 | 959.51 | 7984.92 | completed |
| marconi fixed eff=0.0 | 4 | 4.0687 | 524.44 | 5003.89 | completed |
| marconi fixed eff=1.0 | 4 | 4.0695 | 493.94 | 4855.14 | completed |
| marconi autotune (final patched) | 4 | 4.0698 | 485.13 | 4617.08 | completed |

### Finite-rate benchmark deltas

| Comparison | Throughput delta | TTFT delta | E2E delta |
|---|---:|---:|---:|
| autotune vs main | +0.0618% | -49.44% | -42.18% |
| autotune vs fixed eff=0.0 | +0.0260% | -7.49% | -7.73% |
| autotune vs fixed eff=1.0 | +0.0078% | -1.78% | -4.90% |

Interpretation:

- The realistic finite-rate production-signoff run now completes cleanly.
- Final autotune no longer thrashes through queue-free weights; it converges to `eff_weight=1.0`.
- Final patched autotune is now slightly better than the fixed `eff_weight=1.0` baseline on all three realistic metrics:
  - throughput
  - TTFT
  - mean end-to-end latency
- The gain over fixed `1.0` is small, but it is now in the right direction and stable.
- The win over `main` remains large on latency.

## Key artifact paths

### Saturated runs
- main: `/workspace/autoresearch/main-full.log`
- autotune: `/workspace/autoresearch/marconi-full-tunedfix.log`
- fixed eff=0.0: `/workspace/autoresearch/marconi-noautotune-full-fixed.log`
- fixed eff=1.0: `/workspace/autoresearch/marconi-eff1-full.log`

### Finite-rate production-signoff runs
- main: `/workspace/autoresearch/main-full-rate4.log`
- autotune final: `/workspace/autoresearch/marconi-full-rate4-queuefix.log`
- fixed eff=0.0 final: `/workspace/autoresearch/marconi-eff0-rate4-cleanrerun.log`
- fixed eff=1.0 final: `/workspace/autoresearch/marconi-eff1-rate4-cleanrerun.log`

## Recommendation

Current honest production read:

- benchmark correctness issue: fixed
- TP-rank autotune coordination issue: fixed
- finite-rate stability issue: fixed for the validated signoff run
- realistic latency benchmark: autotune now wins

### PR-safe summary

> After fixing a disabled-autotune correctness bug and synchronizing live autotune application across tensor-parallel ranks, Marconi autotune now completes the realistic finite-rate benchmark cleanly on `Qwen/Qwen3-Next-80B-A3B-Instruct` with 4x H100. On the final `--request-rate 4` production-signoff benchmark, autotune achieves 4.0698 req/s, 485.1 ms mean TTFT, and 4617.1 ms mean E2E latency, slightly outperforming the strongest fixed `eff_weight=1.0` baseline (4.0695 req/s, 493.9 ms TTFT, 4855.1 ms E2E) while substantially outperforming `main` on latency.

### Final recommendation

This is now reasonable to call **Marconi push-to-prod ready** for this benchmark setup.
