# autoresearch for Marconi autotuning

This fork repurposes `autoresearch` from single-GPU language-model training into a fixed benchmark harness for **Marconi autotuning in SGLang**.

The target use case is:
- run on either a `4xH100` node for final validation or a `1xA100` node for cheaper iteration
- point at a checked-out `sglang` repo, typically the `marconi-eviction` branch
- let **Codex** iteratively edit the target Marconi autotune implementation
- use this repo only as the evaluation harness and experiment log

The core design is still intentionally small:
- `prepare.py` is the fixed harness and runtime utilities
- `train.py` runs one Marconi autotune experiment and prints a compact summary
- `program.md` is the Codex research loop

## What this fork is optimizing

The goal is not generic Marconi eviction. The goal is specifically:

- improve **autotune** behavior for Marconi on hybrid models
- keep the public API aligned with `--radix-eviction-policy marconi`
- evaluate against the shared-prefix serving workload that already exposed useful signal for Marconi

This fork is built around the benchmark story you already established:
- fixed nonzero `eff_weight` can outperform `main`
- autotune correctness and stability matter
- autotune should be evaluated by throughput, TTFT, and whether tuning rounds actually finish and apply during serving

## Repository roles

### Fixed harness

`prepare.py` contains:
- target repo discovery
- benchmark configuration
- server launch helpers
- benchmark execution
- summary extraction

This file is treated as fixed infrastructure.

### Mutable evaluation entrypoint

`train.py` runs one experiment against the target `sglang` checkout and prints a summary in a grep-friendly format.

### Codex instructions

`program.md` tells Codex:
- which target repo to edit
- which files are in scope
- how to run one experiment
- how to log and keep/discard results

## Target repo contract

By default the harness expects:
- `SGLANG_REPO=/home/qi/sglang`
- branch under test is handled in the target repo itself
- the target repo is already on the branch you want Codex to work on

The harness runs the target repo by forcing:
- `PYTHONPATH=<target-repo>/python`
- `cwd=<target-repo>`

This avoids accidentally benchmarking some other editable install.

## Default benchmark modes

Two modes are built in:

- `fast`
  - reduced shared-prefix workload for iteration
  - intended for the main autonomous loop
- `full`
  - the original larger saturated shared-prefix workload
  - intended only for promising candidates

Set the mode with:

```bash
MARCONI_EVAL_MODE=fast uv run train.py
MARCONI_EVAL_MODE=full uv run train.py
```

## Required environment

- `4` visible NVIDIA GPUs
- working SGLang checkout with the Marconi branch under test
- model access for `Qwen/Qwen3-Next-80B-A3B-Instruct`
- `uv`

Optional environment variables:

```bash
export SGLANG_REPO=/path/to/sglang
export MARCONI_MODEL_PATH=nvidia/NVIDIA-Nemotron-Nano-9B-v2
export MARCONI_EVAL_MODE=fast
export MARCONI_RESULTS_DIR=/path/to/results
export MARCONI_TP_SIZE=1
export MARCONI_MIN_GPUS=1
export MARCONI_CONTEXT_LENGTH=32768
export MARCONI_EXTRA_SERVER_ARGS="--marconi-eff-weight 0 --disable-marconi-autotune"
export MARCONI_BASE_URL=http://127.0.0.1:30000
```

## Quick start

```bash
uv sync
uv run train.py
```

For Hermes on `1xA100`:

```bash
chmod +x start_hermes_a100.sh
./start_hermes_a100.sh
```

Then paste the contents of `hermes_kickoff_prompt.txt` into Hermes.

The run prints a summary like:

```text
---
status: completed
mode: fast
target_branch: marconi-eviction
target_commit: abc1234
request_throughput: 10.31
mean_ttft_ms: 178966.25
mean_e2e_latency_ms: 190000.00
autotune_applied: 1
autotune_rounds: 1
current_eff_weight: 1.0
last_tuned_eff_weight: 1.0
results_jsonl: /.../result.jsonl
server_log: /.../server.log
```

## Intended Codex workflow

1. Open this repo in Codex.
2. Read `program.md`.
3. Let Codex edit the target `sglang` repo, not this harness, unless the harness itself is wrong.
4. Use `uv run train.py > run.log 2>&1` for one experiment.
5. Log the result in `results.tsv`.
6. Keep only target-repo changes that improve the benchmark under the rules in `program.md`.

## What this fork is not

- not a generic SGLang benchmark runner
- not a replacement for your benchmark notes
- not a distributed orchestration system

It is a narrow research harness for one concrete problem: **Marconi autotune policy quality on real hybrid-model serving workloads**.
