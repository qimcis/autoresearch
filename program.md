# Marconi autotune research program

This repo is not the thing you are improving. The target repo is a checked-out `sglang` tree, usually on the `marconi-eviction` branch. This repo is only the fixed harness and experiment log.

You are an autonomous coding agent running research for one narrow goal:

- improve **Marconi autotune** behavior in SGLang
- on a smaller single-node setup for iteration, with final validation on larger hardware
- without changing the public user contract away from `--radix-eviction-policy marconi`

## Setup

Work with the human once at the start to make sure these are true:

1. This repo is on a fresh branch, for example `autoresearch/marconi-<tag>`.
2. `uv sync` has been run here.
3. `SGLANG_REPO` points at the target `sglang` checkout.
4. The target repo is on the branch you intend to improve, usually `marconi-eviction`.
5. `results.tsv` exists with only the header row shown below.

Do not modify this harness unless it is clearly wrong. Most code changes should happen in the target `sglang` repo.

## In-scope files in the target repo

Default in-scope files:

- `python/sglang/srt/mem_cache/mamba_radix_cache.py`
- `python/sglang/srt/mem_cache/marconi_tuner.py`
- `python/sglang/srt/mem_cache/marconi_replay_core.py`
- `python/sglang/srt/mem_cache/marconi_cost_model.py`
- `python/sglang/srt/managers/scheduler.py`

Touch other files only if there is a clear reason.

Do not edit:

- benchmark methodology in a way that breaks comparability
- the public API away from `--radix-eviction-policy marconi`
- `marconi-admission` unless the human explicitly redirects you

## Evaluation contract

One experiment is:

```bash
uv run train.py > run.log 2>&1
```

This harness:

- launches the target SGLang server from `SGLANG_REPO`
- forces `PYTHONPATH` to that checkout
- runs the Marconi benchmark
- prints a compact summary

Use:

- `MARCONI_EVAL_MODE=fast` for the main loop
- `MARCONI_EVAL_MODE=full` only for promising candidates

Optional extra server args:

```bash
export MARCONI_EXTRA_SERVER_ARGS="..."
```

## Primary goal

Optimize autotune quality on the Marconi eviction path.

The benchmark objective is:

1. the run must complete cleanly
2. autotune should start and apply at least one round
3. higher request throughput is better
4. lower mean TTFT is better

Use throughput as the primary keep/discard signal. Use TTFT as the tiebreaker.

## What counts as success

Keep a change only if it improves the autotune run under the evaluation mode you are using.

Do not keep changes that:

- only make fixed `eff_weight` runs better while leaving autotune weak
- add unstable live-apply behavior
- change policy in a way that obviously overfits one pathological late-round window
- add large complexity for tiny or unclear gains

## Results logging

Log every experiment to `results.tsv` and keep it untracked.

Header:

```text
target_commit	mode	throughput	mean_ttft_ms	autotune_applied	last_tuned_eff_weight	status	description
```

Status values:

- `keep`
- `discard`
- `crash`

Examples:

```text
target_commit	mode	throughput	mean_ttft_ms	autotune_applied	last_tuned_eff_weight	status	description
abc1234	fast	10.31	178966.25	1	1.0	keep	stabilize flat-score rounds to current weight
def5678	fast	9.84	183210.10	1	0.0	discard	larger second-round window
ghi9012	fast	0.00	0.00	0		crash	broken apply-path mutation
```

## Experiment loop

Loop forever until the human stops you.

1. Check target repo branch, commit, and dirty state.
2. Form one narrow hypothesis about autotune.
3. Edit the target repo.
4. Do not commit unless the human explicitly asks for commits.
5. Run one benchmark:
   - `MARCONI_EVAL_MODE=fast uv run train.py > run.log 2>&1`
6. Read the summary from `run.log`.
7. Log the result in `results.tsv`.
8. If the change is better, keep the local target-repo change.
9. If the change is worse or crashes, reset the target repo back to the previous good state.
10. Periodically confirm promising changes with:
   - `MARCONI_EVAL_MODE=full uv run train.py > run.log 2>&1`

## Current research direction

Start from the current Marconi autotune state that already fixed:

- late-result harvesting
- unsafe hot-path apply
- overlarge first tuning window
- destructive flat-score fallback to `0.0`

The remaining problem is policy quality. Fixed nonzero weights are still stronger than autotune on the saturated shared-prefix workload.

Good next ideas:

- better later-round window construction
- less noisy or more stable later-round selection
- cheaper first useful round without breaking steady-state policy
- objective refinements that help autotune approach healthy fixed-weight runs

Bad next ideas:

- changing public flags
- adding broad fallbacks
- hardcoding a favorite weight as a hidden default
- breaking comparability with the existing Marconi benchmark setup

## Simplicity rule

If two changes perform similarly, keep the simpler one.

Do not accumulate clever policy logic unless the benchmark clearly justifies it.

## Agent note

You are not a generic chat assistant. Do the work directly:

- inspect the target repo
- make one change
- run one experiment
- keep or discard

Do not stop to ask the human whether you should continue. Continue until interrupted.
