# Marconi Context

This file is the working handoff for autonomous Marconi autotune research.

Read this after `README.md` and `program.md`.

## What Marconi is

Marconi is a prefix-caching policy for hybrid LLMs.

In this project, Marconi means:

- **selective admission** for hybrid branch checkpoints
- **hybrid-aware cache bookkeeping** for KV cache and recurrent state
- **eviction policy** that combines recency with an efficiency / compute-savings term
- **autotuning** of the efficiency weight used by eviction

The key point is that Marconi is not just “cache more.” It tries to decide:

1. which reusable branch points are worth caching
2. which cached entries are worth keeping under pressure
3. how much to weight reuse efficiency relative to recency

## What branch matters here

The target branch for this research is `marconi-eviction`.

There was an intentional split:

- `marconi-admission`
  - admission / checkpointing / correctness infrastructure
  - not the source of the measured performance win
- `marconi-eviction`
  - Marconi eviction policy
  - autotuner
  - this is the branch we are improving now

Do not spend time on `marconi-admission` unless explicitly redirected.

## Public API constraint

The public user contract must stay aligned with:

```bash
--radix-eviction-policy marconi
```

Do not introduce a separate public Marconi enable flag.

## What reviewers cared about

Two reviewers matter most for this work:

- **Xinyi**
  - wanted the public API to use `--radix-eviction-policy marconi`
  - wanted invalid configs to fail early
  - called out incorrect replay-cost accounting when ancestors are tombstoned
  - called out architecture-specific FLOPs assumptions

- **Rui Pan**
  - author of the Marconi paper
  - wanted admission vs eviction separated
  - wanted autotuning to exist and behave sensibly
  - wanted `eff_weight=0` ablations and weight sweeps
  - warned that saturated TTFT includes queueing and should be interpreted carefully

## What was already fixed before this autoresearch loop

These are **already fixed** on `marconi-eviction` and should be treated as solved infrastructure unless a regression is discovered:

### Autotune lifecycle / scheduling

- finished tuning results are no longer lost near the tail of the run
- the first bootstrap window no longer grows pathologically from pre-eviction history
- tuning results are no longer applied from the eviction hot path
- live weight apply was moved to safer boundaries

### Live-apply correctness

- live switching itself was causing regressions under heavy load
- that specific unsafe apply-path bug was fixed
- fixed manual weights such as `1.6` were shown to be healthy

### Later-round stability

- later rounds could revert to `0.0` on flat score grids simply because `0.0` was the first candidate
- that destructive fallback was fixed
- when the active weight is effectively tied with the nominal winner, the tuner can keep the current weight

### First-round timing

- the first tuning round was made cheaper by using a coarse grid first
- this made the first apply happen materially earlier

## What the benchmark history showed

These numbers are from the earlier `Qwen3-Next-80B-A3B-Instruct` `4xH100` work and remain the main grounding for Marconi behavior.

### Saturated shared-prefix workload

- `main`
  - `10.71 req/s`
  - mean TTFT `177466.98 ms`
- `marconi-admission`
  - `9.78 req/s`
  - mean TTFT `177092.55 ms`
- `marconi-eviction`, `eff_weight=0`, autotune off
  - `9.12 req/s`
  - mean TTFT `186987.42 ms`
- `marconi-eviction`, autotune on after earlier fixes
  - `9.36 req/s`
  - mean TTFT `181315.80 ms`
- fixed nonzero weights
  - `0.5`: `11.99 req/s`, mean TTFT `156059.60 ms`
  - `1.0`: `11.74 req/s`, mean TTFT `157847.51 ms`
  - `1.6`: `11.85 req/s`, mean TTFT `155529.64 ms`
  - `2.0`: `12.14 req/s`, mean TTFT `155651.28 ms`

Main conclusion from those runs:

- **manual nonzero `eff_weight` produced the win**
- autotune was made functional and stable, but it did **not** match the healthy fixed-weight runs on that workload

## What we learned from later autotune debugging

The remaining problem is not basic plumbing anymore.

The remaining problem is **policy quality**.

In particular:

- `fast` mode is often too short and too low-signal for autotune quality
- `full` mode is more informative
- later rounds can become effectively flat
- timing-only fixes helped, but they did not close the gap to the healthy fixed-weight runs

More recent diagnostic conclusions:

- in `fast` mode, replay windows can have almost no pressure-bearing requests
- replay score spreads can be exactly flat across the weight grid
- when that happens, there is little or no real signal for autotune to learn from
- `full` mode showed more meaningful spread in round 1
- later rounds in `full` could still go flat

## Current working hypothesis area

Do **not** spend most effort on:

- API redesign
- admission logic
- more scheduler/lifecycle plumbing
- benchmark harness churn

Do focus on:

- replay objective quality
- window composition and signal quality
- later-round stability
- conditions under which a later round should be skipped or frozen
- small policy changes that make autotune closer to the healthy fixed-weight runs

## Current research setup

We are now using this harness for cheaper iteration on smaller hardware.

Current intended proxy setup:

- hardware: `1xA100`
- model: `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
- target branch: `marconi-eviction`

This is a **proxy**, not final truth.

Why we use it:

- cheaper and easier to iterate than `Qwen3-Next-80B` on `4xH100`
- still a relevant hybrid-model serving setup

What it is good for:

- autotune logic
- replay scoring behavior
- later-round policy experiments
- instrumentation-guided debugging

What it is **not** good for:

- final claims about Marconi on `Qwen3-Next-80B-A3B-Instruct`

Any promising result here should eventually be validated on the real larger setup.

## What success means in this repo

We are optimizing for **autotune-enabled runs**, not fixed-weight-only runs.

Primary metric:

- **request throughput**

Secondary metric:

- **mean TTFT**

A run is only useful if:

- it completes cleanly
- tuning starts and applies safely
- the result improves autotune behavior, not just fixed-weight behavior

If a change makes fixed `eff_weight` runs look better but leaves autotune weak, that is not success for this loop.

## Fast vs full

`fast` mode:

- cheaper
- useful for smoke tests
- often weak signal for autotune quality

`full` mode:

- more expensive
- better for policy-quality validation
- should be used for promising candidates or when `fast` is clearly flat/noisy

Do not overfit the code to the `fast` harness if it is not giving useful autotune signal.

## Working rules for the agent

- Make code changes in the target `sglang` repo, not mainly in this harness.
- Prefer one narrow hypothesis per iteration.
- Keep detailed notes in `results.tsv`.
- Do not commit unless explicitly asked.
- Do not stop after one experiment or one batch of experiments unless explicitly interrupted.
- If `fast` mode repeatedly gives flat-score / no-apply behavior, escalate promising ideas to `full` mode.

## Short version

Marconi is a hybrid prefix-caching policy.

The admission/eviction split already happened.

The remaining problem is **autotune policy quality**.

We already fixed:

- lost tuning results
- unsafe live apply
- oversized bootstrap windows
- destructive flat-score snap-back to `0.0`

We have **not** yet made autotune match the healthy fixed nonzero weights.

In this loop, optimize for:

1. **throughput**
2. **TTFT**

and prefer changes that improve autotune itself, not just fixed-weight ablations.
