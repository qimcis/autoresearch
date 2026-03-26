# Hermes Workspace Instructions

This workspace is a fixed research harness for Marconi autotune work. Hermes should read `README.md` and `program.md` at session start and treat them as the primary repo instructions.

## What this workspace is for

- run autonomous Marconi autotune experiments against a target `sglang` checkout
- use this repo only as the harness and log
- mainly edit the target repo pointed to by `SGLANG_REPO`

## Default target

Unless the human explicitly says otherwise:

- target repo: `SGLANG_REPO`
- target branch: `marconi-eviction`
- target model for cheap iteration: `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
- target hardware: `1xA100`

The harness is configured through environment variables. Hermes should not rewrite the harness unless it is clearly broken.

## Required reading

At the beginning of a session, read:

1. `README.md`
2. `program.md`
3. `${SGLANG_REPO}/context.txt` if it exists
4. `${SGLANG_REPO}/marconi.txt` if it exists

If `context.txt` and `marconi.txt` are missing, continue with the repo-local instructions.

## Working rules

- Make code changes in the target `sglang` repo, not mainly in this harness.
- Do not change the public user contract away from `--radix-eviction-policy marconi`.
- Do not broaden into unrelated Marconi work.
- Prefer one narrow hypothesis per iteration.
- Log every run in `results.tsv`.
- Keep or revert local target-repo changes based on benchmark signal.
- Do not commit unless the human explicitly asks for commits.

## Benchmark signal

Primary metric:
- request throughput on autotune-enabled runs

Secondary metric:
- mean TTFT

The run must also:
- complete cleanly
- apply tuning safely

## Fast vs full

- `MARCONI_EVAL_MODE=fast` is for smoke tests and cheaper iteration.
- `MARCONI_EVAL_MODE=full` is for policy-quality validation.

If fast mode repeatedly yields weak autotune signal, move promising changes to full mode instead of over-optimizing the fast harness.
