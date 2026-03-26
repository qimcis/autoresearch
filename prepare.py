import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import torch


DEFAULT_MODEL_PATH = "Qwen/Qwen3-Next-80B-A3B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:30000"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_TP_SIZE = 4
DEFAULT_CONTEXT_LENGTH = 32768
DEFAULT_PAGE_SIZE = 1
DEFAULT_MAMBA_FULL_MEMORY_RATIO = "0.9"


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    num_prompts: int
    gsp_num_groups: int
    gsp_prompts_per_group: int
    gsp_system_prompt_len: int
    gsp_question_len: int
    gsp_output_len: int
    gsp_num_turns: int
    request_rate: str


FAST_BENCHMARK = BenchmarkConfig(
    name="fast",
    num_prompts=1536,
    gsp_num_groups=96,
    gsp_prompts_per_group=16,
    gsp_system_prompt_len=8192,
    gsp_question_len=512,
    gsp_output_len=128,
    gsp_num_turns=1,
    request_rate="inf",
)

FULL_BENCHMARK = BenchmarkConfig(
    name="full",
    num_prompts=3072,
    gsp_num_groups=192,
    gsp_prompts_per_group=16,
    gsp_system_prompt_len=8192,
    gsp_question_len=512,
    gsp_output_len=128,
    gsp_num_turns=1,
    request_rate="inf",
)


def target_repo() -> Path:
    return Path(os.environ.get("SGLANG_REPO", "/home/qi/sglang")).expanduser().resolve()


def model_path() -> str:
    return os.environ.get("MARCONI_MODEL_PATH", DEFAULT_MODEL_PATH)


def tensor_parallel_size() -> int:
    return int(os.environ.get("MARCONI_TP_SIZE", str(DEFAULT_TP_SIZE)))


def min_gpu_count() -> int:
    return int(os.environ.get("MARCONI_MIN_GPUS", str(tensor_parallel_size())))


def context_length() -> int:
    return int(os.environ.get("MARCONI_CONTEXT_LENGTH", str(DEFAULT_CONTEXT_LENGTH)))


def page_size() -> int:
    return int(os.environ.get("MARCONI_PAGE_SIZE", str(DEFAULT_PAGE_SIZE)))


def mamba_full_memory_ratio() -> str:
    return os.environ.get(
        "MARCONI_MAMBA_FULL_MEMORY_RATIO", DEFAULT_MAMBA_FULL_MEMORY_RATIO
    )


def base_url() -> str:
    return os.environ.get("MARCONI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def results_root() -> Path:
    return Path(os.environ.get("MARCONI_RESULTS_DIR", DEFAULT_RESULTS_DIR)).resolve()


def extra_server_args() -> list[str]:
    value = os.environ.get("MARCONI_EXTRA_SERVER_ARGS", "")
    return shlex.split(value)


def eval_mode() -> str:
    return os.environ.get("MARCONI_EVAL_MODE", "fast").strip().lower()


def benchmark_config(mode: str | None = None) -> BenchmarkConfig:
    mode = (mode or eval_mode()).strip().lower()
    if mode == "fast":
        return FAST_BENCHMARK
    if mode == "full":
        return FULL_BENCHMARK
    raise ValueError(f"Unsupported MARCONI_EVAL_MODE: {mode}")


def ensure_repo_layout(repo: Path) -> None:
    required = [
        repo / ".git",
        repo / "python" / "sglang" / "bench_serving.py",
        repo / "python" / "sglang" / "srt" / "mem_cache" / "mamba_radix_cache.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Target SGLang repo is missing required paths:\n" + "\n".join(missing)
        )


def ensure_gpu_count(min_count: int | None = None) -> None:
    min_count = min_count or min_gpu_count()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    count = torch.cuda.device_count()
    if count < min_count:
        raise RuntimeError(f"Expected at least {min_count} GPUs, found {count}")


def repo_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(repo / "python")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pythonpath
    return env


def repo_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


def target_branch(repo: Path) -> str:
    return repo_git(repo, "branch", "--show-current")


def target_commit(repo: Path) -> str:
    return repo_git(repo, "rev-parse", "--short", "HEAD")


def target_is_dirty(repo: Path) -> bool:
    return bool(repo_git(repo, "status", "--short"))


def wait_for_server(process: subprocess.Popen[str], server_log: Path, timeout_s: int = 1800) -> None:
    deadline = time.time() + timeout_s
    url = base_url() + "/health_generate"
    while time.time() < deadline:
        if process.poll() is not None:
            tail = ""
            if server_log.exists():
                tail = "\n".join(server_log.read_text().splitlines()[-200:])
            raise RuntimeError(
                "Server exited before becoming healthy.\n" + tail
            )
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(f"Server did not become healthy within {timeout_s} seconds")


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def launch_server(repo: Path, run_dir: Path) -> tuple[subprocess.Popen[str], Path]:
    server_log = run_dir / "server.log"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path(),
        "--tp-size",
        str(tensor_parallel_size()),
        "--context-length",
        str(context_length()),
        "--schedule-policy",
        "lpm",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--mamba-full-memory-ratio",
        mamba_full_memory_ratio(),
        "--page-size",
        str(page_size()),
        "--enable-metrics",
        "--radix-eviction-policy",
        "marconi",
        *extra_server_args(),
    ]
    with server_log.open("w", encoding="utf-8") as fout:
        process = subprocess.Popen(
            cmd,
            cwd=repo,
            env=repo_env(repo),
            stdout=fout,
            stderr=subprocess.STDOUT,
            text=True,
        )
    wait_for_server(process, server_log)
    return process, server_log


def run_benchmark(repo: Path, run_dir: Path, config: BenchmarkConfig) -> Path:
    output_file = run_dir / "result.jsonl"
    bench_log = run_dir / "bench.log"
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--dataset-name",
        "generated-shared-prefix",
        "--num-prompts",
        str(config.num_prompts),
        "--gsp-num-groups",
        str(config.gsp_num_groups),
        "--gsp-prompts-per-group",
        str(config.gsp_prompts_per_group),
        "--gsp-system-prompt-len",
        str(config.gsp_system_prompt_len),
        "--gsp-question-len",
        str(config.gsp_question_len),
        "--gsp-output-len",
        str(config.gsp_output_len),
        "--gsp-num-turns",
        str(config.gsp_num_turns),
        "--request-rate",
        config.request_rate,
        "--seed",
        "7",
        "--output-file",
        str(output_file),
        "--output-details",
    ]
    with bench_log.open("w", encoding="utf-8") as fout:
        subprocess.run(
            cmd,
            cwd=repo,
            env=repo_env(repo),
            stdout=fout,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    return output_file


def load_last_result(output_file: Path) -> dict[str, Any]:
    lines = output_file.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"Empty benchmark result file: {output_file}")
    return json.loads(lines[-1])


def _find_first(data: Any, key: str) -> Any:
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            found = _find_first(value, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find_first(value, key)
            if found is not None:
                return found
    return None


def build_summary(result: dict[str, Any], mode: str, repo: Path, run_dir: Path, server_log: Path) -> dict[str, Any]:
    server_info = result.get("server_info") or {}
    summary = {
        "status": "completed",
        "mode": mode,
        "target_repo": str(repo),
        "target_branch": target_branch(repo),
        "target_commit": target_commit(repo),
        "target_dirty": target_is_dirty(repo),
        "request_throughput": result.get("request_throughput"),
        "mean_ttft_ms": result.get("mean_ttft_ms"),
        "mean_e2e_latency_ms": result.get("mean_e2e_latency_ms"),
        "completed_requests": result.get("completed"),
        "results_jsonl": str(run_dir / "result.jsonl"),
        "bench_log": str(run_dir / "bench.log"),
        "server_log": str(server_log),
        "server_info_json": str(run_dir / "server_info.json"),
    }
    for key in [
        "autotune_started",
        "autotune_finished",
        "autotune_applied",
        "autotune_skipped",
        "autotune_rounds",
        "autotune_inflight",
        "current_eff_weight",
        "last_tuned_eff_weight",
        "autotune_status",
        "token_hit_rate",
        "evict_mamba_states",
    ]:
        summary[key] = _find_first(server_info, key)
    (run_dir / "server_info.json").write_text(
        json.dumps(server_info, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def create_run_dir(mode: str) -> Path:
    run_dir = results_root() / time.strftime("%Y%m%d-%H%M%S") / mode
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
