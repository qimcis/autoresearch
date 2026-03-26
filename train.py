import traceback

from prepare import (
    benchmark_config,
    build_summary,
    create_run_dir,
    ensure_gpu_count,
    ensure_repo_layout,
    eval_mode,
    launch_server,
    load_last_result,
    run_benchmark,
    stop_process,
    target_repo,
)


def emit(summary: dict[str, object]) -> None:
    print("---")
    for key, value in summary.items():
        print(f"{key}: {value}")


def main() -> int:
    mode = eval_mode()
    repo = target_repo()
    run_dir = create_run_dir(mode)
    summary: dict[str, object] | None = None
    server = None
    server_log = run_dir / "server.log"

    try:
        ensure_gpu_count()
        ensure_repo_layout(repo)
        config = benchmark_config(mode)
        server, server_log = launch_server(repo, run_dir)
        result_file = run_benchmark(repo, run_dir, config)
        result = load_last_result(result_file)
        summary = build_summary(result, mode, repo, run_dir, server_log)
        emit(summary)
        return 0
    except Exception as exc:
        summary = {
            "status": "crash",
            "mode": mode,
            "target_repo": str(repo),
            "error": str(exc),
            "server_log": str(server_log),
            "traceback": traceback.format_exc(),
        }
        emit(summary)
        return 1
    finally:
        stop_process(server)


if __name__ == "__main__":
    raise SystemExit(main())
