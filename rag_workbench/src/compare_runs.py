import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


METRIC_KEYS = [
    "retrieval_keyword_hit_avg",
    "answer_keyword_hit_avg",
    "citation_presence_avg",
    "citation_coverage_avg",
]

STAGE_KEYS = [
    "retrieval_miss",
    "retrieval_low_signal",
    "generation_empty_or_timeout",
    "ungrounded_answer",
]


def load_runs(path: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Keep benchmark summaries only.
        if isinstance(obj, dict) and "eval_count" in obj and "results" in obj:
            runs.append(obj)
    if not runs:
        raise ValueError("No benchmark summary runs found in log file.")
    return runs


def get_by_index(items: List[Dict[str, Any]], idx: int) -> Dict[str, Any]:
    n = len(items)
    actual = idx if idx >= 0 else n + idx
    if actual < 0 or actual >= n:
        raise IndexError(f"Run index {idx} out of range for {n} runs.")
    return items[actual]


def metric_value(run: Dict[str, Any], key: str) -> float:
    value = run.get(key, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def stage_rate(run: Dict[str, Any], stage_key: str) -> float:
    eval_count = max(1, int(run.get("eval_count", 1)))
    stage_counts = run.get("stage_counts", {}) or {}
    count = int(stage_counts.get(stage_key, 0))
    return count / float(eval_count)


def run_label(run: Dict[str, Any]) -> str:
    return (
        f"ts={run.get('timestamp', '')}; provider={run.get('provider', '')}; "
        f"model={run.get('model', '')}; retriever={run.get('retriever', 'keyword')}; "
        f"reranker={run.get('reranker', 'none')}; eval_count={run.get('eval_count', 0)}"
    )


def compare_runs(base: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    metric_deltas: Dict[str, float] = {}
    for key in METRIC_KEYS:
        metric_deltas[key] = round(metric_value(target, key) - metric_value(base, key), 4)

    stage_rate_deltas: Dict[str, float] = {}
    for key in STAGE_KEYS:
        stage_rate_deltas[key] = round(stage_rate(target, key) - stage_rate(base, key), 4)

    regressions: List[str] = []
    for key in [
        "retrieval_keyword_hit_avg",
        "answer_keyword_hit_avg",
        "citation_presence_avg",
        "citation_coverage_avg",
    ]:
        if metric_deltas[key] < 0:
            regressions.append(f"Metric dropped: {key} ({metric_deltas[key]})")

    for key in STAGE_KEYS:
        if stage_rate_deltas[key] > 0:
            regressions.append(
                f"Failure stage increased: {key} (+{stage_rate_deltas[key]} rate)"
            )

    return {
        "base": base,
        "target": target,
        "metric_deltas": metric_deltas,
        "stage_rate_deltas": stage_rate_deltas,
        "regressions": regressions,
        "is_regression": len(regressions) > 0,
    }


def print_run_list(runs: List[Dict[str, Any]], last_n: int) -> None:
    start = max(0, len(runs) - last_n)
    print("=== Recent Runs ===")
    for i in range(start, len(runs)):
        run = runs[i]
        print(f"[{i}] {run_label(run)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark runs from eval log")
    parser.add_argument("--log-file", default="logs/eval_runs.jsonl")
    parser.add_argument("--base-index", type=int, default=-2)
    parser.add_argument("--target-index", type=int, default=-1)
    parser.add_argument("--list", action="store_true", help="List recent run indices and metadata")
    parser.add_argument("--last-n", type=int, default=10)
    parser.add_argument("--output-file", default="")
    args = parser.parse_args()

    runs = load_runs(Path(args.log_file))

    if args.list:
        print_run_list(runs, args.last_n)
        return

    base = get_by_index(runs, args.base_index)
    target = get_by_index(runs, args.target_index)
    report = compare_runs(base, target)

    print("=== Run Compare ===")
    print("Base:")
    print(run_label(base))
    print("Target:")
    print(run_label(target))
    print()

    print("Metric deltas (target - base):")
    for key in METRIC_KEYS:
        print(f"- {key}: {report['metric_deltas'][key]}")

    print("Stage rate deltas (target - base):")
    for key in STAGE_KEYS:
        print(f"- {key}: {report['stage_rate_deltas'][key]}")

    print()
    if report["is_regression"]:
        print("Regression warnings:")
        for item in report["regressions"]:
            print(f"- {item}")
    else:
        print("No regression warning detected.")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
