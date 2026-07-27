import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METRIC_PATHS = [
    ("point_auprc", ("ranking", "auprc")),
    ("point_f1", ("pointwise", "f1")),
    ("pa_f1", ("pa", "f1")),
    ("window_auprc", ("window_level", "auprc")),
    ("window_f1", ("window_level", "f1")),
]


def get_nested(obj, path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def flatten_metric(path: Path):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "metrics_path": str(path),
        "method": metrics.get("experiment_name"),
        "dataset": metrics.get("dataset"),
        "seed": metrics.get("seed"),
        "eval_unit": metrics.get("eval_unit"),
        "score_mode": metrics.get("score_mode"),
        "threshold_protocol": metrics.get("threshold_protocol"),
        "checkpoint_path": metrics.get("checkpoint_path"),
    }
    details = metrics.get("threshold_details", {})
    win_details = metrics.get("window_level", {}).get("threshold_details", {})
    row["uses_test_scores"] = bool(details.get("uses_test_scores")) or bool(win_details.get("uses_test_scores"))
    row["uses_test_labels"] = bool(details.get("uses_test_labels")) or bool(win_details.get("uses_test_labels"))
    row["uses_test_anomaly_ratio"] = bool(details.get("uses_test_anomaly_ratio")) or bool(
        win_details.get("uses_test_anomaly_ratio")
    )
    for name, nested in METRIC_PATHS:
        value = get_nested(metrics, nested)
        row[name] = "" if value is None else value
    return row


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    vals = [float(v) for v in values if v != "" and v is not None]
    if not vals:
        return "", "", 0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, var ** 0.5, len(vals)


def main():
    parser = argparse.ArgumentParser(description="Summarize paper supplement metrics.")
    parser.add_argument("--result_root", default="results/paper_supplement")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--expected_seeds", nargs="*", type=int, default=[2024, 2025, 2026])
    args = parser.parse_args()

    root = Path(args.result_root)
    output_dir = Path(args.output_dir) if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [flatten_metric(path) for path in sorted(root.rglob("metrics.json"))]

    issues = []
    for row in rows:
        if row["uses_test_scores"] or row["uses_test_labels"] or row["uses_test_anomaly_ratio"]:
            issues.append({"type": "leaky_threshold", **row})
        if row["dataset"] == "ST330IR001_CP001" and row["eval_unit"] != "window":
            issues.append({"type": "st_missing_window_eval", **row})

    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["dataset"], row["eval_unit"], row["score_mode"])].append(row)

    summary = []
    for key, group_rows in sorted(groups.items()):
        seeds = sorted(int(r["seed"]) for r in group_rows if r["seed"] is not None)
        missing = sorted(set(args.expected_seeds) - set(seeds))
        if missing:
            issues.append({"type": "missing_seeds", "method": key[0], "dataset": key[1], "missing": missing})
        out = {"method": key[0], "dataset": key[1], "eval_unit": key[2], "score_mode": key[3], "seeds": " ".join(map(str, seeds))}
        for metric, _ in METRIC_PATHS:
            mean, std, n = mean_std([r[metric] for r in group_rows])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_n"] = n
        summary.append(out)

    long_fields = list(rows[0].keys()) if rows else [
        "metrics_path", "method", "dataset", "seed", "eval_unit", "score_mode", "threshold_protocol",
        "checkpoint_path", "uses_test_scores", "uses_test_labels", "uses_test_anomaly_ratio",
    ] + [name for name, _ in METRIC_PATHS]
    write_csv(output_dir / "summary_long.csv", rows, long_fields)
    summary_fields = list(summary[0].keys()) if summary else [
        "method", "dataset", "eval_unit", "score_mode", "seeds"
    ]
    write_csv(output_dir / "summary_mean_std.csv", summary, summary_fields)
    manifest = {"result_root": str(root), "n_runs": len(rows), "issues": issues}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if issues:
        raise SystemExit(f"Summary completed with {len(issues)} issue(s); see run_manifest.json")
    print(f"Summarized {len(rows)} runs into {output_dir}")


if __name__ == "__main__":
    main()
