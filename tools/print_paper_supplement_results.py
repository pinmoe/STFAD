#!/usr/bin/env python3
"""Print paper-ready tables from paper-supplement experiment metrics.

The reporting protocol follows the manuscript:
* MSL/SMAP/SKAB: point-level AUPRC and point-wise F1 are primary;
  PA-F1 is auxiliary.
* ST330IR001_CP001: window-level AUPRC, precision, recall, and F1 are
  primary because each sample is an independent operating cycle.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


PUBLIC_DATASETS = ("MSL", "SMAP", "SKAB")
ST330 = "ST330IR001_CP001"
METHOD_ORDER = (
    "E1_gaussian",
    "E3_multiscale",
    "E1_gaussian_rec_mean",
    "E3_multiscale_rec_mean",
    "iforest",
    "ocsvm",
    "autoencoder",
)
METHOD_LABELS = {
    "E1_gaussian": "E1 Gaussian",
    "E3_multiscale": "E3 Multiscale",
    "E1_gaussian_rec_mean": "E1 Gaussian + rec_mean",
    "E3_multiscale_rec_mean": "E3 Multiscale + rec_mean",
    "iforest": "Isolation Forest",
    "ocsvm": "One-Class SVM",
    "autoencoder": "Autoencoder",
}


def nested_get(obj, *keys):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def fmt_number(value):
    return "" if value is None else f"{float(value):.4f}"


def mean_std(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std


def fmt_mean_std(values):
    mean, std = mean_std(values)
    if mean is None:
        return "—"
    return f"{mean:.4f}±{std:.4f}"


def method_sort_key(method):
    try:
        return METHOD_ORDER.index(method), method
    except ValueError:
        return len(METHOD_ORDER), method


def read_runs(result_root):
    runs = []
    for path in sorted(result_root.rglob("metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        method = metrics.get("experiment_name") or path.parents[1].name
        dataset = metrics.get("dataset") or path.parents[2].name
        seed = metrics.get("seed")
        if seed is None and path.parent.name.startswith("seed_"):
            seed = path.parent.name.removeprefix("seed_")
        details = metrics.get("threshold_details", {})
        window_details = nested_get(metrics, "window_level", "threshold_details") or {}
        runs.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": int(seed) if seed is not None else None,
                "metrics_path": str(path),
                "eval_unit": metrics.get("eval_unit"),
                "score_mode": metrics.get("score_mode"),
                "threshold_protocol": metrics.get("threshold_protocol"),
                "uses_test_scores": bool(details.get("uses_test_scores"))
                or bool(window_details.get("uses_test_scores")),
                "uses_test_labels": bool(details.get("uses_test_labels"))
                or bool(window_details.get("uses_test_labels")),
                "uses_test_anomaly_ratio": bool(details.get("uses_test_anomaly_ratio"))
                or bool(window_details.get("uses_test_anomaly_ratio")),
                "point_auprc": nested_get(metrics, "ranking", "auprc"),
                "point_precision": nested_get(metrics, "pointwise", "precision"),
                "point_recall": nested_get(metrics, "pointwise", "recall"),
                "point_f1": nested_get(metrics, "pointwise", "f1"),
                "pa_f1": nested_get(metrics, "pa", "f1"),
                "window_auprc": nested_get(metrics, "window_level", "auprc"),
                "window_precision": nested_get(metrics, "window_level", "precision"),
                "window_recall": nested_get(metrics, "window_level", "recall"),
                "window_f1": nested_get(metrics, "window_level", "f1"),
                "n_windows": nested_get(metrics, "window_level", "n_windows"),
                "anomaly_windows": nested_get(metrics, "window_level", "anomaly_windows"),
            }
        )
    return runs


def group_runs(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["dataset"], run["method"])].append(run)
    return groups


def aggregate_public(groups):
    rows = []
    for dataset in PUBLIC_DATASETS:
        methods = sorted(
            (method for ds, method in groups if ds == dataset),
            key=method_sort_key,
        )
        for method in methods:
            items = groups[(dataset, method)]
            rows.append(
                {
                    "数据集": dataset,
                    "方法": METHOD_LABELS.get(method, method),
                    "种子数": len(items),
                    "AUPRC": fmt_mean_std([x["point_auprc"] for x in items]),
                    "逐点Precision": fmt_mean_std([x["point_precision"] for x in items]),
                    "逐点Recall": fmt_mean_std([x["point_recall"] for x in items]),
                    "逐点F1": fmt_mean_std([x["point_f1"] for x in items]),
                    "PA-F1（辅助）": fmt_mean_std([x["pa_f1"] for x in items]),
                }
            )
    return rows


def aggregate_st330(groups):
    rows = []
    methods = sorted(
        (method for ds, method in groups if ds == ST330),
        key=method_sort_key,
    )
    for method in methods:
        items = groups[(ST330, method)]
        rows.append(
            {
                "数据集": "ST330IR001",
                "方法": METHOD_LABELS.get(method, method),
                "种子数": len(items),
                "窗口AUPRC": fmt_mean_std([x["window_auprc"] for x in items]),
                "窗口Precision": fmt_mean_std([x["window_precision"] for x in items]),
                "窗口Recall": fmt_mean_std([x["window_recall"] for x in items]),
                "窗口F1": fmt_mean_std([x["window_f1"] for x in items]),
            }
        )
    return rows


def raw_rows(runs):
    keys = (
        "point_auprc",
        "point_precision",
        "point_recall",
        "point_f1",
        "pa_f1",
        "window_auprc",
        "window_precision",
        "window_recall",
        "window_f1",
    )
    rows = []
    for run in sorted(
        runs,
        key=lambda x: (x["dataset"], method_sort_key(x["method"]), x["seed"] or -1),
    ):
        row = {
            "dataset": run["dataset"],
            "method": run["method"],
            "seed": run["seed"],
            "eval_unit": run["eval_unit"],
            "score_mode": run["score_mode"],
            "threshold_protocol": run["threshold_protocol"],
        }
        row.update({key: fmt_number(run[key]) for key in keys})
        row["metrics_path"] = run["metrics_path"]
        rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    if not rows:
        return "_无数据_"
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def audit(runs, groups, expected_seeds):
    issues = []
    expected = set(expected_seeds)
    for (dataset, method), items in sorted(groups.items()):
        seeds = {item["seed"] for item in items}
        if seeds != expected:
            issues.append(
                f"{dataset}/{method}: seeds={sorted(x for x in seeds if x is not None)}, "
                f"expected={sorted(expected)}"
            )
        if any(
            item["uses_test_scores"]
            or item["uses_test_labels"]
            or item["uses_test_anomaly_ratio"]
            for item in items
        ):
            issues.append(f"{dataset}/{method}: threshold may use test information")
        if dataset == ST330 and any(item["window_f1"] is None for item in items):
            issues.append(f"{dataset}/{method}: missing window-level metrics")
        if dataset in PUBLIC_DATASETS and any(item["point_auprc"] is None for item in items):
            issues.append(f"{dataset}/{method}: missing point-level AUPRC")
    if not runs:
        issues.append("no metrics.json files found")
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Print paper-ready mean±sample-std tables for paper-supplement results."
    )
    parser.add_argument(
        "--result-root",
        default="results/paper_supplement",
        help="Root containing dataset/method/seed_*/metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: RESULT_ROOT/paper_tables).",
    )
    parser.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        default=[2024, 2025, 2026],
    )
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir) if args.output_dir else result_root / "paper_tables"
    runs = read_runs(result_root)
    groups = group_runs(runs)
    public_rows = aggregate_public(groups)
    st330_rows = aggregate_st330(groups)
    issues = audit(runs, groups, args.expected_seeds)

    public_md = markdown_table(public_rows)
    st330_md = markdown_table(st330_rows)
    report = (
        "# 论文结果汇总\n\n"
        "统计口径：三个随机种子的均值±样本标准差（ddof=1）。\n\n"
        "## 公开数据集点级结果\n\n"
        "AUPRC和逐点F1为主要指标，PA-F1仅作辅助参考。\n\n"
        f"{public_md}\n\n"
        "## ST330IR001窗口级结果\n\n"
        "每个窗口对应一个独立工作周期，窗口级指标为主要评价口径。\n\n"
        f"{st330_md}\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_results.md").write_text(report, encoding="utf-8")
    write_csv(output_dir / "public_point_metrics.csv", public_rows)
    write_csv(output_dir / "st330_window_metrics.csv", st330_rows)
    write_csv(output_dir / "all_seed_metrics.csv", raw_rows(runs))
    (output_dir / "audit.json").write_text(
        json.dumps(
            {
                "n_runs": len(runs),
                "expected_seeds": args.expected_seeds,
                "issues": issues,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(report)
    print(f"\n共读取 {len(runs)} 次运行。")
    print(f"文件已保存到：{output_dir}")
    if issues:
        print("\n[检查警告]")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("[检查通过] 随机种子齐全、未发现测试信息选阈值、评价单位完整。")


if __name__ == "__main__":
    main()
