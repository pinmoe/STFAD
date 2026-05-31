#!/usr/bin/env python3
import argparse
import csv
import pathlib
import re

PATTERN_METRIC = re.compile(
    r"Accuracy\s*:\s*([0-9.]+),\s*Precision\s*:\s*([0-9.]+),\s*Recall\s*:\s*([0-9.]+),\s*F-score\s*:\s*([0-9.]+)"
)
PATTERN_THRESH = re.compile(r"Optimal threshold \(grid search\):\s*([0-9.eE+-]+)\s*→\s*expected F1\s*≈\s*([0-9.]+)")
PATTERN_RUN = re.compile(r"\[RUN\]\s+([^\s]+)\s+\(mode=([^,]+),\s*win=([^,]+),\s*k=([^\)]+)\)")

def find_latest_log_root(base: pathlib.Path) -> pathlib.Path:
    cands = sorted(base.glob("thesis_pack_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError("未找到 logs/thesis_pack_* 目录，请先运行实验脚本。")
    return cands[0]

def parse_log(path: pathlib.Path):
    run_tag = mode = win = kval = None
    threshold = exp_f1 = None
    acc = prec = rec = f1 = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = PATTERN_RUN.search(line)
        if m:
            run_tag, mode, win, kval = m.group(1), m.group(2), m.group(3), m.group(4)
        m = PATTERN_THRESH.search(line)
        if m:
            threshold, exp_f1 = m.group(1), m.group(2)
        m = PATTERN_METRIC.search(line)
        if m:
            acc, prec, rec, f1 = m.group(1), m.group(2), m.group(3), m.group(4)
    if f1 is None:
        return None
    return {
        "run_tag": run_tag or path.stem.replace("_test", ""),
        "mode": mode or "",
        "win_size": win or "",
        "k": kval or "",
        "threshold": threshold or "",
        "expected_f1": exp_f1 or "",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "log_file": str(path),
    }

def main():
    ap = argparse.ArgumentParser(description="汇总 run_thesis_minimum_pack.sh 产生的 test 日志")
    ap.add_argument("log_root", nargs="?", default="latest", help="日志目录；不填或填 latest 时自动选最新 logs/thesis_pack_*")
    ap.add_argument("--output", default=None, help="输出 CSV 路径（默认在日志目录下）")
    args = ap.parse_args()

    if args.log_root == "latest":
        root = find_latest_log_root(pathlib.Path("logs"))
    else:
        root = pathlib.Path(args.log_root)
    if not root.exists():
        raise FileNotFoundError(f"日志目录不存在: {root}")

    rows = []
    for p in sorted(root.glob("*_test.log")):
        parsed = parse_log(p)
        if parsed:
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f"未在 {root} 的 *_test.log 中找到可解析指标。")

    out = pathlib.Path(args.output) if args.output else (root / "summary.csv")
    fields = ["run_tag", "mode", "win_size", "k", "threshold", "expected_f1", "accuracy", "precision", "recall", "f1", "log_file"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] 汇总 {len(rows)} 条结果 -> {out}")
    top = sorted(rows, key=lambda x: float(x["f1"]), reverse=True)[:5]
    print("Top-5 by F1:")
    for r in top:
        print(f"  {r['run_tag']:<28} F1={r['f1']} P={r['precision']} R={r['recall']} Acc={r['accuracy']}")

if __name__ == "__main__":
    main()