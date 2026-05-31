#!/usr/bin/env python3
"""汇总所有 logs/thesis_pack_* 目录的结果，合并输出。"""
import argparse, csv, pathlib, re, sys

PATTERN_METRIC = re.compile(
    r"Accuracy\s*:\s*([0-9.]+),\s*Precision\s*:\s*([0-9.]+),\s*Recall\s*:\s*([0-9.]+),\s*F-score\s*:\s*([0-9.]+)"
)
PATTERN_THRESH = re.compile(r"Optimal threshold \(grid search\):\s*([0-9.eE+-]+)\s*→\s*expected F1\s*≈\s*([0-9.]+)")
PATTERN_RUN = re.compile(r"\[RUN\]\s+([^\s]+)\s+\(mode=([^,]+),\s*win=([^,]+),\s*k=([^\)]+)\)")


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
        "log_dir": str(path.parent.name),
        "log_file": str(path),
    }


def main():
    ap = argparse.ArgumentParser(description="汇总所有 thesis_pack_* 目录的 test 日志")
    ap.add_argument("--base", default="logs")
    ap.add_argument("--output", default=None)
    ap.add_argument("--top", type=int, default=20, help="打印 Top-N 条")
    args = ap.parse_args()

    base = pathlib.Path(args.base)
    dirs = sorted(base.glob("thesis_pack_*"))

    if not dirs:
        print("未找到 logs/thesis_pack_* 目录", file=sys.stderr)
        sys.exit(1)

    rows = []
    for d in dirs:
        for p in sorted(d.glob("*_test.log")):
            parsed = parse_log(p)
            if parsed:
                rows.append(parsed)

    if not rows:
        print("未在任何目录中找到 *_test.log 可解析指标", file=sys.stderr)
        sys.exit(1)

    fields = ["run_tag", "mode", "win_size", "k", "threshold", "expected_f1",
              "accuracy", "precision", "recall", "f1", "log_dir", "log_file"]
    out = pathlib.Path(args.output) if args.output else (base / "all_summary.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] 汇总 {len(rows)} 条结果 ({len(dirs)} 个目录) -> {out}")
    top = sorted(rows, key=lambda x: float(x["f1"]), reverse=True)[:args.top]
    print(f"\nTop-{args.top} by F1:")
    for r in top:
        print(f"  {r['run_tag']:<28} F1={r['f1']} P={r['precision']} R={r['recall']} Acc={r['accuracy']}")

    # 按数据集分组统计
    from collections import defaultdict
    by_dataset = defaultdict(list)
    for r in rows:
        ds = r["run_tag"].split("_")[0] if "_" in r["run_tag"] else "?"
        by_dataset[ds].append(r)

    print("\n按数据集分组 Top-3:")
    for ds in sorted(by_dataset.keys()):
        print(f"\n  [{ds}] ({len(by_dataset[ds])} runs)")
        top3 = sorted(by_dataset[ds], key=lambda x: float(x["f1"]), reverse=True)[:3]
        for r in top3:
            print(f"    {r['run_tag']:<28} F1={r['f1']} P={r['precision']} R={r['recall']} Acc={r['accuracy']}")


if __name__ == "__main__":
    main()