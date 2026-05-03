#!/usr/bin/env python3
import argparse
import pathlib
import re

DONE_RE = re.compile(r"Accuracy\s*:\s*[0-9.]+,\s*Precision\s*:\s*[0-9.]+,\s*Recall\s*:\s*[0-9.]+,\s*F-score\s*:\s*[0-9.]+")


def latest_root(base: pathlib.Path) -> pathlib.Path:
    cands = sorted(base.glob("thesis_pack_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError("未找到 logs/thesis_pack_* 目录")
    return cands[0]


def expected_runs(dataset: str, include_sweep: bool):
    runs = [
        f"{dataset}_E1_PA",
        f"{dataset}_E2_PA",
    ]
    if include_sweep:
        runs += [f"{dataset}_E1_win{w}_k3" for w in (50, 100, 150, 200)]
        runs += [f"{dataset}_E1_win100_k{k}" for k in (1, 3, 5, 10)]
    # 去重并保持顺序（win100_k3 会在两个 sweep 中重复出现）
    return list(dict.fromkeys(runs))


def has_final_metric(test_log: pathlib.Path) -> bool:
    if not test_log.exists():
        return False
    txt = test_log.read_text(encoding="utf-8", errors="ignore")
    return DONE_RE.search(txt) is not None


def main():
    ap = argparse.ArgumentParser(description="检查 thesis_pack 实验进度，给出未完成 run 列表")
    ap.add_argument("log_root", nargs="?", default="latest", help="日志目录；默认 latest 自动选最新 logs/thesis_pack_*")
    ap.add_argument("--dataset", default="HAI")
    ap.add_argument("--no-sweep", action="store_true", help="只检查 E1/E2")
    args = ap.parse_args()

    root = latest_root(pathlib.Path("logs")) if args.log_root == "latest" else pathlib.Path(args.log_root)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")

    runs = expected_runs(args.dataset, include_sweep=not args.no_sweep)

    done, partial, missing = [], [], []
    for tag in runs:
        train_log = root / f"{tag}_train.log"
        test_log = root / f"{tag}_test.log"
        ckpt = root / "checkpoints" / tag / f"{args.dataset}_checkpoint.pth"

        if has_final_metric(test_log):
            done.append(tag)
        elif train_log.exists() or test_log.exists() or ckpt.exists():
            partial.append(tag)
        else:
            missing.append(tag)

    print(f"[LOG_ROOT] {root}")
    print(f"[DONE] {len(done)}")
    for t in done:
        print(f"  ✅ {t}")
    print(f"[PARTIAL] {len(partial)}")
    for t in partial:
        print(f"  ⚠️  {t}")
    print(f"[MISSING] {len(missing)}")
    for t in missing:
        print(f"  ❌ {t}")

    if missing or partial:
        print("\n[建议] 继续跑未完成项：")
        for t in partial + missing:
            print(f"  {t}")


if __name__ == "__main__":
    main()