import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PREFIX = "ST330IR001_CP001"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_window(path: Path, win_size: int, n_features: int) -> np.ndarray:
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < n_features:
        raise ValueError(f"{path} has {numeric.shape[1]} numeric columns, expected at least {n_features}")
    arr = numeric.iloc[:, :n_features].to_numpy(dtype=np.float32)
    if arr.shape != (win_size, n_features):
        raise ValueError(f"{path} has shape {arr.shape}, expected {(win_size, n_features)}")
    return arr


def load_windows(paths, win_size: int, n_features: int) -> np.ndarray:
    if not paths:
        return np.empty((0, win_size, n_features), dtype=np.float32)
    return np.stack([read_window(path, win_size, n_features) for path in paths]).astype(np.float32)


def split_normal(paths, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    paths = [paths[i] for i in order]
    n = len(paths)
    n_train = int(n * 0.64)
    n_val = int(n * 0.16)
    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test_normal = paths[n_train + n_val:]
    if not train or not val or not test_normal:
        raise ValueError("Normal split is empty; need enough normal CSV windows for 64/16/20 split.")
    return train, val, test_normal, order.tolist()


def main():
    parser = argparse.ArgumentParser(description="Prepare ST330IR001_CP001 fixed-window arrays without leakage.")
    parser.add_argument("--raw_root", default="data/ST330IR001.CP001")
    parser.add_argument("--output_dir", default="data/ST330IR001.CP001")
    parser.add_argument("--normal_dir", default="normal")
    parser.add_argument("--fault_glob", default="fault_*")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--win_size", type=int, default=56)
    parser.add_argument("--n_features", type=int, default=29)
    parser.add_argument("--shuffle_test", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    normal_paths = sorted((raw_root / args.normal_dir).glob("*.csv"))
    fault_paths = []
    for fault_dir in sorted(raw_root.glob(args.fault_glob)):
        if fault_dir.is_dir():
            fault_paths.extend(sorted(fault_dir.glob("*.csv")))
    if not normal_paths:
        raise FileNotFoundError(f"No normal CSV files under {raw_root / args.normal_dir}")
    if not fault_paths:
        raise FileNotFoundError(f"No fault CSV files matching {raw_root / args.fault_glob}")

    train_paths, val_paths, test_normal_paths, normal_order = split_normal(normal_paths, args.seed)
    test_paths = list(test_normal_paths) + list(fault_paths)
    test_labels = np.array([0] * len(test_normal_paths) + [1] * len(fault_paths), dtype=np.int64)
    test_order = list(range(len(test_paths)))
    if args.shuffle_test:
        rng = np.random.default_rng(args.seed + 1)
        perm = rng.permutation(len(test_paths))
        test_paths = [test_paths[i] for i in perm]
        test_labels = test_labels[perm]
        test_order = perm.tolist()

    train_raw = load_windows(train_paths, args.win_size, args.n_features)
    val_raw = load_windows(val_paths, args.win_size, args.n_features)
    test_raw = load_windows(test_paths, args.win_size, args.n_features)

    scaler = StandardScaler()
    scaler.fit(train_raw.reshape(-1, args.n_features))
    train = scaler.transform(train_raw.reshape(-1, args.n_features)).reshape(train_raw.shape).astype(np.float32)
    val = scaler.transform(val_raw.reshape(-1, args.n_features)).reshape(val_raw.shape).astype(np.float32)
    test = scaler.transform(test_raw.reshape(-1, args.n_features)).reshape(test_raw.shape).astype(np.float32)
    compat_labels = np.repeat(test_labels[:, None, None], args.win_size, axis=1).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{PREFIX}_train.npy", train)
    np.save(output_dir / f"{PREFIX}_val.npy", val)
    np.save(output_dir / f"{PREFIX}_test.npy", test)
    np.save(output_dir / f"{PREFIX}_test_window_label.npy", test_labels)
    np.save(output_dir / f"{PREFIX}_test_label.npy", compat_labels)

    manifest = {
        "dataset": PREFIX,
        "seed": args.seed,
        "win_size": args.win_size,
        "n_features": args.n_features,
        "split": {"normal_train": len(train_paths), "normal_val": len(val_paths),
                  "normal_test": len(test_normal_paths), "fault_test": len(fault_paths)},
        "arrays": {"train": list(train.shape), "val": list(val.shape), "test": list(test.shape),
                   "test_window_label": list(test_labels.shape), "test_label_compat": list(compat_labels.shape)},
        "scaler_fit": "normal_train_windows_only",
        "test_order": test_order,
        "normal_shuffle_order": normal_order,
        "files": {
            "train": [str(p) for p in train_paths],
            "val": [str(p) for p in val_paths],
            "test": [str(p) for p in test_paths],
        },
        "sha256": {str(p): file_sha256(p) for p in train_paths + val_paths + test_paths},
    }
    with (output_dir / f"{PREFIX}_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest["split"], indent=2))
    print(f"Wrote {output_dir / (PREFIX + '_manifest.json')}")


if __name__ == "__main__":
    main()
