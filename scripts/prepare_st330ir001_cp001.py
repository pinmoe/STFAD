"""
Prepare ST330IR001.CP001 welding-gun fault data for data_loader.py.

Input directory:
  data/ST330IR001.CP001/
    normal/*.csv
    fault_21051/*.csv

Each raw CSV is already one fixed-length window with 56 rows. This script keeps
that sample boundary and writes 3D arrays:
  ST330IR001_CP001_train.npy       (N_train, 56, 29)
  ST330IR001_CP001_test.npy        (N_test, 56, 29)
  ST330IR001_CP001_test_label.npy  (N_test, 56, 1)
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


DATASET_PREFIX = "ST330IR001_CP001"
TIME_COLUMN = "datetime"


def _read_window_csv(path, expected_columns=None, expected_rows=None, nan_fill=0.0):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if expected_columns is not None and header != expected_columns:
        raise ValueError(f"{path} has inconsistent columns.")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{path} has {len(rows)} rows, expected {expected_rows}.")
    if TIME_COLUMN not in header:
        raise ValueError(f"{path} is missing required time column: {TIME_COLUMN}")

    time_idx = header.index(TIME_COLUMN)
    feature_names = [col for i, col in enumerate(header) if i != time_idx]
    values = np.empty((len(rows), len(feature_names)), dtype=np.float32)

    for row_idx, row in enumerate(rows):
        if len(row) != len(header):
            raise ValueError(f"{path} row {row_idx + 2} has {len(row)} columns, expected {len(header)}.")
        out_col = 0
        for col_idx, col_name in enumerate(header):
            if col_idx == time_idx:
                continue
            raw = row[col_idx].strip()
            if raw == "":
                value = nan_fill
            else:
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(f"{path} column {col_name} row {row_idx + 2} is not numeric: {raw}") from exc
                if not math.isfinite(value):
                    value = nan_fill
            values[row_idx, out_col] = value
            out_col += 1

    return header, feature_names, values


def _load_folder(folder, label, nan_fill):
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    windows = []
    expected_columns = None
    expected_rows = None
    feature_names = None
    for path in files:
        header, current_feature_names, values = _read_window_csv(
            path,
            expected_columns=expected_columns,
            expected_rows=expected_rows,
            nan_fill=nan_fill,
        )
        if expected_columns is None:
            expected_columns = header
            expected_rows = values.shape[0]
            feature_names = current_feature_names
        windows.append(values)

    data = np.stack(windows, axis=0).astype(np.float32)
    labels = np.full((data.shape[0], data.shape[1], 1), label, dtype=np.float32)
    return data, labels, files, feature_names


def main():
    parser = argparse.ArgumentParser(description="Prepare ST330IR001.CP001 dataset.")
    parser.add_argument("--src_dir", type=str, default="data/ST330IR001.CP001")
    parser.add_argument("--dst_dir", type=str, default="data/ST330IR001.CP001")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of normal windows used for training. The rest are normal test windows.",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--nan_fill", type=float, default=0.0)
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1.")

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    normal_dir = src_dir / "normal"
    fault_dir = src_dir / "fault_21051"
    dst_dir.mkdir(parents=True, exist_ok=True)

    normal_data, normal_labels, normal_files, feature_names = _load_folder(normal_dir, 0.0, args.nan_fill)
    fault_data, fault_labels, fault_files, fault_feature_names = _load_folder(fault_dir, 1.0, args.nan_fill)
    if feature_names != fault_feature_names:
        raise ValueError("Normal and fault CSV files have different feature columns.")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(normal_data.shape[0])
    rng.shuffle(indices)
    split = int(round(normal_data.shape[0] * args.train_ratio))
    train_idx = np.sort(indices[:split])
    test_normal_idx = np.sort(indices[split:])

    train = normal_data[train_idx]
    test = np.concatenate([normal_data[test_normal_idx], fault_data], axis=0).astype(np.float32)
    test_label = np.concatenate([normal_labels[test_normal_idx], fault_labels], axis=0).astype(np.float32)

    np.save(dst_dir / f"{DATASET_PREFIX}_train.npy", train)
    np.save(dst_dir / f"{DATASET_PREFIX}_test.npy", test)
    np.save(dst_dir / f"{DATASET_PREFIX}_test_label.npy", test_label)

    metadata = {
        "dataset": DATASET_PREFIX,
        "source_dir": str(src_dir),
        "window_size": int(train.shape[1]),
        "num_features": int(train.shape[2]),
        "feature_names": feature_names,
        "normal_files": len(normal_files),
        "fault_files": len(fault_files),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "train_windows": int(train.shape[0]),
        "test_windows": int(test.shape[0]),
        "test_normal_windows": int(len(test_normal_idx)),
        "test_fault_windows": int(fault_data.shape[0]),
        "test_anomaly_ratio": float(test_label.mean()),
    }
    with (dst_dir / f"{DATASET_PREFIX}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Saved ST330IR001.CP001 arrays:")
    print(f"  train:      {train.shape} -> {dst_dir / (DATASET_PREFIX + '_train.npy')}")
    print(f"  test:       {test.shape} -> {dst_dir / (DATASET_PREFIX + '_test.npy')}")
    print(f"  test_label: {test_label.shape} -> {dst_dir / (DATASET_PREFIX + '_test_label.npy')}")
    print(f"  anomaly ratio: {test_label.mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
