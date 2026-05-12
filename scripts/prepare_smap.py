"""
SMAP 数据预处理脚本
默认从 data/SMAP_MSL/data/data/{train,test} 读取逐通道 npy，合并为代码需要的三个文件：
  data/SMAP/SMAP_train.npy
  data/SMAP/SMAP_test.npy
  data/SMAP/SMAP_test_label.npy
"""

import os
import ast
import argparse
import numpy as np
import pandas as pd


ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_SRC_DIR = os.path.join(ROOT_DIR, "data", "SMAP_MSL", "data", "data")
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, "data", "SMAP")
DEFAULT_LABEL_CSV = os.path.join(ROOT_DIR, "data", "labeled_anomalies.csv")


def build_smap_dataset(src_dir: str, out_dir: str, label_csv: str) -> None:
    train_dir = os.path.join(src_dir, "train")
    test_dir = os.path.join(src_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"未找到训练目录: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"未找到测试目录: {test_dir}")
    if not os.path.isfile(label_csv):
        raise FileNotFoundError(f"未找到标注文件: {label_csv}")

    label_df = pd.read_csv(label_csv)
    smap_df = label_df[label_df["spacecraft"] == "SMAP"].copy().set_index("chan_id")
    smap_df = smap_df[~smap_df.index.duplicated(keep="first")]

    all_train_chan_ids = {f[:-4] for f in os.listdir(train_dir) if f.endswith(".npy")}
    chan_ids = sorted([cid for cid in smap_df.index if cid in all_train_chan_ids])
    if not chan_ids:
        raise RuntimeError(f"在 {train_dir} 中未发现 npy 文件")

    print(f"找到 {len(chan_ids)} 个 SMAP 子通道")

    # 合并训练数据
    train_arrays = []
    for cid in chan_ids:
        train_path = os.path.join(train_dir, f"{cid}.npy")
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"缺失训练文件: {train_path}")
        arr = np.load(train_path)
        train_arrays.append(arr)
    train_all = np.concatenate(train_arrays, axis=0)

    # 合并测试数据，并根据 labeled_anomalies.csv 生成逐点标签
    test_arrays = []
    label_arrays = []
    for cid in chan_ids:
        test_path = os.path.join(test_dir, f"{cid}.npy")
        if not os.path.isfile(test_path):
            raise FileNotFoundError(f"缺失测试文件: {test_path}")

        arr = np.load(test_path)
        test_arrays.append(arr)

        n = arr.shape[0]
        labels = np.zeros(n, dtype=np.int32)

        if cid in smap_df.index:
            anomaly_text = str(smap_df.loc[cid, "anomaly_sequences"])
            anomaly_seqs = ast.literal_eval(anomaly_text)
            for seq in anomaly_seqs:
                start, end = int(seq[0]), int(seq[1])
                start = max(start, 0)
                end = min(end, n - 1)
                if start <= end:
                    labels[start:end + 1] = 1

        label_arrays.append(labels)

    test_all = np.concatenate(test_arrays, axis=0)
    label_all = np.concatenate(label_arrays, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "SMAP_train.npy"), train_all)
    np.save(os.path.join(out_dir, "SMAP_test.npy"), test_all)
    np.save(os.path.join(out_dir, "SMAP_test_label.npy"), label_all)

    print("\n生成完成:")
    print(f"  SMAP_train.npy: shape={train_all.shape}")
    print(f"  SMAP_test.npy: shape={test_all.shape}")
    print(
        f"  SMAP_test_label.npy: shape={label_all.shape}, 异常比例={label_all.mean() * 100:.2f}%"
    )
    print(f"\n输出目录: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取并合并 SMAP 数据集")
    parser.add_argument(
        "--src_dir",
        default=DEFAULT_SRC_DIR,
        help="源目录，需包含 train/ 和 test/ 子目录",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="输出目录，默认 data/SMAP",
    )
    parser.add_argument(
        "--label_csv",
        default=DEFAULT_LABEL_CSV,
        help="标注 CSV 路径，默认 data/labeled_anomalies.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_smap_dataset(args.src_dir, args.out_dir, args.label_csv)
