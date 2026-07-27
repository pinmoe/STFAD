import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, auc
from sklearn.svm import OneClassSVM

from utils.eval_metrics import pointwise_metrics, resolve_eval_unit, window_level_metrics_from_scores


def load_dataset(data_root, dataset):
    root = Path(data_root)
    train = np.load(root / f"{dataset}_train.npy").astype(np.float32)
    val = np.load(root / f"{dataset}_val.npy").astype(np.float32) if (root / f"{dataset}_val.npy").exists() else train[-max(1, len(train)//5):]
    test = np.load(root / f"{dataset}_test.npy").astype(np.float32)
    if (root / f"{dataset}_test_window_label.npy").exists():
        labels = np.load(root / f"{dataset}_test_window_label.npy").astype(int)
        point_labels = np.repeat(labels[:, None], test.shape[1], axis=1).reshape(-1)
    else:
        point_labels = np.load(root / f"{dataset}_test_label.npy").astype(int).reshape(-1)
    return train, val, test, point_labels


def auprc(labels, scores):
    p, r, _ = precision_recall_curve(labels, scores)
    return float(auc(r, p))


class TinyAE(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        hidden = min(hidden, max(4, dim // 2))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


def run_autoencoder(train, val, test, epochs, seed):
    torch.manual_seed(seed)
    dim = train.shape[-1]
    model = TinyAE(dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_t = torch.tensor(train.reshape(-1, dim), dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(train_t) - train_t) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        val_t = torch.tensor(val.reshape(-1, dim), dtype=torch.float32)
        test_t = torch.tensor(test.reshape(-1, dim), dtype=torch.float32)
        val_scores = ((model(val_t) - val_t) ** 2).mean(dim=1).numpy()
        test_scores = ((model(test_t) - test_t) ** 2).mean(dim=1).numpy()
    return val_scores, test_scores


def run_sklearn(method, train, val, test, seed):
    train_x = train.reshape(-1, train.shape[-1])
    val_x = val.reshape(-1, val.shape[-1])
    test_x = test.reshape(-1, test.shape[-1])
    if method == "iforest":
        model = IsolationForest(random_state=seed, contamination="auto")
    elif method == "ocsvm":
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    else:
        raise ValueError(method)
    model.fit(train_x)
    return -model.score_samples(val_x), -model.score_samples(test_x)


def main():
    parser = argparse.ArgumentParser(description="Run simple baselines with the same split/eval protocol.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", default="ST330IR001_CP001")
    parser.add_argument("--methods", nargs="+", default=["iforest", "ocsvm", "autoencoder"],
                        choices=["iforest", "ocsvm", "autoencoder", "tranad"])
    parser.add_argument("--result_root", default="results/paper_supplement")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    parser.add_argument("--eval_unit", default="auto", choices=["auto", "point", "window"])
    parser.add_argument("--win_size", type=int, default=56)
    parser.add_argument("--ae_epochs", type=int, default=5)
    args = parser.parse_args()

    train, val, test, labels = load_dataset(args.data_root, args.dataset)
    eval_unit = resolve_eval_unit(args.dataset, args.eval_unit)
    for method in args.methods:
        if method == "tranad":
            raise NotImplementedError("TranAD baseline requires an audited external implementation; not faking results.")
        if method in ("iforest", "ocsvm"):
            val_scores, test_scores = run_sklearn(method, train, val, test, args.seed)
        else:
            val_scores, test_scores = run_autoencoder(train, val, test, args.ae_epochs, args.seed)

        out_dir = Path(args.result_root) / args.dataset / method / f"seed_{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if eval_unit == "window":
            wm = window_level_metrics_from_scores(
                test_scores, labels, val_scores, args.win_size, percentile=args.threshold_percentile
            )
            metrics = {"dataset": args.dataset, "experiment_name": method, "seed": args.seed,
                       "eval_unit": eval_unit, "window_level": {k: v for k, v in wm.items() if not k.startswith("_")}}
        else:
            threshold = float(np.percentile(val_scores, args.threshold_percentile))
            pred = (test_scores > threshold).astype(int)
            point = pointwise_metrics(labels, pred)
            metrics = {"dataset": args.dataset, "experiment_name": method, "seed": args.seed,
                       "eval_unit": eval_unit, "threshold": threshold,
                       "threshold_details": {"source": "normal_validation_scores", "uses_test_scores": False,
                                             "uses_test_labels": False, "uses_test_anomaly_ratio": False},
                       "pointwise": point, "ranking": {"auprc": auprc(labels, test_scores)}}
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.save(out_dir / "scores.npy", test_scores)
        print(f"[baseline] wrote {out_dir}")


if __name__ == "__main__":
    main()
