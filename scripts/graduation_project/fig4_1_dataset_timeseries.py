"""
图4.1 - 实验数据集统计特征及示例片段
保存路径: figures/毕业论文用图/第4章/Figure4-1.png
运行方式: python scripts/fig4_1_dataset_timeseries.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family":        "Microsoft YaHei",
    "axes.unicode_minus": False,
    "axes.linewidth":     0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
})

OUT_DIR = "figures/毕业论文用图/第4章"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    {"name": "MSL",  "train": "data/MSL/MSL_train.npy",
     "test": "data/MSL/MSL_test.npy",
     "label": "data/MSL/MSL_test_label.npy",  "color": "#4C78A8"},
    {"name": "SMAP", "train": "data/SMAP/SMAP_train.npy",
     "test": "data/SMAP/SMAP_test.npy",
     "label": "data/SMAP/SMAP_test_label.npy", "color": "#F58518"},
    {"name": "SKAB", "train": "data/SKAB/SKAB_train.npy",
     "test": "data/SKAB/SKAB_test.npy",
     "label": "data/SKAB/SKAB_test_label.npy", "color": "#2CA02C"},
]

WIN_LEN  = 7200
ANOM_CLR = "#E74C3C"
THRESH   = 1.0          # anomaly threshold shown as dashed line


def anomaly_spans(labels):
    spans, in_seg, start = [], False, 0
    for i, v in enumerate(labels):
        if v == 1 and not in_seg:
            in_seg, start = True, i
        elif v == 0 and in_seg:
            in_seg = False
            spans.append((start, i))
    if in_seg:
        spans.append((start, len(labels)))
    return spans


def select_window_and_channel(test, label, n=WIN_LEN):
    """
    Pick (window_start, channel) that maximises visual quality.
    Scores channels on z-score normalised signal in the normal region.
    """
    spans = anomaly_spans(label)
    if not spans:
        return 0, 0

    kern = np.ones(60) / 60
    best_score, best_start, best_ch = -1.0, 0, 0

    for sp_s, sp_e in spans:
        if sp_e - sp_s < 80:
            continue
        mid = (sp_s + sp_e) // 2
        ws  = max(0, mid - n // 2)
        we  = min(len(label), ws + n)
        ws  = max(0, we - n)
        lbl_seg  = label[ws: ws + n]
        anom_cnt = int(lbl_seg.sum())
        if anom_cnt < 300:
            continue
        data_seg  = test[ws: ws + n]
        norm_idx  = np.where(lbl_seg == 0)[0]
        anom_idx  = np.where(lbl_seg == 1)[0]
        if len(norm_idx) < 30:
            continue

        ch_scores = []
        for ch in range(data_seg.shape[1]):
            x  = data_seg[:, ch].astype(float)
            xn = x[norm_idx]
            std = xn.std()
            if std < 1e-6:
                ch_scores.append(0.0)
                continue
            # z-score the normal region, then measure low-freq variance
            zn = (xn - xn.mean()) / std
            lf = (np.convolve(zn, kern, mode="valid").std()
                  if len(zn) > 60 else 0.3)
            # anomaly z-score deviation (how far does anomaly go above threshold)
            za = (x[anom_idx] - xn.mean()) / std if len(anom_idx) >= 5 else np.array([0])
            contrast = max(0.0, za.mean() - THRESH)
            ch_scores.append(lf * (1.0 + 0.4 * min(contrast, 3.0)))

        best_ch_now = int(np.argmax(ch_scores))
        ch_quality  = ch_scores[best_ch_now]
        anom_frac   = anom_cnt / n
        balance     = 1.0 - abs(anom_frac - 0.25) / 0.25
        if ch_quality * max(0.2, balance) > best_score:
            best_score = ch_quality * max(0.2, balance)
            best_start, best_ch = ws, best_ch_now

    return best_start, best_ch


# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=False)
fig.subplots_adjust(hspace=0.45, top=0.96, bottom=0.10,
                    left=0.09, right=0.80)

for ax, cfg in zip(axes, DATASETS):
    train = np.load(cfg["train"])
    test  = np.load(cfg["test"])
    label = np.load(cfg["label"]).astype(int).ravel()

    start, ch = select_window_and_channel(test, label)
    seg       = slice(start, start + WIN_LEN)
    t         = np.arange(WIN_LEN)
    lbl_seg   = label[seg]

    raw  = test[seg, ch].astype(float)
    mu, sigma = raw.mean(), raw.std() + 1e-8
    sig  = (raw - mu) / sigma          # z-score normalisation

    # ── anomaly shading ──────────────────────────────────────────────────
    first = True
    for s, e in anomaly_spans(lbl_seg):
        ax.axvspan(s, e, alpha=0.18, color=ANOM_CLR, linewidth=0,
                   label="异常片段" if first else "")
        first = False

    # ── signal ───────────────────────────────────────────────────────────
    ax.plot(t, sig, color=cfg["color"], linewidth=0.75, alpha=0.9,
            label="原始序列", zorder=3, rasterized=True)

    # ── threshold line ───────────────────────────────────────────────────
    ax.axhline(THRESH, color="#777777", linestyle="--", linewidth=0.9,
               label="阈值", zorder=2)

    # ── axes style ───────────────────────────────────────────────────────
    y_lo = max(-2.8, sig.min() * 1.15)
    y_hi = max(sig.max() * 1.15, THRESH * 1.6)
    y_lo = min(y_lo, -1.5)
    y_hi = max(y_hi,  2.0)
    ax.set_xlim(0, WIN_LEN)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("归一化值", fontsize=10.5)
    ax.set_xlabel("时间步（测试集片段）", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.35, alpha=0.5, color="#AAAAAA")
    ax.set_title(f"{cfg['name']}  数据集",
                 fontsize=12, fontweight="bold", pad=5, loc="left")

    # ── legend (inside, upper left) ──────────────────────────────────────
    ax.legend(loc="upper left", fontsize=8.5,
              frameon=True, framealpha=0.88,
              edgecolor="#CCCCCC", handlelength=1.4,
              prop={"size": 8.5})

    # ── stats box (upper right, outside axes) ────────────────────────────
    n_train   = train.shape[0]
    n_test    = test.shape[0]
    n_ch      = test.shape[1]
    anom_rate = label.mean() * 100
    stats = (f"训练集: {n_train/1e4:.1f} 万步\n"
             f"测试集: {n_test/1e4:.1f} 万步\n"
             f"通道数: {n_ch}\n"
             f"异常率: {anom_rate:.2f}%")
    ax.text(1.025, 0.97, stats,
            transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            linespacing=1.8,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.97, linewidth=0.8))

# ── save ─────────────────────────────────────────────────────────────────────
out = f"{OUT_DIR}/Figure4-1.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)
