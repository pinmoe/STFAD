"""
答辩PPT 第4页配图 — SKAB水泵系统多通道时序图
保存路径: figures/毕业论文用图/答辩PPT/pump_timeseries.png
运行方式: cd C:\\VSCode\\Anomaly-Transformer && python scripts/graduation_project/ppt_fig_pump_timeseries.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":        "Microsoft YaHei",
    "axes.unicode_minus": False,
    "axes.linewidth":     1.0,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
})

OUT_DIR   = "figures/毕业论文用图/答辩PPT"
DATA_FILE = "data/SKAB/valve1/0.csv"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 通道配置 ──────────────────────────────────────────────────────────────────
CHANNELS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]
CH_LABELS = [
    "加速度1 (RMS)",
    "加速度2 (RMS)",
    "电流",
    "压力",
    "温度",
    "热电偶",
    "电压",
    "体积流量 (RMS)",
]
# 通道颜色：蓝系4 + 绿/橙/红/紫各1，答辩场合配色稳重
CH_COLORS = [
    "#1f6eb5", "#4e9ad4",
    "#2a9d5c", "#e07b39",
    "#c0392b", "#8e44ad",
    "#546e7a", "#00796b",
]

# ── 读取数据 ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE, sep=";", parse_dates=["datetime"])

anom_all = df[df["anomaly"] == 1.0].index.tolist()
assert len(anom_all) > 0, f"{DATA_FILE} 中未找到异常标注"

anom_start = anom_all[0]
anom_end   = anom_all[-1]

# 截取窗口：异常前80步 + 异常全程 + 异常后40步
WIN_PRE  = 80
WIN_POST = 40
seg_start = max(0, anom_start - WIN_PRE)
seg_end   = min(len(df) - 1, anom_end + WIN_POST)
df_win    = df.iloc[seg_start:seg_end + 1].reset_index(drop=True)

# 异常在窗口内的局部坐标
local_anom_start = anom_start - seg_start
local_anom_end   = anom_end   - seg_start
x_axis = np.arange(len(df_win))

# ── 绘图 ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    8, 1, figsize=(13, 9.5), sharex=True,
    gridspec_kw={"hspace": 0.08}
)
fig.patch.set_facecolor("white")

for i, (ch, label, color) in enumerate(zip(CHANNELS, CH_LABELS, CH_COLORS)):
    ax   = axes[i]
    vals = df_win[ch].values.astype(float)

    # ── 单变量阈值（正常段均值 ± 3σ）─────────────────────────────────────
    normal_vals = vals[:local_anom_start]
    if normal_vals.std() < 1e-8:
        mu, sigma = normal_vals.mean(), 1.0
    else:
        mu, sigma = normal_vals.mean(), normal_vals.std()
    upper = mu + 3.0 * sigma
    lower = mu - 3.0 * sigma

    # ── 异常区背景 ────────────────────────────────────────────────────────
    ax.axvspan(local_anom_start, local_anom_end,
               alpha=0.13, color="#d62728", linewidth=0, zorder=1)

    # ── 阈值线 ────────────────────────────────────────────────────────────
    ax.axhline(upper, color="#888888", linewidth=0.65, linestyle="--",
               alpha=0.75, zorder=2)
    ax.axhline(lower, color="#888888", linewidth=0.65, linestyle="--",
               alpha=0.75, zorder=2)

    # ── 时序线 ────────────────────────────────────────────────────────────
    ax.plot(x_axis, vals, color=color, linewidth=1.1, zorder=3)

    # ── Y轴标签 ───────────────────────────────────────────────────────────
    ax.set_ylabel(label, fontsize=15, rotation=0, ha="right", va="center")
    ax.yaxis.set_label_coords(-0.09, 0.5)
    ax.tick_params(labelsize=13)
    ax.set_facecolor("#fafafa")

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#dddddd")
    ax.spines["bottom"].set_linewidth(0.5)


# ── 最后一行X轴 ───────────────────────────────────────────────────────────────
axes[-1].set_xlabel("时间步（相对窗口起点）", fontsize=16)
axes[-1].set_xlim(0, len(df_win) - 1)
axes[-1].tick_params(labelsize=13)


# ── 保存 ──────────────────────────────────────────────────────────────────────
out = f"{OUT_DIR}/pump_timeseries.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"已保存: {out}")
plt.close(fig)
