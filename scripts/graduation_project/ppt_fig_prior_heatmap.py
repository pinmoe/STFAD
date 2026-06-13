"""
答辩PPT 第7页配图 — 正常/异常关联矩阵热力图对比
保存路径:
  figures/毕业论文用图/答辩PPT/prior_heatmap_normal.png
  figures/毕业论文用图/答辩PPT/prior_heatmap_abnormal.png
  figures/毕业论文用图/答辩PPT/prior_heatmap_compare.png  (合图，直接插PPT)
运行方式: cd C:\\VSCode\\Anomaly-Transformer && python scripts/graduation_project/ppt_fig_prior_heatmap.py

图片说明：
  正常关联矩阵 = E1 高斯先验（解析计算）：代表模型对"正常时序"的关联期望，
                 对角线平滑衰减，时间局部性清晰。
  异常关联矩阵 = E1 Series 注意力（已保存，来自 SKAB 异常窗口674）：
                 代表模型在异常时刻的实际注意力分布，与先验结构存在偏差。
  两图的差异正是 KL 散度可检测异常的直观证据。
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family":        "Microsoft YaHei",
    "axes.unicode_minus": False,
    "axes.linewidth":     0.8,
})

OUT_DIR      = "figures/毕业论文用图/答辩PPT"
ATTN_PATH    = "pics/results/SKAB_E1_attn_E1E2.npy"   # shape (8, 100, 100)
WIN_SIZE     = 100
VIS_HEAD     = 5   # Head-5：峰值最分散，结构最易读（0.907 vs 其他 head 的 1.000）
os.makedirs(OUT_DIR, exist_ok=True)

# ── 大连理工蓝系渐变色板 ─────────────────────────────────────────────────────
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "dlut_blue",
    ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
)

# ── 1. 正常关联矩阵：E1 高斯先验（解析计算）───────────────────────────────────
# 对应 AnomalyAttention.forward() 中的 prior 计算逻辑
# sigma 取训练收敛后的典型值（sigmoid(1.0*5)+1e-5 ≈ 0.993, 3^0.993-1 ≈ 1.98）
def build_gaussian_prior(win_size: int, sigma: float = 2.0) -> np.ndarray:
    """构造高斯先验矩阵，每行归一化，与 AnomalyAttention 中 prior 一致。"""
    dist = np.abs(
        np.arange(win_size)[:, None] - np.arange(win_size)[None, :]
    ).astype(np.float32)
    mat = (1.0 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(
        -dist ** 2 / (2 * sigma ** 2)
    )
    row_sum = mat.sum(axis=1, keepdims=True)
    return mat / (row_sum + 1e-12)

normal_mat = build_gaussian_prior(WIN_SIZE, sigma=2.0)

# ── 2. 异常关联矩阵：E1 Series 注意力（来自异常窗口 674）──────────────────────
assert os.path.exists(ATTN_PATH), (
    f"找不到 {ATTN_PATH}，请先运行:\n"
    f"  PYTHONPATH=$(pwd) python pics/run_inference.py"
)
attn_all    = np.load(ATTN_PATH)          # (8, 100, 100)
abnormal_mat = attn_all[VIS_HEAD]         # 取单个注意力头

# ── 3. 绘制合图（左右并排，直接插PPT）─────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2, figsize=(12, 4.8),
    facecolor="white",
    gridspec_kw={"wspace": 0.38}
)

def draw_heatmap(ax, mat, title, scale_label=None):
    """两图各自独立色阶，保证内部结构清晰可读。"""
    im = ax.imshow(mat, cmap=CMAP_BLUE, aspect="auto",
                   vmin=0, vmax=mat.max(), interpolation="nearest")
    ax.set_title(title, fontsize=12.5, fontweight="bold",
                 color="#0d3b66", pad=9)
    ax.set_xlabel("键时间步 $t'$", fontsize=9.5)
    ax.set_ylabel("查询时间步 $t$", fontsize=9.5)
    ticks = [0, 20, 40, 60, 80, WIN_SIZE - 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7.5)
    lbl = scale_label or f"注意力权重（max={mat.max():.3f}）"
    cbar.set_label(lbl, fontsize=7.5, rotation=270, labelpad=14)
    return im

draw_heatmap(
    axes[0], normal_mat,
    "正常关联矩阵（高斯先验 $A_{prior}$）",
    scale_label="注意力权重（对角线平滑衰减）",
)
draw_heatmap(
    axes[1], abnormal_mat,
    "异常关联矩阵（Series 注意力 $A_{series}$，Head-5）",
    scale_label="注意力权重（独立色阶）",
)

# 在正常图中标注对角线说明
axes[0].plot([0, WIN_SIZE - 1], [0, WIN_SIZE - 1],
             color="#d62728", linewidth=1.0, linestyle="--", alpha=0.6,
             label="对角线（自相关中心）")
axes[0].legend(loc="lower right", fontsize=7.5,
               frameon=True, framealpha=0.88, edgecolor="#cccccc")


# ── 保存合图 ─────────────────────────────────────────────────────────────────
out_compare = f"{OUT_DIR}/prior_heatmap_compare.png"
fig.savefig(out_compare, dpi=300, bbox_inches="tight", facecolor="white")
print(f"已保存合图: {out_compare}")

# ── 4. 单独保存两张图（插PPT时可分别放左右两侧）────────────────────────────────
for ax, fname in [
    (axes[0], f"{OUT_DIR}/prior_heatmap_normal.png"),
    (axes[1], f"{OUT_DIR}/prior_heatmap_abnormal.png"),
]:
    extent = ax.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    # 向外扩展以包含标题和colorbar
    fig.savefig(fname, dpi=300,
                bbox_inches=extent.expanded(1.22, 1.30),
                facecolor="white")
    print(f"已保存单图: {fname}")

plt.close(fig)
print("=== 热力图生成完毕 ===")
