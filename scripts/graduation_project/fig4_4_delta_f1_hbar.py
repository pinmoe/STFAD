"""
图4.4 - 各实验配置相对 E1 的 ΔF1 水平柱状图
Y轴：实验设置（从上到下 E1→E6）；X轴：ΔF1（相对 E1）
保存路径: figures/毕业论文用图/第4章/Figure4-4.png
运行方式: python scripts/fig4_4_delta_f1_hbar.py
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.linewidth'] = 0.8

OUT_DIR = "figures/毕业论文用图/第4章"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 实验数据（ΔF1 vs E1） ─────────────────────────────────────────────────────
# 顺序: E1, E2, E3, E4, B1, B2, B3, E5, E6（Y 轴从上到下）
LABELS = ['E1', 'E2', 'E3', 'E4', 'B1', 'B2', 'B3', 'E5', 'E6']

data = {
    'MSL': [
        ('E1', +0.0000),
        ('E2', -0.0340),
        ('E3', -0.0580),
        ('E4', -0.0430),
        ('B1', -0.1120),
        ('B2', -0.3590),
        ('B3', -0.0650),
        ('E5', -0.0190),
        ('E6', -0.0498),
    ],
    'SMAP': [
        ('E1', +0.0000),
        ('E2', -0.0210),
        ('E3', -0.0010),
        ('E4', +0.0010),
        ('B1', +0.0410),
        ('B2', +0.1107),
        ('B3', +0.0000),
        ('E5', +0.0520),
        ('E6', +0.0368),
    ],
    'SKAB': [
        ('E1', +0.0000),
        ('E2', +0.0180),
        ('E3', +0.0220),
        ('E4', -0.0150),
        ('B1', +0.0120),
        ('B2', +0.0100),
        ('B3', +0.0100),
        ('E5', -0.0040),
        ('E6', +0.0223),
    ],
}

# ── 颜色方案 ──────────────────────────────────────────────────────────────────
COLOR_NEUTRAL = '#9A9A9A'   # E1（0）
COLOR_POS     = '#5B8F4A'   # 正值（绿）
COLOR_NEG     = '#D95A4A'   # 负值（红）
E6_IDX        = 8           # E6 在列表中的索引


def bar_color(v, idx):
    if idx == 0:        # E1
        return COLOR_NEUTRAL
    elif v > 1e-6:
        return COLOR_POS
    elif v < -1e-6:
        return COLOR_NEG
    else:
        return COLOR_NEUTRAL


# ── 绘图 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
fig.subplots_adjust(wspace=0.12, left=0.08, right=0.98, top=0.88, bottom=0.17)

DATASETS = ['MSL', 'SMAP', 'SKAB']
N = len(LABELS)
y_pos = np.arange(N)[::-1]   # E1 顶部，E6 底部

for ax, ds in zip(axes, DATASETS):
    rows   = data[ds]
    vals   = [r[1] for r in rows]
    colors = [bar_color(v, i) for i, v in enumerate(vals)]

    # ── 水平条形图 ────────────────────────────────────────────────────────
    for i, (yi, vi, ci) in enumerate(zip(y_pos, vals, colors)):
        bar_height = 0.55
        # E6 加黑边框
        ec = 'black' if i == E6_IDX else '#888888'
        lw = 1.4     if i == E6_IDX else 0.5
        ax.barh(yi, vi, height=bar_height,
                color=ci, alpha=0.94,
                edgecolor=ec, linewidth=lw, zorder=3)

        # 数值标注
        fmt = f'{vi:+.4f}'
        if abs(vi) < 1e-6:
            fmt = '+0.0000'
        ha  = 'left'  if vi >= 0 else 'right'
        dx  = 0.0015  if vi >= 0 else -0.0015
        fw  = 'bold'  if i == E6_IDX else 'normal'
        # 对于很小的负值，标注放在 0 右侧
        x_label = vi + dx
        ax.text(x_label, yi, fmt,
                ha=ha, va='center',
                fontsize=7.5, fontweight=fw,
                color='#111111',
                )

    # 零基准线
    ax.axvline(0, color='black', linewidth=0.9, zorder=4)

    # E1 基准标注
    ax.text(0.002, y_pos[0] + 0.38, 'E1 baseline',
            fontsize=7.5, color='#333333',
            va='bottom', )

    # Y 轴刻度（仅第一个子图）
    ax.set_yticks(y_pos)
    if ds == 'MSL':
        ax.set_yticklabels(LABELS, fontsize=9.5, )
        ax.set_ylabel('实验设置', fontsize=11, )
    else:
        ax.set_yticklabels([])

    ax.set_ylim(-0.6, N - 0.4)

    # X 轴范围留白
    v_arr = np.array(vals)
    x_lo  = min(v_arr.min() * 1.30, -0.01)
    x_hi  = max(v_arr.max() * 1.40,  0.005) if v_arr.max() > 0 else 0.02
    ax.set_xlim(x_lo, x_hi)

    ax.set_xlabel('ΔF1（相对 E1）', fontsize=11, )
    ax.tick_params(labelsize=9, direction='in', length=3)

    # 网格
    ax.set_facecolor('white')
    ax.grid(axis='x', linestyle='--', linewidth=0.35, alpha=0.35,
            color='gray', zorder=0)

    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 标题
    ax.set_title(ds, fontsize=13, fontweight='bold', pad=6)

    # E6 行高亮
    ax.axhspan(y_pos[E6_IDX] - 0.45, y_pos[E6_IDX] + 0.45,
               alpha=0.06, color='#333333', zorder=0)

# ── 图例 ─────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=COLOR_NEG, edgecolor='#888888', lw=0.5,
                   label='Negative ΔF1'),
    mpatches.Patch(facecolor=COLOR_POS, edgecolor='#888888', lw=0.5,
                   label='Positive ΔF1'),
    mpatches.Patch(facecolor=COLOR_NEUTRAL, edgecolor='#888888', lw=0.5,
                   label='E1 Baseline (Δ=0)'),
]
fig.legend(handles=legend_elements,
           loc='upper center',
           ncol=3,
           fontsize=9,
           frameon=True,
           framealpha=0.9,
           edgecolor='#AAAAAA',
           bbox_to_anchor=(0.5, 1.01),
           prop={'size': 9})

# ── 保存 ─────────────────────────────────────────────────────────────────────
out = f"{OUT_DIR}/Figure4-4.png"
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved: {out}")
plt.close(fig)
