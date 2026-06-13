"""
图4.2 - 各实验配置在三个数据集上的 F1 分数点图（lollipop chart）
Y轴：实验设置；X轴：F1 分数；以 E1 为基准参考线
保存路径: figures/毕业论文用图/第4章/Figure4-2.png
运行方式: python scripts/fig4_2_f1_dot_plot.py
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.linewidth'] = 0.8

OUT_DIR = "figures/毕业论文用图/第4章"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 实验数据（F1 分数） ──────────────────────────────────────────────────────
# 顺序: E1, E2, E3, E4, B1, B2, B3, E5, E6
LABELS = ['E1', 'E2', 'E3', 'E4', 'B1', 'B2', 'B3', 'E5', 'E6']
N = len(LABELS)

data = {
    'MSL': {
        'f1':     [0.91, 0.88, 0.86, 0.87, 0.80, 0.56, 0.85, 0.90, 0.86],
        'color':  '#4C78A8',
        'annot':  {0: '0.91', 5: '0.56', 8: '0.86'},  # 标注特殊点
        'xrange': (0.40, 0.95),
    },
    'SMAP': {
        'f1':     [0.70, 0.68, 0.70, 0.70, 0.74, 0.81, 0.70, 0.76, 0.74],
        'color':  '#F58518',
        'annot':  {0: '0.70', 5: '0.81', 8: '0.74'},
        'xrange': (0.55, 0.95),
    },
    'SKAB': {
        'f1':     [0.79, 0.81, 0.82, 0.78, 0.80, 0.80, 0.80, 0.79, 0.81],
        'color':  '#54A24B',
        'annot':  {0: '0.79', 2: '0.82', 8: '0.82'},
        'xrange': (0.55, 0.95),
    },
}

DATASETS = ['MSL', 'SMAP', 'SKAB']
E6_IDX   = 8   # E6 在 LABELS 中的索引

# ── 绘图 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
fig.subplots_adjust(wspace=0.12, left=0.07, right=0.98, top=0.90, bottom=0.12)

y_pos = np.arange(N)[::-1]   # E1 在顶部，E6 在底部

for ax, ds in zip(axes, DATASETS):
    cfg    = data[ds]
    f1     = cfg['f1']
    color  = cfg['color']
    xr     = cfg['xrange']
    e1_val = f1[0]

    ax.set_facecolor('white')
    ax.set_xlim(xr)
    ax.set_ylim(-0.6, N - 0.4)

    # 垂直基准线 (E1 F1值)
    ax.axvline(e1_val, color='#333333', linewidth=1.1, zorder=3)

    # 横向辅助网格
    for y in y_pos:
        ax.axhline(y, color='#b0b0b0', linewidth=0.45, alpha=0.3, zorder=0)

    # 水平连线 & 点
    for i, (yi, vi) in enumerate(zip(y_pos, f1)):
        lw   = 2.0 if i == E6_IDX else 1.3
        alph = 0.72 if i == E6_IDX else 0.36
        # 从 E1 参考线到当前点的连线
        ax.plot([e1_val, vi], [yi, yi],
                color=color, linewidth=lw, alpha=alph,
                solid_capstyle='round', zorder=2)

    # 填充圆点（非 E6）
    xs_reg = [f1[i] for i in range(N) if i != E6_IDX]
    ys_reg = [y_pos[i] for i in range(N) if i != E6_IDX]
    ax.scatter(xs_reg, ys_reg,
               s=28, c=color, zorder=5,
               edgecolors='white', linewidths=0.9)

    # E6 特殊标记（空心大圆 + 小实心圆）
    xe6 = f1[E6_IDX]
    ye6 = y_pos[E6_IDX]
    ax.scatter([xe6], [ye6],
               s=80, facecolors='white', edgecolors=color,
               linewidths=1.8, zorder=5)
    ax.scatter([xe6], [ye6],
               s=14, c=color, zorder=6,
               edgecolors='white', linewidths=0.7)

    # 标注文字（只标注部分关键点）
    for idx, txt in cfg['annot'].items():
        vi = f1[idx]
        yi = y_pos[idx]
        # E6 行标注放在标记右侧（错开圆圈宽度）
        if idx == E6_IDX:
            dx, ha = 0.013, 'left'
        elif vi < e1_val:
            dx, ha = -0.006, 'right'
        else:
            dx, ha = 0.006, 'left'
        ax.text(vi + dx, yi, txt,
                fontsize=7.5, va='center', ha=ha,
                color='#222222')

    # Y 轴刻度（仅第一个子图显示）
    ax.set_yticks(y_pos)
    if ds == 'MSL':
        ax.set_yticklabels(LABELS, fontsize=9.5)
        ax.set_ylabel('实验设置', fontsize=11)
    else:
        ax.set_yticklabels([])

    # X 轴
    ax.set_xlabel('F1', fontsize=11)
    ax.tick_params(labelsize=9, direction='in', length=3)

    # 底部横线（替代底部 spine）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#777777')

    # 标题
    ax.set_title(ds, fontsize=13, fontweight='bold', pad=6)

    # 阴影区（E6 行高亮）
    ax.axhspan(y_pos[E6_IDX] - 0.45, y_pos[E6_IDX] + 0.45,
               alpha=0.055, color=color, zorder=0)

# ── 图例 ─────────────────────────────────────────────────────────────────────
dot_e    = mlines.Line2D([], [], color='#555555', marker='o', linestyle='None',
                          markersize=6, markerfacecolor='#555555',
                          markeredgecolor='white', label='E/B 系列（本研究实验）')
dot_e6   = mlines.Line2D([], [], color='#555555', marker='o', linestyle='None',
                          markersize=8, markerfacecolor='white',
                          markeredgecolor='#555555', markeredgewidth=1.8,
                          label='E6（位置编码 DGR 先验）')
ref_line = mlines.Line2D([], [], color='#333333', linewidth=1.1,
                          label='E1 基准线')

fig.legend(handles=[dot_e, dot_e6, ref_line],
           loc='lower center',
           ncol=3,
           fontsize=9,
           frameon=True,
           framealpha=0.9,
           edgecolor='#AAAAAA',
           bbox_to_anchor=(0.5, -0.02),
           prop={'size': 9})

# ── 保存 ─────────────────────────────────────────────────────────────────────
out = f"{OUT_DIR}/Figure4-2.png"
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved: {out}")
plt.close(fig)
