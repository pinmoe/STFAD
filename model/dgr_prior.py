"""dgr_prior.py
Prior association 模块，替代原始高斯核，注入点在 AnomalyTransformer.forward()。

三种实现由 AnomalyTransformer 的 dgr_mode 参数决定，不再使用全局开关：
  'dynamic'    → DGRPrior           （E2）
  'multiscale' → MultiScaleDGRPrior （E3）
  'static'     → StaticDGRPrior     （E4）

注意：全局变量 USE_STATIC_DGR / USE_MULTISCALE_DGR 已删除，
      AnomalyTransformer.py 不再 import 这两个变量。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGRPrior(nn.Module):
    """
    动态 DGR Prior（E2 消融对照）。
    输入: x_seq (B, W, C)
    输出: prior (B, H, W, W)，每行 sum=1

    修复（训练/测试分数分布不一致问题）：
      - head_dim > 1：使用 Xavier 初始化 + 余弦相似度（L2 归一化），
        防止 HAI 等慢变工控数据差分量级极小时先验退化为均匀分布，
        导致训练能量远低于测试能量、阈值失准（E2/E3 on HAI）。
      - head_dim = 1（SKAB: C=8, H=8）：保持 std=0.01 小初始化，
        避免秩-1 相似度矩阵下大权重引入随机噪声，保持现有 SKAB 结果不变。
    """
    def __init__(self, in_channels: int, n_heads: int, dropout: float = 0.1, use_diff: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.use_diff = use_diff
        # 以 in_channels//n_heads 为上限，避免 SKAB(C=8,H=8) 膨胀为 64 维噪声
        self.head_dim = max(in_channels // max(n_heads, 1), 1)
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(in_channels, n_heads * self.head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        # head_dim > 1：Xavier 初始化，使特征从第一个 epoch 起就有结构性尺度。
        # head_dim = 1（SKAB）：保留小初始化，防止秩-1 矩阵产生极端峰值分布。
        if self.head_dim > 1:
            nn.init.xavier_uniform_(self.proj.weight)
        else:
            nn.init.normal_(self.proj.weight, std=0.01)

        # raw 模式（use_diff=False）：不做 L2 归一化，用可学习温度 τ 代替固定 scale。
        # 初始化 log_τ = 0.5*log(head_dim)，即 τ₀ = sqrt(head_dim)，
        # 与 diff 模式的 scale=1/sqrt(head_dim) 在量级上等价（互为倒数，方向一致）。
        # 训练时模型可自主调整 τ 的大小以控制 softmax 的尖锐程度。
        if not use_diff and self.head_dim > 1:
            init_log_tau = 0.5 * math.log(max(self.head_dim, 1))
            self.log_tau = nn.Parameter(torch.tensor(init_log_tau))
        else:
            self.log_tau = None

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, W, C = x_seq.shape
        H = self.n_heads
        if self.use_diff:
            # 差分模式：适合点突变异常（MSL 等），diff[0] = 0
            feat_input = torch.zeros_like(x_seq)
            feat_input[:, 1:, :] = x_seq[:, 1:, :] - x_seq[:, :-1, :]
        else:
            # 原始值模式：适合工控慢变信号（HAI 等），
            # 原始传感器值在正常工况下有稳定的互相关结构，
            # 异常时传感器间关系失调 → 相似度下降 → 先验趋均匀 → KL 升高
            feat_input = x_seq
        feat = self.proj(feat_input).view(B, W, H, self.head_dim).permute(0, 2, 1, 3)
        feat = self.dropout(feat)
        if self.use_diff:
            # diff 模式（MSL/SKAB）：保持原有行为，L2 归一化 + 固定 scale。
            # head_dim=1（SKAB）跳过归一化。
            if self.head_dim > 1:
                feat = F.normalize(feat, dim=-1)
            sim = torch.matmul(feat, feat.transpose(-1, -2)) * self.scale
        else:
            # raw 模式（HAI）：不做 L2 归一化，保留特征幅值信息。
            # 正常段各时刻传感器值相似 → 幅值相似 → 点积均匀 → 先验近均匀（合理）。
            # 异常段某时刻值偏离 → 幅值异常 → 与正常时刻点积更小 → 先验出现结构 → KL 可升高。
            # L2 归一化会把异常时刻强制拉回单位球，消除幅值差异，此处故意不做。
            if self.log_tau is not None:
                tau = torch.exp(self.log_tau).clamp(min=1e-2)  # 防止温度坍缩
                sim = torch.matmul(feat, feat.transpose(-1, -2)) / tau
            else:
                # head_dim=1 fallback
                sim = torch.matmul(feat, feat.transpose(-1, -2)) * self.scale
        return F.softmax(sim, dim=-1)


class StaticDGRPrior(nn.Module):
    """
    静态可学习 DGR Prior（E4）。
    prior 是 (H, W, W) 可学习参数，与输入完全无关。
    输入: x_seq (B, W, C) — 只用于取 B，数值被忽略
    输出: prior (B, H, W, W)，每行 sum=1

    修复（训练/测试分数分布不一致问题）：
      原有 std=0.01 小初始化导致 prior_logits ≈ 全零 → softmax ≈ 均匀分布 →
      series 被 Phase-1 推向均匀 → 训练 KL 极低 → 训练能量仅为 E1 的一半 →
      阈值被拉低至 ~1368（E1 为 ~2640）→ 测试时大量正常点超阈 → HAI 假阳性爆炸。

      修复：将 prior_logits 初始化为高斯结构（sigma = win_size × 10%），
      使静态先验的训练起点与 E1 Gaussian Prior 一致，训练能量保持在合理范围。
      后续训练可在此基础上自由学习适合各数据集的先验形状。
    """
    def __init__(self, win_size: int, n_heads: int):
        super().__init__()
        # 高斯结构初始化：logits_{i,j} = -(i-j)^2 / (2 * sigma^2)，sigma = win_size * 0.1
        # softmax 后即为类高斯先验，与 E1 训练起点一致，消除训练/测试能量分布不匹配。
        positions = torch.arange(win_size).float()
        dist2 = (positions.unsqueeze(0) - positions.unsqueeze(1)).pow(2)
        sigma2 = float(win_size * 0.1) ** 2  # sigma = 窗口长度的 10%（= 10 步，win_size=100）
        gauss_logits = (-dist2 / sigma2).unsqueeze(0).expand(n_heads, -1, -1).contiguous()
        self.prior_logits = nn.Parameter(gauss_logits)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B = x_seq.shape[0]
        prior = F.softmax(self.prior_logits, dim=-1)
        return prior.unsqueeze(0).expand(B, -1, -1, -1)


class MultiScaleDGRPrior(nn.Module):
    """
    多尺度动态 DGR Prior（E3）。
    细粒度（全窗口）与粗粒度（降采样 W/2）加权混合，每行归一化。
    输入: x_seq (B, W, C)
    输出: prior (B, H, W, W)，每行 sum=1
    """
    def __init__(self, in_channels: int, win_size: int, n_heads: int, dropout: float = 0.1, use_diff: bool = True):
        super().__init__()
        self.win_size = win_size
        self.fine = DGRPrior(in_channels, n_heads, dropout, use_diff=use_diff)
        self.coarse = DGRPrior(in_channels, n_heads, dropout, use_diff=use_diff)
        self.mix_logit = nn.Parameter(torch.zeros(1))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, W, C = x_seq.shape
        prior_fine = self.fine(x_seq)
        x_coarse = x_seq[:, ::2, :]
        prior_coarse_small = self.coarse(x_coarse)
        prior_coarse = F.interpolate(
            prior_coarse_small, size=(W, W), mode='bilinear', align_corners=False
        )
        alpha = torch.sigmoid(self.mix_logit)
        prior = alpha * prior_fine + (1 - alpha) * prior_coarse
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return prior


class DGRPriorPE(nn.Module):
    """
    E6：位置编码增强的动态 DGR Prior。

    在 E2（DGRPrior）基础上，向差分特征中注入正弦位置编码，
    修复 E2 因差分特征无时间次序信息而退化为均匀先验的问题。

    关键设计：
    - pe_scale 零初始化：训练起点与 E2 完全一致，无 cold-start 风险。
    - 训练过程中 pe_scale 自主学习注入量，对 MSL（时间局部性主导）
      自然倾向于保留时间结构，对 SMAP/SKAB（通道耦合主导）保留通道信息。
    """
    def __init__(self, in_channels: int, n_heads: int, win_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = max(in_channels // max(n_heads, 1), 1)
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(in_channels, n_heads * self.head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        if self.head_dim > 1:
            nn.init.xavier_uniform_(self.proj.weight)
        else:
            nn.init.normal_(self.proj.weight, std=0.01)

        # 正弦位置编码 (W, C)，固定不学习
        pe = torch.zeros(win_size, in_channels)
        position = torch.arange(win_size).unsqueeze(1).float()
        half = in_channels // 2
        div_term = torch.exp(torch.arange(0, half).float() * (-math.log(10000.0) / half))
        pe[:, 0:2*half:2] = torch.sin(position * div_term)
        pe[:, 1:2*half:2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (W, C)

        # 零初始化：训练开始时 pe_scale=0，行为与 E2 完全相同
        self.pe_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, W, C = x_seq.shape
        H = self.n_heads

        # 差分特征（与 E2 相同）
        feat_input = torch.zeros_like(x_seq)
        feat_input[:, 1:, :] = x_seq[:, 1:, :] - x_seq[:, :-1, :]

        # 注入位置编码
        feat_input = feat_input + self.pe_scale * self.pe[:W, :].unsqueeze(0)

        feat = self.proj(feat_input).view(B, W, H, self.head_dim).permute(0, 2, 1, 3)
        feat = self.dropout(feat)
        if self.head_dim > 1:
            feat = F.normalize(feat, dim=-1)
        sim = torch.matmul(feat, feat.transpose(-1, -2)) * self.scale
        return F.softmax(sim, dim=-1)


class DGRSigmaOffset(nn.Module):
    """
    E5：DGR Sigma 调制先验。
    不替换高斯先验分布，而是预测每个时刻、每个注意力头的 sigma 偏置量，
    叠加到 AnomalyAttention 内部的原始高斯核宽度上。

    设计优势：
    - 零初始化：训练初始完全等价于 E1（Gaussian baseline），无 cold-start
    - 结构保留：任何情况下先验仍保持对角线峰值，不会退化为均匀分布
    - 梯度清晰：网络只需学习"何时拉宽/收窄时间相关性范围"

    输入:  x_seq  (B, W, C)
    输出:  sigma_offset  (B, H, W)，加到 AnomalyAttention 内部 sigma(B,H,W) 上
    """
    def __init__(self, in_channels: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(in_channels, 16)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_heads),
        )
        # 零初始化：训练开始时输出全零，完全退化为 E1
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, W, C) → (B, W, H) → (B, H, W)
        return self.net(x_seq).permute(0, 2, 1)
