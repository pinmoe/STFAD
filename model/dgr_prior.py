"""dgr_prior.py
Prior association 模块，替代原始高斯核，注入点在 AnomalyTransformer.forward()。

三种实现由 AnomalyTransformer 的 dgr_mode 参数决定，不再使用全局开关：
  'dynamic'    → DGRPrior           （E2）
  'multiscale' → MultiScaleDGRPrior （E3）
  'static'     → StaticDGRPrior     （E4）

注意：全局变量 USE_STATIC_DGR / USE_MULTISCALE_DGR 已删除，
      AnomalyTransformer.py 不再 import 这两个变量。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGRPrior(nn.Module):
    """
    动态 DGR Prior（E2 消融对照）。
    输入: x_seq (B, W, C)
    输出: prior (B, H, W, W)，每行 sum=1
    修复：head_dim 不允许超过 in_channels//n_heads（防止低通道数据集如 SKAB 过膨胀到均匀先验）
    """
    def __init__(self, in_channels: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        # 修复：以 in_channels//n_heads 为上限，避免 SKAB(C=8,H=8) 膨胀为 64 维噪声
        self.head_dim = max(in_channels // max(n_heads, 1), 1)
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(in_channels, n_heads * self.head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, W, C = x_seq.shape
        H = self.n_heads
        feat = self.proj(x_seq).view(B, W, H, self.head_dim).permute(0, 2, 1, 3)
        feat = self.dropout(feat)
        sim = torch.matmul(feat, feat.transpose(-1, -2)) * self.scale
        return F.softmax(sim, dim=-1)


class StaticDGRPrior(nn.Module):
    """
    静态可学习 DGR Prior（E4）。
    prior 是 (H, W, W) 可学习参数，与输入完全无关。
    输入: x_seq (B, W, C) — 只用于取 B，数值被忽略
    输出: prior (B, H, W, W)，每行 sum=1
    """
    def __init__(self, win_size: int, n_heads: int):
        super().__init__()
        self.prior_logits = nn.Parameter(torch.zeros(n_heads, win_size, win_size))
        nn.init.normal_(self.prior_logits, std=0.01)

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
    def __init__(self, in_channels: int, win_size: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.win_size = win_size
        self.fine = DGRPrior(in_channels, n_heads, dropout)
        self.coarse = DGRPrior(in_channels, n_heads, dropout)
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
