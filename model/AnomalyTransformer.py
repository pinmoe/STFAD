import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding
from .dgr_prior import DGRPrior, StaticDGRPrior, MultiScaleDGRPrior


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        if self.norm is not None:
            x = self.norm(x)
        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True,
                 use_dgr_prior=False, dgr_mode='none',
                 prior_fusion='replace', prior_alpha=0.5, prior_alpha_learnable=False,
                 dgr_input_mode='raw'):
        """
        dgr_mode 参数说明（优先级高于 use_dgr_prior）：
          'none'       -> E1，原始高斯先验
          'dynamic'    -> E2，DGRPrior 动态先验
          'multiscale' -> E3，MultiScaleDGRPrior 多尺度动态先验
          'static'     -> E4，StaticDGRPrior 静态可学习先验

        prior_fusion 参数说明：
          'replace' -> 仅使用 DGR（兼容现有实现）
          'blend'   -> 高斯先验与 DGR 先验加权融合
        """
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.prior_fusion = prior_fusion
        self.prior_alpha = float(prior_alpha)
        self.dgr_input_mode = dgr_input_mode

        if dgr_mode == 'none':
            self.dgr_mode = 'none'
        else:
            self.dgr_mode = dgr_mode

        if self.dgr_mode == 'none' and use_dgr_prior:
            self.dgr_mode = 'dynamic'

        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout,
                                         output_attention=output_attention),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

        if self.dgr_mode == 'multiscale':
            self.dgr_priors = nn.ModuleList(
                [MultiScaleDGRPrior(enc_in, win_size, n_heads, dropout=dropout)
                 for _ in range(e_layers)]
            )
        elif self.dgr_mode == 'static':
            self.dgr_priors = nn.ModuleList(
                [StaticDGRPrior(win_size, n_heads) for _ in range(e_layers)]
            )
        elif self.dgr_mode == 'dynamic':
            self.dgr_priors = nn.ModuleList(
                [DGRPrior(enc_in, n_heads, dropout=dropout) for _ in range(e_layers)]
            )
        else:
            self.dgr_priors = None

        self.prior_alpha_logits = None
        if self.prior_fusion == 'blend' and self.dgr_priors is not None and prior_alpha_learnable:
            alpha = min(max(self.prior_alpha, 1e-4), 1.0 - 1e-4)
            init_logit = torch.logit(torch.tensor(alpha))
            self.prior_alpha_logits = nn.Parameter(init_logit.repeat(e_layers))

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        gaussian_prior = prior

        if self.dgr_priors is not None:
            if self.dgr_mode == 'static':
                dgr_prior = [self.dgr_priors[u](x) for u in range(len(gaussian_prior))]
            else:
                if self.dgr_input_mode == 'smoothed':
                    dgr_input = x.mean(dim=1, keepdim=True).expand_as(x)
                else:
                    dgr_input = x
                dgr_prior = [self.dgr_priors[u](dgr_input) for u in range(len(gaussian_prior))]

            if self.prior_fusion == 'blend':
                prior = []
                for u in range(len(gaussian_prior)):
                    if self.prior_alpha_logits is not None:
                        alpha = torch.sigmoid(self.prior_alpha_logits[u])
                    else:
                        alpha = torch.tensor(self.prior_alpha, device=x.device, dtype=x.dtype)
                    fused = alpha * gaussian_prior[u] + (1.0 - alpha) * dgr_prior[u]
                    fused = fused / fused.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    prior.append(fused)
            else:
                prior = dgr_prior
        else:
            prior = gaussian_prior

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out
