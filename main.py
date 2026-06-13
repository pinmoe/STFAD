import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = False
    set_random_seed(config.seed)
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--seed', type=int, default=2024)

    # use_dgr_prior 保留向后兼容，但优先使用 dgr_mode
    parser.add_argument('--use_dgr_prior', type=str2bool, default='false')

    # 新增：明确指定 DGR 模式，避免硬编码开关
    # none       → E1，原始高斯先验（use_dgr_prior=False）
    # dynamic    → E2，DGRPrior 动态先验
    # multiscale → E3，MultiScaleDGRPrior 多尺度动态先验
    # static     → E4，StaticDGRPrior 静态可学习先验
    # sigma_offset → E5，DGRSigmaOffset 调制高斯核宽度
    parser.add_argument('--dgr_mode', type=str, default='none',
                        choices=['none', 'dynamic', 'multiscale', 'static', 'sigma_offset', 'dynamic_pe'])

    # 先验融合策略
    # replace:      仅使用 DGR（兼容你当前实现）
    # blend:        高斯先验 + DGR 先验加权融合
    # entropy_gate: 基于 DGR 行熵的逐点自适应门控（高熵→高斯，低熵→DGR）
    parser.add_argument('--prior_fusion', type=str, default='replace',
                        choices=['replace', 'blend', 'entropy_gate'])
    parser.add_argument('--prior_alpha', type=float, default=0.5,
                        help='blend 模式下高斯先验权重，范围建议 [0,1]')
    parser.add_argument('--prior_alpha_learnable', type=str2bool, default='false',
                        help='是否让 prior_alpha 在训练中可学习（每层一个）')

    # DGR 输入方式：raw 更强调传感器间关系，smoothed 兼容旧逻辑
    parser.add_argument('--dgr_input_mode', type=str, default='raw',
                        choices=['raw', 'smoothed'])

    # DGR 特征构建方式：diff=时序差分（适合点突变，MSL/SKAB），raw=原始值（适合慢变工控，HAI）
    parser.add_argument('--dgr_feature_mode', type=str, default='diff',
                        choices=['diff', 'raw'])

    # entropy_gate 融合参数（prior_fusion='entropy_gate' 时生效）
    parser.add_argument('--prior_entropy_tau', type=float, default=0.6,
                        help='熵归一化阈值，高于此值偏向高斯先验，范围建议 [0.4, 0.8]')
    parser.add_argument('--prior_entropy_gamma', type=float, default=12.0,
                        help='门控陡峭度，越大决策边界越硬，建议 [6, 20]')

    # 测试阶段异常评分策略（无需重训练，仅影响 test 模式）
    # combined : (KL_series + KL_prior) + max_rec  原始论文公式（默认）
    # rec_only  : max(MSE, dim=channel)            纯重建误差（适合HAI等值漂移型异常）
    # rec_mean  : mean(MSE, dim=channel)           均值重建误差（比max更平稳，减少噪声通道影响）
    # weighted  : score_alpha*rec + (1-alpha)*KL   可调权重融合
    parser.add_argument('--score_mode', type=str, default='combined',
                        choices=['combined', 'rec_only', 'rec_mean', 'weighted', 'chan_var'],
                        help='测试阶段异常评分公式（chan_var=通道方差倒数加权，适合多传感器工控数据如HAI）')
    parser.add_argument('--score_alpha', type=float, default=1.0,
                        help='weighted 模式下重建误差权重，范围 [0, 1]')
    # 测试后处理：对最终 1D 评分序列做滑动均值平滑（抑制孤立尖峰假阳性）
    # 对持续性异常（如 HAI 工控攻击）有效；点异常数据集保持默认 1（不平滑）
    parser.add_argument('--score_smooth_k', type=int, default=1,
                        help='评分时序平滑窗口大小（1=不平滑，建议尝试 5/10/20）')

    # 差分重建辅助评分（方向A）：对点突变异常在差分域放大信号，无需重训练
    parser.add_argument('--diff_beta', type=float, default=0.0,
                        help='差分重建辅助评分权重（0=不启用，MSL/SKAB/SMAP建议 0.5~2.0）')
    # 局部z-score后处理（方向B）：突出局部异常对比度，抑制背景能量波动带来的假阳性
    parser.add_argument('--score_local_z_win', type=int, default=0,
                        help='局部z-score半窗口大小（0=不启用，MSL建议 200~500）')
    # 训练时差分重建项（方向C）：显式训练模型正确重建局部变化，需重训练
    parser.add_argument('--window_score_mode', type=str, default='mean',
                        choices=['mean', 'max', 'topk_mean', 'median', 'p95'],
                        help='Window-level aggregation for fixed-window datasets such as ST330IR001_CP001')
    parser.add_argument('--window_topk_ratio', type=float, default=0.2,
                        help='Ratio of points used by window_score_mode=topk_mean')
    parser.add_argument('--window_anormly_ratio', type=float, default=None,
                        help='Window-level threshold ratio. Defaults to anormly_ratio when omitted.')
    parser.add_argument('--window_ratio_sweep', type=str, default='',
                        help='Comma-separated window threshold ratios to print, for example 35,40,45.93,50')
    parser.add_argument('--window_score_sweep', type=str2bool, default='false',
                        help='Print window-level metrics for all aggregation modes in one test run')
    parser.add_argument('--export_score_path', type=str, default='',
                        help='Optional .npz path for exporting test-time point scores and labels.')
    parser.add_argument('--lambda_diff', type=float, default=0.0,
                        help='训练时差分重建项权重（0=不启用，MSL建议 0.1~1.0，需重训练）')

    # 记忆库先验净化：从训练集中选取低重建误差的正常样本，替代被异常污染的实时先验。
    # 对高异常率数据集（如 ST330IR001_CP001 45.93%）尤其有效，配合 dynamic/multiscale 使用。
    parser.add_argument('--use_memory_bank', type=str2bool, default='false',
                        help='是否启用正常样本记忆库净化 DGR 先验（仅对 dynamic/multiscale 有效）')

    # === 方案1+3 新增参数 ===
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (0.0=原始, 小数据集如SKAB建议0.2-0.3)')
    parser.add_argument('--temperature', type=float, default=50.0,
                        help='异常分数放大温度 (默认50, MSL/SKAB可尝试20-30)')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Transformer d_model (HAI=512, MSL=256, SKAB=128)')

    config = parser.parse_args()

    # 数据集感知的默认参数覆盖（仅当用户没有显式指定时生效）
    # 通过检查参数是否仍为原始默认值来做，简单有效
    _DATASET_OVERRIDES = {
        'HAI':  {'anormly_ratio': 1.0, 'num_epochs': 10},
        'MSL':  {'anormly_ratio': 10.0, 'num_epochs': 20, 'dropout': 0.1, 'd_model': 256},
        'SKAB': {'anormly_ratio': 5.0, 'num_epochs': 15, 'dropout': 0.2, 'd_model': 128},
        'ST330IR001_CP001': {
            'anormly_ratio': 45.93,
            'win_size': 56,
            'input_c': 29,
            'output_c': 29,
            'd_model': 128,
        },
    }
    if config.dataset in _DATASET_OVERRIDES:
        for k, v in _DATASET_OVERRIDES[config.dataset].items():
            # 只覆盖仍为原始默认值的参数
            orig_default = parser.get_default(k)
            if getattr(config, k) == orig_default:
                setattr(config, k, v)
                print(f'[Auto] {k} overridden to {v} for dataset={config.dataset}')

    # dgr_mode 覆盖 use_dgr_prior，保证两者一致
    if config.dgr_mode != 'none':
        config.use_dgr_prior = True
    else:
        config.use_dgr_prior = False

    # dgr_mode=none 或 sigma_offset 时，prior fusion 无意义，回退原始高斯先验
    if config.dgr_mode in ('none', 'sigma_offset'):
        config.prior_fusion = 'replace'

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
