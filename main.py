import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
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

    # use_dgr_prior 保留向后兼容，但优先使用 dgr_mode
    parser.add_argument('--use_dgr_prior', type=str2bool, default='false')

    # 新增：明确指定 DGR 模式，避免硬编码开关
    # none       → E1，原始高斯先验（use_dgr_prior=False）
    # dynamic    → E2，DGRPrior 动态先验
    # multiscale → E3，MultiScaleDGRPrior 多尺度动态先验
    # static     → E4，StaticDGRPrior 静态可学习先验
    # sigma_offset → E5，DGRSigmaOffset 调制高斯核宽度
    parser.add_argument('--dgr_mode', type=str, default='none',
                        choices=['none', 'dynamic', 'multiscale', 'static', 'sigma_offset'])

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
                        choices=['combined', 'rec_only', 'rec_mean', 'weighted'],
                        help='测试阶段异常评分公式')
    parser.add_argument('--score_alpha', type=float, default=1.0,
                        help='weighted 模式下重建误差权重，范围 [0, 1]')
    # 测试后处理：对最终 1D 评分序列做滑动均值平滑（抑制孤立尖峰假阳性）
    # 对持续性异常（如 HAI 工控攻击）有效；点异常数据集保持默认 1（不平滑）
    parser.add_argument('--score_smooth_k', type=int, default=1,
                        help='评分时序平滑窗口大小（1=不平滑，建议尝试 5/10/20）')

    config = parser.parse_args()

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
