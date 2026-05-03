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
    parser.add_argument('--dgr_mode', type=str, default='none',
                        choices=['none', 'dynamic', 'multiscale', 'static'])

    # === 方案1+3 新增参数 ===
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (0.0=原始, 小数据集如SKAB建议0.2-0.3)')
    parser.add_argument('--temperature', type=float, default=50.0,
                        help='异常分数放大温度 (默认50, MSL/SKAB可尝试20-30)')
    parser.add_argument('--score_mode', type=str, default='assoc+recon',
                        choices=['assoc+recon', 'recon_only', 'assoc_only'],
                        help='异常分数组合: assoc+recon=当前默认, recon_only=仅重构, assoc_only=仅关联差异')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Transformer d_model (HAI=512, MSL=256, SKAB=128)')

    config = parser.parse_args()

    # 数据集感知的默认参数覆盖（仅当用户没有显式指定时生效）
    # 通过检查参数是否仍为原始默认值来做，简单有效
    _DATASET_OVERRIDES = {
        'HAI':  {'anormly_ratio': 1.0, 'num_epochs': 10},
        'MSL':  {'anormly_ratio': 10.0, 'num_epochs': 20, 'dropout': 0.1, 'd_model': 256},
        'SKAB': {'anormly_ratio': 5.0, 'num_epochs': 15, 'dropout': 0.2, 'd_model': 128},
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

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
