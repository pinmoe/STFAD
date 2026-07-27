import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import json
import re
import subprocess
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from utils.eval_metrics import (
    aggregate_window_scores,
    resolve_eval_unit,
    window_labels_from_point_labels,
    window_level_metrics_from_scores,
)


def _local_zscore(arr, half_win):
    """滑动窗口局部 z-score，用 numpy 卷积实现，O(n)。"""
    w = 2 * half_win + 1
    kernel = np.ones(w) / w
    mu   = np.convolve(arr,      kernel, mode='same')
    sq_mu = np.convolve(arr ** 2, kernel, mode='same')
    var  = np.maximum(sq_mu - mu ** 2, 0.0)
    return (arr - mu) / (np.sqrt(var) + 1e-8)


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def _safe_name(value):
    value = str(value).strip()
    if not value:
        return "unnamed"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self._config_snapshot = dict(config)

    def _experiment_name(self):
        explicit = getattr(self, "experiment_name", "")
        if explicit:
            return _safe_name(explicit)
        parts = [
            getattr(self, "dataset", "dataset"),
            getattr(self, "dgr_mode", "none"),
            getattr(self, "prior_fusion", "replace"),
            getattr(self, "dgr_feature_mode", "diff"),
            getattr(self, "score_mode", "combined"),
        ]
        if float(getattr(self, "diff_beta", 0.0)) > 0:
            parts.append("diffbeta_{}".format(getattr(self, "diff_beta")))
        if int(getattr(self, "score_smooth_k", 1)) > 1:
            parts.append("smooth_{}".format(getattr(self, "score_smooth_k")))
        if int(getattr(self, "score_local_z_win", 0)) > 1:
            parts.append("localz_{}".format(getattr(self, "score_local_z_win")))
        if bool(getattr(self, "use_memory_bank", False)):
            parts.append("memorybank")
        return _safe_name("__".join(str(p) for p in parts))

    def _result_path(self):
        return os.path.join(
            getattr(self, "result_dir", "results/paper_main"),
            _safe_name(getattr(self, "dataset", "dataset")),
            self._experiment_name(),
            "seed_{}".format(_safe_name(getattr(self, "seed", "unknown"))),
        )

    def _write_test_outputs(self, metrics, arrays):
        out_dir = self._result_path()
        os.makedirs(out_dir, exist_ok=True)

        config = {k: _jsonable(v) for k, v in self._config_snapshot.items()}
        config["resolved_experiment_name"] = self._experiment_name()
        config["git_commit"] = _git_commit()
        config["checkpoint_path"] = os.path.join(
            str(self.model_save_path),
            str(self.dataset) + "_checkpoint.pth",
        )

        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(metrics), f, indent=2, ensure_ascii=False, sort_keys=True)
        with open(os.path.join(out_dir, "checkpoint_path.txt"), "w", encoding="utf-8") as f:
            f.write(config["checkpoint_path"] + "\n")

        if bool(getattr(self, "save_scores", True)):
            for name, arr in arrays.items():
                np.save(os.path.join(out_dir, name + ".npy"), np.asarray(arr))

        print("[Results] saved structured outputs to {}".format(out_dir))

    def _model_complexity(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
        }

    def _write_train_outputs(self, metrics):
        out_dir = self._result_path()
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(metrics), f, indent=2, ensure_ascii=False, sort_keys=True)

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            d_model=getattr(self, 'd_model', 512),
            dropout=getattr(self, 'dropout', 0.0),
            n_heads=getattr(self, 'n_heads', 8),
            e_layers=getattr(self, 'e_layers', 3),
            use_dgr_prior=self.use_dgr_prior,
            dgr_mode=getattr(self, 'dgr_mode', 'none'),   # 向后兼容
            prior_fusion=getattr(self, 'prior_fusion', 'replace'),
            prior_alpha=getattr(self, 'prior_alpha', 0.5),
            prior_alpha_learnable=getattr(self, 'prior_alpha_learnable', False),
            dgr_input_mode=getattr(self, 'dgr_input_mode', 'raw'),
            prior_entropy_tau=getattr(self, 'prior_entropy_tau', 0.6),
            prior_entropy_gamma=getattr(self, 'prior_entropy_gamma', 12.0),
            dgr_feature_mode=getattr(self, 'dgr_feature_mode', 'diff'),
        )
        # 修复二：对 DGR 参数单独施加 weight_decay，减少过拟合（尤其对 E4/E5）
        if hasattr(self.model, 'dgr_priors') and self.model.dgr_priors is not None:
            dgr_param_ids = {id(p) for p in self.model.dgr_priors.parameters()}
            main_params = [p for p in self.model.parameters() if id(p) not in dgr_param_ids]
            dgr_params = list(self.model.dgr_priors.parameters())
            self.optimizer = torch.optim.Adam([
                {'params': main_params, 'lr': self.lr},
                {'params': dgr_params, 'lr': self.lr, 'weight_decay': 1e-4},
            ])
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                input_data = batch[0]
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach(),
                            series[u])))
                    # 修复一：Phase 2 detach DGR prior，避免 DGR 参数通过 prior_loss 接收梯度
                    prior_u_d = prior[u].detach()
                    prior_loss += (torch.mean(
                        my_kl_loss((prior_u_d / torch.unsqueeze(torch.sum(prior_u_d, dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                              self.win_size)),
                                   series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(),
                                   (prior_u_d / torch.unsqueeze(torch.sum(prior_u_d, dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                              self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)
                loss_1.append((rec_loss - self.k * series_loss).item())
                loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    # ---------------------------------------------------------------------- #
    # 记忆库：净化 DGR 先验中的异常污染
    # 适用场景：高异常率数据集（如 ST330IR001_CP001 45.93%），
    #           动态 DGR 先验从输入实时计算，异常样本会污染先验分布。
    # 方法：训练后从 train_loader 中选取重建误差最低的前 50% 样本，
    #       用这些"干净正常样本"计算平均先验，推理时替换被污染的实时先验。
    # ---------------------------------------------------------------------- #

    def _build_memory_bank(self, max_samples: int = 512):
        """收集训练集中重建误差最低的样本作为正常记忆库。"""
        _crit = nn.MSELoss(reduction='none')
        all_inputs, all_errors = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in self.train_loader:
                inp = batch[0].float().to(self.device)
                out, _, _, _ = self.model(inp)
                err = _crit(inp, out).mean(dim=(1, 2)).cpu()  # (B,) 每样本均值误差
                all_inputs.append(inp.cpu())
                all_errors.append(err)
        all_inputs = torch.cat(all_inputs, dim=0)   # (N, W, C)
        all_errors = torch.cat(all_errors, dim=0)   # (N,)
        n_keep = min(max_samples, max(len(all_errors) // 2, 1))
        _, idx = all_errors.topk(n_keep, largest=False)
        thresh_err = all_errors[idx[-1]].item()
        print(f"[MemoryBank] 保留 {n_keep}/{len(all_errors)} 个低误差样本，"
              f"误差阈值: {thresh_err:.4f}")
        return all_inputs[idx]  # (n_keep, W, C)

    def _compute_mb_prior(self, memory_bank):
        """用记忆库样本计算 DGR 先验，batch 维取均值，返回每层固定先验列表。"""
        n = len(memory_bank)
        bs = min(self.batch_size, n)
        idx = torch.randperm(n)[:bs]
        mb_inp = memory_bank[idx].to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, _, mb_prior, _ = self.model(mb_inp)
        # 对 batch 维平均，得 (1, H, W, W)，推理时 expand 到实际 batch size
        return [p.mean(dim=0, keepdim=True).detach() for p in mb_prior]

    def train(self):

        print("======================TRAIN MODE======================")
        train_start_time = time.time()

        _lambda_diff = float(getattr(self, 'lambda_diff', 0.0))
        if _lambda_diff > 0:
            print(f"[训练增强] lambda_diff={_lambda_diff}（差分重建正则项已启用）")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        completed_epochs = 0
        best_val_loss1 = None
        best_val_loss2 = None

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                input_data = batch[0]

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    # Phase 2 梯度流：所有模式（含 E5 sigma_offset）prior 均参与反向传播。
                    # sigma_offset 的梯度路径：prior_loss → prior → sigma → sigma_ext → DGRSigmaOffset。
                    # Phase 1 的 series_loss 已 detach prior，sigma_offset 只通过 Phase 2 训练——
                    # 这正好是 Minimax 博弈的设计意图：Phase 2 让先验追赶系列注意力分布。
                    prior_u_p2 = prior[u]  # 所有模式统一：prior 参与 Phase 2 梯度
                    _p2_denom = torch.unsqueeze(torch.sum(prior_u_p2, dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)
                    prior_loss += (torch.mean(my_kl_loss(
                        prior_u_p2 / _p2_denom,
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(),
                                   prior_u_p2 / _p2_denom)))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)
                if _lambda_diff > 0:
                    _di = input[:, 1:, :] - input[:, :-1, :]
                    _do = output[:, 1:, :] - output[:, :-1, :]
                    rec_loss = rec_loss + _lambda_diff * self.criterion(_di, _do)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)
            completed_epochs = epoch + 1
            if best_val_loss1 is None or vali_loss1 < best_val_loss1:
                best_val_loss1 = float(vali_loss1)
            if best_val_loss2 is None or vali_loss2 < best_val_loss2:
                best_val_loss2 = float(vali_loss2)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        train_metrics = {
            "dataset": getattr(self, "dataset", None),
            "experiment_name": self._experiment_name(),
            "seed": getattr(self, "seed", None),
            "completed_epochs": int(completed_epochs),
            "requested_epochs": int(getattr(self, "num_epochs", 0)),
            "train_steps_per_epoch": int(train_steps),
            "best_val_loss1": best_val_loss1,
            "best_val_loss2": best_val_loss2,
            "runtime_sec": float(time.time() - train_start_time),
            "model": self._model_complexity(),
            "checkpoint_path": os.path.join(str(self.model_save_path), str(self.dataset) + "_checkpoint.pth"),
        }
        self._write_train_outputs(train_metrics)

    @torch.no_grad()
    def test(self):
        test_start_time = time.time()
        checkpoint_path = os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        temperature = float(getattr(self, 'temperature', 50.0))

        print("======================TEST MODE======================")

        # 记忆库先验净化（仅对 dynamic/multiscale/dynamic_pe 有效）
        _dgr_mode = getattr(self, 'dgr_mode', 'none')
        _use_mb   = getattr(self, 'use_memory_bank', False)
        _mb_prior = None
        if _use_mb and _dgr_mode in ('dynamic', 'multiscale', 'dynamic_pe'):
            print("[MemoryBank] 正在构建正常样本记忆库...")
            _memory_bank = self._build_memory_bank()
            _mb_prior    = self._compute_mb_prior(_memory_bank)
            print(f"[MemoryBank] 清洁先验构建完成，共 {len(_mb_prior)} 层")

        criterion = nn.MSELoss(reduce=False)
        # 测试评分策略参数（无需重训练）
        _smode       = getattr(self, 'score_mode',       'combined')
        _salpha      = float(getattr(self, 'score_alpha',      1.0))
        _smooth      = int(getattr(self,   'score_smooth_k',   1))
        _diff_beta   = float(getattr(self, 'diff_beta',        0.0))
        _local_z_win = int(getattr(self,   'score_local_z_win', 0))
        _threshold_mode = getattr(self, 'threshold_mode', 'val_percentile')
        _threshold_percentile = float(getattr(self, 'threshold_percentile', 95.0))
        _threshold_grid_min = float(getattr(self, 'threshold_grid_min', 75.0))
        _threshold_grid_max = float(getattr(self, 'threshold_grid_max', 99.9))
        _threshold_grid_step = float(getattr(self, 'threshold_grid_step', 0.5))
        if _smode != 'combined' or _smooth > 1 or _diff_beta > 0 or _local_z_win > 1:
            print(f"[评分策略] mode={_smode},  alpha={_salpha},  smooth_k={_smooth},  "
                  f"diff_beta={_diff_beta},  local_z_win={_local_z_win}")
        print(f"[阈值协议] mode={_threshold_mode}, percentile={_threshold_percentile}")

        # chan_var 模式：在训练集上估计每个通道重建误差的方差，
        # 以方差倒数作为通道权重 w_c ∝ 1/(Var_c + ε)，归一化后加权求和。
        # 原理：方差小的通道在正常段重建稳定，偶现误差峰判别力更强；
        #       方差大的通道本身不稳定（如噪声传感器），应降权以减少假阳性。
        # 此步骤仅需额外一次 train_loader 前向，不修改模型参数。
        chan_w = None
        if _smode == 'chan_var':
            print("[chan_var] 正在估计各通道重建误差方差...")
            _rec_sum  = None  # 累积 (C,) 均值，用于 Welford 两趟法
            _rec_sum2 = None  # 累积 (C,) 平方和
            _rec_cnt  = 0
            with torch.no_grad():
                for _cbatch in self.train_loader:
                    _cinput = _cbatch[0].float().to(self.device)
                    _coutput, _, _, _ = self.model(_cinput)
                    _crec = criterion(_cinput, _coutput)  # (B, W, C)
                    _crec_flat = _crec.view(-1, _crec.shape[-1])  # (N, C)
                    if _rec_sum is None:
                        _rec_sum  = _crec_flat.sum(dim=0)
                        _rec_sum2 = (_crec_flat ** 2).sum(dim=0)
                    else:
                        _rec_sum  += _crec_flat.sum(dim=0)
                        _rec_sum2 += (_crec_flat ** 2).sum(dim=0)
                    _rec_cnt += _crec_flat.shape[0]
            # Var = E[x^2] - (E[x])^2
            _chan_mean = _rec_sum / _rec_cnt
            _chan_var  = (_rec_sum2 / _rec_cnt) - _chan_mean ** 2
            _chan_var  = _chan_var.clamp(min=0.0)  # 防止浮点负值
            chan_w = 1.0 / (_chan_var + 1e-8)
            chan_w = chan_w / chan_w.sum()  # 归一化为概率权重，形状 (C,)
            top5_idx = chan_w.topk(5).indices.tolist()
            print(f"[chan_var] 样本数={_rec_cnt}, 通道数={chan_w.shape[0]}")
            print(f"[chan_var] 权重: min={chan_w.min().item():.6f},  "
                  f"max={chan_w.max().item():.6f},  判别力最强5通道={top5_idx}")

        # (1) stastic on the train set
        attens_energy = []
        for i, batch in enumerate(self.train_loader):
            input_data = batch[0]
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            if _mb_prior is not None:
                prior = [p.expand(input.shape[0], -1, -1, -1) for p in _mb_prior]
            _rec = criterion(input, output)
            if _smode == 'chan_var':
                loss = (_rec * chan_w.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            elif _smode == 'rec_mean':
                loss = _rec.mean(dim=-1)
            else:
                loss = torch.max(_rec, dim=-1).values
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            kl_score = series_loss + prior_loss
            if _smode in ('rec_only', 'rec_mean', 'chan_var'):
                score = loss
            elif _smode == 'weighted':
                score = _salpha * loss + (1.0 - _salpha) * kl_score
            else:
                score = kl_score + loss
            if _diff_beta > 0:
                _dr = criterion(input[:, 1:, :] - input[:, :-1, :],
                                output[:, 1:, :] - output[:, :-1, :])
                _dr = torch.cat([torch.zeros(_dr.shape[0], 1, _dr.shape[2], device=input.device), _dr], dim=1)
                score = score + _diff_beta * torch.max(_dr, dim=-1).values
            cri = score.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        if _smooth > 1:
            train_energy = np.convolve(train_energy, np.ones(_smooth) / _smooth, mode='same')
        if _local_z_win > 1:
            train_energy = _local_zscore(train_energy, _local_z_win)

        # (2) find the threshold
        if _threshold_mode in ('val_percentile', 'val_grid'):
            threshold_loader = self.vali_loader
            threshold_source = 'val'
        elif _threshold_mode == 'oracle_ratio':
            threshold_loader = self.thre_loader
            threshold_source = 'test'
        else:
            threshold_loader = None
            threshold_source = 'train'

        attens_energy = []
        threshold_labels = []
        if threshold_loader is not None:
            for i, batch in enumerate(threshold_loader):
                input_data = batch[0]
                labels = batch[1]
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                if _mb_prior is not None:
                    prior = [p.expand(input.shape[0], -1, -1, -1) for p in _mb_prior]

                _rec = criterion(input, output)
                if _smode == 'chan_var':
                    loss = (_rec * chan_w.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
                elif _smode == 'rec_mean':
                    loss = _rec.mean(dim=-1)
                else:
                    loss = torch.max(_rec, dim=-1).values

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                kl_score = series_loss + prior_loss
                if _smode in ('rec_only', 'rec_mean', 'chan_var'):
                    score = loss
                elif _smode == 'weighted':
                    score = _salpha * loss + (1.0 - _salpha) * kl_score
                else:
                    score = kl_score + loss
                if _diff_beta > 0:
                    _dr = criterion(input[:, 1:, :] - input[:, :-1, :],
                                    output[:, 1:, :] - output[:, :-1, :])
                    _dr = torch.cat([torch.zeros(_dr.shape[0], 1, _dr.shape[2], device=input.device), _dr], dim=1)
                    score = score + _diff_beta * torch.max(_dr, dim=-1).values
                cri = score.detach().cpu().numpy()
                attens_energy.append(cri)
                threshold_labels.append(labels)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            threshold_energy = np.array(attens_energy)
            if _smooth > 1:
                threshold_energy = np.convolve(threshold_energy, np.ones(_smooth) / _smooth, mode='same')
            if _local_z_win > 1:
                threshold_energy = _local_zscore(threshold_energy, _local_z_win)
            threshold_labels = np.concatenate(threshold_labels, axis=0).reshape(-1).astype(int)
        else:
            threshold_energy = train_energy
            threshold_labels = np.zeros_like(threshold_energy, dtype=int)

        threshold_details = {
            'mode': _threshold_mode,
            'source': threshold_source,
            'uses_test_scores': bool(_threshold_mode == 'oracle_ratio'),
            'uses_test_labels': False,
            'uses_test_anomaly_ratio': bool(_threshold_mode == 'oracle_ratio'),
        }
        if _threshold_mode == 'train_percentile':
            thresh = np.percentile(train_energy, _threshold_percentile)
            threshold_details['percentile'] = _threshold_percentile
        elif _threshold_mode == 'val_percentile':
            thresh = np.percentile(threshold_energy, _threshold_percentile)
            threshold_details['percentile'] = _threshold_percentile
        elif _threshold_mode == 'val_grid':
            from sklearn.metrics import f1_score
            if threshold_labels.sum() == 0:
                thresh = np.percentile(threshold_energy, _threshold_percentile)
                threshold_details['fallback'] = 'val_labels_have_no_positive_points'
                threshold_details['percentile'] = _threshold_percentile
            else:
                best_f1 = -1.0
                best_percentile = _threshold_grid_min
                best_thresh = np.percentile(threshold_energy, best_percentile)
                percentiles = np.arange(
                    _threshold_grid_min,
                    _threshold_grid_max + 1e-9,
                    _threshold_grid_step,
                )
                for percentile in percentiles:
                    candidate = np.percentile(threshold_energy, percentile)
                    candidate_pred = (threshold_energy > candidate).astype(int)
                    candidate_f1 = f1_score(threshold_labels, candidate_pred, zero_division=0)
                    if candidate_f1 > best_f1:
                        best_f1 = candidate_f1
                        best_percentile = float(percentile)
                        best_thresh = candidate
                thresh = best_thresh
                threshold_details['best_val_f1'] = float(best_f1)
                threshold_details['percentile'] = float(best_percentile)
                threshold_details['uses_test_labels'] = bool(threshold_source == 'test')
        else:
            combined_energy = np.concatenate([train_energy, threshold_energy], axis=0)
            legacy_percentile = 100 - self.anormly_ratio
            thresh = np.percentile(combined_energy, legacy_percentile)
            threshold_details['percentile'] = float(legacy_percentile)
            threshold_details['anormly_ratio'] = float(self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, batch in enumerate(self.thre_loader):
            input_data = batch[0]
            labels = batch[1]
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            if _mb_prior is not None:
                prior = [p.expand(input.shape[0], -1, -1, -1) for p in _mb_prior]

            _rec = criterion(input, output)
            if _smode == 'chan_var':
                loss = (_rec * chan_w.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            elif _smode == 'rec_mean':
                loss = _rec.mean(dim=-1)
            else:
                loss = torch.max(_rec, dim=-1).values

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            kl_score = series_loss + prior_loss
            if _smode in ('rec_only', 'rec_mean', 'chan_var'):
                score = loss
            elif _smode == 'weighted':
                score = _salpha * loss + (1.0 - _salpha) * kl_score
            else:
                score = kl_score + loss
            if _diff_beta > 0:
                _dr = criterion(input[:, 1:, :] - input[:, :-1, :],
                                output[:, 1:, :] - output[:, :-1, :])
                _dr = torch.cat([torch.zeros(_dr.shape[0], 1, _dr.shape[2], device=input.device), _dr], dim=1)
                score = score + _diff_beta * torch.max(_dr, dim=-1).values

            cri = score.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        if _smooth > 1:
            test_energy = np.convolve(test_energy, np.ones(_smooth) / _smooth, mode='same')
        if _local_z_win > 1:
            test_energy = _local_zscore(test_energy, _local_z_win)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        pred_raw = pred.copy()  # PA 调整前，用于逐点指标和 AUPRC

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average='binary', zero_division=0)
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        # ---- 补充评估指标（无需 PA 的更严格视角）----
        from utils.eval_metrics import pointwise_metrics, compute_auprc, event_level_metrics
        pw = pointwise_metrics(gt, pred_raw)
        print("[逐点(无PA)] Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
            pw['precision'], pw['recall'], pw['f1']))
        auprc = None
        try:
            auprc = compute_auprc(gt, test_energy)
            print("[AUPRC]      {:.4f}  (不依赖阈值，越高越好)".format(auprc))
        except Exception:
            pass
        ev = event_level_metrics(gt, pred)
        print("[事件级(PA)] Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
            ev['event_precision'], ev['event_recall'], ev['event_f1']))

        # ---- 窗口级评估（固定窗口数据集专用，如 ST330IR001_CP001）----
        # 每条样本本身就是一个完整窗口，点级 AUPRC 噪声很大（窗口内各步骤得分不同）。
        # 窗口级评估：对窗口内所有步骤的异常分数取均值 → 1 个窗口分 → 与窗口标签比较。
        window_metrics = {}
        win_scores = None
        win_labels = None
        win_pred = None
        if False and self.win_size > 1 and len(test_energy) % self.win_size == 0:
            n_win = len(test_energy) // self.win_size
            win_scores = test_energy.reshape(n_win, self.win_size).mean(axis=1)
            win_labels = test_labels.reshape(n_win, self.win_size).max(axis=1).astype(int)
            n_anom_win = win_labels.sum()
            if 0 < n_anom_win < len(win_labels):
                try:
                    win_auprc = compute_auprc(win_labels, win_scores)
                    print("[窗口级AUPRC] {:.4f}  (n_windows={}, anomaly_windows={})".format(
                        win_auprc, n_win, int(n_anom_win)))
                    # 窗口级 F1：阈值取与 anormly_ratio 一致的百分位
                    win_thresh = np.percentile(win_scores, 100 - self.anormly_ratio)
                    win_pred = (win_scores > win_thresh).astype(int)
                    win_pw = pointwise_metrics(win_labels, win_pred)
                    print("[窗口级F1]   Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
                        win_pw['precision'], win_pw['recall'], win_pw['f1']))
                    window_metrics = {
                        'n_windows': int(n_win),
                        'anomaly_windows': int(n_anom_win),
                        'threshold': float(win_thresh),
                        'auprc': float(win_auprc),
                        'precision': float(win_pw['precision']),
                        'recall': float(win_pw['recall']),
                        'f1': float(win_pw['f1']),
                    }
                except Exception:
                    pass

        resolved_eval_unit = resolve_eval_unit(getattr(self, 'dataset', ''), getattr(self, 'eval_unit', 'auto'))
        if resolved_eval_unit == 'window':
            if getattr(self, 'window_threshold_mode', 'val_percentile') != 'val_percentile':
                raise ValueError("window evaluation currently supports only window_threshold_mode=val_percentile")
            try:
                wm = window_level_metrics_from_scores(
                    test_scores=test_energy,
                    test_labels=test_labels,
                    val_scores=threshold_energy,
                    win_size=int(self.win_size),
                    percentile=float(getattr(self, 'threshold_percentile', 95.0)),
                    agg=getattr(self, 'window_score_agg', 'mean'),
                    topk=int(getattr(self, 'window_score_topk', 5)),
                )
                win_scores = wm.pop('_scores')
                win_labels = wm.pop('_labels')
                win_pred = wm.pop('_pred')
                window_metrics = wm
                print("[Window AUPRC] {:.4f}  (n_windows={}, anomaly_windows={})".format(
                    window_metrics['auprc'], window_metrics['n_windows'], window_metrics['anomaly_windows']))
                print("[Window F1]   Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
                    window_metrics['precision'], window_metrics['recall'], window_metrics['f1']))
            except Exception as exc:
                raise RuntimeError(
                    "Window-level evaluation failed for dataset={} eval_unit={} win_size={}.".format(
                        getattr(self, 'dataset', None), resolved_eval_unit, self.win_size
                    )
                ) from exc

        metrics = {
            'dataset': getattr(self, 'dataset', None),
            'experiment_name': self._experiment_name(),
            'seed': getattr(self, 'seed', None),
            'checkpoint_path': checkpoint_path,
            'threshold': float(thresh),
            'threshold_protocol': _threshold_mode,
            'threshold_details': threshold_details,
            'threshold_percentile': threshold_details.get('percentile'),
            'temperature': float(temperature),
            'score_mode': _smode,
            'score_alpha': _salpha,
            'score_smooth_k': _smooth,
            'diff_beta': _diff_beta,
            'score_local_z_win': _local_z_win,
            'eval_unit': resolved_eval_unit,
            'window_score_agg': getattr(self, 'window_score_agg', 'mean'),
            'window_threshold_mode': getattr(self, 'window_threshold_mode', 'val_percentile'),
            'use_memory_bank': bool(_use_mb),
            'n_points': int(len(gt)),
            'n_anomaly_points': int(gt.sum()),
            'anomaly_ratio': float(gt.mean()) if len(gt) else 0.0,
            'pa': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f_score),
            },
            'pointwise': {
                'accuracy': float(pw['accuracy']),
                'precision': float(pw['precision']),
                'recall': float(pw['recall']),
                'f1': float(pw['f1']),
            },
            'ranking': {
                'auprc': None if auprc is None else float(auprc),
            },
            'event_level': {
                'precision': float(ev['event_precision']),
                'recall': float(ev['event_recall']),
                'f1': float(ev['event_f1']),
            },
            'window_level': window_metrics,
            'runtime_sec': float(time.time() - test_start_time),
            'model': self._model_complexity(),
        }
        arrays = {
            'scores': test_energy,
            'labels': gt,
            'pred_point': pred_raw,
            'pred_pa': pred,
            'train_scores': train_energy,
            'threshold_scores': threshold_energy,
            'threshold_labels': threshold_labels,
        }
        if win_scores is not None:
            arrays['window_scores'] = win_scores
        if win_labels is not None:
            arrays['window_labels'] = win_labels
        if win_pred is not None:
            arrays['window_pred'] = win_pred
        self._write_test_outputs(metrics, arrays)

        return accuracy, precision, recall, f_score
