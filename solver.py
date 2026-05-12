import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment


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

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            e_layers=3,
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

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

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

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)
        # 测试评分策略参数（无需重训练）
        _smode  = getattr(self, 'score_mode',    'combined')
        _salpha = float(getattr(self, 'score_alpha',   1.0))
        _smooth = int(getattr(self, 'score_smooth_k',  1))
        if _smode != 'combined' or _smooth > 1:
            print(f"[评分策略] mode={_smode},  alpha={_salpha},  smooth_k={_smooth}")

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
            cri = score.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        if _smooth > 1:
            train_energy = np.convolve(train_energy, np.ones(_smooth) / _smooth, mode='same')

        # (2) find the threshold
        attens_energy = []
        for i, batch in enumerate(self.thre_loader):
            input_data = batch[0]
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

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
            # Metric
            kl_score = series_loss + prior_loss
            if _smode in ('rec_only', 'rec_mean', 'chan_var'):
                score = loss
            elif _smode == 'weighted':
                score = _salpha * loss + (1.0 - _salpha) * kl_score
            else:
                score = kl_score + loss
            cri = score.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        if _smooth > 1:
            test_energy = np.convolve(test_energy, np.ones(_smooth) / _smooth, mode='same')
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, batch in enumerate(self.thre_loader):
            input_data = batch[0]
            labels = batch[1]
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

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

            cri = score.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        if _smooth > 1:
            test_energy = np.convolve(test_energy, np.ones(_smooth) / _smooth, mode='same')
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)

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
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        return accuracy, precision, recall, f_score