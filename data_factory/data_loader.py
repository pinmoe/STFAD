import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


def _split_train_val(data, val_ratio=0.2):
    if len(data) <= 1:
        return data, data
    n_val = max(1, int(len(data) * val_ratio))
    if n_val >= len(data):
        n_val = 1
    return data[:-n_val], data[-n_val:]


def _zero_window_label(win_size):
    return np.zeros((win_size, 1), dtype=np.float32)


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train, self.val = _split_train_val(data)
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), _zero_window_label(self.win_size)
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), _zero_window_label(self.win_size)
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train, self.val = _split_train_val(data)
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), _zero_window_label(self.win_size)
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), _zero_window_label(self.win_size)
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class HAISegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/HAI_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/HAI_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/HAI_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SKABSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(data_path + "/SKAB_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = np.load(data_path + "/SKAB_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train, self.val = _split_train_val(data)
        self.test_labels = np.load(data_path + "/SKAB_test_label.npy")

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), \
                   _zero_window_label(self.win_size)
        elif self.mode == "val":
            return np.float32(self.val[index:index + self.win_size]), \
                   _zero_window_label(self.win_size)
        elif self.mode == "test":
            return np.float32(self.test[index:index + self.win_size]), \
                   np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                index // self.step * self.win_size:
                index // self.step * self.win_size + self.win_size]), \
                   np.float32(self.test_labels[
                index // self.step * self.win_size:
                index // self.step * self.win_size + self.win_size])

class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])




class BATADALSegLoader(object):
    """
    BATADAL 水务工控系统数据集 Loader。

    文件依赖（由 prepare_batadal.py 生成）：
      {data_path}/BATADAL_train.npy       shape (8761, 43)  float32
      {data_path}/BATADAL_test.npy        shape (4177, 43)  float32
      {data_path}/BATADAL_test_label.npy  shape (4177,)     int32  0/1

    训练集 = dataset03（纯正常数据）
    测试集 = dataset04（ATT_FLAG: -999 → 0 正常, 1 → 1 攻击）
    val 指向 test，与 PSM/MSL/HAI 保持一致。
    """

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = np.load(data_path + "/BATADAL_train.npy")
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(data_path + "/BATADAL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = train_data
        self.val = self.test
        self.test_labels = np.load(
            data_path + "/BATADAL_test_label.npy"
        ).reshape(-1, 1).astype(np.float32)

        print("train:", self.train.shape)
        print("test: ", self.test.shape)
        print(f"test anomaly ratio: {self.test_labels.mean()*100:.2f}%")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (
                np.float32(self.train[index : index + self.win_size]),
                np.float32(self.test_labels[0 : self.win_size]),
            )
        elif self.mode == "val":
            return (
                np.float32(self.val[index : index + self.win_size]),
                np.float32(self.test_labels[0 : self.win_size]),
            )
        elif self.mode == "test":
            return (
                np.float32(self.test[index : index + self.win_size]),
                np.float32(self.test_labels[index : index + self.win_size]),
            )
        else:
            base = index // self.step * self.win_size
            return (
                np.float32(self.test[base : base + self.win_size]),
                np.float32(self.test_labels[base : base + self.win_size]),
            )


class ST330IR001CP001SegLoader(object):
    """
    Loader for the ST330IR001.CP001 welding-gun fault dataset.

    The raw dataset is organized as one fixed-length window per CSV file.
    scripts/prepare_st330ir001_cp001.py converts it to:
      {data_path}/ST330IR001_CP001_train.npy       shape (N_train, 56, 29)
      {data_path}/ST330IR001_CP001_test.npy        shape (N_test, 56, 29)
      {data_path}/ST330IR001_CP001_test_label.npy  shape (N_test, 56, 1)
    """

    prefix = "ST330IR001_CP001"

    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = np.load(os.path.join(data_path, self.prefix + "_train.npy")).astype(np.float32)
        test_data = np.load(os.path.join(data_path, self.prefix + "_test.npy")).astype(np.float32)
        test_labels = np.load(os.path.join(data_path, self.prefix + "_test_label.npy")).astype(np.float32)

        if train_data.ndim != 3 or test_data.ndim != 3:
            raise ValueError("ST330IR001_CP001 arrays must have shape (N, win_size, C).")
        if test_labels.ndim == 2:
            test_labels = test_labels[:, :, None]
        if test_labels.shape[:2] != test_data.shape[:2]:
            raise ValueError("ST330IR001_CP001 labels must match test data windows and length.")
        if train_data.shape[1] != win_size or test_data.shape[1] != win_size:
            raise ValueError(
                f"ST330IR001_CP001 uses fixed windows of {train_data.shape[1]} rows. "
                f"Run with --win_size {train_data.shape[1]}."
            )

        n_features = train_data.shape[-1]
        self.scaler.fit(train_data.reshape(-1, n_features))
        train_scaled = self.scaler.transform(train_data.reshape(-1, n_features)).reshape(train_data.shape)
        self.train, self.val = _split_train_val(train_scaled)
        self.test = self.scaler.transform(test_data.reshape(-1, n_features)).reshape(test_data.shape)
        self.test_labels = test_labels

        print("train:", self.train.shape)
        print("test: ", self.test.shape)
        print(f"test anomaly ratio: {self.test_labels.mean()*100:.2f}%")

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        if self.mode == "val":
            return self.val.shape[0]
        return self.test.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            return (
                np.float32(self.train[index]),
                np.zeros((self.win_size, 1), dtype=np.float32),
            )
        if self.mode == "val":
            return (
                np.float32(self.val[index]),
                np.zeros((self.win_size, 1), dtype=np.float32),
            )
        return (
            np.float32(self.test[index]),
            np.float32(self.test_labels[index]),
        )


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'HAI'):
        dataset = HAISegLoader(data_path, win_size, 1, mode)
    elif dataset == 'SKAB':
        dataset = SKABSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'BATADAL':
        dataset = BATADALSegLoader(data_path, win_size, 1, mode)
    elif dataset in ('ST330IR001_CP001', 'ST330IR001.CP001'):
        dataset = ST330IR001CP001SegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
