# coding=utf-8
'''
    Created on 06.09.2019
    
    @author: Qianhui Liu
    '''

import random
import os
import scipy.io as scio
import h5py
import collections
import pickle
import numpy as np
import bisect

'''
SpikeLoader: 用于读取、预处理和随机选择训练集和测试集
             input: fname[文件名](格式：pkl(Addr, Time, Label))
                    data_size[数据集大小]
                    tr_rate[训练数据集大小/数据集大小]
                    prepared[是否已做预处理]
                    max_time[最大时间]
                    max_addr[最大位置]
                    seed[随机种子，用于选择训练和测试数据集]
             
func: get, 用于获取训练集和测试集，output: TrainData(Addr(n, addr), Time, Label), TestData(Addr, Time, Label)
      dump，存储预处理后的数据集（pkl格式）
      _prepare，用于预处理数据集，包括去除大于max_time的脉冲、排序
      _load, 读取数据集, _load_from_pkl
      _sample, 用于选择训练集和测试集
usage: loader = SpikeLoader(fname)
       (tr_addr, tr_time, tr_label), (te_addr, te_time, te_label) = loader.get()
       option: loader.dump(fpath)
'''
class SpikeLoader(object):
    def __init__(self, fname, data_size=0, tr_rate=0.9, max_time=1., noise=0.0, max_addr=3136, prepared=False, seed=123):
        if seed is not None and type(seed) == int:
            random.seed(seed)
        self._fname = fname
        self._data_size = data_size
        self._tr_rate = tr_rate
        self._max_time = max_time
        self._noise = noise
        self._max_addr = max_addr
        self._data = self._load()
        if not prepared:
            self._data = self._prepare(self._data)

    def get(self):
        return self._sample(self._data, self._data_size, self._tr_rate)

    def _prepare(self, data):
        def prepare(addr, time):
            time = [t - time[0] for t in time]
        #    time, addr = zip(*sorted(zip(time, addr)))
            max_time_idx = bisect.bisect_left(time, self._max_time)
            addr, time = addr[:max_time_idx], time[:max_time_idx]
            time, addr = np.array(time), np.array(addr, dtype=np.int32)
            return addr, time

        Addr, Time, Label = data
        for i in range(len(Addr)):
            addr, time = prepare(Addr[i], Time[i])
            Addr[i], Time[i] = addr, time
        return Addr, Time, Label

    def _load(self):
        assert os.path.exists(self._fname)
        if self._fname.endswith('.mat'):
            return self._load_from_mat(self._fname)
        elif self._fname.endswith('.pkl'):
            return self._load_from_pkl(self._fname)
        raise ValueError('Unexpected spike data file: {}'.format(self._fname))

    def _sample(self, data, size, tr_rate):
        addr, time, y = data
        size = len(y) if size <= 0 or size > len(y) else size

        y_counter = collections.defaultdict(list)
        tr_idx, te_idx = [], []
        for i, yi in enumerate(y):
            y_counter[yi].append(i)
        for yi in y_counter:
            cur = y_counter[yi]
            sampled = random.sample(cur, int(len(cur)*size*1./len(y)))
            tr_idx += sampled[:int(len(sampled)*tr_rate)]
            te_idx += sampled[int(len(sampled)*tr_rate):]
        tr_addr, tr_time, tr_y = [addr[idx] for idx in tr_idx], \
                                 [time[idx] for idx in tr_idx], \
                                 [y[idx] for idx in tr_idx]
        te_addr, te_time, te_y = [addr[idx] for idx in te_idx], \
                                 [time[idx] for idx in te_idx], \
                                 [y[idx] for idx in te_idx]
        return (tr_addr, tr_time, tr_y), (te_addr, te_time, te_y)

    def _load_from_pkl(self, fname):
        with open(fname, 'r') as f:
            data = pickle.load(f)
            if len(data) != 3:
                tr_addr, tr_time, tr_y, te_addr, te_time, te_y = data
                addr = tr_addr + te_addr
                time = tr_time + te_time
                y = tr_y + te_y
            else:
                addr, time, y = data
            return addr, time, y

    def dump(self, fname):
        with open(fname, 'w') as f:
            pickle.dump(self._data, f)
