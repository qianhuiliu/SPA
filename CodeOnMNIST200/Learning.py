# coding=utf-8
'''
    Created on 06.09.2019
    
    @author: Qianhui Liu
    '''

import numpy as np
import pickle
import random
import bisect
import os
import matplotlib.pyplot as plt



'''
basic Tempotron，用于训练和测试
func: train, 用于训练, input: Addr(n_samples, addr((n,), type=np.array)), Time, 
                             Label(n_samples, label(type=int))
      test, 用于测试， input: Addr, Time, Label, multi[输出神经元衰减形式]，
                       output: Accuracy, True rate, True Positive, False Positive
                          usage: model = Tempotorn.load(fname)
      _update_weights, 用于更新权重，input: addr((n,), type=np.array)[位置], 
                                           time((n,), type=np.array)[排序时间], label(type=int)[类别]
                                    usage: 虚函数
      _simulate, 用于模拟电压变化，input: addr, time, multi, inhibition(type=float)[抑制时长]
                                  output: V(n_spikes, n_class, n_neuron_per_class)[各个脉冲时间点输出神经元膜电位]
      _simulate_fires, 用于模拟脉冲发放, input: addr, time, multi, inhibition(type=float)[抑制时长]
                                  output: fires(n_class, n_neuron_per_class)[各个输出神经元脉冲发放数目]
      _backward_event_finder，寻找峰值时间点
'''
class Tempotron(object):
    def __init__(self, tau, tau_m, n_in, n_class, n_neuron_per_class,
                 weight_variance=0.1, weight_offset=0, seed=123,
                 end=1.0, lmd=1e-2, threshold=1.0, min_event=10,
                 inhibition=0.0, **kwargs):
        #basic
        self._weight_variance = weight_variance
        self._weight_offset = weight_offset
        self._seed = seed
        self._lmd = lmd
        self._min_event = min_event
        self._inhibition = inhibition

        #network architecture
        self._tau = tau
        self._tau_m = tau_m
        self._n_in = n_in
        self._n_class = n_class
        self._n_neuron_per_class = n_neuron_per_class
        self._threshold = threshold * np.ones((self._n_class, self._n_neuron_per_class))
        self._init_weight()
        self._coe = 2 ##########################

    def _init_weight(self):
        np.random.seed(self._seed)
        self._Weights = self._weight_variance * np.random.randn\
            (self._n_in, self._n_class, self._n_neuron_per_class) + self._weight_offset
#	print self._Weights[200][1][0:9]
        # np.random.randn生成0为均值，1为标准差的正态分布。这里是生成均值为weight_offset,标准差为weight_variance的权重
    def _is_single_tau(self):
        return not self._tau_m or self._tau_m <= 0

    def _V_leak1(self, t):
        return np.where(t >= 0, np.exp(-1. * t / self._tau), 1)

    def _V_leak2(self, t):
        return np.where(t >= 0, np.exp(-1. * t / self._tau_m), 1)

    def _K(self, t):
        if self._is_single_tau(): return self._V_leak1(t)
        else: return (self._V_leak1(t) - self._V_leak2(t)) * self._coe

    def _simulate_single(self, addr, time):
        number_of_spike = len(addr)
        v = np.zeros((number_of_spike, self._n_class, self._n_neuron_per_class), dtype=np.float32)
        v[0] = self._Weights[addr[0]]
        for i in range(1, number_of_spike):
            delta_t = time[i] - time[i - 1]
            delta_v = self._Weights[addr[i]]
            v[i] = v[i - 1] * self._V_leak1(delta_t) + delta_v
        return v

    def _simulate_fires_single(self, addr, time, inhibition_time=0.):
        number_of_spike = len(addr)
        fires = np.zeros((number_of_spike, self._n_class, self._n_neuron_per_class), dtype=np.int32)
        inhibition = np.zeros((self._n_class, self._n_neuron_per_class), dtype=np.int32)
        v = self._Weights[addr[0]]
        for i in range(1, number_of_spike):
            delta_t = time[i] - time[i - 1]
            delta_v = self._Weights[addr[i]]
            inhibition = np.maximum(inhibition - delta_t, 0)
            v = v * self._V_leak1(delta_t) + (inhibition <= 0) * delta_v
            fires[i] = (v >= self._threshold)
            inhibition = np.where(v >= self._threshold, inhibition_time, inhibition)
            v = np.where(v >= self._threshold, 0, v)
        return fires

    def _simulate_multi(self, addr, time):
        number_of_spike = len(addr)
        v_lut1 = np.zeros((number_of_spike, self._n_class, self._n_neuron_per_class), dtype=np.float32)
        v_lut2 = np.zeros((number_of_spike, self._n_class, self._n_neuron_per_class), dtype=np.float32)
        v_lut1[0] = v_lut2[0] = self._Weights[addr[0]]
        for i in range(1, number_of_spike):
            delta_t = time[i] - time[i - 1]
            delta_v = self._Weights[addr[i]] if addr[i] >= 0 else 0
            v_lut1[i] = v_lut1[i - 1] * self._V_leak1(delta_t) + delta_v
            v_lut2[i] = v_lut2[i - 1] * self._V_leak2(delta_t) + delta_v
        return (v_lut1 - v_lut2) * self._coe

    def _simulate_fires_multi(self, addr, time, inhibition_time=0.):
        number_of_spike = len(addr)
        fires = np.zeros((number_of_spike, self._n_class, self._n_neuron_per_class), dtype=np.int32)
        inhibition = np.zeros((self._n_class, self._n_neuron_per_class), dtype=np.int32)
        v_lut1 = self._Weights[addr[0]]
        v_lut2 = self._Weights[addr[0]]
        for i in range(1, number_of_spike):
            delta_t = time[i] - time[i - 1]
            delta_v = self._Weights[addr[i]] * self._coe if addr[i] >= 0 else 0
            inhibition = np.maximum(inhibition - delta_t, 0)
            v_lut1 = v_lut1 * self._V_leak1(delta_t) + (inhibition <= 0) * delta_v
            v_lut2 = v_lut2 * self._V_leak2(delta_t) + (inhibition <= 0) * delta_v
            cfire = (v_lut1 - v_lut2) >= self._threshold
            fires[i] = cfire
            inhibition = np.where(cfire, inhibition_time, inhibition)
            v_lut1 = np.where(cfire, 0, v_lut1)
            v_lut2 = np.where(cfire, 0, v_lut2)
        return fires

    def _simulate(self, addr, time, is_single_tau = None):
        if is_single_tau is None: is_single_tau = self._is_single_tau()
        if is_single_tau: return self._simulate_single(addr, time)
        else: return self._simulate_multi(addr, time)

    def _simulate_fires(self, addr, time, multi=True, is_single_tau = None, inhibition = None):
        if not multi:
            return np.max(self._simulate(addr, time), axis=0) >= self._threshold
        if is_single_tau is None: is_single_tau = self._is_single_tau()
        if inhibition is None: inhibition = self._inhibition
        if is_single_tau:
            fires = self._simulate_fires_single(addr, time, inhibition_time=inhibition)
        else:
            fires = self._simulate_fires_multi(addr, time, inhibition_time=inhibition)
        return np.sum(fires, axis=0)

    def _backward_event_finder(self, addr, time, get_fire_time=True):
        def _find_last_event(addr):
            real_event_addr = np.where(addr >= 0)[0]
            return real_event_addr[-1] if len(real_event_addr) > 0 else len(addr) - 1

        def _find_fire_event(v):
            fire_event = -1 * np.ones((self._n_class, self._n_neuron_per_class), dtype=np.int32)
            v_max = np.max(v, axis=0)
            for i in range(self._n_class):
                for j in range(self._n_neuron_per_class):
                    if v_max[i, j] >= self._threshold[i, j]:
                        fire_event[i, j] = np.where(v[:, i, j] >= self._threshold[i, j])[0][0]
            return fire_event

        v = self._simulate(addr, time)
        is_fire = np.max(v, axis=0) >= self._threshold
        if get_fire_time:
            idx = np.where(is_fire, _find_fire_event(v), _find_last_event(addr))
        else:
            idx = np.where(is_fire, np.argmax(v, axis=0), _find_last_event(addr))
        vmax = np.max(v, axis=0)
        return vmax, idx

    def _update_weights(self, addr, time, label):
        if len(addr) < self._min_event: return
        v, idx = self._backward_event_finder(addr, time)
        delta = -1 * (v >= self._threshold)
        delta[label] += 1
        for k in range(np.max(idx)):
            if addr[k] >= 0:
                self._Weights[addr[k], :, :] += self._lmd * self._K(time[idx] - time[k]) * delta

    def train(self, Addr, Time, Label):
        number_of_samples = len(Addr)
        order = random.sample(range(number_of_samples), number_of_samples)
	for ii, i in enumerate(order):
            self._update_weights(Addr[i], Time[i], Label[i])

    def test(self, Addr, Time, Label, multi=False):
        def vote(fires, label):
            fcnt = np.sum(fires, axis=1)
            available = np.where(fcnt == np.max(fcnt))[0]
            return 0. if label not in available else 1./len(available)

        number_of_samples = len(Addr)
        Accuracy = []
        #T, TP, FP = [], [], []
        for i in range(number_of_samples):
            addr, time, label = Addr[i], Time[i], Label[i]
            if len(addr) < self._min_event: continue
            try:
                fires = self._simulate_fires(addr, time, multi=multi)
                # tp = np.sum(fires[label] > 0) * 1./ self._n_neuron_per_class
                # t = np.sum(fires > 0) * 1./ (self._n_neuron_per_class * self._n_class)
                # fp = (t * (self._n_class) - tp) / (self._n_class - 1)
                acc = vote(fires, label)
                Accuracy.append(acc)
                # T.append(t)
                # TP.append(tp)
                # FP.append(fp)
            except:
                Accuracy.append(1./self._n_class)
                continue
        return np.mean(Accuracy)#, np.mean(T), np.mean(TP), np.mean(FP)

    def save(self, fname):
        assert isinstance(fname, str)
        dump_dict = {'tau': self._tau, 'tau_m': self._tau_m, 'n_in': self._n_in,
                     'n_class': self._n_class, 'n_neuron_per_class': self._n_neuron_per_class,
                     'weight_variance': self._weight_variance, 'weight_offset': self._weight_offset,
                     'seed': self._seed,  'lmd': self._lmd,
                     'threshold': self._threshold, 'min_event': self._min_event,
                     'inhibition': self._inhibition, 'Weights': self._Weights}
        with open(fname, 'wb') as f:
            pickle.dump(dump_dict, f)

    @classmethod
    def load(cls, fname):
        assert isinstance(fname, str)
        assert os.path.exists(fname)
        with open(fname, 'r') as f:
            load_dict = pickle.load(f)
            tau = load_dict['tau']
            tau_m = load_dict['tau_m']
            n_in = load_dict['n_in']
            n_class = load_dict['n_class']
            n_neuron_per_class = load_dict['n_neuron_per_class']
            weight_variance = load_dict['weight_variance']
            weight_offset = load_dict['weight_offset']
            seed = load_dict['seed']
            lmd = load_dict['lmd']
            threshold = load_dict['threshold']
            min_event = load_dict['min_event']
            inhibition = load_dict['inhibition']
            Weights = load_dict['Weights']
            print(tau, tau_m, n_in, n_class, n_neuron_per_class, weight_variance, weight_offset)
            method = cls(tau=tau, tau_m=tau_m, n_in=n_in, n_class=n_class,
                         n_neuron_per_class=n_neuron_per_class, weight_variance=weight_variance,
                         weight_offset=weight_offset, seed=seed, end=end, lmd=lmd,
                         threshold=threshold, min_event=min_event, inhibition=inhibition)
            method._Weights = Weights
        return method


'''
SegmentAdaptiveTempotron: 继承与AdaptiveTempotron， 第四章学习算法
func: _update_weights: 用于更新权重
'''
class SPA(Tempotron):
    def __init__(self, tau, tau_m, n_in, n_class, n_neuron_per_class,
                 weight_variance=0.1, weight_offset=0, seed=123,
                 end=1.0, lmd=1e-2, threshold=1.0, min_event=10,
                 inhibition=0.0, alpha=0.1, update_step=80000, update_tau=80000, **kwargs):
        super(SPA, self).__init__(tau=tau, tau_m=tau_m, n_in=n_in,
                                                n_class=n_class, n_neuron_per_class=n_neuron_per_class,
                                                weight_variance=weight_variance, weight_offset=weight_offset,
                                                seed=seed, end=end, lmd=lmd, threshold=threshold,
                                                min_event=min_event, inhibition=inhibition,
                                                **kwargs)
        self._lmd *= 5
        self._alpha = alpha
        self._update_step = update_step
        self._update_tau = update_tau

    def _func(self, v):
        return np.where(v >= 10, v, np.where(v <= -20, 0, np.log(np.exp(v) + 1)))

    def _func_dirivative(self, v):
        return np.where(v <= -10, 0., np.where(v >= 20, 1, 1 / (np.exp(-v) + 1)))

    def _update_weights_v(self, v, addr, time, label, l_idx, r_idx):
        fv = self._func(v)
        dv = self._func_dirivative(v)
        fv_sum = np.sum(fv, axis=0, keepdims=True)

        delta = (-1. / (fv_sum + 1e-8)) * np.ones((self._n_class, self._n_neuron_per_class), dtype=np.float32)
        delta[label] += 1. / (fv[label] + 1e-8)
        delta *= dv

        delta = delta *(1 - self._alpha) - self._alpha * (v >= self._threshold)
        delta[label] += self._alpha

        for i in range(self._n_neuron_per_class):
            li = l_idx[0, i]
            ri = r_idx[:, i]
            for k in range(np.max(ri - li)):
                ll = li + k
                if addr[ll] >= 0:
                    self._Weights[addr[ll], :, i] += self._lmd * self._K(time[ri] - time[ll]) * delta[:,i]

    def _update_weights(self, addr, time, label):
        if len(addr) < self._min_event: return
        v = self._simulate(addr, time)

        l_idx= np.zeros((self._n_class, self._n_neuron_per_class), dtype=np.int32)
        r_idx = np.zeros((self._n_class, self._n_neuron_per_class), dtype=np.int32)
        length = len(time)
        while np.max(l_idx) < length:
            r_t = time[l_idx] + self._update_step
            for i in range(self._n_neuron_per_class):
                ri = bisect.bisect_right(time, r_t[0, i])#查找r_t[0, i]将会插入time的位置并返回,当有重复值时插到右边
                li = l_idx[0, i]
                r_idx[:, i] = np.argmax(v[li:ri, :, i], axis=0) + li#argmax返回每列最大的数字对应的行编号,即每个神经元在第li到ri的输入事件下电压最大是什么时候
                #r_idx表示在第li个事件到第ri个事件的刺激下，每个label的第i个分类神经元的电压最大是发生在第几个事件的刺激后

            v_max = np.zeros((self._n_class, self._n_neuron_per_class), dtype=np.float32)
            for i in range(self._n_class):
                for j in range(self._n_neuron_per_class):
                    v_max[i][j] = v[r_idx[i][j], i, j]

            self._update_weights_v(v_max, addr, time, label, l_idx, r_idx)
            #self._update_weights_v(v_max, addr, time, label, l_idx, r_idx)
            r_idx_max = np.minimum(length, np.max(r_idx, axis=0) + 1)
            #每个神经元取出最大的数
            for i in range(self._n_class):
                l_idx[i] = r_idx_max










