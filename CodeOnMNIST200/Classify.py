# coding=utf-8
'''
    Created on 06.09.2019
    
    @author: Qianhui Liu
    '''

from myTempotron import *
import argparse
import random
import os
import collections


'''
Classify: 封装myTempotron
func: test, 指定训练集和测试集，对模型进行测试
            input: tr_data[训练集, tuple](tr_addr, tr_time, tr_label), te_data, cycle
usage: model = Classify(tau=1, tau_m=0.2, n_in=3136, n_class=10, n_neuron_per_class=10, tgt='MNIST_DVS200))
       model.test(tr_data, te_data)
       or model.test((tr_addr, tr_time, tr_label), (te_addr, te_time, te_label)
'''
class Classify(object):
    def __init__(self, tau, tau_m, n_in, n_class, n_neuron_per_class,
                 tgt, method='SPA',learning_rate=0.1,
                 noise=0.0, seed=123, debug=True):
        random.seed(seed)
        # get argument from system input with default of input argument
        parser = argparse.ArgumentParser()
        parser.add_argument('--method', '-m', default=method)
        parser.add_argument('--learning-rate', '-lr', default=learning_rate, type=float)
        parser.add_argument('--noise', '-noise', default=noise, type=float)
        args = parser.parse_args()

        # get argument
        self._method = args.method
        self._learning_rate = args.learning_rate
        self._noise = args.noise
        self._datasize = 0
        self._tau = tau
        self._tau_m = tau_m
        self._n_in = n_in
        self._n_class = n_class
        self._n_neuron_per_class = n_neuron_per_class
        self._tgt = tgt
        self._debug = debug

        #build model
        self._model = eval(self._method)(tau=self._tau, tau_m=self._tau_m, n_in=self._n_in,
                                n_class=self._n_class, n_neuron_per_class=self._n_neuron_per_class,
                                lmd = self._learning_rate)
        self._te_multi_al_history, self._tr_multi_al_history = [], []


    def _init_history(self):
        self._te_multi_history, self._tr_multi_history = [], []
    
    def _update_history(self, tr_multi_acc, te_multi_acc):
        self._tr_multi_history.append(tr_multi_acc)
        self._te_multi_history.append(te_multi_acc)
    
    def _save_history_to_csv(self,cycle):
        if self._te_multi_history:
                self._tr_multi_al_history.append(self._tr_multi_history)
                self._te_multi_al_history.append(self._te_multi_history)
        fformat = 'res/{tgt}_{method}_size_{size}_tau_{tau}_lr_{lr}_' \
                        'noise_{noise}_cycle_{cycle}_history.csv'
        fname = fformat.format(tgt = self._tgt,
                               method = self._method,
                               size=self._datasize,
                               lr=self._learning_rate,
                               tau="{}_{}".format(self._tau, self._tau_m),
                               noise="{:.2f}".format(self._noise),
                               cycle = cycle)
        def write_history(f, history):
            for h in history:
                f.write(','.join(['%.4f'%t for t in h]) + '\n')

        with open(fname, 'w') as f:
            write_history(f, self._tr_multi_al_history)
            write_history(f, self._te_multi_al_history)

    def test(self, tr_data, te_data,cycle):
        self._init_history()
        self._datasize = len(tr_data[0]) + len(te_data[0])
        self._model = eval(self._method)(tau=self._tau, tau_m=self._tau_m, n_in=self._n_in,
                                         n_class=self._n_class, n_neuron_per_class=self._n_neuron_per_class,
                                         lmd=self._learning_rate)

        for i in range(10):
            #test
            self._model.train(*tr_data)
            if self._debug:
                tr_multi_acc = self._model.test(*tr_data, multi=True)
                print('Train_MULTI {} {}th: {}'.format(cycle, i, tr_multi_acc))
                te_multi_acc = self._model.test(*te_data, multi=True)
                print('Test_MULTI {} {}th: {}'.format(cycle, i, te_multi_acc))
                self._update_history(tr_multi_acc, te_multi_acc)
        self._save_history_to_csv(cycle)
        return te_multi_acc
