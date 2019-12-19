# coding=utf-8
'''
    Created on 06.09.2019
    
    @author: Qianhui Liu
    '''

import numpy as np
import math
import collections
import scipy.io as scio
import random
import pickle
import struct
import os
#import cv2 as cv


class FeatureDetector(object):
    def __init__(self, row, col, tau, x_pool=2, y_pool=2, Rot = [0, 45, 90, 135], C1size = [2,2], minFS = 3, maxFS = 9, stepFS = 2, gamma=0.3, circularRF=True, thr = 2):
        self._row = row
        self._col = col
        self._size = (row, col)
        self._Rot = Rot
        self._n_Rot = len(Rot)
        self._C1size = [2,2]
        self._minFS = minFS
        self._maxFS = maxFS
        self._stepFS = stepFS
        self._gamma = gamma
        self._circularRF = circularRF
        self._tau = tau

        self._thr = thr
        self._x_pool = x_pool
        self._y_pool = y_pool

        self._FSize = range(minFS, maxFS + stepFS, stepFS)
        self._n_FSize = len(self._FSize)
        self._n_filters = len(self._FSize) * len(Rot)
        self._Filters = self.genFilters() #shape [n_FSize, n_Rot, fsize, fsize]

        self._V = self.genInitV()

    def setGaborFilter(self, **kwargs):
        self._Rot = kwargs.get('Rot', self._Rot)
        self._minFS = kwargs.get('minFS', self._minFS)
        self._maxFS = kwargs.get('maxFS', self._maxFS)
        self._stepFS = kwargs.get('stepFS', self._stepFS)
        self._gamma = kwargs.get('gamma', self._gamma)
        self._circularRF = kwargs.get('circularRF', self._circularRF)

        self._FSize = range(self._minFS, self._maxFS+self._stepFS, self._stepFS)
        self._n_Rot = len(self._Rot)
        self._n_FSize = len(self._FSize)
        self._n_filters = self._n_FSize * self._n_Rot
        self._Filters = self.genFilters()

    def genFilters(self):
        def calFilter(fsize, rot):
            div = 4 - (fsize - 7)/2 * 0.05
            lamb = fsize*2/div
            sigma = lamb*0.8
            rot = math.radians(rot)

            filter = np.zeros((fsize, fsize))
            center = int(fsize/2)
            for i in range(fsize):
                for j in range(fsize):
                    x_, y_ = i-center, j-center
                    x = x_*math.cos(rot) + y_*math.sin(rot)
                    y = -x_*math.sin(rot) + y_*math.cos(rot)
                    filter[i, j] = 0 if self._circularRF and x_**2 + y_**2 > (fsize/2.)**2 else \
                        math.exp(-(x**2 + (self._gamma*y)**2)/(2*sigma**2)) * math.cos(2*math.pi*x/lamb)
            filter = filter - np.mean(filter)
            filter = filter / np.sqrt(np.sum(filter**2))
            return filter

        Filters = []
        for fsize in self._FSize:
            Filters.append(np.concatenate(tuple([np.reshape(calFilter(fsize, rot), (1, fsize, fsize)) for rot in self._Rot])))
        return Filters

    def genInitV(self):
        return np.zeros((self._n_FSize, self._n_Rot, self._row, self._col))

    def leak(self, v, delta_t, tau = None):
        if not tau: tau = self._tau
        return v * np.exp(-1.* delta_t/tau)

    def S1(self, v, x_addr, y_addr):
        def conv_and_spike(v, x, y):
            addr = []
            for i in range(len(self._FSize)):
                fradius = self._FSize[i] / 2
                x_l, y_l = max(0, x - fradius), max(0, y - fradius)
                x_r, y_r = min(self._row, x + fradius + 1), min(self._col, y + fradius + 1)
                x_fl, y_fl = fradius + x_l - x, fradius + y_l - y
                x_fr, y_fr = fradius + x_r - x, fradius + y_r - y
                v[i, :, x_l:x_r, y_l:y_r] += self._Filters[i][:, x_fl:x_fr, y_fl:y_fr]
                j, x_idx, y_idx = np.where(v[i, :, x_l:x_r, y_l:y_r]>self._thr)
                x_pos, y_pos = x_idx+x_l, y_idx+y_l
                for k in range(len(j)):
                    v[i, j[k], x_pos[k]/self._x_pool*self._x_pool:(x_pos[k]/self._x_pool+1)*self._x_pool, y_pos[k]/self._y_pool*self._y_pool:(y_pos[k]/self._y_pool+1)*self._y_pool] = 0
                pos = (self._row*self._col/(self._x_pool*self._y_pool))*(i*self._n_Rot+j) + x_pos/self._x_pool*(self._col/self._y_pool) + y_pos/self._y_pool
                addr.extend(set(pos))
                # v[i, :, x_l:x_r, y_l:y_r] = np.where(v[i, :, x_l:x_r, y_l:y_r]>self._thr, 0, v[i, :, x_l:x_r, y_l:y_r])
            return v, addr

        if x_addr<0 or x_addr>=self._row: raise AssertionError('x_addr(%d) is out of bound (0, %d).'%(x_addr, self._row))
        if y_addr<0 or y_addr>=self._col: raise AssertionError('y_addr(%d) is out of bound (0, %d).'%(y_addr, self._row))
        x_addr = max(0, min(x_addr, self._row))
        y_addr = max(0, min(y_addr, self._col))
        return conv_and_spike(v, x_addr, y_addr)

    def C1(self, v):
        return np.max(np.reshape(v, (self._n_filters, self._row/self._x_pool, self._x_pool, self._col/self._y_pool, self._y_pool)), axis=(2, 4))

    def genSpikes(self, Timestamps, X_addr, Y_addr):
        if not len(Timestamps): raise AssertionError('Empty AERData.')
        Time, Addr = [], []
        n_spike = len(Timestamps)
        V = self.genInitV()
        V, _ = self.S1(V, X_addr[0], Y_addr[0])
        for i in range(1, n_spike):
            V = self.leak(V, Timestamps[i]-Timestamps[i-1])
            V, addr = self.S1(V, X_addr[i], Y_addr[i])
            Time.extend([Timestamps[i]]*len(addr))
            Addr.extend(addr)
        return Addr, Time

    def getFeatures(self, x, y, ts, p):
        x = np.minimum(np.maximum(x, 0), self._row - 1)
        y = np.minimum(np.maximum(y, 0), self._col - 1)
        addr, time = [], []
        for pp in np.unique(p):
            cx, cy, cts = x[p == pp], y[p == pp], ts[p == pp]
            c_addr, c_time = self.genSpikes(cts, cx, cy)
            addr.append(c_addr)
            time.append(c_time)
        addr = reduce(lambda x, y: x + y, addr, [])
        time = reduce(lambda x, y: x + y, time, [])

        if len(addr) > 1:
            time, addr = zip(*sorted(zip(time, addr)))
        return addr, time


    def getFeaturesAll(self, X, Y, TS, P):
        Addr, Time, V = [], [], []
        for i in range(len(X)):
            x, y, ts, p = X[i], Y[i], TS[i], P[i]
            addr, time = self.getFeatures(x, y, ts, p)
            Addr.append(np.array(addr))
            Time.append(np.array(time))
        return Addr, Time

    @property
    def Filters(self):
        return self._Filters
