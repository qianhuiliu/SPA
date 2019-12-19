# coding=utf-8
'''
    Created on 06.09.2019
    
    @author: Qianhui Liu
    '''

from Classify import *
from FeatureDetector import *
from SpikeLoader import *
from multiprocessing import Process, Queue

def readAER(filepath):
    f = scio.loadmat(filepath)
    Labels = f['Labels']
    Cins = f['CINs']
    x, y, ts, p, labels = [], [], [], [], []
    l = Labels.size
    for i in range(10000): #number of samples
            labels.append(Labels[0, i])
            cur_ts = np.reshape(Cins[0, i][:, 0], newshape=(-1,))
            cur_x = np.reshape(Cins[0, i][:, 3], newshape=(-1,))
            cur_y = np.reshape(Cins[0, i][:, 4], newshape=(-1,))
            cur_p = np.reshape(Cins[0, i][:, 2], newshape=(-1,))
            cur_x = cur_x.astype(np.int32)
            cur_y = cur_y.astype(np.int32)
            print(i, cur_ts.shape, cur_p.shape, max(cur_x), min(cur_x))
            cur_ts, cur_x, cu_y, cur_p = zip(*sorted(zip(cur_ts, cur_x, cur_y, cur_p)))
            cur_ts = cur_ts[:] - cur_ts[0]
            ts.append(np.array(cur_ts))
            x.append(np.array(cur_x))
            y.append(np.array(cur_y))
            p.append(np.array(cur_p))
    return ts,x,y,p,labels

def trainAndtest(cycle,cycle1):
    accMulti = []
    for i in range(1):
        tr_data, te_data = loader.get()
        accM = model.test(tr_data, te_data,cycle)

if __name__ == '__main__':
    # MNIST_DVS200
    tau, tau_m = 80000, 20000
    end = 200000
    thr = 2
    size = 28
    nGroup = 10
    lmd = 1e-2
    noise = 0
    method = 'SPA'
    filepath = '../data/MNIST_DVS_28x28_10000_200ms.mat'
    ts, x, y, p, labels = readAER(filepath)
    s = FeatureDetector(size, size, tau_m,thr = thr)
    addr, time = s.getFeaturesAll(x, y, ts, p)
    featureFile = '../data/MNISTDVS200/MNISTDVS200_tau_' + str(tau_m) + '_thr_' + str(thr) + '_data.pkl'
    with open(featureFile, 'w') as f:
        pickle.dump((addr, time, labels), f)
    #####feature finish
    model = Classify(tau=tau, tau_m=tau_m, n_in=size * size * 4, n_class=nGroup,
                     n_neuron_per_class=10, tgt='MNISTDVS200',
                     learning_rate=lmd, method=method)
    loader = SpikeLoader(featureFile, max_time=end, data_size=10000,
                         noise=0.1, max_addr=size * size * 4)
 #   accMulti = []
 #   for i in range(10):
 #       tr_data, te_data = loader.get()
 #       accM = model.test(tr_data, te_data,i)
 #       accMulti.append(accM)
 #   accMulti = np.array(accMulti).astype(np.float)
 #   print 'multi accuracy --> mean: ', np.mean(accMulti), '--> standard deviation: ', np.std(accMulti)
    #multiprocess
    jobs = []
    for cycle in range(10):
        job = Process(target=trainAndtest, args=(cycle + 1, cycle + 1))
        jobs.append(job)
        job.start()
    for job in jobs:
        job.join()



