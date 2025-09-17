#!/usr/bin/env Python
# coding=utf-8
"""experiments for high-dimensional large-sample dataset"""
import random

import sklearn.metrics
from tqdm import tqdm

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler as MM
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score, pairwise_distances


import MIC
from scipy.io import loadmat
import numpy as np
import pandas as pd

import utils


import time
import warnings

warnings.filterwarnings("ignore")



# set random seed
myseed = 893027724
np.random.seed(myseed)
random.seed(myseed)
# global parase
gbest = None
max_gen = 60  # 200
w0 = 0.9
w1 = 0.4
c = 1.49445
divNum = 5
gamma = 0.8

npseed = np.random.get_state()[1][0]
rdseed = random.getstate()[1][0]


class Particle:
    # init Particle
    def __init__(self, data, pID, divID, maxLen, parLen, VRmax, VRmin, selflag,
                 cls=svm.SVC()):
        self.pID = pID
        self.rankID = pID
        self.divID = divID
        self.maxLen = maxLen
        self.dim = parLen
        self.Pc = 0  # comprehensive learning threshold

        self.pos = np.random.uniform(0, 1, self.maxLen)
        self.vel = np.random.uniform(0, 1, self.maxLen) * (VRmax - VRmin) + VRmin

        self.set = np.random.uniform(0, 0, self.maxLen)

        self.Vmax = VRmax
        self.Vmin = VRmin

        self.fitness = None

        self.pbest = None
        self.pset = None
        self.acc = 0
        self.bacc = 0
        self.examplars = []
        self.impflag = 0  # i^th is changed
        self.selflag = selflag  # feature selected threshold
        self.Db = 0
        self.Dw = 0
        self.pDb = 0
        self.pDw = 0
        self.dst = 0
        self.pdst = 0
        # init feature subset in all dimension
        for i in range(0, self.maxLen):
            self.examplars.append([self.divID, self.pID])
            if i < self.dim:
                if self.pos[i] > self.selflag:
                    self.set[i] = 1
                else:
                    self.set[i] = 0
            else:
                self.set[i] = 0
        self.set = self.set.astype(int)

        self.fit_func(data, cls, self.dim, isDst=False)
        self.pbest = self.fitness
        self.pset = self.set
        self.pacc = self.acc
        self.pbacc = self.bacc
        self.pdst = self.dst

    # update single particle velocity, position
    def updateSingle(self, data, Block, cls, w):

        for i in range(self.dim):
            rand_num = np.random.uniform(0, 1)

            self.vel[i] = w * self.vel[i] + c * rand_num * (
                Block[self.examplars[i][0]][self.examplars[i][1]].pbest - self.pos[i])
            if self.vel[i] > self.Vmax:
                self.vel[i] = self.Vmax
            elif self.vel[i] < self.Vmin:
                self.vel[i] = self.Vmin

            self.pos[i] = self.pos[i] + self.vel[i]

            # boundary processing
            if self.pos[i] < 0:
                self.pos[i] = abs(self.pos[i])
                self.vel[i] = -self.vel[i]

            if self.pos[i] > 1:
                self.pos[i] = self.pos[i] - 1
                self.vel[i] = -self.vel[i]
            # select feature
            if self.pos[i] > self.selflag:
                self.set[i] = 1
            else:
                self.set[i] = 0
            self.set = self.set.astype(int)

        # update fitness
        self.fit_func(data, cls, self.dim)
        # update pbest info
        if self.fitness > self.pbest:
            self.pbest = self.fitness
            self.pset = self.set
            self.pacc = self.acc
            self.pbacc = self.bacc
            self.pDb = self.Db
            self.pDw = self.Dw
            self.pdst = self.dst
        else:
            self.impflag = self.impflag + 1

    def changePdim(self, parLen):
        '''
        change single particle dim in each divison,
        update par info,
        :param parLen:
        :return:
        '''

        if self.dim < parLen:
            for i in range(self.dim, parLen):
                self.pos[i] = np.random.uniform(0, 1)
                self.vel[i] = np.random.uniform(0, 1) * (VRmax - VRmin) + VRmin
                if self.pos[i] > self.selflag:
                    self.set[i] = 1
            self.set = self.set.astype(int)

            self.dim = parLen

            self.fit_func(data, cls, self.dim)
            if self.fitness > self.pbest:
                self.pbest = self.fitness
                self.pset = self.set
                self.pacc = self.acc
                self.pbacc = self.bacc
                self.pDb = self.Db
                self.pDw = self.Dw
                self.pdst = self.dst

        elif self.dim > parLen:
            for i in range(parLen, self.dim):
                self.set[i] = 0
            self.set = self.set.astype(int)

            self.dim = parLen

            self.fit_func(data, cls, self.dim)

            if self.fitness > self.pbest:
                self.pbest = self.fitness
                self.pset = self.set
                self.pacc = self.acc
                self.pbacc = self.bacc
                self.pDb = self.Db
                self.pDw = self.Dw
                self.pdst = self.dst

    def _reset_metrics(self):

        self.fitness = 0
        self.acc = 0
        self.bacc = 0
        self.dst = 0

    def fit_func(self, data, cls, divDim, isDst=True):

        x_train, y_train, x_test, y_test = data
        y_train = np.ravel(y_train)


        mask = self.set[:self.dim].astype(bool)

        if self.maxLen > self.dim:
            extended_mask = np.zeros(self.maxLen, dtype=bool)
            extended_mask[:self.dim] = mask
        else:
            extended_mask = mask

        x_train_subset = x_train[:, extended_mask]

        if x_train_subset.shape[1] == 0:
            self._reset_metrics()
            return


        cls.fit(x_train_subset, y_train)


        y_pred = cls.predict(x_train_subset)

        self.acc = accuracy_score(y_train, y_pred)
        self.bacc = self.acc

        if isDst:
            distance, Dw, Db = fast_cal(x_train_subset, y_train, divDim)
            self.fitness = gamma * self.bacc + (1 - gamma) * distance
            self.dst = distance
            self.Db, self.Dw = Db, Dw
        else:
            self.fitness = gamma * self.bacc
            self.dst = 0


class Swarm:

    def __init__(self, data, pNum, Ndim, w0, w1, VRmax, VRmin, impflag, selflag,
                 classifier, divSize, Parlen):
        """

        :param data:
        :param pNum:
        :param Ndim:
        :param w0:
        :param w1:
        :param VRmax:
        :param VRmin:
        :param impflag:
        :param selflag:
        :param classifier:
        :param divSize:
        :param Parlen:
        """

        self.pNum = pNum
        self.dim = Ndim
        self.w0 = w0
        self.w1 = w1
        self.Vmax = VRmax
        self.Vmin = VRmin

        self.impflag = impflag
        self.selflag = selflag
        self.lenflag = 0
        self.Block = []
        self.blockExamplar = []
        self.divSize = divSize
        self.ParLen = Parlen
        self.cls = classifier

        self.gbest_fit = -99999999
        self.gbest_set = np.random.uniform(0, 0, Ndim)
        self.gbest_acc = 0
        self.gbest_bacc = 0

        self.maxLen = Ndim
        self.bestLen = Ndim
        self.bestBlock = None
        self.bestParticle = None
        self.breakflag = 0
        self.gDb = 0
        self.gDw = 0
        self.gdst = 0
        self.test_acc = 0
        self.test_bacc = 0
        # init particles
        for divID in tqdm(range(divNum)):
            Particles = []
            if divID < divNum - 1:
                # init each Block
                for pID in range(divSize[divID]):

                    Particles.append(
                        Particle(data, pID, divID, Ndim, Parlen[divID], self.Vmax, self.Vmin, selflag))
            elif pNum % divNum == 0:
                for pID in range(divSize[divID]):

                    Particles.append(Particle(data, pID, divID, Ndim, Parlen[divID], self.Vmax, self.Vmin, selflag))
            else:
                temp_num = pNum % divNum
                for pID in range(temp_num):
                    Particles.append(Particle(data, pID, divID, Ndim, Parlen[divID], self.Vmax, self.Vmin, selflag))
            self.Block.append(Particles)
        self.checkRank()

    def updateSwarm(self, data, divNum):

        w = 0.8

        flag = 0
        for divID in range(divNum):
            Block = self.Block
            Particles = self.Block[divID]
            pnum = len(Particles)

            for pID in range(pnum):
                if Particles[pID].impflag > self.impflag:

                    for d in range(Particles[pID].dim):

                        rand_num = np.random.uniform(0, 1)
                        if rand_num < Particles[pID].Pc:
                            div_rd1 = np.random.uniform(0, 1)
                            div_rd2 = np.random.uniform(0, 1)
                            div_id1 = np.ceil(div_rd1 * (divNum - 1))
                            div_id2 = np.ceil(div_rd2 * (divNum - 1))

                            div_id1 = div_id1.astype(int)
                            div_id2 = div_id2.astype(int)

                            for i in range(pNum):
                                if self.ParLen[div_id1] >= Particles[pID].dim and self.ParLen[div_id2] >= Particles[
                                    pID].dim:
                                    p_rd1 = np.random.uniform(0, 1)
                                    p_rd2 = np.random.uniform(0, 1)
                                    p_id1 = np.ceil(p_rd1 * (self.divSize[div_id1] - 1))
                                    p_id2 = np.ceil(p_rd2 * (self.divSize[div_id2] - 1))

                                    p_id1 = p_id1.astype(int)
                                    p_id2 = p_id2.astype(int)
                                    # select examplar
                                    if self.Block[div_id1][p_id1].pbest >= self.Block[div_id2][p_id2].pbest:
                                        Particles[pID].examplars[d] = [div_id1, p_id1]
                                    else:
                                        Particles[pID].examplars[d] = [div_id2, p_id2]

                                    break
                                else:
                                    div_rd1 = np.random.uniform(0, 1)
                                    div_rd2 = np.random.uniform(0, 1)
                                    div_id1 = np.ceil(div_rd1 * (divNum - 1))
                                    div_id2 = np.ceil(div_rd2 * (divNum - 1))

                                    div_id1 = div_id1.astype(int)
                                    div_id2 = div_id2.astype(int)

                        else:
                            Particles[pID].examplars[d] = [divID, pID]
                    Particles[pID].impflag = 0
                Particles[pID].updateSingle(data, Block, cls, w)

                self.checkRank()
                # self.checkBlockExamplar()
                # check gbest
                if Particles[pID].pbest > self.gbest_fit:
                    self.gbest_set = Particles[pID].pset
                    self.gbest_fit = Particles[pID].pbest
                    self.gbest_acc = Particles[pID].pacc
                    self.gbest_bacc = Particles[pID].pbacc
                    self.bestBlock = divID
                    self.bestParticle = pID
                    self.gDb = Particles[pID].pDb
                    self.gDw = Particles[pID].pDw
                    self.gdst = Particles[pID].pdst
                    self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test, y_test, cls)
                    print(
                        'global best:Block {} particle {}, fitness={:.10f},acc={:.4f},bacc={:.4f},dst={},Db={},Dw={}\n select_num={},test_acc={},test_bacc={}'.format(
                            self.bestBlock, pID, self.gbest_fit,
                            self.gbest_acc, self.gbest_bacc, self.gdst, self.gDb,
                            self.gDw,
                            np.where(
                                self.gbest_set == 1)[
                                0].shape[0], self.test_acc, self.test_bacc))
                    self.lenflag = 0
                    data_dict['fold'].append(f)
                    data_dict['epoch'].append(e)
                    data_dict['divID'].append(self.bestBlock)
                    data_dict['pID'].append(self.bestParticle)
                    data_dict['gbest_fit'].append(self.gbest_fit)
                    data_dict['gbest_acc'].append(self.gbest_acc)
                    data_dict['gbest_bacc'].append(self.gbest_bacc)
                    data_dict['gbest_set'].append(np.where(self.gbest_set == 1)[0])
                    data_dict['select_num'].append(np.where(self.gbest_set == 1)[0].shape[0])
                    data_dict['bestLen'].append(self.bestLen)
                    data_dict['distance'].append(self.gdst)
                    data_dict['test_acc'].append(self.test_acc)
                    data_dict['test_bacc'].append(self.test_bacc)
                    flag = flag + 1

        if flag == 0:
            self.lenflag = self.lenflag + 1

    def cal_test_bacc(self, x_train, y_train, x_test, y_test, cls):
        set = self.gbest_set

        mask = set.astype(bool)

        extended_mask = mask

        x_train_subset = x_train[:, extended_mask]
        x_test_subset = x_test[:, extended_mask]
        cls.fit(x_train_subset, y_train)


        y_pred = cls.predict(x_test_subset)

        acc = accuracy_score(y_test, y_pred)
        bacc = acc

        return acc, bacc

    def find_bestBlock(self):

        block_fit = []
        for divID in range(divNum):
            Particles = self.Block[divID]
            plen = len(Particles)
            temp = 0
            for p in Particles:
                temp = temp + p.fitness
            ave_fit = temp / plen
            block_fit.append(ave_fit)
        bestBlock = np.argmax(block_fit)

        return bestBlock

    def LenChange(self):
        '''
        change length for each block
        :return:
        '''

        avgBlock = self.find_bestBlock()
        if self.ParLen[avgBlock] == self.maxLen:
            self.breakflag = self.breakflag + 1
        else:
            self.breakflag = 0
        if self.bestBlock > avgBlock:
            print('best block is not included,{}-{}'.format(self.bestBlock, avgBlock))
        self.bestLen = self.ParLen[avgBlock]
        print(',gbest block:{},gbest length:{},max length:{}'
              ' --> avg block:{},new max length:{}'.format(self.bestBlock, self.ParLen[self.bestBlock], self.maxLen,
                                                           avgBlock, self.bestLen))
        if self.bestLen != self.maxLen:

            times = 0
            for i in range(divNum):
                if i != avgBlock:
                    newLen = np.ceil(self.bestLen * (times + 1) / divNum)
                    newLen = int(newLen)
                    self.ParLen[i] = newLen

                    Particles = self.Block[i]
                    times = times + 1

                    for p in Particles:
                        p.changePdim(newLen)
            self.maxLen = self.bestLen
        else:
            if self.breakflag >= beta:
                self.bestLen = self.ParLen[int((self.bestBlock + avgBlock) / 2)]
                times = 0
                for i in range(divNum):
                    if i != int((self.bestBlock + avgBlock) / 2):
                        newLen = np.ceil(self.bestLen * (times + 1) / divNum)
                        newLen = int(newLen)
                        self.ParLen[i] = newLen

                        Particles = self.Block[i]
                        times = times + 1

                        for p in Particles:
                            p.changePdim(newLen)
                self.maxLen = self.bestLen

    def checkRank(self):
        for div in range(divNum):
            Particles = self.Block[div]
            collect_fit = []
            for p in Particles:
                collect_fit.append(p.fitness)
            collect_fit = np.array(collect_fit)
            index_sorted = np.argsort(-1 * collect_fit)
            for i in range(len(Particles)):
                Particles[i].rankID = index_sorted[i]
                Particles[i].Pc = 0.05 + 0.45 * (np.exp((10 * index_sorted[i]) / (divSize[div] - 1)) - 1) / (
                    np.exp(10) - 1)




def fast_cal(features, labels, divDim):
    """
    cal distance between each distance base feature subset
    :param features: must be ndarray
    :param labels:
    :return:
    """

    norms = np.linalg.norm(features, axis=1)  # get norm for each row vector(instance vce)

    Mandst_matrix = (pairwise_distances(features, metric='manhattan')/features.shape[1]).astype(np.float32)
    Euldst_matrix = pairwise_distances(features, metric='euclidean')
    Cosdst_matrix = 1 - np.dot(features, features.T) / (norms[:, np.newaxis] * norms[np.newaxis, :])


    if True in np.isnan(Cosdst_matrix):
        Cosdst_matrix[np.where(np.isnan(Cosdst_matrix) == True)] = 0

    prior = features.shape[1]
    label_types = np.unique(labels)

    type_index = []
    for i in label_types:
        type_index.append(np.where(labels == i)[0])

    Dwmin = []
    for i in range(len(label_types)):
        Mtemp = Mandst_matrix[type_index[i]]
        Mtemp = Mtemp[:, type_index[i]]
        Etemp = Euldst_matrix[type_index[i]]
        Etemp = Etemp[:, type_index[i]]
        Ctemp = Cosdst_matrix[type_index[i]]
        Ctemp = Ctemp[:, type_index[i]]

        row, col = utils.z_score(Etemp)
        Etemp[row, col] = 0
        row, col = utils.z_score(Mtemp)
        Mtemp[row, col] = 0
        row, col = utils.z_score(Ctemp)
        Ctemp[row, col] = 0
        X = np.vstack((np.max(Mtemp, axis=1), np.max(Etemp, axis=1), np.max(Ctemp, axis=1)))
        cov = np.cov(X)
        inv_cov = (np.linalg.pinv(cov)).astype(np.float32)
        score = np.sqrt(np.dot(np.dot(X.T, inv_cov), X))
        diag_score = np.diagonal(score)
        Dwmin.append(np.mean(diag_score))
    Dw = np.mean(Dwmin)


    dif_type_index = []
    for i in label_types:
        dif_type_index.append(np.where(labels != i)[0])
    Dbmax = []
    for i in range(len(label_types)):
        Mtemp = Mandst_matrix[type_index[i]]
        Mtemp = Mtemp[:, dif_type_index[i]]
        Etemp = Euldst_matrix[type_index[i]]
        Etemp = Etemp[:, dif_type_index[i]]
        Ctemp = Cosdst_matrix[type_index[i]]
        Ctemp = Ctemp[:, dif_type_index[i]]

        row, col = utils.z_score(Etemp)
        Etemp[row, col] = 99999
        row, col = utils.z_score(Mtemp)
        Mtemp[row, col] = 99999
        row, col = utils.z_score(Ctemp)
        Ctemp[row, col] = 1.1
        X = np.vstack((np.min(Mtemp, axis=1), np.min(Etemp, axis=1), np.min(Ctemp, axis=1)))

        cov = np.cov(X)
        inv_cov = (np.linalg.pinv(cov)).astype(np.float32)
        score = np.sqrt(np.dot(np.dot(X.T, inv_cov), X))
        diag_score = np.diagonal(score)
        Dbmax.append(np.mean(diag_score))
    Db = np.mean(Dbmax)

    delta = 5
    if prior <= 20:
        distance = 0
    else:
        distance = 1 / (1 + np.exp(-1 * (Dw - Db) / delta / np.sqrt(prior ** 3)))
    if distance > 0.9:
        distance = 0.0
    return distance, Dw, Db


def Div_Nbr(pNum, divNum, Ndim):
    """
    Return particles num in each division, each len of particle in single division
    :param pNum:
    :param divNum:
    :param Ndim:
    :return:
    """
    divSize = pNum // divNum

    parLen = np.array([np.ceil(Ndim * (i + 1) / divNum) for i in range(divNum)])
    parLen = parLen.astype(int)
    return divSize, parLen


def save_data(data, c):
    # print(data)
    with open('./result/forPancancer{}.txt'.format(run_time), c) as f:
        for k, v in data.items():
            f.write(k + ':' + str(v))
            f.write('\n')


def _pairwise_euclidean(X, out_dtype=np.float32):

    X = np.asarray(X, dtype=np.float32, order="C")
    row_norm2 = np.einsum('ij,ij->i', X, X)
    G = X @ X.T  # Gram
    sq = row_norm2[:, None] + row_norm2[None, :] - 2.0 * G
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq, dtype=np.float64).astype(out_dtype, copy=False)

def read_txt_file_pandas(file_path, sep=' '):


    with open(file_path, "r") as f:
        features = []
        labels = []
        for line in f.readlines():
            line = line.strip()
            line = line.split()

            line = [np.float64(x) for x in line]
            feature = line[:-1]
            label = line[-1]

            features.append(feature)
            labels.append(label)
    features = np.array(features).astype(np.float32)
    labels = np.array(labels).astype(int)

    return features, labels


if __name__ == '__main__':

    run_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    selflag = 0.6
    impflag = 7
    # preprocessing data
    "data. The data of the arff file, accessible by attribute names." \
    "meta Contains information about the arff file " \
    "such as name and type of attributes, the relation (name of the dataset"

    data = loadmat('dataset/Pancancer.mat')

    df = pd.DataFrame(data['DATA'])
    data_arr = df.values
    data_arr_T = data_arr.T
    features = data_arr_T[:-1].T
    labels = data_arr_T[-1]


    features = features.astype(np.float32)
    labels = labels.astype(int)
    cls = svm.SVC()
    scaling = MM(feature_range=(0, 1))
    features = scaling.fit_transform(features)

    print("Class types:", Counter(labels))

    pNum = 30

    VRmax = 0.6
    VRmin = -0.6
    gbest_fit = []
    gbest_acc = []
    beta = 5  #
    alpha = 7
    # K-FOLD EXPERIMENT

    kf = KFold(n_splits=5, shuffle=True)
    k_acc = 0.0
    k_size = 0.0
    # myseed = np.random.get_state()[1][0]
    parameter_dict = {'seed': [rdseed, npseed], 'max_gen': max_gen, 'w': 0.8, 'beta': beta, 'c': c, 'divNum': divNum,
                      'gamma': gamma, }
    data_dict = {'fold': [], 'epoch': [], 'divID': [], 'pID': [], 'gbest_fit': [], 'gbest_acc': [], 'gbest_bacc': [],
                 'distance': [], 'select_num': [], 'bestLen': [], 'gbest_set': [], 'test_acc': [], 'test_bacc': [], }
    # global f
    for f, (train_index, test_index) in enumerate(kf.split(features, labels)):

        # load k-fold data
        x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
            test_index]

        mic = MIC.MIC(x_train, y_train)

        index_sorted = np.argsort(-mic)

        # resort features
        x_train_T, x_test_T = x_train.T, x_test.T
        x_train_T, x_test_T = x_train_T[index_sorted[0:int(0.3 * len(index_sorted))]], x_test_T[
            index_sorted[0:int(0.1 * len(index_sorted))]]
        x_train, x_test = x_train_T.T, x_test_T.T
        Ndim = len(x_train_T)
        data = [x_train, y_train, x_test, y_test]
        # div param
        divSize, Parlen = Div_Nbr(pNum, divNum, Ndim)
        temp = np.array([1, 2, 3, 4, 5])
        divSize = 5 + np.around(5 / 2 * (1 + np.tanh((temp - np.median(temp)) ** 3 / 10)))
        divSize = divSize.astype(int)
        # init Particles
        Particles = Swarm(data, pNum, Ndim, w0, w1, VRmax, VRmin, impflag, selflag, cls, divSize, Parlen)

        # search features subset
        for e in tqdm(range(max_gen)):
            Particles.updateSwarm(data, divNum)

            gbest_fit.append(Particles.gbest_fit)
            gbest_acc.append(Particles.test_bacc)

            if Particles.lenflag >= beta:
                print('len changed', end=' ')
                Particles.LenChange()
                Particles.lenflag = 0
            # save_data(data_dict, 'w')

        print('{} fold---gbest fitness:{:.4f},gbest acc:{:.4f},select num:{}'.format(f, gbest_fit[-1], gbest_acc[-1],
                                                                                     Counter(Particles.gbest_set == 1)[
                                                                                         1]))

        save_data(data_dict, 'w')
        # cal average value
        k_acc = k_acc + Particles.test_bacc
        k_size = k_size + Counter(Particles.gbest_set == 1)[1]
    save_data(parameter_dict, 'a')
    print('average precision:{:.4f}\taverage size:{:.2f}'.format(k_acc / 5.0, k_size / 5.0))
    print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
