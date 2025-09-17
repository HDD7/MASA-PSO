import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler as MM
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score,balanced_accuracy_score

# from sklearn.metrics import roc_auc_score as AUC
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import utils

import time
import warnings

warnings.filterwarnings("ignore")


seed = 3027724
np.random.seed(seed)
random.seed(seed)

max_gen = 100  # 200
w0 = 0.9
w1 = 0.4
c = 1.49445
divNum = 5
gamma = 0.8

npseed = np.random.get_state()[1][0]
rdseed = random.getstate()[1][0]


mycount_ep=[]
mycount_num=[]
cf1=[]
cf2=[]
cf3=[]

class Particle:
    # init Particle
    def __init__(self, data, pID, divID, maxLen, parLen, VRmax, VRmin, selflag,
                 cls=svm.SVC()):
        self.pID = pID
        self.rankID = pID
        self.divID = divID
        self.maxLen = maxLen
        self.dim = parLen
        self.Pc = 0

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
        self.impflag = 0
        self.selflag = selflag
        self.Db = 0
        self.Dw = 0
        self.pDb = 0
        self.pDw = 0
        self.dst = 0
        self.pdst = 0
        self.forget = 0
        self.data = data
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
                self.pos[i] = 0.0
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

            if self.pos[i] < 0:
                self.pos[i] = abs(self.pos[i])
                self.vel[i] = -self.vel[i]

            if self.pos[i] > 1:
                self.pos[i] = self.pos[i] - 1
                self.vel[i] = -self.vel[i]

            if self.pos[i] > self.selflag:
                self.set[i] = 1
            else:
                self.set[i] = 0
            self.set = self.set.astype(int)

        self.fit_func(data, cls, self.dim)

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
                self.pos[i] = 0.0
            self.set = self.set.astype(int)

            self.dim = parLen

            self.forget = self.fitness
            self.fit_func(data, cls, self.dim)
            self.forget = -self.forget + self.fitness

            if self.fitness > self.pbest:
                self.pbest = self.fitness
                self.pset = self.set
                self.pacc = self.acc
                self.pbacc = self.bacc
                self.pDb = self.Db
                self.pDw = self.Dw
                self.pdst = self.dst

    def EFL(self, EliteStatic):

        Examplar = EliteStatic[:self.dim]

        total_counts = np.sum(Examplar)
        probabilities = Examplar / total_counts
        selected_index = np.unique(np.random.choice(len(Examplar), int(0.6 * self.dim), p=probabilities))

        x_train, y_train = self.data[0], self.data[1]
        col_list = []
        for col in range(0, self.dim):
            if col not in selected_index:
                col_list.append(col)
        if len(col_list) >= self.dim:
            tempfitness = 0
        else:

            for col in range(self.dim, self.maxLen):
                col_list.append(col)
            x_train_subset = np.delete(x_train, col_list, axis=1)
            cls.fit(x_train_subset, np.ravel(y_train))
            y_pred = cls.predict(x_train_subset)
            acc = 0
            balance_acc = recall_score(y_train, y_pred, average='macro')
            distance, Dw, Db = fast_cal(x_train_subset, y_train, 0)
            tempfitness = gamma * balance_acc + (1 - gamma) * distance


        if tempfitness > self.fitness:

            for i in range(self.dim):
                if i in selected_index:
                    self.pos[i] = np.random.uniform(0.6, 1)
                else:
                    self.pos[i] = np.random.uniform(0, 0.6)
            self.set[np.where(self.pos > self.selflag)] = 1
            self.set[np.where(self.pos <= self.selflag)] = 0
            self.set = self.set.astype(int)
            self.fitness = tempfitness
            self.dst = distance
            self.bacc = balance_acc
            self.acc = acc
            self.Db = Db


            if self.fitness > self.pbest:
                self.pbest = self.fitness
                self.pset = self.set
                self.pacc = acc
                self.pbacc = balance_acc
                self.pDb = Db
                self.pDw = Dw
                self.pdst = distance

    def EOL(self, lowBoundary, upBoundary):
        tempPos = np.random.uniform(0, 1, self.dim)

        for i in range(self.dim):
            tempPos[i] = np.random.uniform(low=lowBoundary[i], high=upBoundary[i]) * (upBoundary[i] + lowBoundary[i]) - \
                         self.pos[i]

        x_train, y_train = self.data[0], self.data[1]
        col_list = []
        for col in range(0, self.dim):
            if tempPos[col] < self.selflag:
                col_list.append(col)
        if len(col_list) >= self.dim:
            tempfitness = 0
        else:

            for col in range(self.dim, self.maxLen):
                col_list.append(col)
            x_train_subset = np.delete(x_train, col_list, axis=1)
            cls.fit(x_train_subset, np.ravel(y_train))
            y_pred = cls.predict(x_train_subset)
            acc = 0
            balance_acc = recall_score(y_train, y_pred, average='macro')
            distance, Dw, Db = fast_cal(x_train_subset, y_train, 0)
            tempfitness = gamma * balance_acc + (1 - gamma) * distance


        if tempfitness > self.fitness:

            for i in range(self.dim):
                self.pos[i] = tempPos[i]
            self.set[np.where(self.pos > self.selflag)] = 1
            self.set[np.where(self.pos <= self.selflag)] = 0
            self.fitness = tempfitness
            self.dst = distance
            self.bacc = balance_acc
            self.acc = acc
            self.Db = Db
            self.Dw = Dw

            if self.fitness > self.pbest:
                self.pbest = self.fitness
                self.pset = self.set
                self.pacc = acc
                self.pbacc = balance_acc
                self.pDb = Db
                self.pDw = Dw
                self.pdst = distance

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

        self.acc = 0
        self.bacc = balanced_accuracy_score(y_train, y_pred)

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

        self.pNum = divSize
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

        for divID in tqdm(range(divNum)):
            Particles = []

            for pID in range(divSize[divID]):

                Particles.append(
                    Particle(data, pID, divID, Ndim, Parlen[divID], self.Vmax, self.Vmin, selflag))

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

        acc = 0
        bacc = balanced_accuracy_score(y_test, y_pred)

        return acc, bacc

    def getEliteBoundary(self, avgBlock):

        EliteBlock = self.Block[avgBlock]

        features_list = []
        for p in EliteBlock:
            features_list.append(p.pos)
        features_list = np.array(features_list)
        features_list_T = features_list.T
        del_index = np.where(features_list_T < self.selflag)

        features_list_T[del_index] = np.nan
        upBoundary = np.nanmax(features_list_T, axis=1)
        lowBoundary = np.nanmin(features_list_T, axis=1)

        return upBoundary, lowBoundary

    def getDeBoundary(self, Degration):

        # Degration = self.Block[]
        features_list = []
        for p in Degration:
            features_list.append(p.pos)
        features_list = np.array(features_list)
        features_list_T = features_list.T
        del_index = np.where(features_list_T == 0)

        features_list_T[del_index] = np.nan
        upBoundary = np.nanmax(features_list_T, axis=1)
        lowBoundary = np.nanmin(features_list_T, axis=1)

        return upBoundary, lowBoundary

    def getEliteSet(self, avgBlock):

        set_list = []
        for div in range(len(self.ParLen)):
            if self.ParLen[div] <= self.ParLen[avgBlock]:
                for p in self.Block[div]:
                    set_list.append(p.pset)

        set_list = np.array(set_list)  # (pNum,Ndim)
        set_static = np.sum(set_list, axis=0)

        return set_static

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
            total = 0
            avg1=0
            avg2=0
            avg3=0

            EliteStatic = self.getEliteSet(avgBlock)
            times = 0
            total_forget = 0
            for i in range(divNum):
                if i != avgBlock:
                    newLen = np.ceil(self.bestLen * (times + 1) / divNum)
                    newLen = int(newLen)
                    self.ParLen[i] = newLen

                    Particles = self.Block[i]
                    times = times + 1

                    collect_forget = []

                    avgfit1 = 0
                    avgfit2 = 0
                    num = 0
                    for p in Particles:
                        num += 1
                        avgfit1 = avgfit1 + p.fitness
                        p.changePdim(newLen)
                        avgfit2 = avgfit2 + p.fitness

                        if p.pbest > self.gbest_fit:
                            self.gbest_set = p.pset
                            self.gbest_fit = p.pbest
                            self.gbest_acc = p.pacc
                            self.gbest_bacc = p.pbacc
                            self.bestBlock = i
                            self.bestParticle = p.pID
                            self.gDb = p.pDb
                            self.gDw = p.pDw
                            self.gdst = p.pdst
                            self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test, y_test, cls)

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

                            self.lenflag = 0

                        collect_forget.append(p.forget)
                    avgfit1 = avgfit1 / num
                    avgfit2 = avgfit2 / num
                    avg1+=avgfit1
                    collect_forget = np.array(collect_forget)
                    forget_index = np.where(collect_forget < 0)[0]
                    if forget_index.shape[0] > 0:

                        avgfit3 = 0
                        avgfit4 = 0
                        num2 = 0
                        forget_index = forget_index.astype(int)
                        Particles = np.array(Particles)
                        DeUpBoundary, DeLowBoundary = self.getDeBoundary(Particles[forget_index])
                        index_sorted = np.argsort(collect_forget)
                        lb = int(0.2 * len(index_sorted))

                        for idx in range(len(index_sorted)):
                            id = index_sorted[idx]
                            if idx < lb and collect_forget[id] < 0:
                                num2 += 1
                                avgfit3 += Particles[id].fitness
                                Particles[id].EOL(DeLowBoundary, DeUpBoundary)
                                avgfit4 += Particles[id].fitness
                                if Particles[id].pbest > self.gbest_fit:
                                    self.gbest_set = Particles[id].pset
                                    self.gbest_fit = Particles[id].pbest
                                    self.gbest_acc = Particles[id].pacc
                                    self.gbest_bacc = Particles[id].pbacc
                                    self.bestBlock = i
                                    self.bestParticle = id
                                    self.gDb = Particles[id].pDb
                                    self.gDw = Particles[id].pDw
                                    self.gdst = Particles[id].pdst
                                    self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test, y_test,
                                                                                       cls)

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
                            elif idx >= lb and idx < 2 * lb and collect_forget[id] < 0:
                                num2 += 1
                                # 精英策略,学习每个分区最有希望的
                                avgfit3 += Particles[id].fitness
                                Particles[id].EFL(EliteStatic)
                                avgfit4 += Particles[id].fitness
                                if Particles[id].pbest > self.gbest_fit:
                                    self.gbest_set = Particles[id].pset
                                    self.gbest_fit = Particles[id].pbest
                                    self.gbest_acc = Particles[id].pacc
                                    self.gbest_bacc = Particles[id].pbacc
                                    self.bestBlock = i
                                    self.bestParticle = id
                                    self.gDb = Particles[id].pDb
                                    self.gDw = Particles[id].pDw
                                    self.gdst = Particles[id].pdst
                                    self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test, y_test,
                                                                                       cls)

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
                        avgfit3 /= num2
                        avgfit4 /= num2
                        total += forget_index.shape[0]
                        avg2+=avgfit3
                        avg3+=avgfit4
                        print('Block-{}\t Foreget num: {}'.format(i, forget_index.shape[0]))
                        print('Block-{}\tBefore:{}\tAfter:{}'.format(i, avgfit1, avgfit2))
                        print('Block-{}\tBE:{}\tAE:{}'.format(i, avgfit3, avgfit4))
            print('epoch:{}\tforget num: {}\t forget rate: {}'.format(e, total, total/100))
            avg1/=4
            avg2/=4
            avg3/=4
            mycount_ep.append(e)
            mycount_num.append(total)
            cf1.append(avg1)
            cf2.append(avg2)
            cf3.append(avg3)
            self.maxLen = self.bestLen
        else:
            if self.breakflag >= beta:
                self.bestLen = self.ParLen[int((self.bestBlock + avgBlock) / 2)]
                times = 0

                EliteSet = self.getEliteSet(int((self.bestBlock + avgBlock) / 2))
                for i in range(divNum):
                    if i != int((self.bestBlock + avgBlock) / 2):
                        newLen = np.ceil(self.bestLen * (times + 1) / divNum)
                        newLen = int(newLen)
                        self.ParLen[i] = newLen

                        Particles = self.Block[i]
                        times = times + 1

                        collect_forget = []
                        for p in Particles:
                            p.changePdim(newLen)

                            if p.pbest > self.gbest_fit:
                                self.gbest_set = p.pset
                                self.gbest_fit = p.pbest
                                self.gbest_acc = p.pacc
                                self.gbest_bacc = p.pbacc
                                self.bestBlock = i
                                self.bestParticle = p.pID
                                self.gDb = p.pDb
                                self.gDw = p.pDw
                                self.gdst = p.pdst
                                self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test, y_test,
                                                                                   cls)
                                print(
                                    'global best:Block {} particle {}, fitness={:.10f},acc={:.4f},bacc={:.4f},dst={},Db={},Dw={}\n select_num={},test_acc={},test_bacc={}'.format(
                                        self.bestBlock, p.pID, self.gbest_fit,
                                        self.gbest_acc, self.gbest_bacc, self.gdst, self.gDb,
                                        self.gDw,
                                        np.where(
                                            self.gbest_set == 1)[
                                            0].shape[0], self.test_acc, self.test_bacc))

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

                                self.lenflag = 0

                            collect_forget.append(p.forget)

                        collect_forget = np.array(collect_forget)

                        forget_index = np.where(collect_forget < 0)[0]
                        if forget_index.shape[0] > 0:
                            forget_index = forget_index.astype(int)
                            Particles = np.array(Particles)
                            DeUpBoundary, DeLowBoundary = self.getDeBoundary(Particles[forget_index])
                            index_sorted = np.argsort(collect_forget)
                            lb = int(0.2 * len(index_sorted))

                            for idx in range(len(index_sorted)):
                                id = index_sorted[idx]
                                if idx < lb and collect_forget[id] < 0:
                                    Particles[id].EOL(DeLowBoundary, DeUpBoundary)
                                    if Particles[id].pbest > self.gbest_fit:
                                        self.gbest_set = Particles[id].pset
                                        self.gbest_fit = Particles[id].pbest
                                        self.gbest_acc = Particles[id].pacc
                                        self.gbest_bacc = Particles[id].pbacc
                                        self.bestBlock = i
                                        self.bestParticle = id
                                        self.gDb = Particles[id].pDb
                                        self.gDw = Particles[id].pDw
                                        self.gdst = Particles[id].pdst
                                        self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test,
                                                                                           y_test,
                                                                                           cls)
                                        print(
                                            'global best:Block {} particle {}, fitness={:.10f},acc={:.4f},bacc={:.4f},dst={},Db={},Dw={}\n select_num={},test_acc={},test_bacc={}'.format(
                                                self.bestBlock, id, self.gbest_fit,
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
                                elif idx >= lb and idx < 2 * lb and collect_forget[id] < 0:

                                    Particles[id].EFL(EliteSet)
                                    if Particles[id].pbest > self.gbest_fit:
                                        self.gbest_set = Particles[id].pset
                                        self.gbest_fit = Particles[id].pbest
                                        self.gbest_acc = Particles[id].pacc
                                        self.gbest_bacc = Particles[id].pbacc
                                        self.bestBlock = i
                                        self.bestParticle = id
                                        self.gDb = Particles[id].pDb
                                        self.gDw = Particles[id].pDw
                                        self.gdst = Particles[id].pdst
                                        self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test,
                                                                                           y_test,
                                                                                           cls)
                                        print(
                                            'global best:Block {} particle {}, fitness={:.10f},acc={:.4f},bacc={:.4f},dst={},Db={},Dw={}\n select_num={},test_acc={},test_bacc={}'.format(
                                                self.bestBlock, id, self.gbest_fit,
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

    def tsne(self, x_train, y_train, x_test, y_test):
        set = self.gbest_set

        mask = set.astype(bool)

        extended_mask = mask

        x_train_subset = x_train[:, extended_mask]
        x_test_subset = x_test[:, extended_mask]
        combined_data = np.vstack([x_train_subset, x_test_subset])
        t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(combined_data)
        train_embedded=t_sne_features[:len(x_train_subset)]
        test_embedded=t_sne_features[len(x_train_subset):]
        unique_classes = np.unique(np.concatenate((y_train, y_test)))
        cmap = plt.get_cmap('RdYlGn', len(unique_classes))

        for i, cls in enumerate(unique_classes):
            mask_train = y_train == cls
            plt.scatter(train_embedded[mask_train, 0],
                        train_embedded[mask_train, 1],
                        c=[cmap(i)],
                        marker='o',
                        label=f'Train {cls}',
                        alpha=0.7,
                        edgecolors='w',
                        s=70)

        # Plot test set (triangles)
        for i, cls in enumerate(unique_classes):
            mask_test = y_test == cls
            plt.scatter(test_embedded[mask_test, 0],
                        test_embedded[mask_test, 1],
                        c=[cmap(i)],
                        marker='^',
                        label=f'Test {cls}',
                        alpha=0.7,
                        edgecolors='k',
                        s=70)
        plt.title('(d) Prostate')
        # Combine legend handles and avoid duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Remove duplicates
        plt.legend(by_label.values(), by_label.keys(),
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   title='Class Groups')

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def fast_cal(features, labels, divDim):
    """
    cal distance between each distance base feature subset
    :param features: must be ndarray
    :param labels:
    :return:
    """
    col_len = features.shape[1]
    if col_len <= 20:
        distance = 0
        Dw = 0
        Db = 0
    else:
        norms = np.linalg.norm(features, axis=1)
        dif_features = features - features[:, np.newaxis]
        Mandst_matrix = np.mean(np.abs(dif_features), axis=2)
        Euldst_matrix = np.sqrt(np.sum(dif_features ** 2, axis=2))
        Cosdst_matrix = 1 - np.dot(features, features.T) / (norms[:, np.newaxis] * norms[np.newaxis, :])

        if True in np.isnan(Cosdst_matrix):
            Cosdst_matrix[np.where(np.isnan(Cosdst_matrix) == True)] = 0


        label_types = np.unique(labels)
        # 同类标签下标
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
            # 剔除距离异常的离群点
            row, col = utils.z_score(Etemp)
            Etemp[row, col] = 0
            row, col = utils.z_score(Mtemp)
            Mtemp[row, col] = 0
            row, col = utils.z_score(Ctemp)
            Ctemp[row, col] = 0
            X = np.vstack((np.max(Mtemp, axis=1), np.max(Etemp, axis=1), np.max(Ctemp, axis=1)))

            cov = np.cov(X)
            inv_cov = np.linalg.pinv(cov)
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
            inv_cov = np.linalg.pinv(cov)
            score = np.sqrt(np.dot(np.dot(X.T, inv_cov), X))
            diag_score = np.diagonal(score)
            Dbmax.append(np.mean(diag_score))
        Db = np.mean(Dbmax)

        delta = 5

        distance = 1 / (1 + np.exp(-1 * (Dw - Db) / delta / np.sqrt(col_len ** 3)))
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
    with open('./result/forProTumor_{}_{}.txt'.format(t, run_time), c) as f:
        for k, v in data.items():
            f.write(k + ':' + str(v))
            f.write('\n')

def save_gbest(data, c):
    # print(data)
    with open('./result/forPro_{}.txt'.format(run_time), c) as f:
        for k, v in data.items():
            f.write(k + ':' + str(v))
            f.write('\n')
import MIC

if __name__ == '__main__':
    gbest_line = {'fold': [], 'epoch': [], 'gbest_fit': [], 'select_num': []}
    for t in range(60):
        print('{}th run of experiments'.format(t))
        run_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        selflag = 0.6
        impflag = 7


        # select classifier
        cls=KNN(n_neighbors=3)
        # cls=svm.SVC()
        # cls = XGBClassifier(n_estimators=30, max_depth=1)
        # cls = GaussianNB()
        data = pd.read_csv('../../dataset/Prostate_Tumor.csv')


        data_arr = data.values
        data_arr_T = data_arr.T
        features = data_arr_T[:-1].T
        scaling = MM(feature_range=(0, 1))
        features = scaling.fit_transform(features)
        labels = data_arr_T[-1]
        features = features.astype(np.float32)
        labels = labels.astype(int)  # bytes to int
        print("Class types:", Counter(labels))
        Ndim = features.shape[1]

        pNum = 100  # default 30 population size
        VRmax = 0.6
        VRmin = -0.6
        gbest_fit = []
        gbest_acc = []
        beta = 5  #
        alpha = 7
        # K-FOLD EXPERIMENT
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        # kf = KFold(n_splits=10, shuffle=True)
        k_acc = 0.0
        k_size = 0.0
        # myseed = np.random.get_state()[1][0]
        parameter_dict = {'seed': [rdseed, npseed], 'max_gen': max_gen, 'w': 0.8, 'beta': beta, 'c': c,
                          'divNum': divNum,
                          'gamma': gamma, 'avg_bacc': [], 'avg_size': []}
        data_dict = {'fold': [], 'epoch': [], 'divID': [], 'pID': [], 'gbest_fit': [], 'gbest_acc': [],
                     'gbest_bacc': [],
                     'distance': [], 'select_num': [], 'bestLen': [], 'gbest_set': [], 'test_acc': [],
                     'test_bacc': [], }

        mic = MIC.MIC(features, labels)
        # Ranking based su
        index_sorted = np.argsort(-mic)
        features_T = features.T
        features_T = features_T[index_sorted[0:int(0.3 * len(index_sorted))]]
        features = features_T.T
        # global f
        for f, (train_index, test_index) in enumerate(kf.split(features, labels)):

            # load k-fold data
            x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
                test_index]
            print(Counter(y_train), '\t', Counter(y_test))
            # x_train, x_test = x_train_T.T, x_test_T.T
            Ndim = x_train.shape[1]
            data = [x_train, y_train, x_test, y_test]
            # div param
            divSize, Parlen = Div_Nbr(pNum, divNum, Ndim)
            temp = np.array([1, 2, 3, 4, 5])
            divSize = 10 + np.around(20 / 2 * (1 + np.tanh((temp - np.median(temp)) ** 3 / 10)))
            divSize = divSize.astype(int)
            # init Particles
            Particles = Swarm(data, pNum, Ndim, w0, w1, VRmax, VRmin, impflag, selflag, cls, divSize, Parlen)

            # search features subset
            for e in tqdm(range(max_gen)):
                Particles.updateSwarm(data, divNum)
                gbest_fit.append(Particles.gbest_fit)
                gbest_acc.append(Particles.test_bacc)
                # Particles.LenChange()
                if Particles.lenflag >= beta:
                    print('len changed', end=' ')
                    Particles.LenChange()
                    Particles.lenflag = 0
                save_data(data_dict, 'w')
                gbest_line['fold'].append(f)
                gbest_line['epoch'].append(e)
                gbest_line['gbest_fit'].append(Particles.gbest_fit)
                sn = np.where(Particles.gbest_set == 1)[0].shape[0]
                gbest_line['select_num'].append(sn)
            print(
                '{} fold---gbest fitness:{:.4f},gbest acc:{:.4f},select num:{}'.format(f, gbest_fit[-1], gbest_acc[-1],
                                                                                       Counter(
                                                                                           Particles.gbest_set == 1)[
                                                                                           1]))

            data = pd.DataFrame({
                'EPOCH': mycount_ep,
                'RATE': mycount_num,
                'F1': cf1,
                'F2:': cf2,
                'F3': cf3
            })
            save_gbest(gbest_line, 'w')
            data.to_excel('./hyperresult/HE/FP2.xlsx', index=False)
            save_data(data_dict, 'w')
            # cal average value
            k_acc = k_acc + Particles.test_bacc
            k_size = k_size + Counter(Particles.gbest_set == 1)[1]
        parameter_dict['avg_bacc'].append(k_acc / 10.0)
        parameter_dict['avg_size'].append(k_size / 10.0)
        save_data(parameter_dict, 'a')
        print('average precision:{:.4f}\taverage size:{:.2f}'.format(k_acc / 10.0, k_size / 10.0))
        print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
