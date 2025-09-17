#!/usr/bin/env Python
# coding=utf-8
import random

import numpy as np
import pandas as pd
import sklearn.metrics
from tqdm import tqdm

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold,LeaveOneOut,cross_val_predict
from sklearn.preprocessing import MinMaxScaler as MM
from collections import Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, pairwise_distances
from sklearn.base import clone
from scipy.io import arff
import copent

from sklearn.manifold import TSNE
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
max_gen = 100
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
        # pbest
        self.pbest = None
        self.pset = None
        self.acc = 0
        self.bacc = 0
        self.examplars = []  # examplar pID
        self.impflag = 0  # i^th is changed
        self.selflag = selflag  # feature selected threshold
        self.Db = 0
        self.Dw = 0
        self.pDb = 0
        self.pDw = 0
        self.dst = 0
        self.pdst = 0
        self.forget = 0
        self.data=data
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

        self.fit_func2(data, isDst=False)
        self.pbest = self.fitness
        self.pset = self.set
        self.pacc = self.acc
        self.pbacc = self.bacc
        self.pdst = self.dst

    # update single particle velocity, position
    def updateSingle(self, data, Block,  w):

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
        self.fit_func2(data)
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
        # append more features
        if self.dim < parLen:
            for i in range(self.dim, parLen):
                self.pos[i] = np.random.uniform(0, 1)
                self.vel[i] = np.random.uniform(0, 1) * (VRmax - VRmin) + VRmin
                if self.pos[i] > self.selflag:
                    self.set[i] = 1
            self.set = self.set.astype(int)

            self.dim = parLen

            self.fit_func2(data)


            if self.fitness > self.pbest:
                self.pbest = self.fitness
                self.pset = self.set
                self.pacc = self.acc
                self.pbacc = self.bacc
                self.pDb = self.Db
                self.pDw = self.Dw
                self.pdst = self.dst

        # remove redundance features
        elif self.dim > parLen:
            for i in range(parLen, self.dim):
                self.set[i] = 0
            self.set = self.set.astype(int)

            self.dim = parLen

            self.forget = self.fitness
            self.fit_func2(data)
            self.forget = self.forget - self.fitness  # ，<0 is forget

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

        # roulette
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
            # out of boundary
            for col in range(self.dim, self.maxLen):
                col_list.append(col)
            x_train_subset = np.delete(x_train, col_list, axis=1)
            # select classifier
            cls = KNN(n_neighbors=1, p=1)
            # cls=svm.SVC()
            # cls = XGBClassifier(n_estimators=30, max_depth=1)
            # cls=KNN(n_neighbors=3)
            # cls = GaussianNB()
            cls.fit(x_train_subset, np.ravel(y_train))
            y_pred = cls.predict(x_train_subset)
            acc = 0 # accuracy_score(y_train, y_pred)
            balance_acc = balanced_accuracy_score(y_train, y_pred)
            pair_dis = (pairwise_distances(x_train_subset, metric='manhattan') / x_train_subset.shape[1]).astype(
                np.float32)  # TPR,average='macro'
            distance, Dw, Db = fast_cal(x_train_subset, y_train, pair_dis)
            tempfitness = gamma * balance_acc + (1 - gamma) * distance

        # elite strategy
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

        # cal fitness
        x_train, y_train = self.data[0], self.data[1]
        col_list = []
        for col in range(0, self.dim):
            if tempPos[col] < self.selflag:
                col_list.append(col)
        if len(col_list) >= self.dim:
            tempfitness = 0
        else:
            # out of boundary
            for col in range(self.dim, self.maxLen):
                col_list.append(col)
            x_train_subset = np.delete(x_train, col_list, axis=1)
            # select classifier
            cls = KNN(n_neighbors=1, p=1)
            # cls=svm.SVC()
            # cls = XGBClassifier(n_estimators=30, max_depth=1)
            # cls=KNN(n_neighbors=3)
            # cls = GaussianNB()
            cls.fit(x_train_subset, np.ravel(y_train))
            y_pred = cls.predict(x_train_subset)
            acc = 0 #accuracy_score(y_train, y_pred)
            balance_acc = balanced_accuracy_score(y_train, y_pred)
            pair_dis = (pairwise_distances(x_train_subset, metric='manhattan')/x_train_subset.shape[1]).astype(np.float32)
            distance, Dw, Db = fast_cal(x_train_subset, y_train, pair_dis)
            tempfitness = gamma * balance_acc + (1 - gamma) * distance

        # elite strategy
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
        """reset particle"""
        self.fitness = 0
        self.acc = 0
        self.bacc = 0
        self.dst = 0

    def fit_func2(self, data,  isDst=True):

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

        numclass = 11
        confusion_matrix = np.zeros((numclass, numclass))
        nbr_inst = len(y_train)
        pair_dis = (pairwise_distances(x_train_subset, metric='manhattan')/x_train_subset.shape[1]).astype(np.float32)


        for i in range(nbr_inst):

            nearest_ins = 0
            min_distance = 1.7976931348623157E308  # define MAX VALUE
            for j in range(nbr_inst):
                if i != j:

                    if pair_dis[i][j] < min_distance:
                        min_distance = pair_dis[i][j]
                        nearest_ins = j

            try:
                confusion_matrix[y_train[i]][y_train[nearest_ins]] = confusion_matrix[y_train[i]][
                                                                                     y_train[nearest_ins]] + 1
            except:
                print()
        bacc = unbalanceAcc(confusion_matrix)

        self.acc = 0
        self.bacc = bacc

        if isDst:
            distance, Dw, Db = fast_cal(x_train_subset, y_train, pair_dis)
            self.fitness = gamma * self.bacc + (1 - gamma) * distance
            self.dst = distance
            self.Db, self.Dw = Db, Dw
        else:
            self.fitness = gamma * self.bacc
            self.dst = 0

def Manha(x, y):
    '''man dst cal'''
    return np.mean(np.abs(x - y))

def unbalanceAcc(confusion_matrix):
    '''for loovc test'''
    nbr_class_has_instances = 0.0
    acc = 0.0
    for i in range(len(confusion_matrix)):
        sum_row = 0
        for j in range(len(confusion_matrix[i])):
            sum_row = sum_row + confusion_matrix[i][j]
        if sum_row != 0:
            nbr_class_has_instances = nbr_class_has_instances + 1
            acc = acc + confusion_matrix[i][i] / sum_row
    return acc / nbr_class_has_instances

class Swarm:

    def __init__(self, data, pNum, Ndim, w0, w1, VRmax, VRmin, impflag, selflag,
                  divSize, Parlen):
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
        # x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
        self.pNum = pNum
        self.dim = Ndim
        self.w0 = w0
        self.w1 = w1
        self.Vmax = VRmax
        self.Vmin = VRmin
        # self.generation = 0
        self.impflag = impflag
        self.selflag = selflag
        self.lenflag = 0
        self.Block = []
        self.blockExamplar = []
        self.divSize = divSize
        self.ParLen = Parlen
        # self.cls = classifier

        self.gbest_fit = -99999999
        self.gbest_set = np.random.uniform(0, 0, Ndim)
        self.gbest_acc = 0
        self.gbest_bacc = 0
        # self.gbest_acc = []
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
        # init particles,
        for divID in tqdm(range(divNum)):
            Particles = []
            if divID < divNum - 1:
                # init each Block
                for pID in range(divSize[divID]):
                    # Pc = 0.05 + 0.45 * (np.exp((10 * pID) / (divSize - 1)) - 1) / (np.exp(10) - 1)
                    Particles.append(
                        Particle(data, pID, divID, Ndim, Parlen[divID], self.Vmax, self.Vmin, selflag))
            elif pNum % divNum == 0:
                for pID in range(divSize[divID]):
                    # Pc = 0.05 + 0.45 * (np.exp((10 * pID) / (divSize - 1)) - 1) / (np.exp(10) - 1)
                    Particles.append(Particle(data, pID, divID, Ndim, Parlen[divID], self.Vmax, self.Vmin, selflag))
            else:
                temp_num = pNum % divNum
                for pID in range(temp_num):
                    Particles.append(Particle(data, pID, divID, Ndim, Parlen[divID], self.Vmax, self.Vmin, selflag))
            self.Block.append(Particles)
        self.checkRank()  # init Pc and rank


    def updateSwarm(self, data, divNum):

        w = 0.8
        flag = 0
        for divID in range(divNum):
            Block = self.Block
            Particles = self.Block[divID]
            pnum = len(Particles)

            # updata particles of each blocks
            for pID in range(pnum):
                if Particles[pID].impflag > self.impflag:

                    for d in range(Particles[pID].dim):
                        # update exemplar
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
                Particles[pID].updateSingle(data, Block,  w)

                self.checkRank()
                # self.checkBlockExamplar()
                # check gbest
                if Particles[pID].pbest > self.gbest_fit:
                    self.gbest_set = Particles[pID].pset[:Particles[pID].dim]
                    self.gbest_fit = Particles[pID].pbest
                    self.gbest_acc = Particles[pID].pacc
                    self.gbest_bacc = Particles[pID].pbacc
                    self.bestBlock = divID
                    self.bestParticle = pID
                    self.gDb = Particles[pID].pDb
                    self.gDw = Particles[pID].pDw
                    self.gdst = Particles[pID].pdst
                    self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test, y_test)
                    print(
                        'global best:Block {} particle {}, fitness={:.10f},acc={:.4f},bacc={:.4f},dst={},Db={},Dw={}\n select_num={},test_acc={},test_bacc={}'.format(
                            self.bestBlock, pID, self.gbest_fit,
                            self.gbest_acc, self.gbest_bacc, self.gdst, self.gDb,
                            self.gDw,
                            np.where(
                                self.gbest_set == 1)[
                                0].shape[0], self.test_acc, self.test_bacc))
                    self.lenflag = 0
                    # data_dict['fold'].append(f)
                    # data_dict['epoch'].append(e)
                    # data_dict['divID'].append(self.bestBlock)
                    # data_dict['pID'].append(self.bestParticle)
                    # data_dict['gbest_fit'].append(self.gbest_fit)
                    # data_dict['gbest_acc'].append(self.gbest_acc)
                    # data_dict['gbest_bacc'].append(self.gbest_bacc)
                    # data_dict['gbest_set'].append(np.where(self.gbest_set == 1)[0])
                    # data_dict['select_num'].append(np.where(self.gbest_set == 1)[0].shape[0])
                    # data_dict['bestLen'].append(self.bestLen)
                    # data_dict['distance'].append(self.gdst)
                    # data_dict['test_acc'].append(self.test_acc)
                    # data_dict['test_bacc'].append(self.test_bacc)
                    flag = flag + 1

        if flag == 0:
            self.lenflag = self.lenflag + 1

    def cal_test_bacc(self, x_train, y_train, x_test, y_test):
        # select classifier
        cls = KNN(n_neighbors=1,p=1)
        # cls=svm.SVC()
        # cls = XGBClassifier(n_estimators=30, max_depth=1)
        # cls=KNN(n_neighbors=3)
        # cls = GaussianNB()

        set = self.gbest_set

        f_num=x_train.shape[1]

        extended_mask = np.pad(set.astype(bool),(0, f_num - len(set)),
                      mode='constant', constant_values=False)

        x_train_subset = x_train[:, extended_mask]
        x_test_subset = x_test[:, extended_mask]
        cls.fit(x_train_subset, y_train)


        y_pred = cls.predict(x_test_subset)

        acc = 0
        bacc = balanced_accuracy_score(y_test, y_pred)

        return acc, bacc

    def getEliteBoundary(self, avgBlock):
        """
        get elite particle set
        :param avgBlock:
        :return:
        """
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
        """
        get the low and up boundary
        :param Degration:
        :return: boundary
        """


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

        set_list = np.array(set_list)
        set_static = np.sum(set_list, axis=0)

        return set_static

    def find_bestBlock(self):
        '''
        find Best ave_fitness BLOCK
        divNum,
        :return:bestdivID
        '''
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

        avgBlock = self.find_bestBlock()  # find Bid of best Block
        if self.ParLen[avgBlock] == self.maxLen:
            self.breakflag = self.breakflag + 1
        else:
            self.breakflag = 0

        self.bestLen = self.ParLen[avgBlock]
        print(',gbest block:{},gbest length:{},max length:{}'
              ' --> avg block:{},new max length:{}'.format(self.bestBlock, self.ParLen[self.bestBlock], self.maxLen,
                                                           avgBlock, self.bestLen))

        if self.bestLen != self.maxLen:

            EliteStatic = self.getEliteSet(avgBlock)
            times = 0
            for i in range(divNum):
                if i != avgBlock:
                    newLen = np.ceil(self.bestLen * (times + 1) / divNum)
                    newLen = int(newLen)
                    self.ParLen[i] = newLen

                    Particles = self.Block[i]
                    times = times + 1

                    collect_forget = []
                    for p in Particles:
                        p.changePdim(newLen)

                        if p.pbest > self.gbest_fit:
                            self.gbest_set = p.pset[:p.dim]
                            self.gbest_fit = p.pbest
                            self.gbest_acc = p.pacc
                            self.gbest_bacc = p.pbacc
                            self.bestBlock = i
                            self.bestParticle = p.pID
                            self.gDb = p.pDb
                            self.gDw = p.pDw
                            self.gdst = p.pdst
                            self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test, y_test)
                            print(
                                'global best:Block {} particle {}, fitness={:.10f},acc={:.4f},bacc={:.4f},dst={},Db={},Dw={}\n select_num={},test_acc={},test_bacc={}'.format(
                                    self.bestBlock, p.pID, self.gbest_fit,
                                    self.gbest_acc, self.gbest_bacc, self.gdst, self.gDb,
                                    self.gDw,
                                    np.where(
                                        self.gbest_set == 1)[
                                        0].shape[0], self.test_acc, self.test_bacc))

                            # data_dict['fold'].append(f)
                            # data_dict['epoch'].append(e)
                            # data_dict['divID'].append(self.bestBlock)
                            # data_dict['pID'].append(self.bestParticle)
                            # data_dict['gbest_fit'].append(self.gbest_fit)
                            # data_dict['gbest_acc'].append(self.gbest_acc)
                            # data_dict['gbest_bacc'].append(self.gbest_bacc)
                            # data_dict['gbest_set'].append(np.where(self.gbest_set == 1)[0])
                            # data_dict['select_num'].append(np.where(self.gbest_set == 1)[0].shape[0])
                            # data_dict['bestLen'].append(self.bestLen)
                            # data_dict['distance'].append(self.gdst)
                            # data_dict['test_acc'].append(self.test_acc)
                            # data_dict['test_bacc'].append(self.test_bacc)

                            self.lenflag = 0

                        collect_forget.append(p.forget)
                    # sort for forget
                    collect_forget = np.array(collect_forget)

                    forget_index = np.where(collect_forget < 0)[0]
                    if forget_index.shape[0] > 0:
                        forget_index = forget_index.astype(int)
                        Particles = np.array(Particles)
                        DeUpBoundary, DeLowBoundary = self.getDeBoundary(Particles[forget_index])
                        index_sorted = np.argsort(collect_forget)  # ascending sort
                        lb = int(0.2 * len(index_sorted))

                        # elite opposite-based learning
                        for idx in range(len(index_sorted)):
                            id = index_sorted[idx]
                            if idx < lb and collect_forget[id] < 0:
                                Particles[id].EOL(DeLowBoundary, DeUpBoundary)
                                if Particles[id].pbest > self.gbest_fit:
                                    self.gbest_set = Particles[id].pset[:Particles[id].dim]
                                    self.gbest_fit = Particles[id].pbest
                                    self.gbest_acc = Particles[id].pacc
                                    self.gbest_bacc = Particles[id].pbacc
                                    self.bestBlock = i
                                    self.bestParticle = id
                                    self.gDb = Particles[id].pDb
                                    self.gDw = Particles[id].pDw
                                    self.gdst = Particles[id].pdst
                                    self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test, y_test)
                                    print(
                                        'global best:Block {} particle {}, fitness={:.10f},acc={:.4f},bacc={:.4f},dst={},Db={},Dw={}\n select_num={},test_acc={},test_bacc={}'.format(
                                            self.bestBlock, id, self.gbest_fit,
                                            self.gbest_acc, self.gbest_bacc, self.gdst, self.gDb,
                                            self.gDw,
                                            np.where(
                                                self.gbest_set == 1)[
                                                0].shape[0], self.test_acc, self.test_bacc))
                                    self.lenflag = 0
                                    # data_dict['fold'].append(f)
                                    # data_dict['epoch'].append(e)
                                    # data_dict['divID'].append(self.bestBlock)
                                    # data_dict['pID'].append(self.bestParticle)
                                    # data_dict['gbest_fit'].append(self.gbest_fit)
                                    # data_dict['gbest_acc'].append(self.gbest_acc)
                                    # data_dict['gbest_bacc'].append(self.gbest_bacc)
                                    # data_dict['gbest_set'].append(np.where(self.gbest_set == 1)[0])
                                    # data_dict['select_num'].append(np.where(self.gbest_set == 1)[0].shape[0])
                                    # data_dict['bestLen'].append(self.bestLen)
                                    # data_dict['distance'].append(self.gdst)
                                    # data_dict['test_acc'].append(self.test_acc)
                                    # data_dict['test_bacc'].append(self.test_bacc)
                            elif idx >= lb and idx < 2 * lb and collect_forget[id] < 0:
                                # elite forward learning
                                Particles[id].EFL(EliteStatic)
                                if Particles[id].pbest > self.gbest_fit:
                                    self.gbest_set = Particles[id].pset[:Particles[id].dim]
                                    self.gbest_fit = Particles[id].pbest
                                    self.gbest_acc = Particles[id].pacc
                                    self.gbest_bacc = Particles[id].pbacc
                                    self.bestBlock = i
                                    self.bestParticle = id
                                    self.gDb = Particles[id].pDb
                                    self.gDw = Particles[id].pDw
                                    self.gdst = Particles[id].pdst
                                    self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test, y_test)
                                    print(
                                        'global best:Block {} particle {}, fitness={:.10f},acc={:.4f},bacc={:.4f},dst={},Db={},Dw={}\n select_num={},test_acc={},test_bacc={}'.format(
                                            self.bestBlock, id, self.gbest_fit,
                                            self.gbest_acc, self.gbest_bacc, self.gdst, self.gDb,
                                            self.gDw,
                                            np.where(
                                                self.gbest_set == 1)[
                                                0].shape[0], self.test_acc, self.test_bacc))
                                    self.lenflag = 0
                                    # data_dict['fold'].append(f)
                                    # data_dict['epoch'].append(e)
                                    # data_dict['divID'].append(self.bestBlock)
                                    # data_dict['pID'].append(self.bestParticle)
                                    # data_dict['gbest_fit'].append(self.gbest_fit)
                                    # data_dict['gbest_acc'].append(self.gbest_acc)
                                    # data_dict['gbest_bacc'].append(self.gbest_bacc)
                                    # data_dict['gbest_set'].append(np.where(self.gbest_set == 1)[0])
                                    # data_dict['select_num'].append(np.where(self.gbest_set == 1)[0].shape[0])
                                    # data_dict['bestLen'].append(self.bestLen)
                                    # data_dict['distance'].append(self.gdst)
                                    # data_dict['test_acc'].append(self.test_acc)
                                    # data_dict['test_bacc'].append(self.test_bacc)

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
                                self.gbest_set = p.pset[:p.dim]
                                self.gbest_fit = p.pbest
                                self.gbest_acc = p.pacc
                                self.gbest_bacc = p.pbacc
                                self.bestBlock = i
                                self.bestParticle = p.pID
                                self.gDb = p.pDb
                                self.gDw = p.pDw
                                self.gdst = p.pdst
                                self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test, y_test)
                                print(
                                    'global best:Block {} particle {}, fitness={:.10f},acc={:.4f},bacc={:.4f},dst={},Db={},Dw={}\n select_num={},test_acc={},test_bacc={}'.format(
                                        self.bestBlock, p.pID, self.gbest_fit,
                                        self.gbest_acc, self.gbest_bacc, self.gdst, self.gDb,
                                        self.gDw,
                                        np.where(
                                            self.gbest_set == 1)[
                                            0].shape[0], self.test_acc, self.test_bacc))

                                # data_dict['fold'].append(f)
                                # data_dict['epoch'].append(e)
                                # data_dict['divID'].append(self.bestBlock)
                                # data_dict['pID'].append(self.bestParticle)
                                # data_dict['gbest_fit'].append(self.gbest_fit)
                                # data_dict['gbest_acc'].append(self.gbest_acc)
                                # data_dict['gbest_bacc'].append(self.gbest_bacc)
                                # data_dict['gbest_set'].append(np.where(self.gbest_set == 1)[0])
                                # data_dict['select_num'].append(np.where(self.gbest_set == 1)[0].shape[0])
                                # data_dict['bestLen'].append(self.bestLen)
                                # data_dict['distance'].append(self.gdst)
                                # data_dict['test_acc'].append(self.test_acc)
                                # data_dict['test_bacc'].append(self.test_bacc)

                                self.lenflag = 0

                            collect_forget.append(p.forget)
                        # sort for forget
                        collect_forget = np.array(collect_forget)

                        forget_index = np.where(collect_forget < 0)[0]
                        if forget_index.shape[0] > 0:
                            forget_index = forget_index.astype(int)
                            Particles = np.array(Particles)
                            DeUpBoundary, DeLowBoundary = self.getDeBoundary(Particles[forget_index])
                            index_sorted = np.argsort(collect_forget)  # ascending sort
                            lb = int(0.2 * len(index_sorted))
                            # elite opposite-based learning
                            for idx in range(len(index_sorted)):
                                id = index_sorted[idx]
                                if idx < lb and collect_forget[id] < 0:
                                    Particles[id].EOL(DeLowBoundary, DeUpBoundary)
                                    if Particles[id].pbest > self.gbest_fit:
                                        self.gbest_set = Particles[id].pset[:Particles[id].dim]
                                        self.gbest_fit = Particles[id].pbest
                                        self.gbest_acc = Particles[id].pacc
                                        self.gbest_bacc = Particles[id].pbacc
                                        self.bestBlock = i
                                        self.bestParticle = id
                                        self.gDb = Particles[id].pDb
                                        self.gDw = Particles[id].pDw
                                        self.gdst = Particles[id].pdst
                                        self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test,y_test)
                                        print(
                                            'global best:Block {} particle {}, fitness={:.10f},acc={:.4f},bacc={:.4f},dst={},Db={},Dw={}\n select_num={},test_acc={},test_bacc={}'.format(
                                                self.bestBlock, id, self.gbest_fit,
                                                self.gbest_acc, self.gbest_bacc, self.gdst, self.gDb,
                                                self.gDw,
                                                np.where(
                                                    self.gbest_set == 1)[
                                                    0].shape[0], self.test_acc, self.test_bacc))
                                        self.lenflag = 0
                                        # data_dict['fold'].append(f)
                                        # data_dict['epoch'].append(e)
                                        # data_dict['divID'].append(self.bestBlock)
                                        # data_dict['pID'].append(self.bestParticle)
                                        # data_dict['gbest_fit'].append(self.gbest_fit)
                                        # data_dict['gbest_acc'].append(self.gbest_acc)
                                        # data_dict['gbest_bacc'].append(self.gbest_bacc)
                                        # data_dict['gbest_set'].append(np.where(self.gbest_set == 1)[0])
                                        # data_dict['select_num'].append(np.where(self.gbest_set == 1)[0].shape[0])
                                        # data_dict['bestLen'].append(self.bestLen)
                                        # data_dict['distance'].append(self.gdst)
                                        # data_dict['test_acc'].append(self.test_acc)
                                        # data_dict['test_bacc'].append(self.test_bacc)
                                elif idx >= lb and idx < 2 * lb and collect_forget[id] < 0:
                                    # elite forward learning
                                    Particles[id].EFL(EliteSet)
                                    if Particles[id].pbest > self.gbest_fit:
                                        self.gbest_set = Particles[id].pset[:Particles[id].dim]
                                        self.gbest_fit = Particles[id].pbest
                                        self.gbest_acc = Particles[id].pacc
                                        self.gbest_bacc = Particles[id].pbacc
                                        self.bestBlock = i
                                        self.bestParticle = id
                                        self.gDb = Particles[id].pDb
                                        self.gDw = Particles[id].pDw
                                        self.gdst = Particles[id].pdst
                                        self.test_acc, self.test_bacc = self.cal_test_bacc(x_train, y_train, x_test,y_test)
                                        print(
                                            'global best:Block {} particle {}, fitness={:.10f},acc={:.4f},bacc={:.4f},dst={},Db={},Dw={}\n select_num={},test_acc={},test_bacc={}'.format(
                                                self.bestBlock, id, self.gbest_fit,
                                                self.gbest_acc, self.gbest_bacc, self.gdst, self.gDb,
                                                self.gDw,
                                                np.where(
                                                    self.gbest_set == 1)[
                                                    0].shape[0], self.test_acc, self.test_bacc))
                                        self.lenflag = 0
                                        # data_dict['fold'].append(f)
                                        # data_dict['epoch'].append(e)
                                        # data_dict['divID'].append(self.bestBlock)
                                        # data_dict['pID'].append(self.bestParticle)
                                        # data_dict['gbest_fit'].append(self.gbest_fit)
                                        # data_dict['gbest_acc'].append(self.gbest_acc)
                                        # data_dict['gbest_bacc'].append(self.gbest_bacc)
                                        # data_dict['gbest_set'].append(np.where(self.gbest_set == 1)[0])
                                        # data_dict['select_num'].append(np.where(self.gbest_set == 1)[0].shape[0])
                                        # data_dict['bestLen'].append(self.bestLen)
                                        # data_dict['distance'].append(self.gdst)
                                        # data_dict['test_acc'].append(self.test_acc)
                                        # data_dict['test_bacc'].append(self.test_bacc)

                self.maxLen = self.bestLen

    def checkRank(self):
        for div in range(divNum):
            Particles = self.Block[div]
            collect_fit = []
            for p in Particles:
                collect_fit.append(p.fitness)
            collect_fit = np.array(collect_fit)
            index_sorted = np.argsort(-1 * collect_fit)  # ascending sort
            for i in range(len(Particles)):
                Particles[i].rankID = index_sorted[i]
                Particles[i].Pc = 0.05 + 0.45 * (np.exp((10 * index_sorted[i]) / (divSize[div] - 1)) - 1) / (
                    np.exp(10) - 1)






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def manhattan_dist(x1, x2):
    return np.sum(np.abs(x1 - x2))

def fast_cal(features, labels, pair_dis):
    """
    cal distance between each distance base feature subset
    :param features: must be ndarray
    :param labels:
    :return:
    """
    prior = features.shape[1]
    if prior <= 45: # SRBCT-40, BT2-75
        distance = 0
        Dw=0
        Db=0
    else:
        norms = np.linalg.norm(features, axis=1)  # get norm for each row vector(instance vce)

        Mandst_matrix = pair_dis
        Euldst_matrix = pairwise_distances(features, metric='euclidean')
        Cosdst_matrix = 1 - np.dot(features, features.T) / (norms[:, np.newaxis] * norms[np.newaxis, :])

        # nan to 0 in cos dst
        if True in np.isnan(Cosdst_matrix):
            Cosdst_matrix[np.where(np.isnan(Cosdst_matrix) == True)] = 0


        label_types = np.unique(labels)

        type_index = []
        for i in label_types:
            type_index.append(np.where(labels == i)[0])
        # within class
        Dwmin = []
        for i in range(len(label_types)):
            if len(type_index[i]) <= 1:
                continue
            else:
                Mtemp = Mandst_matrix[type_index[i]]
                Mtemp = Mtemp[:, type_index[i]]
                Etemp = Euldst_matrix[type_index[i]]
                Etemp = Etemp[:, type_index[i]]
                Ctemp = Cosdst_matrix[type_index[i]]
                Ctemp = Ctemp[:, type_index[i]]
                # outlier eliminate
                row, col = utils.z_score(Etemp)
                Etemp[row, col] = 0
                row, col = utils.z_score(Mtemp)
                Mtemp[row, col] = 0
                row, col = utils.z_score(Ctemp)
                Ctemp[row, col] = 0
                X = np.vstack((np.max(Mtemp, axis=1), np.max(Etemp, axis=1), np.max(Ctemp, axis=1)))
                cov = np.cov(X)


                inv_cov = np.linalg.pinv(cov).astype(np.float32)
                score = np.sqrt(np.dot(np.dot(X.T, inv_cov), X))
                diag_score = np.diagonal(score)  # aggregation score
                Dwmin.append(np.mean(diag_score))
        Dw = np.mean(Dwmin)

        # between class
        dif_type_index = []
        for i in label_types:
            dif_type_index.append(np.where(labels != i)[0])
        Dbmax = []
        for i in range(len(label_types)):
            if len(type_index[i]) <= 1:
                continue
                # Dwmin.append(np.mean(diag_score))
            else:
                Mtemp = Mandst_matrix[type_index[i]]
                Mtemp = Mtemp[:, dif_type_index[i]]
                Etemp = Euldst_matrix[type_index[i]]
                Etemp = Etemp[:, dif_type_index[i]]
                Ctemp = Cosdst_matrix[type_index[i]]
                Ctemp = Ctemp[:, dif_type_index[i]]
                # remove outliers
                row, col = utils.z_score(Etemp)
                Etemp[row, col] = 99999
                row, col = utils.z_score(Mtemp)
                Mtemp[row, col] = 99999
                row, col = utils.z_score(Ctemp)
                Ctemp[row, col] = 1.1
                X = np.vstack((np.min(Mtemp, axis=1), np.min(Etemp, axis=1), np.min(Ctemp, axis=1)))

                cov = np.cov(X)
                inv_cov = np.linalg.pinv(cov).astype(np.float32)
                score = np.sqrt(np.dot(np.dot(X.T, inv_cov), X))  # aggre score
                diag_score = np.diagonal(score)
                Dbmax.append(np.mean(diag_score))
        Db = np.mean(Dbmax)

        delta = 5

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
    with open('./result/forSRBCT{}.txt'.format(run_time), c) as f:
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


def convert_ndarray_labels(arr):
    result = arr.copy()

    pso_mask = np.char.find(arr.astype(str), 'POS') >= 0
    result[pso_mask] = 1


    neg_mask = np.char.find(arr.astype(str), 'NEG') >= 0
    result[neg_mask] = -1

    return result

def readtxt2(file):
    df = pd.read_csv(file, sep='\t')
    features=df.values
    features=(features.T)[3:-1]
    labels=df.keys()[3:-1]
    labels=labels.values
    labels=convert_ndarray_labels(labels)
    labels=labels.astype(int)

    return features, labels


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



import MIC


if __name__ == '__main__':

    run_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    selflag = 0.6
    impflag = 7
    # preprocessing data
    "data. The data of the arff file, accessible by attribute names." \
    "meta Contains information about the arff file " \
    "such as name and type of attributes, the relation (name of the dataset"
    file = ["SRBCT","11_Tumors", "Leukemia1", "9_Tumors", "Brain_Tumor1",
            "DLBCL", "Leukemia2", "Prostate_Tumor", "Brain_Tumor2", "Lung_Cancer"]
    data = pd.read_csv('./datasets/low-sample/{}.csv'.format(file[8]))

    data_arr = data.values
    data_arr_T = data_arr.T
    features = data_arr_T[:-1].T
    scaling = MM(feature_range=(0, 1))
    features = scaling.fit_transform(features)
    labels = data_arr_T[-1]

    features = features.astype(np.float32)
    labels = labels.astype(int)  # bytes to int
    print("Class types:", Counter(labels))


    for r in range(50):

        pNum = 100  # default 100 population size

        VRmax = 0.6
        VRmin = -0.6
        gbest_fit = []
        gbest_acc = []
        beta = 5  #
        alpha = 7
        # K-FOLD EXPERIMENT
        kf = KFold(n_splits=10)
        # kf = StratifiedKFold(n_splits=10, shuffle=True)
        k_acc = 0.0
        k_size = 0.0
        # myseed = np.random.get_state()[1][0]
        parameter_dict = {'seed': [rdseed, npseed], 'max_gen': max_gen, 'w': 0.8, 'beta': beta, 'c': c, 'divNum': divNum,
                          'gamma': gamma, }
        data_dict = {'fold': [], 'epoch': [], 'divID': [], 'pID': [], 'gbest_fit': [], 'gbest_acc': [], 'gbest_bacc': [],
                     'distance': [], 'select_num': [], 'bestLen': [], 'gbest_set': [], 'test_acc': [], 'test_bacc': [], }

        for f, (train_index, test_index) in enumerate(kf.split(features,labels)):

            # load k-fold data
            x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
                test_index]

            mic = MIC.MIC(x_train, y_train)
            # Ranking based mic
            index_sorted = np.argsort(-mic)

            # resort features
            x_train_T, x_test_T = x_train.T, x_test.T
            x_train_T, x_test_T = x_train_T[index_sorted[0:int(0.3 * len(index_sorted))]], x_test_T[
                index_sorted[0:int(0.3 * len(index_sorted))]]
            x_train, x_test = x_train_T.T, x_test_T.T
            Ndim = len(x_train_T)
            data = [x_train, y_train, x_test, y_test]
            # div param
            divSize, Parlen = Div_Nbr(pNum, divNum, Ndim)
            temp = np.array([1, 2, 3, 4, 5])
            divSize = 10 + np.around(20 / 2 * (1 + np.tanh((temp - np.median(temp)) ** 3 / 10)))
            divSize = divSize.astype(int)
            # init Particles
            Particles = Swarm(data, pNum, Ndim, w0, w1, VRmax, VRmin, impflag, selflag, divSize, Parlen)
            s = time.time()
            # search features subset
            for e in tqdm(range(max_gen)):
                Particles.updateSwarm(data, divNum)

                gbest_fit.append(Particles.gbest_fit)
                gbest_acc.append(Particles.test_bacc)

                if Particles.lenflag >= beta:

                    Particles.LenChange()
                    Particles.lenflag = 0
                # save_data(data_dict, 'w')

            print('{} fold---gbest fitness:{:.4f},gbest acc:{:.4f},select num:{}'.format(f, Particles.gbest_fit, Particles.test_bacc,
                                                                                         Counter(Particles.gbest_set == 1)[
                                                                                             1]))

            save_data(data_dict, 'w')
            # cal average value
            k_acc = k_acc + Particles.test_bacc
            k_size = k_size + Counter(Particles.gbest_set == 1)[1]

            save_data(parameter_dict, 'a')

        print('average precision:{:.4f}\taverage size:{:.2f}'.format(k_acc / 10.0, k_size / 10.0))
        print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
