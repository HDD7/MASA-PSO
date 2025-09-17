import time

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.preprocessing import MinMaxScaler as MM
import random
from minepy import MINE

import warnings
# import mictools
warnings.filterwarnings("ignore")

# set random seed
myseed = 893027724
np.random.seed(myseed)
random.seed(myseed)
import copent


def MIC(features, labels):
    mine = MINE()
    features_T = features.T
    mic_list = []
    tic_list=[]
    for f in features_T:
        mine.compute_score(f, labels)
        mic = mine.mic()

        mic_list.append(mic)

    mic_list = np.array(mic_list)

    return mic_list


def MEC(features, labels):
    mine = MINE()
    features_T = features.T
    mec_list = []
    tec_list=[]
    for f in features_T:
        X = np.vstack([f, labels]).T
        CDF_X = copent.construct_empirical_copula(X)
        f_CDF = CDF_X[:, 0]
        l = CDF_X[:, 1]
        mine.compute_score(f_CDF, l)
        mec = mine.mic()
        tec=mine.tic()
        mec_list.append(mec)
        tec_list.append(tec)
    mec_list = np.array(mec_list)
    tec_list = np.array(tec_list)
    return mec_list,tec_list


def div_Bin(x, y, xNum, yNUm):
    """
    cal X,Y joint p matrix
    :param x:
    :param y:
    :param xNum:
    :param yNUm:
    :return:
    """
    p_matrix = np.zeros([xNum, yNUm])
    xBin = np.linspace(np.min(x), np.max(x) + 1, xNum + 1)
    yBin = np.linspace(np.min(y), np.max(y) + 1, yNUm + 1)
    N = x.shape[0]
    for i in range(xNum):
        for j in range(yNUm):
            p_matrix[i, j] = np.sum(
                [1 if x[idx] < xBin[i + 1] and x[idx] >= xBin[i] and y[idx] < yBin[j + 1] and y[idx] >= yBin[j] else 0
                 for idx in range(N)])
    p_matrix = p_matrix / N

    return p_matrix


def div_Bin_fast(x, y, xNum, yNum):
    """
    Calculate joint probability matrix for X and Y
    :param x:
    :param y:
    :param xNum: number of bins for X
    :param yNum: number of bins for Y
    :return: joint probability matrix
    """
    # Calculate bin edges for x and y
    x_bins = np.linspace(np.min(x), np.max(x) + 1, xNum + 1)
    y_bins = np.linspace(np.min(y), np.max(y) + 1, yNum + 1)

    # Calculate the indices of the bins to which each value in x and y
    x_indices = np.digitize(x, x_bins) - 1
    y_indices = np.digitize(y, y_bins) - 1

    # Count the number of occurrences of each pair of indices
    p_matrix = np.histogram2d(x_indices, y_indices, bins=[range(xNum + 1), range(yNum + 1)])[0]

    # Normalize the joint p_matrix
    p_matrix = p_matrix / len(x)

    return p_matrix


# step 2
def cal_MI(p_matrix):
    MI = 0
    p_matrix = np.array(p_matrix)
    for i in range(p_matrix.shape[0]):
        for j in range(p_matrix.shape[1]):
            if p_matrix[i, j] != 0:
                MI = MI + p_matrix[i, j] * np.log2(p_matrix[i, j] / (p_matrix[i, :].sum() * p_matrix[:, j].sum()))

    return MI / np.log2(min(p_matrix.shape))


def cal_MI_fast(p_matrix):
    p_matrix = np.array(p_matrix)

    # cal margin probability Px,Py
    p_x = p_matrix.sum(axis=1, keepdims=True)
    p_y = p_matrix.sum(axis=0, keepdims=True)

    # cal MI
    mi_matrix = p_matrix * np.log2(p_matrix / (p_x @ p_y))
    mi_matrix[p_matrix == 0] = 0  # set 0 elements
    mi = np.sum(mi_matrix) / np.log2(min(p_matrix.shape))

    return mi



def cal_MIC(features, labels):
    features_T = features.T
    mic_list = []

    x_maxNum = int(round(features.shape[0] ** 0.3, 0))
    y_maxNum = min(x_maxNum, 15)
    x_minNum = 2
    y_minNum = 2

    for f in features_T:  # f(Num,),labels(N,)
        f_mic_list = []
        # X = np.vstack([f, labels]).T
        # CDF_X = copent.construct_empirical_copula(X)
        # f_CDF = CDF_X[:, 0]
        # l_CDF = CDF_X[:, 1]
        for xblocks in range(x_minNum, x_maxNum + 1):
            for yblocks in range(y_minNum, y_maxNum + 1):
                p_matrix = div_Bin_fast(f, labels, xblocks, yblocks)
                mic = cal_MI_fast(p_matrix)
                f_mic_list.append(mic)
        f_maxMic = max(f_mic_list)
        mic_list.append(f_maxMic)
    return mic_list

