import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.preprocessing import MinMaxScaler as MM

def entropy(feature, base=2):
    '''
    return empirical entropy H(F)
    '''
    _, vec = np.unique(feature, return_counts=True)
    prob_vec = np.array(vec / float(sum(vec)))
    if base == 2:
        logb = np.log2
    elif base == 10:
        logb = np.log10
    else:
        logb = np.log
    Hf = prob_vec.dot(-logb(prob_vec))
    return Hf


def conditional_entropy(feature, labels):
    '''
    return H(F|Y)
    '''
    feature = np.array(feature)
    uy, uyc = np.unique(labels, return_counts=True)
    prob_uyc = uyc / float(sum(uyc))
    cond_entropy = []
    for i in uy:
        a = np.where(labels == i)[0]
        cond_entropy.append(entropy(feature[a]))

    return prob_uyc.dot(cond_entropy)


def mutual_information(feature, labels):
    """
    return information gain IG(F|C)
    :param feature:
    :param labels:
    :return:
    """
    return entropy(feature) - conditional_entropy(feature, labels)


def SU(features, labels):
    """
    Return SU(F|C)
    :param features: features set <ndarray> (fNum,Ndim)
    :param labels: <ndarray>
    :return: ig<ndarray>, su<ndarray>
    """
    features_T = features.T
    su = []  # symmetric uncertainty value
    # ig = []  # information gain value
    for feature in features_T:
        su_val = 2 * mutual_information(feature, labels) / (entropy(feature) + entropy(labels))
        # ig.append(ig_val)
        su.append(su_val)

    # ig = np.array(ig)
    su = np.array(su)
    return su

import CE
