import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.preprocessing import MinMaxScaler as MM

import minepy as MINE

import copent
def CE(features, labels,k=2,dtype='euclidean'):
    features_T = features.T
    ce_list = []
    for f in features_T:
        X = np.vstack([f, labels]).T
        ce = copent.copent(X,k)
        ce_list.append(ce)
    ce_list = np.array(ce_list)
    return ce_list






