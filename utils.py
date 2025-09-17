import math
import time

import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler as MM
from collections import Counter

from scipy.io import arff
import pandas as pd

import SU




def ManDistance(f1, f2):

    dist = np.mean(np.abs(f1 - f2))

    return dist


def EuDistance(f1, f2):

    dist = np.sqrt(np.sum((f1 - f2) ** 2))

    return dist


def CosDistance(f1, f2):

    if np.all(f1 == 0) or np.all(f2 == 0):
        dist = 0.0
    else:
        dist = f1.dot(f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))

    return dist


def Tri_Distance(f1, f2):
    Eul = EuDistance(f1, f2)
    Man = ManDistance(f1, f2)
    Cos = CosDistance(f1, f2)

    X = [Man, Eul, Cos]

    return X


def z_score(distance):
    avg = np.mean(distance, axis=1)
    z_score = np.abs(distance - avg[:, np.newaxis])
    row, col = np.where(z_score > 3)  # return index
    return row, col

def nanz_score(distance):

    np.fill_diagonal(distance, np.nan)

    avg = np.nanmean(distance, axis=1)
    z_score = np.abs(distance - avg[:, np.newaxis])
    row, col = np.where(z_score > 3)  # return index
    return row, col


def IQR(distance):
    q1 = np.quantile(distance, 0.25, axis=1)
    q3 = np.quantile(distance, 0.75, axis=1)
    IQR = q3 - q1
    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR
    lower_bound_col = lower_bound[:, np.newaxis]
    upper_bound_col = upper_bound[:, np.newaxis]
    up_index = np.where(distance - upper_bound_col >= 0)
    low_index = np.where(distance - lower_bound_col <= 0)
    return up_index, low_index


def nanIQR(distance):

    np.fill_diagonal(distance, np.nan)


    q1 = np.nanquantile(distance, 0.25, axis=1)
    q3 = np.nanquantile(distance, 0.75, axis=1)
    IQR = q3 - q1


    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR


    up_index = np.where(distance - upper_bound[:, np.newaxis] >= 0)
    low_index = np.where(distance - lower_bound[:, np.newaxis] <= 0)

    return up_index, low_index




