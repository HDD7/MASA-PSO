import numpy as np
from scipy.stats import wilcoxon

import numpy as np


def compare_algorithms(res1, res2, confidegree=0.05):
    """

    results1:
    results2:
    confidegree: 0.05

    Reuturn:

    """

    stat, p_value = wilcoxon(res1, res2)

    if p_value > confidegree:
        return 0
    elif np.mean(res1) > np.mean(res2):
        return 1
    else:
        return -1

