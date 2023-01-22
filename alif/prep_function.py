import numpy as np
import pandas as pd

def test_preprocessing(data, coefs):
    remove_col = coefs[-1]
    coefs = coefs[:-1]
    n = len(coefs)
    for i in range(n-1):
        if i not in remove_col:
            if i in [1, 3, 4, 7, 8, 9, 12, 13]:
                data.iloc[:, i].replace(coefs[i][0], inplace=True)
            else:
                data.iloc[:, i] = (data.iloc[:, i] - coefs[i][0]) / coefs[i][1] 
            data.iloc[:, i].fillna(coefs[i][-1], inplace=True)
    
    data = data.drop(data.columns.values[remove_col], axis=1)
    data = np.c_[np.ones(shape=(data.shape[0], 1)), data.values[:, :-1]]
    return data