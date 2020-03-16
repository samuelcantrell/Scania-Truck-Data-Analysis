# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:57:33 2020

@author: scant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


data_train = pd.read_csv("train_new.csv")
data_test = pd.read_csv("test_new.csv")
# print(data_train.head())

data_train.pop('cd_000')
data_test.pop('cd_000')

X_train = data_train.to_numpy()[:, 2:]
X_test = data_test.to_numpy()[:, 2:]
# print(X_train)

N_train, D_train = X_train.shape
N_test, D_test = X_test.shape

if D_train != D_test:
    print("Something went horribly wrong!")

else:
    for i in range(1, D_train):
        up = max(np.max(X_train[:, i]), np.max(X_test[:, i]))
        down = min(np.min(X_train[:, i]), np.min(X_test[:, i]))

        if (up - down) != 0:
            X_train[:, i] = (X_train[:, i] - down) / (up - down)
            X_test[:, i] = (X_test[:, i] - down) / (up - down)

    df = pd.DataFrame(X_train)
    df.columns = data_train.columns.values[2:]
    df.to_csv("train_new_norm.csv", header=True, index=False)

    df = pd.DataFrame(X_test)
    df.columns = data_test.columns.values[2:]
    df.to_csv("test_new_norm.csv", header=True, index=False)
