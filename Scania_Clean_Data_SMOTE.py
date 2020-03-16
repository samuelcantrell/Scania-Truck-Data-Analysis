# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:47:22 2020

@author: scant
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# cost = y_hat[y != y_hat and y == 0]*500
data_train = pd.read_csv("train_new_norm.csv")
# print(data_train.head())

X_train = data_train.to_numpy()

N_train, D_train = X_train.shape

pert = 1e-4  # Order of magnitude of the perturbation.
ratio = 0.3  # Ratio of Min to Majority cases.

n_train = (round(len(X_train[X_train[:, 0] == 0])*ratio) -
           len(X_train[X_train[:, 0] == 1]))
# print("Number of Train Cases For Ratio: " + str(n_train))

# Now find how many copies to make
n_copies = round(n_train / len(X_train[X_train[:, 0] == 1]))
n_train = n_copies * len(X_train[X_train[:, 0] == 1])
print("\nNumber of Train Cases Imputed of Minority Class: " + str(n_train))

X_carbon = X_train[X_train[:, 0] == 1]  # Template for SMOTE
N_carbon, D_carbon = X_carbon.shape
X_add = np.ones(shape=(N_carbon, D_carbon))

if D_train != D_carbon:  # Sanity check
    print("\nSomething has gone horribly wrong!")

else:
    for z in range(n_copies):
        for j in range(1, D_carbon):
            for i in range(N_carbon):
                X_add[i][j] = X_carbon[i][j] + np.random.randn()*pert
        X_train = np.append(X_train, X_add, axis=0)

    X_train = np.abs(X_train)
    np.random.shuffle(X_train)
    df = pd.DataFrame(X_train)
    df.columns = data_train.columns.values
    df.to_csv("train_new_norm_SMOTE.csv", header=True, index=False)
    print("\nImputation Complete!")
