# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:47:22 2020

@author: scant
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


data_train = pd.read_csv("train_new_norm.csv")
# print(data_train.head())

X_train = data_train.to_numpy()
N_train, D_train = X_train.shape

ratio = 0.6  # Ratio of Min to Majority cases.

n_train = (round((len(X_train[X_train[:, 0] == 0]) / (1 - ratio)) -
                 len(X_train)))
# print("Number of Train Cases For Ratio: " + str(n_train))

# Now find how many copies to make
n_copies = round(n_train / len(X_train[X_train[:, 0] == 1]))
n_train = n_copies * len(X_train[X_train[:, 0] == 1])
print("\nNumber of Train Cases Imputed of Minority Class: " + str(n_train))

X_carbon = X_train[X_train[:, 0] == 1]  # Template for SMOTE
N_carbon, D_carbon = X_carbon.shape

if D_train != D_carbon:  # Sanity check
    print("\nSomething has gone horribly wrong!")

else:
    for z in range(n_copies):
        X_train = np.append(X_train, X_carbon, axis=0)

    X_train = np.abs(X_train)
    np.random.shuffle(X_train)
    df = pd.DataFrame(X_train)
    df.columns = data_train.columns.values
    df.to_csv("train_new_norm_oversampled.csv", header=True, index=False)
    print("\nImputation Complete!")
