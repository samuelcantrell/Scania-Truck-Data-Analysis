# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:43:13 2020

@author: scant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Generate and plot correlation matricies for the two imputation methods.
"""
data_mean = pd.read_csv("impute_mean.csv")
Cor_mean = data_mean.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(Cor_mean.abs(), annot=False)
plt.axes().set_title("Mean")
plt.show()
"""
data_median = pd.read_csv("impute_median.csv")
Cor_median = data_median.corr()
"""
plt.figure(figsize=(12, 8))
sns.heatmap(Cor_median.abs(), annot=False)
plt.axes().set_title("Median")
plt.show()
"""
# Let's look to the median matrix for further analysis.
Cor_median = Cor_median.abs()
target = np.stack((Cor_median.columns.values, Cor_median['class'].to_numpy()), axis=1)
target = target[np.argsort(target[:, 1])][::-1]

X = Cor_median.to_numpy()
N, D = X.shape  # Row, Col
trash = np.empty(shape=(0, 3))

for j in range(D):
    for i in range(N-j):  # Square, symmetric
        if X[i][j] >= 0.9 and Cor_median.columns.values[i] != Cor_median.columns.values[j]:
            trash = np.append(trash, np.array([Cor_median.columns.values[i],
                                              Cor_median.columns.values[j],
                                              X[i][j]]).reshape(1, 3), axis=0)  # Give pairs

df = pd.DataFrame(target)
df.to_csv("Scania_Target.csv", header=False, index=False)

df = pd.DataFrame(trash)
df.to_csv("Scania_Trash.csv", header=False, index=False)
