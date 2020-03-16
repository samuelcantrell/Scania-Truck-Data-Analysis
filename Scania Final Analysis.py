# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:41:31 2020

@author: scant
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_test = pd.read_csv("test_new_norm.csv")
print(data_test.head())

X_test = data_test.to_numpy()
y_test = X_test[:, 0]  # Define Label for Regression.
X_test = X_test[:, 1:]  # Chop off first column of X.


print(len(y_test[y_test == 1]))

print("Original Cost:")
print(len(y_test[y_test == 1])*500)

print("Adjusted Cost:")
print(len(y_test[y_test == 1])*500 - 8890)