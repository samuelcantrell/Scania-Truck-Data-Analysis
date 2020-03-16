# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:52:28 2020

@author: scant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SimpleLinReg():
    def fit(self, X, y):
        # D = X.shape
        # N = D[0]
        d = np.mean(X**2) - np.mean(X)**2

        self.w_0 = (np.mean(y) * np.mean(X**2) - np.mean(X) * np.mean(X*y)) / d
        self.w_1 = (np.mean(X*y) - np.mean(X) * np.mean(y)) / d

    def predict(self, X):
        return self.w_0 + self.w_1 * X


class MultipleLinearRegression():
    def fit(self, X, y):
        self.w = np.linalg.solve(X.T@X, X.T@y)

    def predict(self, X):
        return np.matmul(X, self.w)


class GeneralizedLinearRegression():
    def fit(self, X, Y):
        self.w = np.linalg.lstsq(np.dot(X.T, X), np.dot(X.T, Y))[0]

    def predict(self, X):
        return np.dot(X, self.w)


def R2(X, y_hat):  # Return Average R2 Value
    return np.mean(np.corrcoef(X, y_hat)[0, 1]**2)


def NMSE(X, y_hat):  # Return Normalized MSE
    return (((y_hat - X) ** 2).mean() /
            (max(np.max(X), np.max(y_hat)) -
            min(np.min(X), np.min(y_hat))))


data_train = pd.read_csv("train_clean_norm_SMOTE.csv")
data_test = pd.read_csv("test_clean_norm.csv")
print(data_train.head())

X_train = data_train.to_numpy()
X_test = data_test.to_numpy()

# Do weird numpy dimension stuff, because y_train needs to have dimension 1
# and y_test needs to have dimension 0.
y_train = X_train[:, 0:1]  # Define Label for Regression.
y_test = X_test[:, 0]  # Define Label for Regression.

X_train = X_train[:, 1:]  # Chop off first column of X.
X_test = X_test[:, 1:]  # Chop off first column of X.

slr = SimpleLinReg()
slr.fit(X_train, y_train)
y_hat_slr = slr.predict(X_test)

mlr = MultipleLinearRegression()
mlr.fit(X_train, y_train)
y_hat_mlr = mlr.predict(X_test)

glr = GeneralizedLinearRegression()
glr.fit(X_train, y_train)
y_hat_glr = glr.predict(X_test)

print(f"\nSLR NMSE Total Value: {NMSE(X_test, y_hat_slr):0.4f}")
print(f"MLR NMSE Total Value: {NMSE(X_test, y_hat_mlr):0.4f}")
print(f"GLR NMSE Total Value: {NMSE(X_test, y_hat_glr):0.4f}")

print(f"\nSLR NMSE y=0 Value: {NMSE(X_test[y_test == 0], y_hat_slr[y_test == 0]):0.4f}")
print(f"MLR NMSE y=0 Value: {NMSE(X_test[y_test == 0], y_hat_mlr[y_test == 0]):0.4f}")
print(f"GLR NMSE y=0 Value: {NMSE(X_test[y_test == 0], y_hat_glr[y_test == 0]):0.4f}")

print(f"\nSLR NMSE y=1 Value: {NMSE(X_test[y_test == 1], y_hat_slr[y_test == 1]):0.4f}")
print(f"MLR NMSE y=1 Value: {NMSE(X_test[y_test == 1], y_hat_mlr[y_test == 1]):0.4f}")
print(f"GLR NMSE y=1 Value: {NMSE(X_test[y_test == 1], y_hat_glr[y_test == 1]):0.4f}")

plt.figure(figsize=(12, 8))
plt.scatter(X_test[:, 3], X_test[:, 2], c=y_test, alpha=0.5)
#plt.plot(y_hat_glr[:, 3], y_hat_glr[:, 2], color='#00FF00')
plt.axis((0, 0.001, 0, 0.001))
plt.show()
