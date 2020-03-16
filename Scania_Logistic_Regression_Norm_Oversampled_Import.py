# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:52:28 2020

@author: scant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def softmax(h):
    return (np.exp(h.T) / np.sum(np.exp(h), axis=1)).T


def cross_entropy(Y, P_hat):
    return -(1/len(Y))*np.sum(np.sum(Y*np.log(P_hat), axis=1), axis=0)


def accuracy(y, y_hat):
    return np.mean(y == y_hat)


def indices_to_one_hot(data, nb_classes):
    # Converts an interable of indices to one hot labels
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


class GenLogisticRegression():
    def __init__(self):
        self.W = None
        self.B = None

    def fit(self, X, y, eta, epochs, show_curve=False):
        N, D = X.shape
        epochs = int(epochs)
        K = len(np.unique(y))
        self.y_values = np.unique(y, return_index=False)
        Y = indices_to_one_hot(y, K).astype(int)
        # self.W = np.random.randn(D, K)
        # self.B = np.random.randn(N, K)
        # self.B = np.random.randn(K)

        J = np.zeros(int(epochs))

        for epoch in range(epochs):
            P_hat = self.__forward__(X)
            J[epoch] = cross_entropy(Y, P_hat)

            self.W -= eta*(1/N)*X.T@(P_hat - Y)
            self.B -= eta*(1/N)*np.sum(P_hat - Y, axis=0)

            if epoch % (epochs*0.05) == 0.0:  # Print progress to console.
                print("Percent Complete: " + str(round((epoch / epochs)*100)) + "%")

        if show_curve:
            plt.figure()
            plt.plot(J)
            plt.xlabel("epochs")
            plt.ylabel("$\mathcal{J}$")
            plt.title("Training Curve")
            plt.show()

    def __forward__(self, X):
        return softmax(X @ self.W + self.B)

    def predict(self, X):
        return np.argmax(self.__forward__(X), axis=1)


data_train = pd.read_csv("train_new_norm_oversampled.csv")
data_test = pd.read_csv("test_new_norm.csv")
print(data_train.head())

X_train = data_train.to_numpy()
X_test = data_test.to_numpy()

# Do weird numpy dimension stuff, because y_train needs to have dimension 1
# and y_test needs to have dimension 0.
y_train = X_train[:, 0:1].astype(int)  # Define Label for Regression.
y_test = X_test[:, 0]  # Define Label for Regression.

X_train = X_train[:, 1:]  # Chop off first column of X.
X_test = X_test[:, 1:]  # Chop off first column of X.

log_reg = GenLogisticRegression()
# Start from imported weights:
log_reg.W = pd.read_csv("GenLogisticRegression-W.csv").to_numpy()
log_reg.B = pd.read_csv("GenLogisticRegression-B.csv").to_numpy()
log_reg.B = log_reg.B[:, 0]  # Make dimensionless for broadcasting

# Update weights and train more:
eta = 1e-1
epochs = 1e5
# log_reg.fit(X_train, y_train, eta, epochs, show_curve=True)
y_hat = log_reg.predict(X_test)

# Give me mah metrics!
print(log_reg.W)
print(f"\nOverall Test Accuracy: {accuracy(y_test, y_hat):0.4f}")
print(f"\nNegative Test Accuracy: {accuracy(y_test[y_test == 0], y_hat[y_test == 0]):0.4f}")
print(f"\nPositive Test Accuracy: {accuracy(y_test[y_test == 1], y_hat[y_test == 1]):0.4f}")

plt.figure(figsize=(12, 8))
plt.scatter(X_test[:, 3], X_test[:, 2], c=y_test, alpha=0.5)
# plt.scatter(X_test[:, 3], X_test[:, 2], c=y_hat, alpha=0.5)
plt.axis((0, 0.000001, 0, 0.00001))
plt.show()
print("Cost Score:")
print(len(y_hat[(y_hat != y_test) & (y_test == 0)])*10 +
      len(y_hat[(y_hat != y_test) & (y_test == 1)])*500)

"""
df = pd.DataFrame(log_reg.W)
df.to_csv("GenLogisticRegression-W.csv", header=True, index=False)
df = pd.DataFrame(log_reg.B)
df.to_csv("GenLogisticRegression-B.csv", header=True, index=False)
"""

# Produce confusion matrix:
plt.figure(figsize=(12, 8))
y_true = pd.Series(y_test.astype(int), name="Actual Label")
y_pred = pd.Series(y_hat, name="Predicted Label")
sns.heatmap(pd.crosstab(y_true, y_pred), annot=True, fmt="d", linewidths=0.25)
plt.ylim(len(set(y_test)), 0)  # Fix limits, matplotlib bugged (ver. 3.11)
