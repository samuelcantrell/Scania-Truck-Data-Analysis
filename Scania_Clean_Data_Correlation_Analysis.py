# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:43:13 2020

@author: scant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


data_train = pd.read_csv("train_clean_norm.csv")
Cor_train = data_train.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(Cor_train.abs(), annot=False)
plt.axes().set_title("Clean Training Set")
plt.show()
