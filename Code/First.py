import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series

# %matplotlib inline
# import warnings
# warnings.filterwarnings('ignore')

# print('libraries imported')

train = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\Desktop\Time Series Analysis- ALV\Data\Train_SU63ISt.csv")
test = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\Desktop\Time Series Analysis- ALV\Data\Test_0qrQsBZ.csv")


# Dataset Structure and content


train_original = train.copy()
test_original = test.copy()

print(train.head())
print('*'*40)
print(test.head())

print(train.columns)
print('*'*40)
print(test.columns)

print(train.dtypes)
print('*'*40)
print(test.dtypes)

print(train.shape)
print('*'*40)
print(test.shape)
