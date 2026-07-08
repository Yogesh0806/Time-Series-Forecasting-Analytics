import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series

# %matplotlib inline
# import warnings
# warnings.filterwarnings('ignore')

# print('libraries imported')

train = pd.read_csv('Train_SU63IST.csv')
test = pd.read_csv('Test_0qrQsBZ.csv')
print(train.head())
print('*'*40)
print(test.head())