import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series

train = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\Desktop\Time Series Analysis- ALV\Data\Train_SU63ISt.csv")
test = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\Desktop\Time Series Analysis- ALV\Data\Test_0qrQsBZ.csv")


train['Datetime'] = pd.to_datetime(train['Datetime'], format='%d-%m-%Y %H:%M')
train.set_index('Datetime', inplace=True)


Train = train.loc['2012-08-25':'2014-06-24']
valid = train.loc['2014-06-25':'2014-09-25']

# Train.Count.plot(figsize=(20,8), title= 'Daily Ridership', fontsize=14, label='train')
# valid.Count.plot(figsize=(20,8), title= "Daily Ridership", fontsize=14, label= 'valid')
# plt.xlabel('Datetime')
# plt.ylabel('Passenger count')
# plt.legend(loc='best')
# plt.show()


'''Modeling Techniques'''

'''Naive Approach'''

dd = np.asarray(Train.Count)
y_hat = valid.copy()
y_hat['naive'] = dd[len(dd)- 1]
plt.figure(figsize=(10,6))
plt.plot(Train.index, Train['Count'], label= 'Train')
plt.plot(valid.index, valid['Count'], label= 'Valid')
plt.plot(y_hat.index, y_hat['naive'], label = 'Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()

# Calculate RMSE(Root Mean Square Error)

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(valid.Count, y_hat.naive))
print(rms)
        # 186.37368520699056