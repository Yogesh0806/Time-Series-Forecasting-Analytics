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

# dd = np.asarray(Train.Count)
# y_hat = valid.copy()
# y_hat['naive'] = dd[len(dd)- 1]
# plt.figure(figsize=(10,6))
# plt.plot(Train.index, Train['Count'], label= 'Train')
# plt.plot(valid.index, valid['Count'], label= 'Valid')
# plt.plot(y_hat.index, y_hat['naive'], label = 'Naive Forecast')
# plt.legend(loc='best')
# plt.title("Naive Forecast")
# plt.show()

# # Calculate RMSE(Root Mean Square Error)

from sklearn.metrics import mean_squared_error
from math import sqrt
# rms = sqrt(mean_squared_error(valid.Count, y_hat.naive))
# print(rms)
#         # 186.37368520699056
        
'''Moving Average'''     

# y_hat_avg = valid.copy()
# y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(10).mean().iloc[-1]       #Average of last 10 points

# plt.figure(figsize=(12,7))
# plt.plot(Train['Count'], label= 'Train')
# plt.plot(valid['Count'], label='Valid')
# plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations')
# plt.legend(loc = 'best')
# plt.show()


# y_hat_avg = valid.copy()
# y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(20).mean().iloc[-1]           #Average of last 20 obsevations 

# plt.figure(figsize=(12,7))
# plt.plot(Train['Count'], label = 'Train')
# plt.plot(valid['Count'], label='Valid')
# plt.plot(y_hat_avg['moving_avg_forecast'], label= 'Moving Average Forecast using 20 observations')
# plt.legend(loc = 'best')
# plt.show()

# y_hat_avg = valid.copy()
# y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(50).mean().iloc[-1]       # Average of last 50

# plt.figure(figsize=(12,7))
# plt.plot(Train['Count'], label = 'Train')
# plt.plot(valid['Count'], label='Valid')
# plt.plot(y_hat_avg['moving_avg_forecast'], label = 'Moving Average Forecast using 50 observations')
# plt.legend(loc = 'best')
# plt.show()

# rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.moving_avg_forecast))
# print(rms)

#         #  186.5773761711873
        

'''Simple Exponential Smoothing'''

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing,Holt

y_hat_avg = valid.copy()
fit2 = SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level=0.6,optimized=False) 
y_hat_avg['SES'] = fit2.forecast(len(valid))

plt.figure(figsize=(12,7))
plt.plot(Train['Count'], label = 'Train')
plt.plot(valid['Count'], label = 'Valid')
plt.plot(y_hat_avg['SES'], label = 'SES')
plt.legend(loc = 'best')
plt.show()


rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.SES))
print(rms)

        # 186.40944652452376