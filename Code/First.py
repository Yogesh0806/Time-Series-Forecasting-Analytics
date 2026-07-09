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

'''Dataset Structure and content'''


train_original = train.copy()
test_original = test.copy()

# print(train.head())
# print('*'*40)
# print(test.head())

# print(train.columns)
# print('*'*40)
# print(test.columns)

# print(train.dtypes)
# print('*'*40)
# print(test.dtypes)

# print(train.shape)
# print('*'*40)
# print(test.shape)


''' Feature Extraction'''
# ==========================================================
# Workflow
# ==========================================================
# 1. Import required libraries.
# 2. Load the train and test datasets.
# 3. Create backup copies of the datasets.
# 4. Convert the 'Datetime' column to datetime format.
# 5. Extract Year, Month, Day, and Hour from Datetime.
# 6. Create additional features:
#    - Day of Week
#    - Weekend (1 = Weekend, 0 = Weekday)
# 7. Set 'Datetime' as the DataFrame index.
# 8. Remove the 'ID' column.
# 9. Select the 'Count' column as the time series.
# 10. Visualize the time series using a line plot.
# ==========================================================


train['Datetime'] = pd.to_datetime(train.Datetime, format = '%d-%m-%Y %H:%M')
test['Datetime'] = pd.to_datetime(test.Datetime, format = '%d-%m-%Y %H:%M')
test_original['Datetime'] = pd.to_datetime(test_original.Datetime, format='%d-%m-%Y %H:%M')
train_original['Datetime'] = pd.to_datetime(train_original.Datetime, format='%d-%m-%Y %H:%M') 

for i in (train, test, test_original, train_original):
    i['year'] = i.Datetime.dt.year
    i['month'] = i.Datetime.dt.month
    i['day'] = i.Datetime.dt.day
    i['hour'] = i.Datetime.dt.hour
    
train['day of week'] = train['Datetime'].dt.dayofweek
temp = train['Datetime']


def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0

temp2 = train['Datetime'].apply(applyer)
train['weekend'] = temp2


train.index = train['Datetime']
df = train.drop('ID',axis=1)
ts = df['Count']

plt.figure(figsize=(12,6))
plt.plot(ts, label='Passenger Count')
plt.title('Time Series')
plt.xlabel('Time(year-month)')
plt.ylabel('Passenger Count')
plt.legend(loc = 'best')
plt.show()
