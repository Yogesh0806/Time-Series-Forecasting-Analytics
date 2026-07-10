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

# plt.figure(figsize=(12,6))
# plt.plot(ts, label='Passenger Count')
# plt.title('Time Series')
# plt.xlabel('Time(year-month)')
# plt.ylabel('Passenger Count')
# plt.legend(loc = 'best')
# plt.show()


'''Exploratory Analysis'''
#Hypo 1 : Traffic will increase as the years pass by

# train.groupby('year')['Count'].mean().plot(kind='bar')
# plt.show()

#Hypo 2 : Traffic increase from May to October.

# train.groupby('month')['Count'].mean().plot(kind='bar')
# plt.show()

# temp = train.groupby(['year', 'month'])['Count'].mean()
# temp.plot(figsize=(15,5),tile='Passenger Count(Monthwise)', fontsize=14)
# plt.show()


#Hypo 3 : Traffic as Daywise

# train.groupby('day')['Count'].mean().plot(kind='bar')
# plt.show()

# Traffic status during peak hour

# train.groupby('hour')['Count'].mean().plot(kind='bar')
# plt.show()

# Hypo 4 : Traffic on weekdays will be more 

# train.groupby('weekend')['Count'].mean().plot(kind='bar')
# plt.show()

# Hypo 5 : Day wise pessenger count

# train.groupby('day of week')['Count'].mean().plot(kind='bar')
# plt.show()

train.timestamp = pd.to_datetime(train.Datetime, format= '%d-%m-%Y %H:%M')
train.index = train.timestamp

hourly = train.resample('H').mean()

daily = train.resample('D').mean() 

weekly = train.resample('W').mean() 
monthly = train.resample('M').mean() 

# fig, axs = plt.subplots(4,1)
# hourly.Count.plot(figsize=(10,4), title='Hourly', fontsize=14, ax=axs[0])
# daily.Count.plot(figsize=(10,4), title='daily', fontsize=14, ax=axs[1])
# weekly.Count.plot(figsize=(10,4), title='weekly', fontsize=14, ax=axs[2])
# monthly.Count.plot(figsize=(10,4), title='monthly', fontsize=14, ax=axs[3])
# plt.show()

fig, axs = plt.subplots(4, 1, figsize=(7, 9))
hourly.Count.plot(title='Hourly',fontsize=12,linewidth=2,color='royalblue',ax=axs[0])
axs[0].set_ylabel("Count", fontsize=11)
axs[0].grid(alpha=0.3)
axs[0].legend(['Hourly'], fontsize=10)

daily.Count.plot(title='Daily',fontsize=12,linewidth=2,color='darkorange',ax=axs[1])
axs[1].set_ylabel("Count", fontsize=11)
axs[1].grid(alpha=0.3)
axs[1].legend(['Daily'], fontsize=10)

weekly.Count.plot(title='Weekly',fontsize=12,linewidth=2,color='green',ax=axs[2])
axs[2].set_ylabel("Count", fontsize=11)
axs[2].grid(alpha=0.3)
axs[2].legend(['Weekly'], fontsize=10)

monthly.Count.plot(title='Monthly',fontsize=12,linewidth=2,color='crimson',ax=axs[3])
axs[3].set_ylabel("Count", fontsize=11)
axs[3].set_xlabel("Date", fontsize=12)
axs[3].grid(alpha=0.3)
axs[3].legend(['Monthly'], fontsize=10)

plt.tight_layout()
plt.show()

