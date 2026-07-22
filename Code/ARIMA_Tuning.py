'''Parameter tuning for ARIMA model'''


'''Dickey-Fuller test'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from matplotlib.pylab import rcParams


rcParams['figure.figsize'] = (20, 8)

train = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\Desktop\Time Series Analysis- ALV\Data\Train_SU63ISt.csv")

test = pd.read_csv(r"C:\Users\HP\OneDrive\Documents\Desktop\Time Series Analysis- ALV\Data\Test_0qrQsBZ.csv")

# Backup Copies
train_original = train.copy()
test_original = test.copy()


train['Datetime'] = pd.to_datetime(train['Datetime'],format='%d-%m-%Y %H:%M')

test['Datetime'] = pd.to_datetime(test['Datetime'],format='%d-%m-%Y %H:%M')

train_original['Datetime'] = pd.to_datetime(train_original['Datetime'],format='%d-%m-%Y %H:%M')

test_original['Datetime'] = pd.to_datetime(test_original['Datetime'],format='%d-%m-%Y %H:%M')


for df in [train, test, train_original, test_original]:
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day
    df['Hour'] = df['Datetime'].dt.hour


train.set_index('Datetime', inplace=True)
train_original.set_index('Datetime', inplace=True)


def test_stationarity(timeseries):

    rolling_mean = timeseries.rolling(window=24).mean()
    rolling_std = timeseries.rolling(window=24).std()

    # plt.figure(figsize=(20,8))
    # plt.plot(timeseries, label='Original', color='blue')
    # plt.plot(rolling_mean, label='Rolling Mean', color='red')
    # plt.plot(rolling_std, label='Rolling Std', color='black')

    # plt.title("Rolling Mean & Rolling Standard Deviation")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # print("=" * 60)
    # print("Augmented Dickey-Fuller Test")
    # print("=" * 60)

    result = adfuller(timeseries.dropna(), autolag='AIC')

    labels = ['Test Statistic','p-value','# Lags Used','Number of Observations']

    output = pd.Series(result[0:4], index=labels)

    for key, value in result[4].items():
        output[f'Critical Value ({key})'] = value

    # print(output)

#     print("\nConclusion")

#     if result[1] <= 0.05:
#         print("Data is Stationary.")
#     else:
#         print("Data is NOT Stationary.")

# test_stationarity(train_original['Count'])

#      OUTPUT

# Test Statistic               -4.456561
# p-value                       0.000235
# # Lags Used                  45.000000
# Number of Observations    18242.000000
# Critical Value (1%)          -3.430709
# Critical Value (5%)          -2.861698
# Critical Value (10%)         -2.566854
# dtype: float64



'''Removing Trend'''

Train = train.loc['2012-08-25':'2014-06-24']
valid = train.loc['2014-06-25':'2014-09-25']

Train_log = np.log(Train['Count'])
valid_log = np.log(valid['Count'])

moving_avg = Train_log.rolling(window=24).mean()

# # plt.plot(Train_log)
# # plt.plot(moving_avg, color='red')
# # plt.show()

train_log_moving_avg_diff = Train_log - moving_avg
train_log_moving_avg_diff.dropna(inplace = True)
test_stationarity(train_log_moving_avg_diff)

# ============================================================
# Augmented Dickey-Fuller Test
# ============================================================
# Test Statistic              -22.470949
# p-value                       0.000000
# # Lags Used                  43.000000
# Number of Observations    15989.000000
# Critical Value (1%)          -3.430759
# Critical Value (5%)          -2.861721
# Critical Value (10%)         -2.566866
# dtype: float64
train_log_diff = Train_log - Train_log.shift(1)
test_stationarity(train_log_diff.dropna())

'''Removing Seasonality'''

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(pd.DataFrame(Train_log).Count.values, period= 24)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# plt.subplot(411)
# plt.plot(Train_log, label ='Original')
# plt.legend(loc = 'best')
# plt.subplot(412)
# plt.plot(trend, label = 'Trend')
# plt.legend(loc = 'best')
# plt.subplot(413)
# plt.plot(seasonal, label= 'Seasonality')
# plt.legend(loc = 'best')
# plt.subplot(414)
# plt.plot(residual, label = 'Residuals')
# plt.legend(loc = 'best')
# plt.tight_layout()
# plt.show()

# Stationarity of residuals 

train_log_decompse = pd.DataFrame(residual)
train_log_decompse['date'] = Train_log.index
train_log_decompse.set_index('date', inplace=True)
train_log_decompse.dropna(inplace=True)
test_stationarity(train_log_decompse[0])

# Output
# ============================================================
# Augmented Dickey-Fuller Test
# ============================================================
# Test Statistic              -31.326116
# p-value                       0.000000
# # Lags Used                  43.000000
# Number of Observations    15988.000000
# Critical Value (1%)          -3.430759
# Critical Value (5%)          -2.861721
# Critical Value (10%)         -2.566866
# dtype: float64


'''Forcasting the Time Series using ARIMA'''


from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(train_log_diff.dropna(), nlags=25)
lag_pacf = pacf(train_log_diff.dropna(), nlags =25, method='ols')

'''ACF and PACF plot'''
# plt.plot(lag_acf)
# plt.axhline(y=0, linestyle ='--', color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color ='gray')
# plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color ='gray')
# plt.title('Autocorrelation Function')
# plt.show()
# plt.plot(lag_pacf)
# plt.axhline(y=0, linestyle ='--', color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color ='gray')
# plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color ='gray')
# plt.title('Partial Autocorrelation Function')
# plt.show()

'''Lets Start Final ARIMA Model'''


'''AR MODEL'''


from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error
# from math import sqrt

model = ARIMA(Train_log, order=(2,1,0))
result_AR = model.fit()

# # Forecast the validation period
# forecast = result_AR.forecast(steps=len(valid))

# # Convert from log scale back to original scale
# AR_predict = np.exp(forecast)

# # Calculate RMSE
# rmse = sqrt(mean_squared_error(valid['Count'], AR_predict))

# # Plot
# plt.figure(figsize=(15,6))

# plt.plot(valid.index, valid['Count'], label='Actual')
# plt.plot(valid.index, AR_predict, color='red', label='ARIMA Prediction')

# plt.xlabel("Date")
# plt.ylabel("Passenger Count")
# plt.title(f"ARIMA Forecast\nRMSE = {rmse:.4f}")

# plt.legend()
# plt.grid(True)

# plt.show()

# print(f"RMSE: {rmse:.4f}")

'''Validation curve for AR Model'''

from sklearn.metrics import mean_squared_error
from math import sqrt

# Predict on validation period
# prediction = result_AR.get_prediction(start=valid.index[0],end=valid.index[-1])

# # Predict on validation period
# prediction = result_AR.get_prediction(
#     start=valid.index[0],
#     end=valid.index[-1]
# )

# # Predicted values (log scale)
# AR_predict = prediction.predicted_mean

# # Convert to original scale
# AR_predict = np.exp(AR_predict)

# # Match validation index
# AR_predict.index = valid.index

# # Calculate RMSE
# rmse = sqrt(mean_squared_error(valid['Count'], AR_predict))

# from sklearn.metrics import mean_squared_error
# from math import sqrt

# # Predict on validation period
# prediction = result_AR.get_prediction(
#     start=valid.index[0],
#     end=valid.index[-1]
# )

# # Predicted values (log scale)
# AR_predict = prediction.predicted_mean

# # Convert to original scale
# AR_predict = np.exp(AR_predict)

# # Match validation index
# AR_predict.index = valid.index

# # Calculate RMSE
# rmse = sqrt(mean_squared_error(valid['Count'], AR_predict))

# # Validation Plot
# plt.figure(figsize=(12,5))

# plt.plot(valid['Count'], label='Valid')
# plt.plot(AR_predict, color='red', label='Predict')

# plt.legend(loc='best')
# plt.title('RMSE : %.4f' % rmse)
# plt.show()

'''MA MODEL'''

model = ARIMA(Train_log, order=(0,1,2))     # Here the p value is zero since it is just the MA model
result_MA = model.fit()

# plt.plot(train_log_diff.dropna(), label='original')
# plt.plot(result_MA.fittedvalues, color = 'red', label='prediction')
# plt.legend(loc ='best')
# plt.show()

MA_predict = result_MA.predict(start = "2014-06-25", end ="2014-09-25")
MA_predict = MA_predict.cumsum().shift().fillna(0)
MA_predict1 = pd.Series(np.ones(valid.shape[0]) * np.log(valid['Count'])[0], index =valid.index)
MA_predict = MA_predict1.add(MA_predict,fill_value=0)
MA_predict = np.exp(MA_predict1)

# plt.plot(valid['Count'], label = "Valid")
# plt.plot(MA_predict, color = 'red', label = 'Predict')
# plt.legend(loc='best')
# plt.title('RMSE : %.4f'% (np.sqrt(np.dot(MA_predict, valid['Count']))/valid.shape[0]))
# plt.show()


'''Combined model'''

model = ARIMA(Train_log, order =(2,1,2))
result_ARIMA = model.fit()

# plt.plot(train_log_diff.dropna(), label ='original')
# plt.plot(result_ARIMA.fittedvalues, color ='red', label ='predicted')
# plt.legend(loc = 'best')
# plt.show()

def check_prediction_diff(predict_diff, given_set):
    predict_diff = predict_diff.cumsum().shift().fillna(0)
    predict_base = pd.Series(np.ones(given_set.shape[0])* np.log(given_set['Count'])[0], index = given_set.index)
    predict_log = predict_base.add(predict_diff, fill_value=0)
    predict = np.exp(predict_log)
    
    plt.plot(given_set['Count'], label = "Given Set")
    plt.plot(predict, color = 'red', label = 'Predict')
    plt.legend(loc = 'best')
    # plt.title('RMSE : %.4f'% (np.sqrt(np.sqrt(predict, given_set['Count']))/given_set.shape[0]))
    rmse = np.sqrt(np.mean((predict - given_set['Count'])**2))

    plt.title('RMSE : %.4f' % rmse)
    plt.show()
    
# def chack_prediction_log(predict_log, given_set):
#     predict = np.exp(predict_log)
    
    # plt.plot(given_set['Count'], label = "Given Set")
    # plt.plot(predict, color = 'red', label = 'Predict')
    # plt.legend(loc = 'best')
    # plt.title('RMSE : %.4f'% (np.sqrt(np.sqrt(predict, given_set['Count']))/given_set.shape[0]))
    # plt.show()
    

ARIMA_predict_diff = result_ARIMA.predict(start = "2014-06-25", end = "2014-09-25")
check_prediction_diff(ARIMA_predict_diff, valid)

'''SARIMAX Model on daily time series'''
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt

# Copy validation dataframe
y_hat_avg = valid.copy()

# Train SARIMA model
fit1 = sm.tsa.statespace.SARIMAX(Train["Count"],order=(2,1,4),seasonal_order=(0,1,1,7),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)

# Forecast validation period
pred = fit1.get_forecast(steps=len(valid))

# Predicted values
y_hat_avg["SARIMA"] = pred.predicted_mean.values

# Plot
plt.figure(figsize=(15,6))
plt.plot(Train["Count"], label="Train")
plt.plot(valid["Count"], label="Validation")
plt.plot(valid.index, y_hat_avg["SARIMA"], color="red", label="SARIMA Forecast")
plt.legend()
plt.grid(True)
plt.title("SARIMA Forecast")
plt.show()


# RMSE
rms = sqrt(mean_squared_error(valid["Count"], y_hat_avg["SARIMA"]))

print("RMSE :", rms)

        # RMSE : 183.8043105936754
        
