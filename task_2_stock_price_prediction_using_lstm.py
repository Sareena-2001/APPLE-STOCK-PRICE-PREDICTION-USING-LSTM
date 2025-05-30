# -*- coding: utf-8 -*-
"""TASK 2: STOCK PRICE PREDICTION USING LSTM

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Nlx8I2nE8BDQ_oNgSOnSxHTPQgJtkDFf
"""

# THE DATASET FOR STOCK PRICE PREDICTION OF THE RENOWNED APPLE COMPANY IS
#FETCHED FROM KAGGLE

#LSTM (LONG-SHORT TERM MEMORY) IS USED TO ACCOMPLISH THIS TASK

#LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
stockprediction_dataset = pd.read_csv('/content/sample_data/apple_stock.csv', parse_dates=['Date'], index_col='Date')

stockprediction_dataset.head(8)

stockprediction_dataset.tail(6)

stockprediction_dataset.shape

stockprediction_dataset.columns

stockprediction_dataset.info()

stockprediction_dataset.describe()

#we have to see if there are missing values in the dataset

print("the number of missing values in the dataset is=")
print(stockprediction_dataset.isnull().sum())

# the missing value of a column will be replaced with the column's mean

stockprediction_dataset.fillna(stockprediction_dataset.mean(), inplace=True)

# histograms will be shown for each column

dataset_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
plt.figure(figsize=(17, 11))
for j, colu in enumerate(dataset_columns):
    plt.subplot(2, 3, j + 1)
    sns.histplot(stockprediction_dataset[colu], kde=True, color='green')
    plt.title(f'Distribution of the Column named {colu}')
plt.tight_layout()
plt.show()

# use the 'close price' column for making predictions

mydataset =stockprediction_dataset [['Close']].values

# data normalization

sca = MinMaxScaler(feature_range=(0, 1))
sca_data = sca.fit_transform(mydataset) #used to represent scaled data

# now ,seqences will be created

def sequence_creation(mydataset, sequence_length):
    xs, ys = [], []
    for i in range(len(mydataset) - sequence_length):
        x = mydataset[i:i + sequence_length]
        y = mydataset[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 60
X, y = sequence_creation(sca_data, sequence_length)

# the data will be split into training and testing set

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#AN LSTM (LONG-SHORT TERM MEMORY) MODEL WILL BE BUILT NOW

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# MODEL TRAINING

history = model.fit(X_train, y_train, epochs=65, batch_size=32, validation_split=0.1)

# PREDICTIONS
pred = model.predict(X_test)
pred = sca.inverse_transform(pred)
y_test = sca.inverse_transform(y_test)

# scores/evaluation metrics
rmse_score = np.sqrt(mean_squared_error(y_test, pred))
mse_score = mean_squared_error(y_test, pred)
mae_score = mean_absolute_error(y_test, pred)
r2_score = r2_score(y_test, pred)
mape_score = mean_absolute_percentage_error(y_test, pred)

#displaying the metrics achieved


print(f'RMSE score for LSTM= {rmse_score}')
print(f'MSE score for LSTM= {mse_score}')
print(f'MAE score for LSTM= {mae_score}')
print(f'R2 score for LSTM= {r2_score}')
print(f'MAPE score for LSTM={mape_score}')

# A DATAFRAME WILL BE CREATED FOR PREDICTIONS

predictions_dataframe = pd.DataFrame({
    'Date': stockprediction_dataset.index[-len(y_test):],
    'Actual Price': y_test.flatten(),
    'Predicted Price': pred.flatten()
})

print("\nPredictions vs Actual Stock Prices:")
print(predictions_dataframe.head())

# a plot for predictions

plt.figure(figsize=(13, 6))
plt.plot(stockprediction_dataset.index[-len(y_test):], y_test, label='Actual Prices')
plt.plot(stockprediction_dataset.index[-len(pred):], pred, label='Predicted Prices')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Scatter plot for representing actual and predicted prices

plt.figure(figsize=(13, 6))
plt.scatter(stockprediction_dataset.index[-len(y_test):], y_test, color='purple', label='Actual Prices', alpha=0.6)
plt.scatter(stockprediction_dataset.index[-len(pred):], pred, color='green', label='Predicted Prices', alpha=0.6)
plt.title('Scatter Plot for Actual vs Predicted Prices ')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()