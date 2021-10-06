import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.pylab import rcParams
import seaborn as sns
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense
from keras.layers import LSTM
import glob
from datetime import datetime
from keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from math import sqrt
import sys


#Read the csv file
#df = pd.read_csv('GE.csv')
df = pd.read_csv('dataset.csv')
# print(df.head()) #7 columns, including the Date.

#Separate dates for future plotting
train_dates = pd.to_datetime(df['timestamp'])
#print(train_dates.tail(15)) #Check last few dates.

#Variables for training
cols = list(df)[1:4]

#Date and volume columns are not used in training.
print(cols)

#New dataframe with only training data - 5 columns
df_for_training = df[cols].astype(float)

#Use heatmap to see corelation between variables
# sns.heatmap(df.corr(),annot=True,cmap='viridis')
# plt.title('Heatmap of co-relation between variables',fontsize=16)
# plt.show()


####################################################################################
# print (df_for_training.head())
#
# ''' Dividing data in test and train sets '''
# dataset = df_for_training.disk_io.values #numpy.ndarray
# dataset = dataset.astype('float32')
# dataset = np.reshape(dataset, (-1, 1))
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# train_size = int(len(dataset) * 0.70)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# print(len(train), len(test))
#
# def create_dataset(dataset, look_back):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back):
#         a = dataset[i:(i + look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     print(len(dataY))
#     return np.array(dataX), np.array(dataY)
#
# look_back = 8
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
#
#
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# print('Training data size:',trainX.shape)
# print('Test data size:',testX.shape)
#
# model = Sequential()
# model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# history = model.fit(trainX, trainY, epochs=50, batch_size=100, validation_data=(testX, testY), verbose=1, shuffle=False)
#
# yhat = model.predict(testX)
#
# pyplot.figure(figsize=(20,8))
# pyplot.plot(yhat[:500], label='predict')
# pyplot.plot(testY[:500], label='true')
# pyplot.legend()
#
# pyplot.ylabel('Relative Humidity', size=15)
# pyplot.xlabel('Time step', size=15)
# pyplot.legend(fontsize=15)
#
# pyplot.show()
#
#
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# #pyplot.legend()
#
# pyplot.title('model loss',size=15)
# pyplot.ylabel('loss',size=15)
# pyplot.xlabel('epochs',size=15)
# pyplot.legend(loc='upper right',fontsize=15)
#
# pyplot.show()
#
#
# sys.exit()
####################################################################################

#dataset = df_for_training.disk_io.values
#dataset = df_for_training.values
#dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.fit_transform(df_for_training)

train_size = int(len(df_for_training_scaled) * 0.70)
test_size = len(df_for_training_scaled) - train_size
train, test = df_for_training_scaled[0:train_size,:], df_for_training_scaled[train_size:len(df_for_training),:] #dataset for len if I keep the dataset
print(len(train), len(test))

#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
#In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training).

#Empty lists to be populated using formatted training data
trainX = []
trainY = []
testX = []
testY = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    testX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    testY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])


trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

testX, testY = np.array(testX), np.array(testY)

print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))

#trainX shape == (169, 14, 3)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

#testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#print('Training data size:',trainX.shape)
#print('Test data size:',testX.shape)



sys.exit()
''' Fitting the data in LSTM Deep Learning model '''
model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(trainX, trainY, epochs=10, batch_size=100, validation_data=(testX, testY), verbose=1, shuffle=False)



#n_days_for_prediction=500  #let us predict past 15 days
#yhat = model.predict(trainX[-n_days_for_prediction:])


yhat = model.predict(testX)


''' Plotting the first 500 entries to see prediction '''
pyplot.figure(figsize=(20,8))
pyplot.plot(yhat[:500], label='predict')
pyplot.plot(trainY[:500], label='true')
pyplot.legend()
pyplot.ylabel('Disk IO', size=15)
pyplot.xlabel('Time step', size=15)
pyplot.legend(fontsize=15)

#pyplot.show()


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.title('model loss',size=15)
pyplot.ylabel('loss',size=15)
pyplot.xlabel('epochs',size=15)
pyplot.legend(loc='upper right',fontsize=15)

#pyplot.show()

# mae = mean_absolute_error(testY, yhat)
# print('Test Score: %.2f MAE' % (mae))
# mse = mean_squared_error(testY, yhat)
# print('Test Score: %.2f MSE' % (mse))
# rmse = sqrt(mse)
# print('Test Score: %.2f RMSE' % (rmse))
# r2 = r2_score(testY, yhat)
# print('Test Score: %.2f R2' % (r2))


