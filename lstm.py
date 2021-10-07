
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

from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from math import sqrt
import sys



df = pd.read_csv('dataset.csv')
# print(df.head())



# Variables for training
cols = list(df)[1:4]

print(cols)

# New dataframe with only training data
df_for_training = df[cols].astype(float)

# Use heatmap to see corelation between variables
# sns.heatmap(df.corr(),annot=True,cmap='viridis')
# plt.title('Heatmap of co-relation between variables',fontsize=16)
# plt.show()



# dataset = df_for_training.disk_io.values
# dataset = df_for_training.values
# dataset = np.reshape(dataset, (-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
train_size = int(len(df_for_training_scaled) * 0.80)
test_size = len(df_for_training_scaled) - train_size
train, test = df_for_training_scaled[0:train_size,:], df_for_training_scaled[train_size:len(df_for_training),:]

print("Length (train-test)",len(train), len(test))

X_train = []
Y_train = []
X_test = []
Y_test = []

n_future = 1   # Number of days we want to look into the future based on the past days. -> # out
n_past = 14    # Number of past days we want to use to predict the future. -> #step
features = 3   # Number of features

# x ram bandwodth
# y cpu

def column(matrix, i):
    return [row[i] for row in matrix]

def split_sequence(seq, steps, out):
    X, Y = list(), list()
    for i in range(len(seq)):
        end = i + steps
        outi = end + out
        if outi > len(seq)-1:
            break
        seqx, seqy = seq[i:end], column(seq[end:outi],1) # o arithmos tis stilsi pou thelo na kano tin provleppsi
        X.append(seqx)
        Y.append(seqy)
    return np.array(X), np.array(Y)


# split into samples
X_train, Y_train = split_sequence(train, n_past, n_future)
X_test, Y_test = split_sequence(test, n_past, n_future)
X_train = X_train.reshape((Y_train.shape[0], X_train.shape[1], features))

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)



model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=False))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae', 'mape'])
history = model.fit(X_train, Y_train, epochs=100, batch_size=100, validation_data=(X_test, Y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
                    verbose=1, shuffle=False)



yhat = model.predict(X_test)
prediction_copies = np.repeat(yhat, df_for_training.shape[1], axis=-1) # https://stackoverflow.com/questions/42997228/lstm-keras-error-valueerror-non-broadcastable-output-operand-with-shape-67704
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]

prediction_copies_Actual = np.repeat(Y_test, df_for_training.shape[1], axis=-1)
y_actual = scaler.inverse_transform(prediction_copies_Actual)[:,0]


''' Plotting the first 500 entries to see prediction '''
pyplot.figure(figsize=(20,8))
pyplot.plot(y_pred_future[:100], label='predict')
pyplot.plot(y_actual[:100], label='true')
pyplot.legend()
pyplot.ylabel('CPU', size=15)
pyplot.xlabel('Time step', size=15)
pyplot.legend(fontsize=15)

pyplot.show()

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.title('model loss',size=15)
pyplot.ylabel('loss',size=15)
pyplot.xlabel('epochs',size=15)
pyplot.legend(loc='upper right',fontsize=15)

pyplot.show()

mae = mean_absolute_error(y_actual, y_pred_future)
print('Test Score: %.2f MAE' % (mae))
mse = mean_squared_error(y_actual, y_pred_future)
print('Test Score: %.2f MSE' % (mse))
rmse = sqrt(mse)
print('Test Score: %.2f RMSE' % (rmse))
r2 = r2_score(y_actual, y_pred_future)
print('Test Score: %.2f R2' % (r2))


