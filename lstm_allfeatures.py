import pandas as pd
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from math import sqrt
import sys



df = pd.read_csv('testdataset.csv')
# print(df.head())

#Separate dates for future plotting
train_dates = pd.to_datetime(df['timestamp'])

#Variables for training
cols = list(df)[1:4]

print(cols)

#New dataframe with only training data - 5 columns
df_for_training = df[cols].astype(float)

#Use heatmap to see corelation between variables
# sns.heatmap(df.corr(),annot=True,cmap='viridis')
# plt.title('Heatmap of co-relation between variables',fontsize=16)
# plt.show()

#dataset = df_for_training.disk_io.values
#dataset = df_for_training.values
#dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
df_for_training_scaled = scaler.fit_transform(df_for_training)

train_size = int(len(df_for_training_scaled) * 0.70)
test_size = len(df_for_training_scaled) - train_size
train, test = df_for_training_scaled[0:train_size,:], df_for_training_scaled[train_size:len(df_for_training),:] #dataset for len if I keep the dataset

print("len(train-test)",len(train), len(test))

#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
#In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training).

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
print ('-----------------------------------------')

testX, testY = np.array(testX), np.array(testY)

print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))

# train_X = trainX.reshape((trainX.shape[0], 14, trainX.shape[2]))
# test_X = testX.reshape((testX.shape[0], 14, testX.shape[2]))
# print(train_X.shape, trainY.shape, test_X.shape, trainY.shape)


''' Fitting the data in LSTM Deep Learning model '''
model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=False))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae', 'mape'])
history = model.fit(trainX, trainY, epochs=1000, batch_size=100, validation_data=(testX, testY), callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
                    verbose=1, shuffle=False)



#n_days_for_prediction=500  #let us predict past 15 days
#yhat = model.predict(trainX[-n_days_for_prediction:])


# yhat = model.predict(testX)

yhat = model.predict(testX)
prediction_copies = np.repeat(yhat, df_for_training.shape[1], axis=-1) # https://stackoverflow.com/questions/42997228/lstm-keras-error-valueerror-non-broadcastable-output-operand-with-shape-67704
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]

prediction_copies_Actual = np.repeat(testY, df_for_training.shape[1], axis=-1)
y_actual = scaler.inverse_transform(prediction_copies_Actual)[:,0]


''' Plotting the first 500 entries to see prediction '''
pyplot.figure(figsize=(20,8))
pyplot.plot(y_pred_future[:100], label='predict')
pyplot.plot(y_actual[:100], label='true')
pyplot.legend()
pyplot.ylabel('Disk IO', size=15)
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


