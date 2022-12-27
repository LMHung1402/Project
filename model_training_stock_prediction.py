import math
import pandas_datareader as web
import numpy as np
import pandas as pandas_datareader
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#get stock quote
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-01-01')

#data dimension
print(df.shape)


#create new data frame with close column
data = df.filter(['Close'])
#convert to numpy array
dataset = data.values
#number of rows to train
training_data_len = math.ceil(len(dataset)*.8)

print(training_data_len)

#scaling data to between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#print(scaled_data)


#create training data set

train_data = scaled_data[0: training_data_len, :]
#split data into x_train and y_train
x_train = []
y_train = []

for i in range(60, len(train_data)):
	x_train.append(train_data[i-60:i,0])
	y_train.append(train_data[i,0])

#convert x and y to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

#reshaping data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#build LSTM model 1.0
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(50))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))

#model 2.0
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(units=1))

#compile model
model.compile(optimizer='adam', loss='mean_squared_error')

#train model
model.fit(x_train, y_train, batch_size=25, epochs=25)

#create testing dataset
#create new array of scaled values
test_data = scaled_data[training_data_len-60: , :]

#create x_test y_test
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(60, len(test_data)):
	x_test.append(test_data[i-60:i, 0])

#convert
x_test = np.array(x_test)

#reshape x_test
x_test = np.reshape(x_test, ( x_test.shape[0], x_test.shape[1], 1 ))

#get the model predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)		#inverse 0-1 to normal value

#get the root mean squared error(RMSE)
rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")