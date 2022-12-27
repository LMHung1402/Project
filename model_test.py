import math
from pandas_datareader import data as web
import numpy as np
import pandas as pandas_datareader
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import yfinance as yfin
yfin.pdr_override()

#get stock quote
df = web.get_data_yahoo('AAPL', start='2012-01-01', end='2020-01-01')

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")


#create new data frame with close column
data = df.filter(['Close'])
#convert to numpy array
dataset = data.values
#number of rows to train
training_data_len = math.ceil(len(dataset)*.8)

#scaling data to between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#create testing dataset
#create new array of scaled values
test_data = scaled_data[training_data_len-60: , :]

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


#graph the error evaluation
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('MODEL TEST WITH APPLE STOCK PRICE(1 day prediction)')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price in USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
