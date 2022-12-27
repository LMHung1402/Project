import math
from pandas_datareader import data as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from tkinter import *
import datetime 
import yfinance as yfin

yfin.pdr_override()

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")


def stock_prediction(t, s_year, s_month, s_day, e_year, e_month, e_day):
	start = datetime.date(int(s_year), int(s_month), int(s_day))
	end = datetime.date(int(e_year), int(e_month), int(e_day))
	today = datetime.date.today()
	pred_date = today + datetime.timedelta(days=1)

	future_days = math.ceil((end-today).days)
	if future_days <= 0:
		future_days = 0
			
	#get stock quote / format: YYYY-MM-DD
	quote = web.get_data_yahoo(t, start='2010-01-01', end=today)

	#create new data frame with close column
	df = quote.filter(['Close'])

	result = []

	for i in range(future_days):
		#last 60 days data
		last_60_days = df[-60:]

		#scaling
		scaler = MinMaxScaler(feature_range=(0,1))
		last_60_days_scaled = scaler.fit_transform(last_60_days)

		#create empty list
		x_test = []

		#append 60 days to x_test
		x_test.append(last_60_days_scaled)

		#convert x_test to np array
		x_test = np.array(x_test)

		#reshape to 3d
		x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

		#get predicted price
		pred_price = model.predict(x_test)
		pred_price = scaler.inverse_transform(pred_price)

		#add new record to dataframe
		temp = pred_price[0][0]
		pred_date += datetime.timedelta(days=1)
		df_temp = pd.DataFrame(temp, index=[pred_date], columns=['Close'])
		df = pd.concat([df,df_temp], ignore_index = False)


	#data graphing
	p1 = -((today-start).days-future_days)
	p2 = -abs((today-end).days)

	past = df[p1:p2]

	future = df[-future_days-1:]
	if future_days == 0:
		future = df[-1:]

	plt.figure(figsize=(16,8))
	plt.xlabel('Date', fontsize=18)
	plt.ylabel('Price', fontsize=18)
	plt.plot(past['Close'])
	plt.plot(future['Close'])
	plt.legend(['past', 'future'], loc='lower right')
	plt.show()


root = Tk()

def myClick():
	t = ticker.get()
	s_year = start_year.get()
	s_month = start_month.get()
	s_day = start_day.get()
	e_year = end_year.get()
	e_month = end_month.get()
	e_day = end_day.get()
	stock_prediction(t, s_year, s_month, s_day, e_year, e_month, e_day)


#entry boxes
ticker = Entry(root)
start_year = Entry(root, width=20)
start_month = Entry(root, width=10)
start_day = Entry(root, width=10)
end_year = Entry(root, width=20)
end_month = Entry(root, width=10)
end_day = Entry(root, width=10)

#introducing text
ticker_label = Label(root, text='Ticker:')
start_label = Label(root, text='Start date:')
end_label = Label(root, text='End date:')
time_format = Label(root, text='Time = YYYY/MM/DD')

#show graph button
myButton = Button(root, text = "Show Graph", command=myClick)


#grid elements 
ticker_label.grid(row=0,column=0)
start_label.grid(row=1,column=0)
end_label.grid(row=2,column=0)
time_format.grid(row=0,column=2)

ticker.grid(row=0,column=1)
start_year.grid(row=1,column=1)
start_month.grid(row=1,column=2)
start_day.grid(row=1,column=3)
end_year.grid(row=2,column=1)
end_month.grid(row=2,column=2)
end_day.grid(row=2,column=3)

myButton.grid(row=3,column=1)

root.mainloop()
