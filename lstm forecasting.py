import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error ,r2_score

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.6)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:,:]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


n = 80
trainX, trainY = create_dataset(train, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))



# create and fit the LSTM network
model = Sequential([Input(shape = (1, look_back))])
model.add(LSTM(n, return_sequences = True,activation='relu'))
model.add(LSTM(n,activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(trainX, trainY, epochs=150000, batch_size=1, verbose=False)

model = tf.keras.models.load_model('my_model.keras')

def forecasting(step,test):
    look_back = 3
    testX, testY= create_dataset(test, look_back)# reshape input to be [samples, time steps, features]
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


    
    test_predictions = []
    new_test = testX [0]
    for i in range(testX.shape[0]):
        y_pred = model.predict(new_test.reshape(1,1,look_back))
        test_predictions.append(y_pred)
        if (i<testX.shape[0]-1):
            if (i%step == step-1):
                new_test = testX[i+1]
            else:
                new_test = np.append(new_test,y_pred, axis =1)[:,1:]



    testPredict = np.array(test_predictions).reshape(len(test_predictions),1)  
  
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    

    # Evaluate R2
    r2_test= r2_score(testY[0], testPredict[:,0])

    #MAPE

    testMAPE = mean_absolute_percentage_error(testY[0], testPredict[:,0])
    
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train)+look_back:len(dataset), :] = testPredict

    # plot baseline and predictions
    plt.figure(figsize=(14, 6))
    plt.plot(scaler.inverse_transform(dataset),label = "True data", color = '#53565A' , alpha = 0.9)
    plt.plot(testPredictPlot, label = "Test Prediction", color = '#D9027D')
    
    plt.xlabel('Days',fontsize = 14)
    plt.ylabel('Number of new cases',fontsize = 14)
    plt.title("LSTM Predition for %s days ahead"%step,fontsize = 14)
    plt.legend()
    plt.show()

    return testPredict,testScore,r2_test,testMAPE