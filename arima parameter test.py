# Test for ARIMA hyper parameters

# Define range for p, q and i 
# Output table of MAPE for the tested parameters in ascending order 

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def ARIMA_param_test(data,p_range,q_range,i_range):
    data_array = data.values
    avg_errors = []
    for p in range(p_range):
        for q in range(q_range):
            for i in range(i_range):
                errors = []
                tscv = TimeSeriesSplit(test_size=10)
                for train_index, test_index in tscv.split(data_array):
                    X_train, X_test = data_array[train_index], data_array[test_index]
                    X_test_orig = X_test
                    fcst = []
                    for step in range(10):




                        try:
                            mod = ARIMA(X_train, order=(p,i,q))
                            res = mod.fit()
                            fcst = np.append(fcst,res.forecast(steps=1))
                        except:
                            
                            fcst = np.append(fcst,100)
                        X_train = np.concatenate((X_train, X_test[0:1,:]))
                        X_test = X_test[1:]
                    errors.append(mean_absolute_percentage_error(X_test_orig.reshape(-1,), fcst))
                pq_result = [p, i, q, np.mean(errors)]
                avg_errors.append(pq_result)
    avg_errors = pd.DataFrame(avg_errors)
    avg_errors.columns = ['p', 'i', 'q', 'error']
    avg_errors.sort_values('error', ascending=True)

    return avg_errors