#ARIMA Forecasting 

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np



def forecasting(step, data, p,d,q):
    step = step
    step_plot = step
    X = data.values
    size = int(len(X) * 0.60)
    
    train, test = X[0:size], X[size:]

    predictions_rolling = []
    conf_ints_rolling = []  # To store confidence intervals

    # Fit the model only once on the training set
    model = ARIMA(train, order=(p,d,q))
    model_fit = model.fit()

    # Generate rolling forecasts
    for i in range(0, len(test), step):
        if i + step > len(test):
            step = len(test) - i  # Adjust step size if it exceeds the test set

        # Get forecast for 'step' ahead
        output = model_fit.get_forecast(steps=step)
        predictions_rolling.extend(output.predicted_mean)
        
        # Extract confidence intervals
        conf_int = output.conf_int(alpha=0.05)
        conf_ints_rolling.extend(conf_int)

        # Update the model with the latest observed data
        model_fit = model_fit.append(test[i:i+step],refit = False )

     # Prepare for plotting
    testPredictPlot = np.empty_like(data['num_infected'])
    testPredictPlot[:] = np.nan
    testPredictPlot[len(train):] = predictions_rolling
    
    # Convert confidence intervals to arrays
    conf_ints_rolling = np.array(conf_ints_rolling)
    lower_limits = np.empty_like(data['num_infected'])
    upper_limits = np.empty_like(data['num_infected'])
    lower_limits[:] = np.nan
    upper_limits[:] = np.nan
    lower_limits[len(train):] = conf_ints_rolling[:, 0]
    upper_limits[len(train):] = conf_ints_rolling[:, 1]

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(test, predictions_rolling))
    mape = mean_absolute_percentage_error(test, predictions_rolling)

    # Plot actual values, predictions, and confidence intervals
    plt.plot(np.array(data['num_infected']), color = '#53565A',label='Actual')
    plt.plot(testPredictPlot, color='#FF4500', label='Predictions')
    plt.fill_between(np.arange(len(data)), lower_limits, upper_limits, color='coral', alpha=0.2, label='95% CI')
    plt.xlabel('Day')
    plt.ylabel('Number Infected')
    plt.title(f'Predicting {step_plot} time steps ahead')
    plt.legend()
    plt.show()
    
    return rmse,mape,testPredictPlot,lower_limits,upper_limits