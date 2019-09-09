from sklearn import linear_model, preprocessing, model_selection, cross_validation
import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np
import math
import seaborn as seabornInstance
import matplotlib.pyplot as plt


def prepare_data(df, forecast_col, forecast_out, test_size):
    # creating new column called label with the last 5 rows are nan
    df['label'] = df[forecast_col].shift(-forecast_out)
    X = np.array(df[[forecast_col]])  # creating the feature array
    X = preprocessing.scale(X)  # processing the feature array
    X_lately = X[-forecast_out:]  # creating the column i want to use later in the predicting method
    X = X[:-forecast_out]  # X that will contain the training and testing
    label.dropna(inplace=True)  # dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
        X, y, test_size=test_size)  # cross validation

    response = [X_train, X_test, Y_train, Y_test, X_lately]
    return response


start = datetime.datetime(2008, 1, 1)
end = datetime.datetime(2019, 1, 1)
df = web.DataReader("AAPL", "yahoo", start, end)
df = df.reset_index()
df.plot(x='Date', y='Adj Close', style='o')
plt.title('AAPL Adj Close Price over time')
plt.xlabel('Close Date')
plt.ylabel('Adj Close Price')
plt.show()

forecast_col = 'Adj Close'  # choosing which column to forecast
forecast_out = 1  # how far to forecast
test_size = 0.2  # the size of my test set

# calling the method were the cross validation and data preperation is in
X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size)

linear_reg = linear_model.LinearRegression()
linear_reg.fit(X_train, y_train)
# To retrieve the intercept:
print(linear_reg.intercept_)
# For retrieving the slope:
print(linear_reg.coef_)

score = linear_reg.score(X_test, Y_test)
y_pred = linear_reg.predict(X_test)

response = {}  # creting json object
response['test_score'] = score
response['forecast_set'] = y_pred

print(response['forecast_set'])
