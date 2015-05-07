import numpy as np
import pandas as pd
import pandas.io.data as web
import datetime
import matplotlib.pyplot as plt
import os, requests

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2015,4,30)
# get ticker data from yahoo finance
f = web.DataReader("GLD", 'yahoo', start,end)

def trendline(data, start, day, n):
    """ basic linear regression for trend line
    day: current date
    start: start of dataset
    n: last *n* days we want trend line for
    data: DataFrame
    """
    last_n = data['Adj Close'][start:day].tail(n+1)[0:n] # last n days

    y = [d for d in last_n] # data
    xi = np.arange(last_n.shape[0])
    A = np.array([ xi, np.ones(last_n.shape[0])])
    w = np.linalg.lstsq(A.T, y)[0]
    line = w[0] * xi + w[1]

    return w # slope w[0]

def rate_of_change(data, day, n):
    """
    rate of change
    P(x) - P(x - n) / P(x -n)
    """
    price = data['Adj Close'][day]
    n_days_ago_price = data['Adj Close'].shift(n).ix[day]
    return (price - n_days_ago_price) / n_days_ago_price

def ratio_ROC(data, day, n, m):
    """ ratio of rate of change
    ROCn / ROCm
    """
    return (rate_of_change(data, day, n)) / (rate_of_change(data, day, m))

def stochastic_oscillator(data, start, day, n=14):
    """ Stochastic Oscillator - just another technical indicator
    L_n = Lowest price over past n days
    H_n = Highest price over past n days
    P(x) = price on days x
    %K = (P(X) - L_n / (H_n - L_n))
    """
    last_n = data['Adj Close'][start:day].tail(n+1)[0:n]
    L = np.min(last_n)
    H = np.max(last_n)
    price = data['Adj Close'][day]
    K = ((price - L) / (H - L))
    return K

def make_features(data, nums, start='20100104'):
    """ takes a dataframe and appends features to each row.
    nums is array of time periods we want each feature for."""
    df = data # copy
    # TODO: grab start date from first date of DF, instead of explicitly
    df['date'] = df.index
    for n in nums:
        # slope of trendline for n in nums
        df['trend_slope_'+str(n)] = df['date'].apply(
                lambda day: trendline(data, start, day, n)[0])
        # ROCn for each n in nums
        df['RoC_'+str(n)] = df['date'].apply(
                lambda day: rate_of_change(data, day, n))
        # ROC1 / ROCm for m in nums
        df['ratio_ROC_'+str(n)] = df['date'].apply(
                lambda day: ratio_ROC(data, day, 1, n))

        # value %K for period 14
        df['%K-14'] = df['date'].apply(
            lambda day: stochastic_oscillator(data, start, day))

    return df


