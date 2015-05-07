import numpy as np
import pandas as pd
import pandas.io.data as web
import datetime


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
    # line = w[0] * xi + w[1]

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

def split_and_label(data):
    dd = data
    dd['IsTomorrowUp'] = dd['Adj Close'].shift(-1) > dd['Adj Close']
    # turn into text label
    dd['IsTomorrowUp'] = dd['IsTomorrowUp'].apply(
            lambda label: 'UP' if label else 'DOWN')
    # drops first n rows where n is largest n in nums
    # eg. RoC15 will be NaN if we don't have data on 15 days prior
    dd = dd.dropna()

def training_prediction_api_format(df, filename='main_training.csv'):
    """ properly format to Prediction API then save file"""
    cols = list(df) # list of columns
    # dont need any of these features
    col_to_del = ['Open', 'Adj Close', 'High', 'Low', 'Close', 'Volume', 'date']
    keep = [col for col in cols if col not in col_to_del]
    # move label to front for proper Prediction API format
    keep.insert(0, keep.pop(keep.index('IsTomorrowUp')))
    df = df.ix[:, keep] # reorder columns

    # save without index or header
    df.to_csv(filename, sep=',', header=False, index=False)
    print('saved %s' % (filename))
    return df

def testing_prediction_api_format(df, filename='main_testing.csv'):
    cols = list(df)
    col_to_del = ['Open', 'Adj Close', 'High', 'Low', 'Close','Volume', 'date']
    keep = [col for col in cols if col not in col_to_del]
    keep.insert(0, keep.pop(keep.index('IsTomorrowUp')))
    df = df.ix[:, keep]

    df.to_csv(filename, sep=',')
    print('saved %s' % (filename))
    return df



# TODO: grab start, end, ticker and nums array from system
start = datetime.datetime(2010,1,1)
end = datetime.datetime(2015,4,30)
# get ticker data from yahoo finance
f = web.DataReader("GLD", 'yahoo', start,end)

updated_df = make_features(f, [2,5,15])
# arbitrarily splitting training and testing data
training_data = training_prediction_api_format(
        split_and_label(updated_df[0:1100]))

testing_data = testing_prediction_api_format(
        split_and_label(updated_df[1100:]))


