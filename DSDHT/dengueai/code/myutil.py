import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels as stats
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
import theano 
import tensorflow as tf
import keras

def plot_cols(arr):
    plt.figure(figsize=(15,40))
    for i in np.arange(0, arr.shape[1]):
        plt.subplot(arr.shape[1], 1, i+1)
        plt.plot(arr[:, i])
        
def plot_cols2(df):
    # won't plot certain columns, string, constant
    plotcolumns = set(df.columns) - {'city', 'year', 'weekofyear', 'week_start_date'}
    i = 1
    plt.figure(figsize=(25,60))
    for column in plotcolumns: 
        s = df[column].apply(pd.to_numeric)
        plt.subplot(len(df.columns), 1, i)
        s.plot(kind='line', color='blue', lw=2)
        s.rolling(window=12, center=False).mean().plot(kind='line', color='red', lw=0.8)
        plt.title(column, y=0.8, loc='right', fontsize=18)
        #s.rolling(window=12, center=False).std().plot(kind='line', color='black', lw=0.8)
        i += 1

def set_index(df):
    df['yearweekofyear'] = df[['year','weekofyear']].\
                           apply(lambda x: str(x[0]) + str(x[1]).zfill(2), axis=1)
    df.set_index('yearweekofyear', inplace=True)
    return df

def get_indexed_dataset(path):
    df = pd.read_csv(path)
    return set_index(df)

def split_dataset_by_city(df):
    return df[df['city']=='iq'], df[df['city']=='sj']

def prep_interpolate(df):
    dfnanfixdriver = pd.DataFrame(df.count()).reset_index()
    dfnanfixdriver.columns = ('colname','rowcount')
    target = dfnanfixdriver['rowcount'].max()
    collist = list(dfnanfixdriver.colname.values)
    for col in collist:
        non_nan_count = dfnanfixdriver[dfnanfixdriver['colname']==col]['rowcount'].values[0]
        if non_nan_count < target:
            df[col].interpolate(method='linear', axis=0, inplace=True)  
    return df

# preprocess train data
def preprocess(df, timesteps=1):

    # step 3: interpolate nans
    df = prep_interpolate(df.iloc[:,4:])

    # step 4: split array into features (starting at col 5) and labels 
    X = df.values[:,:-1].astype('float32')
    y = df.values[:,-1].reshape(X.shape[0],1)

    # step 5: normalize all features 
    scaler = MinMaxScaler(feature_range=(0,1))
    X_scaled = scaler.fit_transform(X)

    # shifts features one row at a time and pads them to the left of existing feature set
    feature_count = X.shape[1]
    X_scaled= X_scaled[:-1,:feature_count] 
    for i in range(1, timesteps):
        leftadd = X_scaled[:-1,:feature_count] 
        X_scaled = np.concatenate((leftadd, X_scaled[1:,:]), axis=1)

    return np.concatenate((X_scaled, y[timesteps:]), axis=1) 

# preprocess test data
def preprocess_test(df_train, df_test, timesteps=1):

    df_test_rowcount = df_test.shape[0]

    # will append training data, which preceeds test data in time
    # so we can create sequences using previous periods data for our predictions
    # just like we did during training 
    Xtrain = df_train.values[:,4:-1].astype('float32')

    df_test = prep_interpolate(df_test.iloc[:,4:])
    df_test.fillna(0, inplace=True)

    X = np.concatenate((Xtrain, df_test.values.astype('float32')), axis=0)
    scaler = MinMaxScaler(feature_range=(0,1))
    X_scaled = scaler.fit_transform(X)

    # save n rows (n=timesteps)
    #top_rows = X_scaled[0:timesteps] 

    # shifts features one row at a time and pads them to the left of existing feature set
    feature_count = X.shape[1]
    X_scaled= X_scaled[:-1,:feature_count] 
    for i in range(1, timesteps):
        leftadd = X_scaled[:-1,:feature_count] 
        X_scaled = np.concatenate((leftadd, X_scaled[1:,:]), axis=1)

    # X_scaled is timesteps rows short, need to add n=timestep rows back at the top
    #for i in range(timesteps, 0, -1):
    #   to_add = np.repeat(top_rows[i-1], timesteps, axis=0).\
    #            reshape(1,feature_count*timesteps)
    #   X_scaled = np.concatenate((to_add, X_scaled), axis=0)

    return X_scaled[-df_test_rowcount:,:] 

