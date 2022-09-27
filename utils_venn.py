import pandas as pd
import numpy as np
import matplotlib as plt
import datetime
import time

PATH_MAIN = "data/"
PATH_DATA_RAW = PATH_MAIN + "raw/"
PATH_DATA_PROCESSED = PATH_MAIN + "processed/"
PATH_DATA_SUBMIT = PATH_MAIN + "submit/"
PATH_DATA_MODEL = PATH_MAIN + "model/"

def divide_time(df,time_stamp):
    df['tmp'] = df[time_stamp].astype('str').apply\
    (lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M').timestamp())
    df['year'] = df['tmp'].apply(lambda x: datetime.datetime.fromtimestamp(x).year)
    df['month'] = df['tmp'].apply(lambda x: datetime.datetime.fromtimestamp(x).month)
    df['weekday'] = df['tmp'].apply(lambda x: datetime.datetime.fromtimestamp(x).weekday())
    df['day'] = df['tmp'].apply(lambda x: datetime.datetime.fromtimestamp(x).day)
    df['hour'] = df['tmp'].apply(lambda x: datetime.datetime.fromtimestamp(x).hour) + df['tmp'].apply(lambda x: datetime.datetime.fromtimestamp(x).minute) / 60.
    del df['tmp']
    del df['time_stamp']
    return df

def get_useful_wifi(df):
    df['wifi'] = df['wifi_infos'].str.extract('(?P<ssid>b_[0-9]+\|-[0-9]+\|true)', expand=True)
    df['ssid'] = df['wifi'].str.split('|').str[0]
    df['intensity'] = df['wifi'].str.split('|').str[1]
    df['wifi'] = df['wifi'].str.split('|').str[2]
    df['wifi'][df['wifi'].isnull() == False] = 1
    df['wifi'][df['wifi'].isnull() == True] = 0
    del df['wifi_infos']
    return df

def label_encoding(train,test,column,le):
    col = train[column]
    col = col.append(test[column])
    le.fit(col)
    train[column] = le.transform(train[column])
    test[column] = le.transform(test[column])
    return train,test,le
    