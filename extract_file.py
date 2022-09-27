#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:04:31 2017

@author: apple
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import os
os.chdir('/Users/apple/Desktop/NewA/kaggle/')
####################### others
PATH_DATA_RAW = '/Users/apple/Desktop/NewA/kaggle/'
holidays = pd.read_csv(PATH_DATA_RAW + "holidays_events.csv", parse_dates=['date'])
stores = pd.read_csv(PATH_DATA_RAW + "stores.csv"
                     , dtype={'store_nbr':np.uint8, 'cluster':np.uint8})
transactions = pd.read_csv(PATH_DATA_RAW + "transactions.csv"
                           , parse_dates=['date']
                           , dtype={'store_nbr':np.uint8, 'transactions':np.uint16})
oil = pd.read_csv(PATH_DATA_RAW + "oil.csv", parse_dates=['date'])
sample_submission = pd.read_csv(PATH_DATA_RAW + "sample_submission.csv")

####################### label Encoder
def LabelEncoder_pro(series):
    le = LabelEncoder()
    series_le = le.fit_transform(series)
    print("min:{0}, max:{1}".format(np.min(series_le), np.max(series_le)))
    return series_le, le

def pickle_load(filename):
    with open(filename, 'rb') as f:
        # https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
        return pickle.load(f, encoding='latin1')
    
def pickle_dump(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

####################### items 
items = pd.read_csv(PATH_DATA_RAW + "items.csv")
items_items_led, items_le = LabelEncoder_pro(items.item_nbr)
items.item_nbr = items_items_led

items.item_nbr = items.item_nbr.astype(np.uint16)
items.loc[:,'class'] = items['class'].astype(np.uint16)

items.perishable = items.perishable.astype(bool)

# pickle_dump(items_le, PATH_DATA_MODEL + "items_le.p")
#items_le = pickle_load(PATH_DATA_MODEL + "items_le.p")

####################### train
# before 3.7 GB --> 1.8 GB
train = pd.read_csv(PATH_DATA_RAW + 'train.csv'
                    , usecols=[1,2,3,4]
                    , parse_dates=['date']
                   )

####################### test
# before 80.4+ MB --> 38.6
test = pd.read_csv(PATH_DATA_RAW + 'test.csv'
                   , usecols=[1,2,3,4]
                   , parse_dates=['date'])


train.item_nbr = items_le.transform(train.item_nbr)
test.item_nbr = items_le.transform(test.item_nbr)

train.store_nbr = train.store_nbr.astype(np.uint8)
train.item_nbr = train.item_nbr.astype(np.uint16)
train.unit_sales = train.unit_sales.astype(np.float32)

test.store_nbr = test.store_nbr.astype(np.uint8)
test.item_nbr = test.item_nbr.astype(np.uint16)
