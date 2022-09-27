#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir('/Users/apple/Desktop/NewA/kaggle/')
from datetime import date, timedelta
import calendar as ca
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from imp import reload
import utils_venn; reload(utils_venn); from utils_venn import *;

PATH_DATA_RAW = '/Users/apple/Desktop/NewA/kaggle/'
df_train = pd.read_csv(PATH_DATA_RAW +
    'train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)
item_nbr_u = df_train[df_train.date>pd.datetime(2017,8,10)].item_nbr.unique()

df_test = pd.read_csv(PATH_DATA_RAW +
    "test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)
items = pd.read_csv(PATH_DATA_RAW +
    "items.csv",
).set_index("item_nbr")

df_2017 = df_train.loc[df_train.date>=pd.datetime(2016,12,26)]
del df_train

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)
items = items.reindex(df_2017.index.get_level_values(1))


def label_encoding(train,column,le):
    col = train[column]
    le.fit(col)
    train[column] = le.transform(train[column])
    return train,le

# change items['family'] from text to categorical
le_family = LabelEncoder()
items, le_family = label_encoding(items,'family',le_family)

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def get_nearwd(date,b_date):
    date_list = pd.date_range(date-timedelta(140),periods=21,freq='7D').date
    result = date_list[date_list<=b_date][-1]
    return result
def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        'family':items.reset_index().family,
        'class':items.reset_index()['class'],
        'perishable':items.reset_index().perishable,
        'item_nbr':promo_2017.reset_index().item_nbr,
        'store_nbr':promo_2017.reset_index().store_nbr,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values,
        "unpromo_16aftsum_2017":(1-get_timespan(promo_2017, t2017+timedelta(16), 16, 16)).iloc[:,1:].sum(axis=1).values, 
    })

    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
        for j in [14,60,140]:
            X["aft_promo_{}{}".format(i,j)] = (promo_2017[
                t2017 + timedelta(days=i)]-1).values.astype(np.uint8)
            X["aft_promo_{}{}".format(i,j)] = X["aft_promo_{}{}".format(i,j)]\
                                        *X['promo_{}_2017'.format(j)]
        if i ==15:
            X["bf_unpromo_{}".format(i)]=0
        else:
            X["bf_unpromo_{}".format(i)] = (1-get_timespan(
                    promo_2017, t2017+timedelta(16), 16-i, 16-i)).iloc[:,1:].sum(
                            axis=1).values / (15-i) * X['promo_{}'.format(i)]

    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        #X['mean_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i, 12, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values        
        
        date = get_nearwd(t2017+timedelta(i),t2017)
        ahead = (t2017-date).days
        if ahead!=0:
            X['ahead0_{}'.format(i)] = get_timespan(df_2017, date+timedelta(ahead), ahead, ahead).mean(axis=1).values
            X['ahead7_{}'.format(i)] = get_timespan(df_2017, date+timedelta(ahead), ahead+7, ahead+7).mean(axis=1).values
        X["day_1_2017_{}1".format(i)]= get_timespan(df_2017, date, 1, 1).values.ravel()
        X["day_1_2017_{}2".format(i)]= get_timespan(df_2017, date-timedelta(7), 1, 1).values.ravel()
        for m in [3,7,14,30,60,140]:
            X["mean_{}_2017_{}1".format(m,i)]= get_timespan(df_2017, date,m, m).\
                mean(axis=1).values
            X["mean_{}_2017_{}2".format(m,i)]= get_timespan(df_2017, date-timedelta(7),m, m).\
                mean(axis=1).values
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X


# add store features
stores = pd.read_csv(PATH_DATA_RAW + "stores.csv")
le_city = LabelEncoder()
le_state = LabelEncoder()
le_type = LabelEncoder()
le_cluster = LabelEncoder()
stores, le_city = label_encoding(stores, 'city',le_city)
stores, le_state = label_encoding(stores, 'state',le_state)
stores, le_type = label_encoding(stores, 'type',le_type)
stores, le_cluster = label_encoding(stores, 'cluster',le_cluster)

def get_n_days_stat(t2017, n,level):
    tmp_item_city = get_timespan(df_2017, t2017, n, n).reset_index()
    tmp_item_city = pd.merge(tmp_item_city, stores, how='left', on='store_nbr')
    tmp_item_city_mean = tmp_item_city.groupby(['item_nbr',level]).mean().reset_index()
    mean_name = level + '_mean_' + str(n)
    median_name = level + '_median_' + str(n)
    std_name = level + '_std_' + str(n)
    tmp_item_city_mean[mean_name] = tmp_item_city_mean.drop(['item_nbr', 'city', 'store_nbr',
                           'state', 'type', 'cluster'], axis = 1).mean(axis = 1)
    tmp_item_city_mean[median_name] = tmp_item_city_mean.drop(['item_nbr', 'city', 'store_nbr',
                           'state', 'type', 'cluster'], axis = 1).median(axis = 1)
    tmp_item_city_mean[std_name] = tmp_item_city_mean.drop(['item_nbr', 'city', 'store_nbr',
                       'state', 'type', 'cluster'], axis = 1).std(axis = 1)
    tmp_item_city_mean = pd.DataFrame({'item_nbr':tmp_item_city_mean.item_nbr,\
                                       level:tmp_item_city_mean[level],\
                                      mean_name:tmp_item_city_mean[mean_name],\
                                      median_name: tmp_item_city_mean[median_name],\
                                      std_name: tmp_item_city_mean[std_name]})
    return tmp_item_city_mean

def get_n_days_stat_global(t2017, n):
    tmp_item_city = get_timespan(df_2017, t2017, n, n).reset_index()
    tmp_item_city = pd.merge(tmp_item_city, stores, how='left', on='store_nbr')
    tmp_item_city_mean = tmp_item_city.groupby(['item_nbr']).mean().reset_index()
    mean_name = '_mean_' + str(n)
    median_name = '_median_' + str(n)
    std_name = '_std_' + str(n)
    tmp_item_city_mean[mean_name] = tmp_item_city_mean.drop(['item_nbr', 'city', 'store_nbr',
                           'state', 'type', 'cluster'], axis = 1).mean(axis = 1)
    tmp_item_city_mean[median_name] = tmp_item_city_mean.drop(['item_nbr', 'city', 'store_nbr',
                           'state', 'type', 'cluster'], axis = 1).median(axis = 1)
    tmp_item_city_mean[std_name] = tmp_item_city_mean.drop(['item_nbr', 'city', 'store_nbr',
                       'state', 'type', 'cluster'], axis = 1).std(axis = 1)
    tmp_item_city_mean = pd.DataFrame({'item_nbr':tmp_item_city_mean.item_nbr,\
                                      mean_name:tmp_item_city_mean[mean_name],\
                                      median_name: tmp_item_city_mean[median_name],\
                                      std_name: tmp_item_city_mean[std_name]})
    return tmp_item_city_mean

def achieve_pre_n_days_unique(t2017, n):
    tmp_unique_value = get_timespan(df_2017, t2017, n, n).reset_index()
    tmp_unique_value = tmp_unique_value.drop(['store_nbr', 'item_nbr'], axis = 1)
    return tmp_unique_value.apply(lambda x: len(x.unique()), axis=1)

def achieve_no_sales_rate(t2017, n):
    tmp_unique_value = get_timespan(df_2017, t2017, n, n).reset_index()
    tmp_unique_value = tmp_unique_value.drop(['store_nbr', 'item_nbr'], axis = 1)
    tmp_unique_value[tmp_unique_value == 0] = -1
    tmp_unique_value[tmp_unique_value != -1] = 0
    tmp_unique_value[tmp_unique_value == -1] = 1
    return tmp_unique_value.sum(axis = 1)/n


print("Preparing dataset...")
t2017 = date(2017, 6, 7)
X_l, y_l = [], []
for i in range(6):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_tmp = pd.merge(X_tmp, stores, how='left', on='store_nbr')
    
    tmp_item_city_mean = get_n_days_stat(t2017 + delta, 7,'city')
    tmp_item_city_mean_14 = get_n_days_stat(t2017 + delta, 14,'city')
    tmp_item_city_mean_31 = get_n_days_stat(t2017 + delta, 31,'city')
    tmp_item_s_mean = get_n_days_stat_global(t2017 + delta, 7)
    tmp_item_s_mean_14 = get_n_days_stat_global(t2017 + delta, 14)
    tmp_item_s_mean_31 = get_n_days_stat_global(t2017 + delta, 31)
    
    X_tmp = pd.merge(X_tmp, tmp_item_city_mean, how='left', on=['item_nbr','city'])
    X_tmp = pd.merge(X_tmp, tmp_item_city_mean_14, how='left', on=['item_nbr','city'])
    X_tmp = pd.merge(X_tmp, tmp_item_city_mean_31, how='left', on=['item_nbr','city'])
    X_tmp = pd.merge(X_tmp, tmp_item_s_mean, how='left', on=['item_nbr'])
    X_tmp = pd.merge(X_tmp, tmp_item_s_mean_14, how='left', on=['item_nbr'])
    X_tmp = pd.merge(X_tmp, tmp_item_s_mean_31, how='left', on=['item_nbr'])
    
    X_tmp['unique_140'] = achieve_pre_n_days_unique(t2017 + delta, 140)
    
    X_tmp['no_sales_7'] = achieve_no_sales_rate(t2017 + delta, 7)
    X_tmp['no_sales_14'] = achieve_no_sales_rate(t2017 + delta, 14)
    X_tmp['no_sales_30'] = achieve_no_sales_rate(t2017 + delta, 30)
    X_tmp['no_sales_60'] = achieve_no_sales_rate(t2017 + delta, 60)
    X_tmp['no_sales_140'] = achieve_no_sales_rate(t2017 + delta, 140)
    
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)