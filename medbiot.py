#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:37:02 2020

@author: mayraju
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold,mutual_info_classif,mutual_info_regression
from sklearn.feature_selection import SelectKBest,SelectPercentile
names=["MI_dir_5_weight","MI_dir_5_mean","MI_dir_5_std","MI_dir_3_weight",
       "MI_dir_3_mean","MI_dir_3_std","MI_dir_1_weight","MI_dir_1_mean",
       "MI_dir_1_std","MI_dir_0.1_weight","MI_dir_0.1_mean",
       "MI_dir_0.1_std","MI_dir_0.01_weight","MI_dir_0.01_mean",
       "MI_dir_0.01_std","HH_5_weight_0","HH_5_mean_0","HH_5_std_0",
       "HH_5_radius_0_1","HH_5_magnitude_0_1","HH_5_covariance_0_1",
       "HH_5_pcc_0_1","HH_3_weight_0","HH_3_mean_0","HH_3_std_0",
       "HH_3_radius_0_1","HH_3_magnitude_0_1","HH_3_covariance_0_1",
       "HH_3_pcc_0_1","HH_1_weight_0","HH_1_mean_0","HH_1_std_0","HH_1_radius_0_1",
       "HH_1_magnitude_0_1","HH_1_covariance_0_1","HH_1_pcc_0_1","HH_0.1_weight_0",
       "HH_0.1_mean_0","HH_0.1_std_0","HH_0.1_radius_0_1","HH_0.1_magnitude_0_1",
       "HH_0.01_covariance_0_1","HH_0.1_pcc_0_1","HH_0.01_weight_0","HH_0.01_mean_0",
       "HH_0.01_std_0","HH_0.01_radius_0_1","HH_0.01_magnitude_0_1","HH_0.01_covariance_0_1",
       "HH_0.01_pcc_0_1","HH_jit_5_weight","HH_jit_5_mean","HH_jit_5_std",
       "HH_jit_3_weight","HH_jit_3_mean","HH_jit_3_std","HH_jit_1_weight",
       "HH_jit_1_mean","HH_jit_1_std","HH_jit_0.1_weight","HH_jit_0.1_mean",
       "HH_jit_0.1_std","HH_jit_0.01_weight","HH_jit_0.01_mean","HH_jit_0.01_std",
       "HpHp_5_weight_0","HpHp_5_mean_0","HpHp_5_std_0","HpHp_5_radius_0_1",
       "HpHp_5_magnitude_0_1","HpHp_5_covariance_0_1","HpHp_5_pcc_0_1","HpHp_3_weight_0",
       "HpHp_3_mean_0","HpHp_3_std_0","HpHp_3_radius_0_1","HpHp_3_magnitude_0_1",
       "HpHp_3_covariance_0_1","HpHp_3_pcc_0_1","HpHp_1_weight_0","HpHp_1_mean_0",
       "HpHp_1_std_0","HpHp_1_radius_0_1","HpHp_1_magnitude_0_1","HpHp_1_covariance_0_1",
       "HpHp_1_pcc_0_1","HpHp_0.1_weight_0","HpHp_0.1_mean_0","HpHp_0.1_std_0",
       "HpHp_0.1_radius_0_1","HpHp_0.1_magnitude_0_1","HpHp_0.1_covariance_0_1",
       "HpHp_0.1_pcc_0_1","HpHp_0.01_weight_0","HpHp_0.01_mean_0","HpHp_0.01_std_0",
       "HpHp_0.01_radius_0_1","HpHp_0.01_magnitude_0_1","HpHp_0.01_covariance_0_1",
       "HpHp_0.01_pcc_0_1"]

print(len(names))

print("process starting")


bash_data=pd.read_csv('/media/mayraju/92C22FBEC22FA587/medBiot/torii_mal_all.csv')
print(bash_data.shape)

#for a in bash_data.columns:
 #   print(a)
print("dataset_uploaded")
bash_data.fillna(0,inplace=True)
bash_data['labels'] = 'torii'
bash_trainleg_cc = bash_data.head(80000)
bash_testleg_cc = bash_data.tail(50000)
print(bash_testleg_cc.shape)
print(bash_trainleg_cc.shape)
print(bash_trainleg_cc.head(10))
X = bash_trainleg_cc.drop('labels',axis=1)
y = bash_trainleg_cc['labels']
#print(y)
print(np.dtype(y))
X_test = bash_testleg_cc.drop('labels',axis=1)
y1 = bash_testleg_cc['labels']
#print(y1)
print(type(y1))
print(type(y))
X_train_T = y.T
y_train = pd.DataFrame(X_train_T)
X_test_T = y1.T
y_test = pd.DataFrame(X_test_T)
#X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0,stratify = y)
##constant feature removall

constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X)

print(constant_filter.get_support().sum())

constant_list = [not temp  for temp in constant_filter.get_support()]
print(constant_list)
print(X.columns[constant_list])
X_train_filter = constant_filter.transform(X)
X_test_filter = constant_filter.transform(X_test)
print(X_train_filter.shape)
print(X_test_filter.shape)
print(X.shape)

##Quasi constant feature removal
quasi_constant_filter = VarianceThreshold(threshold=0.01)
quasi_constant_filter.fit(X_train_filter)
print(quasi_constant_filter.get_support().sum())
X_train_quasi_filter = quasi_constant_filter.transform(X_train_filter)
X_test_quasi_filter = quasi_constant_filter.transform(X_test_filter)

print(X_train_quasi_filter.shape)
print(X_test_quasi_filter.shape)


##remove duplicate features

X_train_T = X_train_quasi_filter.T
X_test_T = X_test_quasi_filter.T

X_train_T  = pd.DataFrame(X_train_T)
X_Test_T = pd.DataFrame(X_test_T)

print(X_train_T.shape)
print(X_test_T.shape)

print(X_train_T.duplicated().sum())

duplicated_features = X_train_T.duplicated()
print(duplicated_features)
features_to_keep = [not index for index in duplicated_features]
print(features_to_keep)

X_train_unique = X_train_T[features_to_keep].T
X_test_unique = X_test_T[features_to_keep].T
X_test1 = pd.DataFrame(X_test_unique)
print(X_train_unique.shape)
print(X_test_unique.shape)
print(X_test1)
print(type(X_train_unique))
print(type(X_test1))
print(X.shape)

#Y2 = pd.DataFrame(np.array(y).T)
#Y3 = pd.DataFrame(np.array(y1).T)
df_col = pd.concat([X_train_unique,y_train], axis=1)
df_col1 = pd.concat([X_test1,y_test], axis=1)
df_col.to_csv('/media/mayraju/92C22FBEC22FA587/medBiot/splited_data/torai_mall_train.csv',sep=',')
df_col1.to_csv('/media/mayraju/92C22FBEC22FA587/medBiot/splited_data/torai_mall_test.csv',sep=',')

