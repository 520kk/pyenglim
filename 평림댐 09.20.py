#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install silence_tensorflow')


# In[4]:


#get_ipython().system('pip install db')


# In[2]:


import sys
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import warnings
warnings.filterwarnings(action='ignore')
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import tensorflow as tf
import joblib
import pickle


# In[36]:


weekly_data = pd.read_csv('data/TB_PYEONGLIM_WEEKLY.csv', encoding='utf-8')
monthly_data = pd.read_csv('data/TB_PYEONGLIM_MONTHLY.csv', encoding='utf-8')


# In[37]:


monthly_data['water_depth'].describe()


# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[39]:


sns.distplot(monthly_data['water_depth'])


# In[40]:


sns.set(rc = {'figure.figsize':(20,20)})
sns.heatmap(monthly_data[['water_depth','wq_ss', 'wq_tcoli', 'wq_tn', 'wq_tp', 'wq_trans',
    'wq_phenol', 'wq_ec', 'wq_no3n', 'wq_nh3n', 'wq_ecoli', 'wq_pop',
    'wq_dtn', 'wq_dtp', 'wq_toc']].corr(), annot= True)


# In[41]:


sns.set(rc = {'figure.figsize':(10,10)})
sns.kdeplot(monthly_data['water_depth'])


# In[42]:


weekly_data['time'] = pd.to_datetime(weekly_data['time'], utc=False)
monthly_data['time'] = pd.to_datetime(monthly_data['time'], utc=False)


# In[43]:


weekly_data.set_index('time', inplace=True)
monthly_data.set_index('time', inplace=True)


# In[44]:


weekly_data.info() #528*50


# In[45]:


monthly_data.info() #118*50


# In[15]:


#weekly_data = weekly_data.fillna(method="ffill").fillna(method="bfill")
#monthly_data = monthly_data.fillna(method="ffill").fillna(method="bfill")


# In[46]:


feature = ['min_ps', 'avg_rhm', 'avg_ts', 'avg_pv', 'sum_ss_hr', 'ss_dur',
    'avg_ps', 'sum_sml_ev', 'avg_tca', 'avg_td', 'avg_ta', 'hr24_sum_rws',
    'sum_gsr', 'avg_pa', 'avg_ws', 'sum_lrg_ev', 'avg_lmac',
    'water_level', 'wq_cloa', 'wq_temp', 'wq_ph', 'wq_doc', 'wq_bod',
    'wq_cod', 'wq_ss', 'wq_tcoli', 'wq_tn', 'wq_tp', 'wq_trans',
    'wq_phenol', 'wq_ec', 'wq_no3n', 'wq_nh3n', 'wq_ecoli', 'wq_pop',
    'wq_dtn', 'wq_dtp', 'wq_toc']

target = 'water_depth'


# In[47]:


target


# In[48]:


weekly_data = weekly_data[feature+[target]]
monthly_data = monthly_data[feature+[target]]
"""
sum_rn feature 누락
"""


# In[49]:


weekly_data #528 rows × 39 columns


# In[50]:


weekly_data['day'] = weekly_data.index.day
weekly_data['month'] = weekly_data.index.month
weekly_data['year'] = weekly_data.index.year
monthly_data['day'] = monthly_data.index.day
monthly_data['month'] = monthly_data.index.month
monthly_data['year'] = monthly_data.index.year


# In[51]:


feature += ['day', 'month', 'year']


# In[52]:


weekly_data #528 rows × 42 columns


# In[53]:


monthly_data  #118 rows × 42 columns


# In[55]:


sns.barplot(x='year', y='water_depth', data=weekly_data)


# In[59]:


sns.barplot(x='month', y='water_depth', data=monthly_data)


# In[61]:


sns.lineplot(x='year', y='water_depth', data=weekly_data)


# In[70]:


sns.set(rc = {'figure.figsize':(20,5)})
fig, ax = plt.subplots(ncols=5)
sns.scatterplot(x='wq_ec',y= "avg_ps", data= weekly_data, ax=ax[0])
sns.regplot(x='wq_no3n', y="avg_ps", data= weekly_data, ax=ax[1])
sns.regplot(x='wq_nh3n', y="avg_ps", data= weekly_data, ax=ax[2])
sns.regplot(x='wq_ecoli', y="avg_ps", data= weekly_data, ax=ax[3])
sns.regplot(x='wq_pop', y="avg_ps", data= weekly_data, ax=ax[4])


# In[ ]:





# In[27]:


# 데이터 스플릿
weekly_window_size = 12
monthly_window_size = 4

weekly_data['label'] = weekly_data[target].shift(periods=weekly_window_size)
monthly_data['label'] = monthly_data[target].shift(periods=monthly_window_size)


train_ratio = 0.9
cut_point = int(len(weekly_data)*train_ratio)

weekly_data.reset_index(inplace=True)
monthly_data.reset_index(inplace=True)


# In[24]:


pd.set_option('display.max_columns', None) # 전체 칼럼을 보여줌
pd.set_option('display.max_rows', None) # 전체 로우를 보여줌
weekly_data


# In[34]:


# 데이터 스플릿
print("\n\n****** Weekly model Train ******")

weekly_scale = MinMaxScaler()
weekly_scale.fit(weekly_data.loc[:cut_point, feature])
joblib.dump(weekly_scale, './model/weekly_scaler.pkl')

weekly_x = []
weekly_y = []

for i in range(len(weekly_data)):

    if (i+weekly_window_size) < len(weekly_data):
        weekly_x.append(np.array(weekly_scale.transform(weekly_data.iloc[i:i+weekly_window_size,][feature])))
        weekly_y.append(np.array(weekly_data.iloc[i+weekly_window_size,-1].astype(float)))

weekly_x = np.array(weekly_x)
weekly_y = np.array(weekly_y)

weekly_train_x = weekly_x[:cut_point, ]
weekly_test_x = weekly_x[cut_point:, ]
weekly_train_y = weekly_y[:cut_point, ]
weekly_test_y = weekly_y[cut_point:, ]

# 모델구축, 학습
### weekly model
weekly_model = Sequential()
weekly_model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=[weekly_train_x.shape[1], weekly_train_x.shape[2]]))
weekly_model.add(Dropout(0.1))
weekly_model.add(LSTM(100, activation='relu'))
weekly_model.add(Dropout(0.4))
weekly_model.add(Dense(1))

weekly_model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=0.02),metrics=['mae'])
weekly_model.fit(weekly_train_x, weekly_train_y, epochs=100, batch_size=32, validation_data=[weekly_test_x, weekly_test_y])

train_result = weekly_model.evaluate(weekly_train_x, weekly_train_y)
print("\n=======Weekly Train result=======")
print("- MSE :", train_result[0])
print("- MAE :", train_result[1])

test_result = weekly_model.evaluate(weekly_test_x, weekly_test_y)
print("\n=======Weekly Test result=======")
print("- MSE :", test_result[0])
print("- MAE :", test_result[1])

weekly_model.save("./model/weekly_model.h5")


# In[35]:


### monthly model

print("\n\n****** Monthly model Train ******")

cut_point = int(len(monthly_data) * train_ratio)
monthly_scale = MinMaxScaler()
monthly_scale.fit(monthly_data.loc[:cut_point, feature])
joblib.dump(monthly_scale, './model/monthly_scaler.pkl')

monthly_x = []
monthly_y = []

for i in range(len(monthly_data)):
    if (i+monthly_window_size) < len(monthly_data):
        monthly_x.append(np.array(monthly_scale.transform(monthly_data.iloc[i:i+monthly_window_size, ][feature])))
        monthly_y.append(np.array(monthly_data.iloc[i+monthly_window_size, -1].astype(float)))

monthly_x = np.array(monthly_x)
monthly_y = np.array(monthly_y)

monthly_train_x = monthly_x[:cut_point, ]
monthly_test_x = monthly_x[cut_point:, ]
monthly_train_y = monthly_y[:cut_point, ]
monthly_test_y = monthly_y[cut_point:, ]

monthly_model = Sequential()
monthly_model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=[monthly_train_x.shape[1], monthly_train_x.shape[2]]))
monthly_model.add(Dropout(0.1))
monthly_model.add(LSTM(100, activation='relu'))
monthly_model.add(Dropout(0.4))
monthly_model.add(Dense(1))

monthly_model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=0.02), metrics=['mae'])
monthly_model.fit(monthly_train_x, monthly_train_y, epochs=100, batch_size=32, validation_data=[monthly_test_x, monthly_test_y])

train_result = monthly_model.evaluate(monthly_train_x, monthly_train_y)
print("\n=======Monthly Train result=======")
print("- MSE :", train_result[0])
print("- MAE :", train_result[1])

test_result = monthly_model.evaluate(monthly_test_x, monthly_test_y)
print("\n=======Monthly Test result=======")
print("- MSE :", test_result[0])
print("- MAE :", test_result[1])

monthly_model.save("./model/monthly_model.h5")

print("\n\n** two model saved at model dir! **\n\n")


# In[87]:


# RandomForest, XGBoost, LSTM
# 상관관계
# 데이터 시각화
# 시간, target
# 상관관계가 높은 특성, target

