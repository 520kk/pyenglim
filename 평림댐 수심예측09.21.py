#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install openpyxl')


# In[41]:


get_ipython().system('pip install statsmodels')


# In[2]:


get_ipython().system('pip install plotly')


# In[15]:


import pandas as pd
import numpy as np
import os
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math


# In[16]:


df = pd.read_excel('./dam/total.xlsx') # 평림댐 데이터를 불러옴


# In[17]:


mpl.rc('font', family = 'Malgun Gothic') # matplotlib에 한글 적용


# In[18]:


df.info(), df.columns # 정보와 칼럼명 확인


# In[19]:


df = pd.DataFrame(df) # 불러온 df를 dataframe으로 만들어줌.
df.info() # df의 정보를 확인함.


# In[20]:


df['일시'] = pd.to_datetime(df['일시']) # 데이터 타입이 object였던 ['일시']를 datetime으로 변환함.


# In[21]:


df1 = df.sort_values(by=['일시']) # datetime인 ['일시']를 오름차순으로 정렬함.


# In[22]:


pd.set_option('display.max_columns', None) # 전체 칼럼을 보여줌
pd.set_option('display.max_rows', None) # 전체 로우를 보여줌
#원상복귀 하려면
#pd.options.display.max_rows = 60
#pd.options.display.max_columns = 20


# In[23]:


pd.options.display.max_rows = 60
pd.options.display.max_columns = 20


# In[24]:


df1


# In[25]:


df1['일시'].describe() #datetime은 describe()를 따로 해야함.


# In[26]:


df1['Year'] = df1['일시'].dt.year # df1['일시']의 연도
df1['Month'] = df1['일시'].dt.month # df1['일시']의 달
# df1['Week'] = df1['일시'].dt.week # df1['일시']의 주
# df1['Day_of_Week'] = df1['일시'].dt.day_name() # df1['일시']의 요일


# In[27]:


# df1 = df1.drop(['Day_of_Week'], axis=1) 


# In[28]:


df1


# In[53]:


df1 = pd.get_dummies(data = df1, columns = ['Year'], prefix = 'Year') # 원핫인코딩 연도


# In[54]:


df1 = pd.get_dummies(data = df1, columns = ['Month'], prefix = 'Month') # 원핫인코딩 월


# In[81]:


df1.info()


# In[82]:


df1.describe() # 기술적 통계 확인


# In[29]:


df1.columns # 컬럼 확인


# In[30]:


df1 = df1.reset_index(drop=True) # 정렬된 데이터의 인덱스를 재설정


# In[31]:


df1


# In[32]:


# 데이터셋을 정제할 때, 특성별로 데이터 스케일이 다르면 안되기 때문에 이 작업을 통해 모든 특성의 범위를 같게 만들어줘야함
# 전체 데이터가 아닌 훈련 데이터에 대해서만 fit() 적용

df1.columns


# In[95]:


train = df1[:3288]
test = df1[3288:]
train_X = train[['수위 (EL.m)', '저수율 (%)', '강우량 (mm)', '유입량 (㎥/s)','총방류량 (㎥/s)']]
train_y = train[['저수량 (백만/㎥)']]
test_X = train[['수위 (EL.m)', '저수율 (%)', '강우량 (mm)', '유입량 (㎥/s)','총방류량 (㎥/s)']]
test_y = train[['저수량 (백만/㎥)']]


# In[99]:


params = {
    'n_estimators':(100, 500),
    'max_depth' : (5, 15),
    'min_samples_leaf' : (3, 18),
    'min_samples_split' : (3, 18)
}

rf_run = RandomForestRegressor(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_run, param_grid=params, cv=2, n_jobs=-1)
grid_cv.fit(train_X, train_y)
 
 
print('최적 하이퍼 파라미터:', grid_cv.best_params_)
print('최적 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))


# In[100]:


rf_run = RandomForestRegressor(random_state=0, max_depth=15, min_samples_leaf=3, min_samples_split=3, n_estimators=100)
rf_run.fit(train_X, train_y)


# In[101]:


train_predict = rf_run.predict(train_X)
print("RMSE':{}".format(math.sqrt(mean_squared_error(train_predict, train_y))))


# In[102]:


rf_run_predict = rf_run.predict(test_X)
print("RMSE':{}".format(math.sqrt(mean_squared_error(rf_run_predict, test_y))) )


# In[103]:


ftr_importances_values = rf_run.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=train_X.columns)
ftr_top = ftr_importances.sort_values(ascending=False)[:10]
 
plt.figure(figsize=(8, 6))
sns.barplot(x=ftr_top, y=ftr_top.index)
plt.show()


# In[104]:


df1.corr()


# In[33]:


import plotly.express as px

fig = px.line(df1, x="일시", y="저수량 (백만/㎥)", title='평림댐 저수량')
fig.show()


# In[36]:


fig = px.line(df1, x="일시", y="수위 (EL.m)", title='평림댐 수위')
fig.show()


# In[38]:


fig = px.line(df1, x="일시", y="저수율 (%)", title='평림댐 저수율')
fig.show()


# In[39]:


fig = px.line(df1, x="일시", y="강우량 (mm)", title='평림댐 강우량')
fig.show()


# In[40]:


fig = px.line(df1, x="일시", y="유입량 (㎥/s)", title='평림댐 유입량')
fig.show()


# In[44]:


fig = px.line(df1, x="일시", y="총방류량 (㎥/s)", title='평림댐 총방류랑')
fig.show()


# In[46]:


sns.kdeplot(df1['저수율 (%)'])

