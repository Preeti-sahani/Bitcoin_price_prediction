#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


# # Import data

# In[4]:


df = pd.read_csv('bitcoin_dataset.csv')
df.head(10)


# # EDA

# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.info()


# # Dealing with correlation

# In[9]:


df.corr()


# In[14]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr() , annot=True , cmap=plt.cm.Accent_r,annot_kws={'fontsize':10})
plt.show()


# In[18]:


def correlation(data , threshold):
    corr = data.corr()['btc_market_price'].sort_values(ascending=False)[1:]
    abs_corr = abs(corr)
    relevant_features = abs_corr[abs_corr>threshold]
    return relevant_features


# In[19]:


corr_features = correlation(df,0.81)


# In[20]:


corr_features


# In[22]:


sns.jointplot(df['btc_market_cap'] , df['btc_market_price'],color='red')


# In[23]:


# btc_market_price is highly correlated with btc_market_cap
df1 = df[corr_features.index]
df1


# In[24]:


df1.shape


# # Dealing with null values

# In[25]:


df1.isnull().sum()


# In[26]:


df1['btc_difficulty'] = df1['btc_difficulty'].fillna(df1['btc_difficulty'].mean())
df1['btc_trade_volume'] = df1['btc_trade_volume'].fillna(df1['btc_trade_volume'].mean())


# In[27]:


df1.isnull().sum()


# # Split data into independent and dependent features

# In[28]:


X = df1
y = df['btc_market_price']


# # Train Model

# In[29]:


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2 , random_state=51)


# In[30]:


X_train.shape ,  X_test.shape , y_train.shape , y_test.shape


# # Feature Scaling

# In[31]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[32]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Train a model

# In[33]:


lr = LinearRegression()


# In[34]:


lr.fit(X_train , y_train)


# # Test model

# In[36]:


pred = lr.predict(X_test)
pred[0]


# In[37]:


y_test.iloc[0]


# # ERROR EVALUATION

# In[39]:


from sklearn.metrics import mean_squared_error , mean_absolute_error
import math


# In[40]:


print(f'Mean Squared Error = {mean_squared_error(y_test,pred)}')
print(f'Root Mean Squared Error = {math.sqrt(mean_squared_error(y_test,pred))}')
print(f'Mean Absolute Error = {mean_absolute_error(y_test,pred)}')


# In[ ]:




