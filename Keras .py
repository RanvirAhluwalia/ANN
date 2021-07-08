#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('../DATA/fake_reg.csv')


# In[3]:


df.head()


# In[4]:


x = df[['feature1', 'feature2']].values


# In[5]:


y = df['price'].values


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 100)


# In[8]:


x_train.shape


# In[9]:


y_train.shape


# In[10]:


x_test.shape


# In[11]:


y_test.shape


# In[12]:


import tensorflow as tf


# In[13]:


from sklearn.preprocessing import MinMaxScaler


# In[14]:


scaler = MinMaxScaler()


# In[15]:


scaler.fit(x_train)


# In[16]:


x_train = scaler.transform(x_train)


# In[17]:


x_test = scaler.transform(x_test)


# In[18]:


x_train


# In[19]:


x_test


# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation


# In[21]:


model = Sequential()


# In[22]:


model.add(Dense(4,activation = 'relu'))


# In[23]:


model.add(Dense(4,activation = 'relu'))


# In[24]:


model.add(Dense(4,activation = 'relu'))


# In[25]:


model.add(Dense(1))


# In[27]:


model.compile(optimizer = 'rmsprop', loss = 'mse')


# In[28]:


model.fit(x = x_train, y = y_train, epochs = 250)


# In[29]:


model.history.history


# In[30]:


df = pd.DataFrame(model.history.history)


# In[31]:


df.plot()


# In[32]:


test_predictions = model.predict(x_test)


# In[33]:


test_predictions


# In[34]:


y_test


# In[37]:


plt.scatter(y_test,test_predictions)


# In[50]:


df1 = pd.DataFrame(y_test)


# In[51]:


df1


# In[52]:


df1.columns = ['Real Values']


# In[53]:


df2 = pd.DataFrame(test_predictions)


# In[54]:


df2.columns = ['Model Predictions']


# In[56]:


df3 = pd.concat([df1,df2], axis = 1)


# In[57]:


df3


# In[58]:


df3['Error'] = df3['Real Values'] - df3['Model Predictions']


# In[59]:


df3


# In[62]:


sns.set_style('whitegrid')
sns.distplot(df3['Error'], bins = 30, kde = False)


# In[65]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[69]:


print(mean_absolute_error(df3['Real Values'], df3['Model Predictions']))


# In[71]:


print(mean_squared_error(df3['Real Values'], df3['Model Predictions']))


# In[72]:


print(np.sqrt(mean_squared_error(df3['Real Values'], df3['Model Predictions'])))


# In[73]:


new_gem = [[998,1000]]


# In[74]:


new_gem = scaler.transform(new_gem)


# In[75]:


new_g = model.predict(new_gem)


# In[76]:


new_g


# In[77]:


from tensorflow.keras.models import load_model
model.save('Prediction Model')


# In[78]:


new_model = load_model('Prediction Model')


# In[79]:


new_model.predict(new_gem)


# In[ ]:
