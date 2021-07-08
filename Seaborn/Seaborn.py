#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


tips = sns.load_dataset('tips')


# In[37]:


tips.head(3)


# In[4]:


sns.distplot(tips['total_bill'])


# In[5]:


sns.distplot(tips['total_bill'], kde = False)


# In[6]:


sns.distplot(tips['total_bill'],kde = False, bins = 30)


# In[12]:


sns.jointplot(x = 'total_bill', y = 'tip', data = tips, kind = 'kde')


# In[13]:


sns.pairplot(tips)


# In[18]:


sns.pairplot(tips,hue = 'sex',palette = 'coolwarm')


# In[29]:


sns.barplot(x = 'total_bill', y = 'tip', data = tips,estimator = np.std)


# In[22]:


import numpy as np


# In[28]:


sns.barplot(x = 'total_bill', y = 'tip', data = tips, estimator = np.std)


# In[36]:


sns.boxplot(x = 'day', y = 'total_bill', data = tips,palette = 'rainbow')


# In[38]:


sns.boxplot(x="", y="total_bill", data=tips,palette='rainbow')


# In[40]:


flights = sns.load_dataset('flights')


# In[41]:


flights.head()


# In[44]:


flights.pivot_table(values = 'passengers', index = 'month', columns = 'year')


# In[45]:


pvflights = flights.pivot_table(values = 'passengers',index = 'month', columns = 'year')
sns.heatmap(pvflights)


# In[48]:


sns.clustermap(pvflights, linecolor = 'white', linewidth = 2)


# In[ ]:
