#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

df=pd.read_csv('~sbox/Curriculums/DA430/Data/bank-additional-full.csv',sep=';')


# In[4]:


print(df.head())


# In[6]:


df['conversion']=df['y'].apply(lambda x:1 if x=='yes' else 0)

df.head()


# In[7]:


# total number of conversions
df.conversion.sum()


# In[8]:


# total number of clients in the data (= number of rows in the data)
df.shape[0]


# In[10]:


print('total conversion rate is %f: %i out of %i' %(df.conversion.sum()/df.shape[0],df.conversion.sum(),df.shape[0]))


# In[12]:


conversions_by_age = df.groupby(by='age')['conversion'].sum() / df.groupby(by='age')['conversion'].count() * 100.0
conversions_by_age


# In[14]:


from matplotlib import pyplot as plt

ax = conversions_by_age.plot(grid=True,figsize=(10, 7),title='Conversion Rates by Age')

ax.set_xlabel('age')
ax.set_ylabel('conversion rate (%)')
plt.show()


# In[15]:


df['age_group'] = df['age'].apply(lambda x: '[18, 30)' if x < 30 else '[30, 40)' if x < 40                                   else '[40, 50)' if x < 50 else '[50, 60)' if x < 60                                   else '[60, 70)' if x < 70 else '70+')
df['age_group']


# In[16]:


conversions_by_age_group = df.groupby(by='age_group')['conversion'].sum() / df.groupby(by='age_group')['conversion'].count() * 100.0
conversions_by_age_group


# In[17]:


ax = conversions_by_age_group.loc[['[18, 30)', '[30, 40)', '[40, 50)', '[50, 60)', '[60, 70)', '70+']].plot(
       kind='bar',
       color='skyblue',
       grid=True,
       figsize=(10, 7),
       title='Conversion Rates by Age Groups'
)
ax.set_xlabel('age')
ax.set_ylabel('conversion rate (%)')
plt.show()
 


# In[18]:


age_marital_df = df.groupby(['age_group', 'marital'])['conversion'].sum().unstack('marital').fillna(0)
age_marital_df = age_marital_df.divide(df.groupby(by='age_group')['conversion'].count(),axis=0)
age_marital_df


# In[19]:


ax = age_marital_df.loc[['[18, 30)', '[30, 40)', '[40, 50)', '[50, 60)', '[60, 70)', '70+']].plot(kind='bar',grid=True,figsize=(10,7))
ax.set_title('Conversion rates by Age & Marital Status')
ax.set_xlabel('age group')
ax.set_ylabel('conversion rate (%)')
plt.show()


# In[20]:


ax = age_marital_df.loc[['[18, 30)', '[30, 40)', '[40, 50)', '[50, 60)', '[60, 70)', '70+']].plot(kind='bar',stacked=True,grid=True,figsize=(10,7))
ax.set_title('Conversion rates by Age & Marital Status')
ax.set_xlabel('age group')
ax.set_ylabel('conversion rate (%)')
plt.show()


# In[ ]:




