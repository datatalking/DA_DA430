#!/usr/bin/env python
# coding: utf-8

# In[18]:

get_ipython().system("pip install 'rfpimp==1.3.7' as rfpimp")
get_ipython().system('pip install pandas')
get_ipython().system('pip install sklearn.ensemble')
get_ipython().system('pip install sklearn.model_selection')
get_ipython().system("pip install 'scikit-learn==0.22.2'")


from rfpimp import importances, plot_importances
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# df_orig = pd.read_csv("/Users/vanessawilson/sbox/Curriculum/DA_DA430/rent.csv")


# In[ ]:df_orig = pd.read_csv("/Users/vanessawilson/sbox/Curriculum/DA_DA430/bank-additional/bank-additional-full.csv")

import numpy as np

df = df_orig.copy()

df['conversion'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
df.head()
#%%

# attentuate affect of outliers in age .... not price
df['duration'] = np.log(df['duration'])


# In[ ]:

df

# In[ ]:


df_train, df_test = train_test_split(df_orig, test_size=0.20)
features = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
            'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
            'cons.conf.idx', 'euribor3m', 'nr.employed', 'conversion']
df_train = df_train[features]
df_test = df_test[features]

X_train, y_train = df_train.drop('conversion', axis=1), df_train['conversion']
X_test, y_test = df_test.drop('conversion', axis=1), df_test['conversion']
# Add column of random numbers
X_train['random'] = np.random.random(size=len(X_train))
X_test['random'] = np.random.random(size=len(X_test))

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, oob_score=True)
rf.fit(X_train, y_train)

imp = importances(rf, X_test, y_test, n_samples=-1)
viz = plot_importances(imp)
viz.view()
