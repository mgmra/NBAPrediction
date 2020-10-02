#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from glob import glob


# In[56]:


files = glob('Data\*.csv')


# In[60]:


files


# In[124]:



# In[120]
#test_df2 = test_df.set_index([test_df.columns[4], test_df.columns[3]])
test_df.columns[4]


# In[107]:





# In[89]:


dfs = {}

for file in [files[i] for i in [0,4]]:
    dfs[file] = pd.read_csv(file).drop('Season', axis='columns').rename(columns={'Unnamed: 0': 'season', 'Unnamed: 1':'Number'}).set_index(['Game Date', 'Match Up'], inplace=True)
for file in files[1:3]:
    dfs[file] = pd.read_csv(file).rename(columns={'Unnamed: 0':'Number'}).set_index(['Game Date', 'Match Up'], inplace = True)


# In[70]:


df_merged = pd.DataFrame()

for i in range(len(dfs.values())-1):
    df_merged = list(dfs.values())[i].join(other = list(dfs.values())[i+1], on=['Match Up', 'Game Date'], how='inner')


# In[74]:


dfs


# In[17]:


test_adv = pd.read_csv('advanced_scores_96.csv').drop('Season', axis='columns').rename(columns={'Unnamed: 0': 'Season', 'Unnamed: 1':'Number'}).set_index('Number')


# In[18]:


test_adv.head()


# In[19]:


test_adv.head()


# In[ ]:




