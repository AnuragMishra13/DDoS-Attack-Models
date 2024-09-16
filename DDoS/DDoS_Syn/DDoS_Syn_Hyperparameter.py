#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from imblearn.over_sampling import SMOTE
from joblib import dump , parallel_backend
import gc
from sklearn.model_selection import GridSearchCV
import optuna
from sklearn.metrics import accuracy_score,f1_score
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split


# In[2]:


train_df = pd.read_csv("Syn.csv").drop(columns=['Unnamed: 0', ' Timestamp', 'SimillarHTTP', 'Flow ID', ' Source IP', ' Destination IP'])
train_df.columns = train_df.columns.str.strip()


# In[3]:


train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df = train_df.dropna()



# In[4]:


label_encoder = LabelEncoder()
train_df['Label'] = label_encoder.fit_transform(train_df['Label'])


# In[5]:


# Prepare features and labels
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values


# In[6]:


# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=27)
X, y = smote.fit_resample(X, y)


# In[7]:


# Scale features
scaler = MaxAbsScaler()
X = scaler.fit_transform(X)


# In[8]:


del train_df
gc.collect()


# In[9]:


with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)


# In[10]:


model = CatBoostClassifier(
    learning_rate=0.2025501646477492,
    depth=5,
    n_estimators=258,
    l2_leaf_reg=2.846674015362996,
    boosting_type="Ordered",
    task_type = 'GPU',
    gpu_ram_part=0.9
)


# In[11]:


model.fit(X,y)


# In[12]:


dump(model,"catboost_model.pkl")


# In[ ]:




