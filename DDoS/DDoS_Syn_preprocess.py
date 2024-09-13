#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


# In[3]:


df = pd.read_csv("Traning Datasets/Syn.csv").drop(columns=['Unnamed: 0',' Timestamp','SimillarHTTP','Flow ID',' Source IP',' Destination IP'])
df.columns = df.columns.str.strip()


# In[4]:


df.shape


# In[5]:


class_counts = df['Label'].value_counts()
print(class_counts)


# In[6]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()


# In[7]:


df.shape


# In[8]:


class_counts = df['Label'].value_counts()
print(class_counts)


# In[9]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder();
df['Label'] = label_encoder.fit_transform(df['Label'])


# In[10]:


# from sklearn.model_selection import KFold
# import category_encoders as ce

# columns_to_encode = ['Flow ID','Source IP','Destination IP']
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# batch_size = 100000
# encoding_map = {}


# In[11]:


# def process_in_batches(train_df, test_df):
#     # Iterate over the test data in chunks to reduce memory usage
#     for start in range(0, len(test_df), batch_size):
#         # Define end index for the current batch
#         end = min(start + batch_size, len(test_df))
        
#         # Extract the current batch
#         batch_test_df = test_df.iloc[start:end]
        
#         # Transform the batch and update the columns in place
#         df.loc[batch_test_df.index, ['Flow ID', 'Source IP', 'Destination IP']] = encoder.transform(batch_test_df[['Flow ID', 'Source IP', 'Destination IP']]).values


# In[12]:


# # K-Fold processing
# for fold, (train_index, test_index) in enumerate(kf.split(df)):
#     train_df = df.iloc[train_index]
#     test_df = df.iloc[test_index]
    
#     # Fit the encoder on the training data
    
#     encoder = ce.TargetEncoder(cols=columns_to_encode)
#     encoder.fit(train_df[['Flow ID', 'Source IP', 'Destination IP']], train_df['Label'])
    
#     # Save the encoding mapping for this fold
#     encoding_map[fold] = encoder
#     # Process the test data in batches and update the columns in place
#     process_in_batches(train_df, test_df)


# In[13]:


X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[14]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = 27)
X , y = smote.fit_resample(X,y)


# In[15]:


from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X = scaler.fit_transform(X)


# In[16]:


with open('scaler.pkl','wb') as f:
    pickle.dump(scaler,f)


# In[17]:


# df.to_csv("Processed Datasets/DDoS_Syn.csv" , index=False)


# In[18]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1)
model.fit(X, y)


# In[19]:


from joblib import dump

dump(model,"Models/Syn_model.joblib")


# In[20]:


class_counts = df['Label'].value_counts()
print(class_counts)


# In[ ]:




