#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[4]:


df = pd.read_csv("Traning Datasets/DrDoS_DNS.csv").drop(columns=['Unnamed: 0',' Timestamp','SimillarHTTP'])
df.columns = df.columns.str.strip()


# In[5]:


df.head()


# In[6]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()


# In[7]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder();
df['Label'] = label_encoder.fit_transform(df['Label'])


# In[8]:


from sklearn.model_selection import KFold
import category_encoders as ce

kf = KFold(n_splits=5, shuffle=True, random_state=42)
encoder = ce.TargetEncoder(cols=['Flow ID','Source IP','Destination IP'])
batch_size = 100000


# In[9]:


def process_in_batches(train_df, test_df):
    # Iterate over the test data in chunks to reduce memory usage
    for start in range(0, len(test_df), batch_size):
        # Define end index for the current batch
        end = min(start + batch_size, len(test_df))
        
        # Extract the current batch
        batch_test_df = test_df.iloc[start:end]
        
        # Transform the batch and update the columns in place
        df.loc[batch_test_df.index, ['Flow ID', 'Source IP', 'Destination IP']] = encoder.transform(batch_test_df[['Flow ID', 'Source IP', 'Destination IP']]).values


# In[10]:


# K-Fold processing
for train_index, test_index in kf.split(df):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    
    # Fit the encoder on the training data
    encoder.fit(train_df[['Flow ID', 'Source IP', 'Destination IP']], train_df['Label'])
    
    # Process the test data in batches and update the columns in place
    process_in_batches(train_df, test_df)


# In[11]:


# for x in list(df.columns):
#     Q1 = df[x].quantile(0.25)
#     Q3 = df[x].quantile(0.75)
#     IQR = Q3 - Q1

#     # Filter out rows where values are not outliers
#     df= df[~((df[x] < (Q1 - 1.5 * IQR)) | (df[x] > (Q3 + 1.5 * IQR)))]


# In[12]:


df.to_csv("Processed Datasets/DrDoS_DNS.csv" , index=False)


# In[13]:


class_counts = df['Label'].value_counts()
print(class_counts)


# In[14]:


df.shape


# In[ ]:




