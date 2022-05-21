#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
import sys

import pandas as pd
import numpy as np

import logging
import mlflow.sklearn
import mlflow
from urllib.parse import urlparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


sys.path.append(os.path.abspath(os.path.join('../scripts')))
from ml import Ml
from preprocess import Preprocess


# In[4]:


ml = Ml()
preprocess = Preprocess()


# In[5]:


df = pd.read_csv('../data/AdSmartABdata.csv')
df


# In[ ]:


# Get URL from DVC
# path = 'data/AdSmartABdata.csv'
# repo = 'https://github.com/jedisam/abtest-mlops'
# version = '6db449393c9626c4fbca44946dfa103660685a27'


# In[ ]:


# Load data from dvc using the dvc.api.Dataset class
# data_url = dvc.api.get_url(
#     path=path,
#     repo=repo,
#     rev=version
# )
# data_url


# In[ ]:


# Read CSV file from remote repository
# data = pd.read_csv(data_url, sep=',')
# data


# In[6]:


# change the date column to datetime
# from preprocess import Preprocess
data = preprocess.convert_to_datetime(df, 'date')
data


# # Exploring the categorical columns

# In[7]:


numerical_column = preprocess.get_numerical_columns(df)
categorical_column = preprocess.get_categorical_columns(df)


# In[8]:


# drop auction_id from categorical_column
categorical_column.remove('auction_id')


# In[9]:


# Get column names have less than 10 more than 2 unique values
to_one_hot_encoding = [col for col in categorical_column if df[col].nunique() <= 10 and df[col].nunique() > 2]

# Get Categorical Column names thoose are not in "to_one_hot_encoding"
to_label_encoding = [col for col in categorical_column if not col in to_one_hot_encoding]


# In[11]:


# Label encoding
label_encoded_columns = preprocess.label_encode(df, to_label_encoding)


# In[14]:


# Copy our DataFrame to X variable
X = df.copy()

# Droping Categorical Columns,
# "inplace" means replace our data with new one
# Don't forget to "axis=1"
X.drop(categorical_column, axis=1, inplace=True)

# Merge DataFrames
X = pd.concat([X, label_encoded_columns], axis=1)


# In[15]:


# Select only rows with responses
X = X.query('yes == 1 | no == 1')


# In[16]:


# Drop auction_id column
X.drop(["auction_id"], axis=1, inplace=True)


# In[17]:


X['target'] = [1] * X.shape[0]
X.loc[X['no'] == 1, 'target'] = 0
y = X['target']
X.drop(["target"], axis=1, inplace=True)
X.drop(['yes', 'no'], axis=1, inplace=True)


# In[18]:


# Get the day of the week from the date column as a new column
X['day'] = X['date'].dt.dayofweek
X.drop(["date"], axis=1, inplace=True)


# # Logistic Regression

# In[28]:


logistic_regression_model = LogisticRegression(random_state=0)
logistic_regression_result = ml.cross_validation(logistic_regression_model, X, y, 5)
logistic_regression_result


# Write scores to file
with open("train/metrics.txt", 'w') as outfile:
    outfile.write(
        f"Training data accuracy: {logistic_regression_result['Training Accuracy scores'][0]}")
    outfile.write(
        f"Validation data accuracy: {logistic_regression_result['Validation Accuracy scores'][0]}")


# Plot accuacy results to cml

# Plot Accuracy Result
model_name = "Decision Tree"
ml.plot_result(model_name, "Accuracy", "Accuracy scores in 5 Folds",
               logistic_regression_result["Training Accuracy scores"],
               logistic_regression_result["Validation Accuracy scores"],
               'train/logistic_regression_accuracy.png')

# Precision Results

# Plot Precision Result
ml.plot_result(model_name, "Precision", "Precision scores in 5 Folds",
               logistic_regression_result["Training Precision scores"],
               logistic_regression_result["Validation Precision scores"],
               'train/logistic_regression_preicision.png')

# Recall Results plot

# Plot Recall Result
ml.plot_result(model_name, "Recall", "Recall scores in 5 Folds",
               logistic_regression_result["Training Recall scores"],
               logistic_regression_result["Validation Recall scores"],
               'train/logistic_regression_recall.png')


# f1 Score Results

# Plot F1-Score Result
ml.plot_result(model_name, "F1", "F1 Scores in 5 Folds",
               logistic_regression_result["Training F1 scores"],
               logistic_regression_result["Validation F1 scores"],
               'train/logistic_regression_f1_score.png')




