#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, fbeta_score


# In[2]:


# file path
csv_file_path = "loan_default.csv"

# load file
df = pd.read_csv(csv_file_path, dtype={'MyColumn': 'str'})


# In[3]:


# preprocessing
# remove months from term column and convert numbers 36 and 60 to integers
df['term'] = df['term'].str.extract('(\d+)').astype(int)

# encode grade column
grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

# apply the mapping
df['grade'] = df['grade'].map(grade_mapping)

# encode emp_length column
emp_length_mapping = {'< 1 year': 1, '1 year': 2, '2 years': 3, '3 years': 4, '4 years': 5, '5 years': 6,
                      '6 years': 7, '7 years': 8, '8 years': 9, '9 years': 10, '10+ years': 11}
# apply the mapping
df['emp_length'] = df['emp_length'].map(emp_length_mapping)

# fill NaN values of emp_length column
df['emp_length'].fillna(0, inplace=True)

# count number of months for each earliest credit line entry up to Dec 2011
def count_months(start, end):
    start_date = datetime.strptime(start, '%b-%y')
    end_date = datetime.strptime(end, '%b-%y')
    months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
    return months

# Calculate the number of months for each entry
df['earliest_cr_line'] = df.apply(lambda row: count_months(row['earliest_cr_line'], 'Dec-11'), axis=1)

# Use one-hot encoding for home_ownership, verification_status, purpose columns
# supress warnings
import warnings
warnings.filterwarnings('ignore')

# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Fit the encoder on the categorical column and transform it
encoded_data = encoder.fit_transform(df[['home_ownership', 'verification_status', 'purpose']])

# Create a new DataFrame with one-hot encoded columns
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['home_ownership', 'verification_status', 'purpose']))

# remove irrelevant features
df = df.drop(columns=['id'])

# remove redundant categorical features
df = df.drop(columns=['home_ownership', 'verification_status', 'purpose'])

# extract lables from dataset and remove 'class' column
labels = df['class'].values
df = df.drop('class', axis=1)

# concat encoded dataframe with original dataframe
data = pd.concat([df, encoded_df], axis=1).to_numpy()


# In[4]:


# split data into train and test 
trainx, testx, trainy, testy = train_test_split(data, labels, test_size=0.2, random_state=42)

# normaliztion
scaler = StandardScaler()
trainx = scaler.fit_transform(trainx)
# Transform the testing data based on the fitted scaler
testx = scaler.transform(testx)


# In[5]:


def compute_correlation(x, corrtype):
    # number of features
    n_features = x.shape[1]
    
    # create correlation matrix
    corr = np.zeros((n_features, n_features))
    
    if corrtype == 'pearson':
        for i in range(n_features):
            for j in range(n_features):
                corr[i, j], _ = pearsonr(x[:, i], x[:, j])
    elif corrtype == 'kendall':
        for i in range(n_features):
            for j in range(n_features):
                corr[i, j], _ = kendalltau(x[:, i], x[:, j])
    elif corrtype == 'spearman':
        for i in range(n_features):
            for j in range(n_features):
                corr[i, j], _ = spearmanr(x[:, i], x[:, j])
                
    return corr

# Pearson
corr = compute_correlation(trainx, 'pearson')
plt.figure(figsize=(8, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[6]:


columns_to_remove = [0, 2, 6, 10, 12, 19, 21, 26, 28]

# Remove the specified columns
trainx = np.delete(trainx, columns_to_remove, axis=1)
testx = np.delete(testx, columns_to_remove, axis=1)


# In[7]:


# rank correlation
# number of features
n_features = trainx.shape[1]

# create correlation vector
corr = np.zeros(n_features)

for i in range(n_features):
    corr[i], _ = pearsonr(trainx[:, i], trainy)

# calculate absolute values
# corr = abs(corr)
print(corr)

# We take k to be 25!


# In[8]:


def rank_correlation(x, y):
    # number of features
    n_features = x.shape[1]
    
    # create correlation vector
    corr = np.zeros(n_features)
    
    for i in range(n_features):
        corr[i], _ = pearsonr(x[:, i], y)
        
    # calculate absolute values
    corr = abs(corr)
    
    # find indices in descending order
    sorted_indices = np.argsort(corr)[::-1]
    
    return sorted_indices

# select features form rank correlation function
k = 25
selected_columns = rank_correlation(trainx, trainy)[:k]
trainx = trainx[:, selected_columns]
testx = testx[:, selected_columns]


# In[9]:


# use PCA to whiten data
# Initialize PCA
pca = PCA(whiten=True)

# Fit and transform the train data
trainx = pca.fit_transform(trainx)

# Transform the test data
testx = pca.transform(testx)


# In[10]:


def tune_nn(x, y, hiddenparams, actparams, alphaparams):
    # solvers = 'lbfgs', 'sgd', 'adam
    # lr = 'constant', 'adaptive'
    # warm_start=True
    # create MLP model
    model = MLPClassifier(max_iter=100, solver='adam', learning_rate='adaptive')
    
    # Define the hyperparameter grid to search
    param_grid = {
        'hidden_layer_sizes': hiddenparams,  
        'activation': actparams, 
        'alpha': alphaparams, 
    }
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', verbose=2)
    
    # Perform grid search with cross-validation
    grid_search.fit(x, y)
    
    # Get the best hyperparameters
    best_hidden = grid_search.best_params_['hidden_layer_sizes']
    best_activation = grid_search.best_params_['activation']
    best_alpha = grid_search.best_params_['alpha']
    best_score = grid_search.best_score_
    
    # Access the cross-validation results
    cv_results = grid_search.cv_results_
    
    return {'best-hidden': best_hidden, 'best-activation': best_activation, 'best-alpha': best_alpha, 
            'best-valid-AUC': best_score, 'grid_search': grid_search}


# In[11]:


hidden_size = [(10, 30, 10), (100, 50), (40, 20), (20, ), (40, )]
activation = ['relu', 'logistic', 'tanh']
alpha = [0.001, 0.01, 0.1, 1, 10]
result = tune_nn(trainx, trainy, hidden_size, activation, alpha)


# In[12]:


result


# In[13]:


import time
start_time = time.time()
# train MLP using best parameters
# Create a Decision Tree classifier
model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='logistic', alpha=0.01, 
                      max_iter=100, solver='adam', learning_rate='adaptive')

# train model
model.fit(trainx, trainy)

# make prediction on test data
testy_pred = model.predict(testx)
trainy_pred = model.predict(trainx)

# Calculate AUC, F1-score, F2-score
test_auc_score = roc_auc_score(testy, testy_pred) #AUC
train_auc_score = roc_auc_score(trainy, trainy_pred) #AUC
test_f1 = f1_score(testy, testy_pred) # F1-score
train_f1 = f1_score(trainy, trainy_pred) # F1-score
test_f2 = fbeta_score(testy, testy_pred, beta=2) #F2-score
train_f2 = fbeta_score(trainy,trainy_pred, beta=2) #F2-score
duration = time.time() - start_time

print('test AUC: ', test_auc_score)
print('test F1-score: ', test_f1)
print('test F2-score: ', test_f2)
print('-----------------------------')
print('train AUC: ', train_auc_score)
print('train F1-score: ', train_f1)
print('train F2-score: ', train_f2)
print('runtime: ', duration)

