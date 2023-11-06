#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, fbeta_score


# In[2]:


# file path
csv_file_path = "loan_default.csv"

# load file
df = pd.read_csv(csv_file_path, dtype={'MyColumn': 'str'})


# In[12]:


df['term'].value_counts()


# In[3]:


def preprocess(df):
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
    
    def fix_year(year):
        year = int(year)
        if year > 20:
            return year + 1900
        else:
            return year + 2000

    # Convert the string values to datetime objects
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'].apply(lambda x: x[:3] + '-'+ str(fix_year(x[4:]))), format='%b-%Y')

    # Specify the end date as December 2011
    end_date = pd.to_datetime('Dec-2011', format='%b-%Y')

    # Calculate the number of months between the dates and the end date
    df['earliest_cr_line'] = (end_date.year - df['earliest_cr_line'].dt.year) * 12 + (end_date.month - df['earliest_cr_line'].dt.month)

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
    df1 = pd.concat([df, encoded_df], axis=1)
    data = pd.concat([df, encoded_df], axis=1).to_numpy()
    
    return data, labels, df1


# In[4]:


data, labels, df1 = preprocess(df)


# In[5]:


# split data into train and test 
trainx_b, testx_b, trainy, testy = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[6]:


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
corr = compute_correlation(trainx_b, 'pearson')
plt.figure(figsize=(8, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[6]:


columns_to_remove = [0, 2, 6, 10, 12, 19, 21, 26, 28]

# Remove the specified columns
trainx_d = np.delete(trainx_b, columns_to_remove, axis=1)
testx_d = np.delete(testx_b, columns_to_remove, axis=1)


# In[7]:


# rank correlation
# number of features
n_features = trainx_d.shape[1]

# create correlation vector
corr = np.zeros(n_features)

for i in range(n_features):
    corr[i], _ = pearsonr(trainx_d[:, i], trainy)

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
selected_columns = rank_correlation(trainx_d, trainy)[:k]
trainx_d = trainx_d[:, selected_columns]
testx_d = testx_d[:, selected_columns]


# In[9]:


def run_pca(train_x, test_x):
    # normaliztion
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    # Transform the testing data based on the fitted scaler
    test_x = scaler.transform(test_x)
    # PCA
    pca = PCA(n_components = 0.95)
    pca.fit(train_x)
    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)
    # number of PCs
    nPC = train_x.shape[1]
    # PCs
    PC = pca.components_
    return (nPC, PC, train_x, test_x)

(nPC, PC, trainx_h, testx_h) = run_pca(trainx_b, testx_b)


# In[11]:


print('Number of PCs to capture 95% of the variance in data: ', nPC)
print('First 3 PCs are: ')
print('PC1: ', PC[1, :])
idx = np.argsort(-np.abs(PC[1, :]))
print('important indices are: ', idx[:5])
print(df1.columns[idx[:5]])
print('PC1: ', PC[2, :])
idx = np.argsort(-np.abs(PC[2, :]))
print('important indices are: ', idx[:5])
print(df1.columns[idx[:5]])
print('PC1: ', PC[3, :])
idx = np.argsort(-np.abs(PC[3, :]))
print('important indices are: ', idx[:5])
print(df1.columns[idx[:5]])


# In[10]:


def run_nmf(train_x, test_x, k):
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    nmf = NMF(n_components=k, init='random')
    W_train = nmf.fit_transform(train_x)
    H = nmf.components_
    W_test = nmf.transform(test_x)
    train_x_reconstruct = np.dot(W_train, H)
    error = np.linalg.norm(train_x - train_x_reconstruct)
    return (error, H, W_train, W_test)


# In[13]:


errors = []
for k in range(1, trainx_b.shape[1] + 1):
    (err, _, _, _) = run_nmf(trainx_b, testx_b, k)
    errors.append(err)
k = np.arange(1, trainx_b.shape[1] + 1)
plt.plot(k, errors)
plt.xlabel('k')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs. k for NMF')
plt.show()


# In[11]:


(error, H, trainx_k, testx_k) = run_nmf(trainx_b, testx_b, 30)


# In[16]:


print('First 3 factors are: ')
print('F1: ', H[1, :])
idx = np.argsort(-np.abs(H[1, :]))
print('important indices are: ', idx[:5])
print(df1.columns[idx[:5]])

print('F2: ', H[2, :])
idx = np.argsort(-np.abs(H[2, :]))
print('important indices are: ', idx[:5])
print(df1.columns[idx[:5]])

print('F3: ', H[3, :])
idx = np.argsort(-np.abs(H[3, :]))
print('important indices are: ', idx[:5])
print(df1.columns[idx[:5]])


# In[12]:


def build_logr(train_x, train_y, test_x, test_y):
    # split train into train and validation 
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
    # create lr model
    model = LogisticRegression()
    # train model
    model.fit(train_x, train_y)
    # predict
    train_y_pred = model.predict(train_x)
    val_y_pred = model.predict(val_x)
    test_y_pred = model.predict(test_x)
    # AUC
    train_auc = roc_auc_score(train_y, train_y_pred)
    val_auc = roc_auc_score(val_y, val_y_pred)
    test_auc = roc_auc_score(test_y, test_y_pred)
    # F1 score
    train_f1 = f1_score(train_y, train_y_pred)
    val_f1 = f1_score(val_y, val_y_pred)
    test_f1 = f1_score(test_y, test_y_pred)
    # F2 score
    train_f2 = fbeta_score(train_y, train_y_pred, beta=2)
    val_f2 = fbeta_score(val_y, val_y_pred, beta=2)
    test_f2 = fbeta_score(test_y, test_y_pred, beta=2)
    return {'train-auc': train_auc, 'train-f1': train_f1, 'train-f2': train_f2, 
            'val-auc': val_auc, 'val-f1': val_f1, 'val-f2': val_f2, 
            'test-auc': test_auc, 'test-f1': test_f1, 'test-f2': test_f2, 
            'params': {}}


# In[18]:


print('Dataset b: ')
print(build_logr(trainx_b, trainy, testx_b, testy))


# In[19]:


print('Dataset d: ')
print(build_logr(trainx_d, trainy, testx_d, testy))


# In[20]:


print('Dataset h: ')
print(build_logr(trainx_h, trainy, testx_h, testy))


# In[21]:


print('Dataset k: ')
print(build_logr(trainx_k, trainy, testx_k, testy))


# In[13]:


def build_dt(train_x, train_y, test_x, test_y): 
    # Create a Decision Tree classifier
    model = DecisionTreeClassifier()

    # Define the hyperparameter grid to search
    n = 15
    param_grid = {
        'max_depth': list(np.arange(2, n+1)),  
        'min_samples_leaf': list(np.arange(2, n+1))  
    }
    
    # Create a custom scoring function
    scoring = {'auc': 'roc_auc', 'f1': 'f1', 'f2': make_scorer(fbeta_score, beta=2)}
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='auc', verbose=2,
                               return_train_score=True)
    
    # Perform grid search with cross-validation
    grid_search.fit(train_x, train_y)
    
    # predict on test data
    test_y_pred = grid_search.predict(test_x)
    
    # index pf best classifier in grid search
    idx = grid_search.best_index_
    
    # Get the best hyperparameters
    best_depth = grid_search.best_params_['max_depth']
    best_leaf_samples = grid_search.best_params_['min_samples_leaf']
    
    # AUC
    results = grid_search.cv_results_
    train_auc = results['mean_train_auc'][idx] 
    val_auc = results['mean_test_auc'][idx] 
    test_auc = roc_auc_score(test_y, test_y_pred) 
    # F1
    train_f1 = results['mean_train_f1'][idx] 
    val_f1 = results['mean_test_f1'][idx] 
    test_f1 = f1_score(test_y, test_y_pred) 
    # F2
    train_f2 = results['mean_train_f2'][idx] 
    val_f2 = results['mean_test_f2'][idx] 
    test_f2 = fbeta_score(test_y, test_y_pred, beta=2)
    
    return {'train-auc': train_auc, 'train-f1': train_f1, 'train-f2': train_f2, 
            'val-auc': val_auc, 'val-f1': val_f1, 'val-f2': val_f2, 
            'test-auc': test_auc, 'test-f1': test_f1, 'test-f2': test_f2, 
            'params': param_grid, 'best-depth':best_depth, 'best-leaf-samples':best_leaf_samples}


# In[24]:


build_dt(trainx_b, trainy, testx_b, testy)


# In[25]:


build_dt(trainx_d, trainy, testx_d, testy)


# In[26]:


build_dt(trainx_h, trainy, testx_h, testy)


# In[27]:


build_dt(trainx_k, trainy, testx_k, testy)


# In[14]:


def build_rf(train_x, train_y, test_x, test_y): 
    # Create a Decision Tree classifier
    model = RandomForestClassifier()

    # Define the hyperparameter grid to search
    n = 20
    param_grid = {
        'n_estimators': [20, 50, 100],
        'max_depth': list(np.arange(2, n+1)),  
        'min_samples_leaf': list(np.arange(2, n+1))  
    }
    
    # Create a custom scoring function
    scoring = {'auc': 'roc_auc', 'f1': 'f1', 'f2': make_scorer(fbeta_score, beta=2)}
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='auc', verbose=2,
                               return_train_score=True)
    
    # Perform grid search with cross-validation
    grid_search.fit(train_x, train_y)
    
    # predict on test data
    test_y_pred = grid_search.predict(test_x)
    
    # index pf best classifier in grid search
    idx = grid_search.best_index_
    
    # Get the best hyperparameters
    best_depth = grid_search.best_params_['max_depth']
    best_leaf_samples = grid_search.best_params_['min_samples_leaf']
    best_n_estimator = grid_search.best_params_['n_estimators']
    
    # AUC
    results = grid_search.cv_results_
    train_auc = results['mean_train_auc'][idx] 
    val_auc = results['mean_test_auc'][idx] 
    test_auc = roc_auc_score(test_y, test_y_pred) 
    # F1
    train_f1 = results['mean_train_f1'][idx] 
    val_f1 = results['mean_test_f1'][idx] 
    test_f1 = f1_score(test_y, test_y_pred) 
    # F2
    train_f2 = results['mean_train_f2'][idx] 
    val_f2 = results['mean_test_f2'][idx] 
    test_f2 = fbeta_score(test_y, test_y_pred, beta=2)
    
    return {'train-auc': train_auc, 'train-f1': train_f1, 'train-f2': train_f2, 
            'val-auc': val_auc, 'val-f1': val_f1, 'val-f2': val_f2, 
            'test-auc': test_auc, 'test-f1': test_f1, 'test-f2': test_f2, 
            'params': param_grid, 'best-depth':best_depth, 'best-leaf-samples':best_leaf_samples, 
            'best-n-estimator': best_n_estimator}


# In[29]:


build_rf(trainx_b, trainy, testx_b, testy)


# In[30]:


build_rf(trainx_d, trainy, testx_d, testy)


# In[31]:


build_rf(trainx_h, trainy, testx_h, testy)


# In[32]:


build_rf(trainx_k, trainy, testx_k, testy)


# In[15]:


# normaliztion
scaler = StandardScaler()
trainx_b_n = scaler.fit_transform(trainx_b)
# Transform the testing data based on the fitted scaler
testx_b_n = scaler.transform(testx_b)

# normaliztion
scaler = StandardScaler()
trainx_d_n = scaler.fit_transform(trainx_d)
# Transform the testing data based on the fitted scaler
testx_d_n = scaler.transform(testx_d)


# In[16]:


def build_svm(train_x, train_y, test_x, test_y): 
    # Create a Decision Tree classifier
    model = SVC()

    # Define the hyperparameter grid to search
    param_grid = {'C': [0.01, 0.1, 1, 2],
                  'gamma': [0.01, 0.1, 1],
                  'kernel': ['linear', 'poly'],
                  'degree': [2, 3]}
    
    # Create a custom scoring function
    scoring = {'auc': 'roc_auc', 'f1': 'f1', 'f2': make_scorer(fbeta_score, beta=2)}
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='auc', verbose=2,
                               return_train_score=True)
    
    # Perform grid search with cross-validation
    grid_search.fit(train_x, train_y)
    
    # predict on test data
    test_y_pred = grid_search.predict(test_x)
    
    # index pf best classifier in grid search
    idx = grid_search.best_index_
    
    # Get the best hyperparameters
    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']
    best_kernel = grid_search.best_params_['kernel']
    best_degree = grid_search.best_params_['degree']
    
    # AUC
    results = grid_search.cv_results_
    train_auc = results['mean_train_auc'][idx] 
    val_auc = results['mean_test_auc'][idx] 
    test_auc = roc_auc_score(test_y, test_y_pred) 
    # F1
    train_f1 = results['mean_train_f1'][idx] 
    val_f1 = results['mean_test_f1'][idx] 
    test_f1 = f1_score(test_y, test_y_pred) 
    # F2
    train_f2 = results['mean_train_f2'][idx] 
    val_f2 = results['mean_test_f2'][idx] 
    test_f2 = fbeta_score(test_y, test_y_pred, beta=2)
    
    return {'train-auc': train_auc, 'train-f1': train_f1, 'train-f2': train_f2, 
            'val-auc': val_auc, 'val-f1': val_f1, 'val-f2': val_f2, 
            'test-auc': test_auc, 'test-f1': test_f1, 'test-f2': test_f2, 
            'params': param_grid, 'best-C':best_C, 'best-gamma':best_gamma,
            'best-kernel': best_kernel, 'best-degree': best_degree}


# In[24]:


build_svm(trainx_b_n, trainy, testx_b_n, testy)


# In[15]:


build_svm(trainx_d_n, trainy, testx_d_n, testy)


# In[26]:


build_svm(trainx_h, trainy, testx_h, testy)


# In[16]:


build_svm(trainx_k, trainy, testx_k, testy)


# In[17]:


def build_nn(train_x, train_y, test_x, test_y): 
    # Create a Decision Tree classifier
    model = MLPClassifier(max_iter=100, solver='adam', learning_rate='adaptive')

    # Define the hyperparameter grid to search
    param_grid = {
        'hidden_layer_sizes': [(10, 30, 10), (100, 50), (40, 20), (20, ), (40, )],  
        'activation': ['relu', 'logistic', 'tanh'], 
        'alpha': [0.001, 0.01, 0.1, 1, 10], 
    }
    
    # Create a custom scoring function
    scoring = {'auc': 'roc_auc', 'f1': 'f1', 'f2': make_scorer(fbeta_score, beta=2)}
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='auc', verbose=2,
                               return_train_score=True)
    
    # Perform grid search with cross-validation
    grid_search.fit(train_x, train_y)
    
    # predict on test data
    test_y_pred = grid_search.predict(test_x)
    
    # index pf best classifier in grid search
    idx = grid_search.best_index_
    
    # Get the best hyperparameters
    best_hidden = grid_search.best_params_['hidden_layer_sizes']
    best_activation = grid_search.best_params_['activation']
    best_alpha = grid_search.best_params_['alpha']
    
    # AUC
    results = grid_search.cv_results_
    train_auc = results['mean_train_auc'][idx] 
    val_auc = results['mean_test_auc'][idx] 
    test_auc = roc_auc_score(test_y, test_y_pred) 
    # F1
    train_f1 = results['mean_train_f1'][idx] 
    val_f1 = results['mean_test_f1'][idx] 
    test_f1 = f1_score(test_y, test_y_pred) 
    # F2
    train_f2 = results['mean_train_f2'][idx] 
    val_f2 = results['mean_test_f2'][idx] 
    test_f2 = fbeta_score(test_y, test_y_pred, beta=2)
    
    return {'train-auc': train_auc, 'train-f1': train_f1, 'train-f2': train_f2, 
            'val-auc': val_auc, 'val-f1': val_f1, 'val-f2': val_f2, 
            'test-auc': test_auc, 'test-f1': test_f1, 'test-f2': test_f2, 
            'params': param_grid, 'best-hidden': best_hidden,
            'best-activation': best_activation, 'best-alpha': best_alpha}


# In[20]:


build_nn(trainx_b_n, trainy, testx_b_n, testy)


# In[21]:


build_nn(trainx_d_n, trainy, testx_d_n, testy)


# In[18]:


build_nn(trainx_h, trainy, testx_h, testy)


# In[19]:


build_nn(trainx_k, trainy, testx_k, testy)


# In[28]:


# model = LogisticRegression()
# model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2)
# model = RandomForestClassifier(max_depth=7, min_samples_leaf=8, n_estimators=20)
# model = SVC(C=0.01, gamma=0.01, kernel='linear')
model = MLPClassifier(max_iter=100, solver='adam', learning_rate='adaptive',
                      activation='logistic', hidden_layer_sizes=(100, 50), alpha=0.01)


# In[29]:


start = time.time()
model.fit(trainx_d_n, trainy)
model.predict(testx_d_n)
print(time.time() - start)

