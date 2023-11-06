#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import mutual_info_classif


# In[2]:


# file path
csv_file_path = "loan_default.csv"

# load file
df = pd.read_csv(csv_file_path, dtype={'MyColumn': 'str'})


# In[3]:


df.select_dtypes(include=['object'])[:10]


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
df = df.drop(columns=['id', 'earliest_cr_line'])

# remove redundant categorical features
df = df.drop(columns=['home_ownership', 'verification_status', 'purpose'])

# extract lables from dataset and remove 'class' column
labels = df['class'].values
df = df.drop('class', axis=1)

# concat encoded dataframe with original dataframe
data = pd.concat([df, encoded_df], axis=1).to_numpy()


# In[4]:


# Split the data into training and testing sets
trainx, testx, trainy, testy = train_test_split(data, labels, test_size=0.2, random_state=42)


# In[4]:


# # extract information from earliest_cr_line column. we will calculate number of months up to Dec 2018. 
# # Every applicant who has more credit history is more likely to pay back for the loans. 

# # Function to parse dates and calculate months
# def calculate_months_between_dates(date_str):
#     # Standardize the date format to "DD-MMM-YY" for consistent parsing
#     if date_str[0].isalpha():
#         # Add a '1-' prefix for dates in "MMM-YY" format
#         date_str = '1-' + date_str
#     else:
#         # add a '-18' suffix for dates in "DD-MMM" format
#         date_str = date_str + '-18'

#     # Parse the date
#     date = datetime.strptime(date_str, '%d-%b-%y')

#     # Calculate the number of months between the date and December 2018
#     end_date = datetime(2018, 12, 31)
#     months = (end_date.year - date.year) * 12 + end_date.month - date.month

#     return int(months)

# df['earliest_cr_line'] = df['earliest_cr_line'].apply(calculate_months_between_dates)
# df


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
                
    return corrdef compute_correlation(x, corrtype):


# In[6]:


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


# In[7]:


def rank_mutual(x, y):
    # Calculate mutual information between features and the target variable y
    mutual_info = mutual_info_classif(x, y)

    # Get the indices that would sort mutual_info in descending order
    sorted_indices = np.argsort(mutual_info)[::-1]

    return sorted_indices


# In[25]:


# Pearson
corr = compute_correlation(trainx, 'pearson')
plt.figure(figsize=(8, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[28]:


plt.figure(figsize=(16, 16))
sns.heatmap(corr[:30, :30], annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[33]:


df


# In[9]:


# Spearman
corr = compute_correlation(trainx, 'spearman')
plt.figure(figsize=(8, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[10]:


plt.figure(figsize=(16, 16))
sns.heatmap(corr[:30, :30], annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[31]:


# Spearman
corr = compute_correlation(trainx, 'kendall')
plt.figure(figsize=(8, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[32]:


plt.figure(figsize=(16, 16))
sns.heatmap(corr[:30, :30], annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[34]:


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


# In[35]:


# Calculate mutual information between features and the target variable y
mutual_info = mutual_info_classif(trainx, trainy)
print(mutual_info)


# In[11]:


def tune_dt(x, y, dparams, lsparams):
    # Create a Decision Tree classifier
    model = DecisionTreeClassifier()

    # Define the hyperparameter grid to search
    param_grid = {
        'max_depth': dparams,  
        'min_samples_leaf': lsparams  
    }
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', verbose=2)
    
    # Perform grid search with cross-validation
    grid_search.fit(x, y)
    
    # Get the best hyperparameters
    best_depth = grid_search.best_params_['max_depth']
    best_leaf_samples = grid_search.best_params_['min_samples_leaf']
    best_score = grid_search.best_score_
    
    # Access the cross-validation results
    cv_results = grid_search.cv_results_
    
    return {'best-depth':best_depth, 'best-leaf-samples':best_leaf_samples, 'best-valid-AUC':best_score, 
           'cv-results':cv_results}


# In[48]:


# determine max number of leaves and depth of the tree
n = 20
dparams = list(np.arange(2, n+1))
lsparams = list(np.arange(2, n+1))

result = tune_dt(trainx, trainy, dparams, lsparams)


# In[51]:


get_ipython().run_line_magic('matplotlib', 'notebook')

# Create a 3D plot for scores and parameters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Set integer ticks on the x, y, and z axes
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
X, Y = np.meshgrid(dparams, lsparams)
mean_test_scores = np.array(result['cv-results']['mean_test_score']).reshape(len(dparams), -1)

ax.plot_surface(X, Y, mean_test_scores, cmap='viridis')
ax.set_xlabel('Max Depth')
ax.set_ylabel('Max Leaf Nodes')
ax.set_zlabel('Mean AUC Score Validation')

plt.show()


# In[61]:


# train decision tree using best parameters
max_depth = result['best-depth']
min_samples_leaf = result['best-leaf-samples']

# Create a Decision Tree classifier
model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

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

print('Best Max Depth: ', max_depth)
print('Best Min Samples of Leaf: ', min_samples_leaf)
print('-----------------------------')
print('test AUC: ', test_auc_score)
print('test F1-score: ', test_f1)
print('test F2-score: ', test_f2)
print('-----------------------------')
print('train AUC: ', train_auc_score)
print('train F1-score: ', train_f1)
print('train F2-score: ', train_f2)


# In[62]:


# Create a plot of the Decision Tree
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
plot_tree(model, filled=True)  # Replace with your feature names
plt.show()


# In[68]:


# remove features that are highly correlated with each other
# Indices of the columns to be removed
columns_to_remove = [0, 2, 6, 10, 12, 19, 21, 26, 28]

# Remove the specified columns
trainx_corr = np.delete(trainx, columns_to_remove, axis=1)
testx_corr = np.delete(testx, columns_to_remove, axis=1)


# In[64]:


# determine max number of leaves and depth of the tree
n = 20
dparams = list(np.arange(2, n+1))
lsparams = list(np.arange(2, n+1))

result = tune_dt(trainx_corr, trainy, dparams, lsparams)


# In[73]:


# train decision tree using best parameters
max_depth = result['best-depth']
min_samples_leaf = result['best-leaf-samples']

# Create a Decision Tree classifier
model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

# train model
model.fit(trainx_corr, trainy)

# make prediction on test data
testy_pred = model.predict(testx_corr)
trainy_pred = model.predict(trainx_corr)

# Calculate AUC, F1-score, F2-score
test_auc_score = roc_auc_score(testy, testy_pred) #AUC
train_auc_score = roc_auc_score(trainy, trainy_pred) #AUC
test_f1 = f1_score(testy, testy_pred) # F1-score
train_f1 = f1_score(trainy, trainy_pred) # F1-score
test_f2 = fbeta_score(testy, testy_pred, beta=2) #F2-score
train_f2 = fbeta_score(trainy,trainy_pred, beta=2) #F2-score

print('Best Max Depth: ', max_depth)
print('Best Min Samples of Leaf: ', min_samples_leaf)
print('-----------------------------')
print('test AUC: ', test_auc_score)
print('test F1-score: ', test_f1)
print('test F2-score: ', test_f2)
print('-----------------------------')
print('train AUC: ', train_auc_score)
print('train F1-score: ', train_f1)
print('train F2-score: ', train_f2)


# In[18]:


# select features form rank correlation function
k = 15
selected_columns = rank_correlation(trainx, trainy)[:k]
trainx_rank_corr = trainx[:, selected_columns]
testx_rank_corr = testx[:, selected_columns]


# In[20]:


# determine max number of leaves and depth of the tree
n = 20
dparams = list(np.arange(2, n+1))
lsparams = list(np.arange(2, n+1))

result = tune_dt(trainx_rank_corr, trainy, dparams, lsparams)


# In[30]:


# train decision tree using best parameters
max_depth = result['best-depth']
min_samples_leaf = result['best-leaf-samples']

# Create a Decision Tree classifier
model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

# train model
model.fit(trainx_rank_corr, trainy)

# make prediction on test data
testy_pred = model.predict(testx_rank_corr)
trainy_pred = model.predict(trainx_rank_corr)

# Calculate AUC, F1-score, F2-score
test_auc_score = roc_auc_score(testy, testy_pred) #AUC
train_auc_score = roc_auc_score(trainy, trainy_pred) #AUC
test_f1 = f1_score(testy, testy_pred) # F1-score
train_f1 = f1_score(trainy, trainy_pred) # F1-score
test_f2 = fbeta_score(testy, testy_pred, beta=2) #F2-score
train_f2 = fbeta_score(trainy,trainy_pred, beta=2) #F2-score

print('Best Max Depth: ', max_depth)
print('Best Min Samples of Leaf: ', min_samples_leaf)
print('-----------------------------')
print('test AUC: ', test_auc_score)
print('test F1-score: ', test_f1)
print('test F2-score: ', test_f2)
print('-----------------------------')
print('train AUC: ', train_auc_score)
print('train F1-score: ', train_f1)
print('train F2-score: ', train_f2)


# In[36]:


# select features form rank correlation function
k = 10
selected_columns = rank_mutual(trainx, trainy)[:k]
trainx_rank_mut = trainx[:, selected_columns]
testx_rank_mut = testx[:, selected_columns]


# In[39]:


# determine max number of leaves and depth of the tree
n = 20
dparams = list(np.arange(5, n+1))
lsparams = list(np.arange(5, n+1))

result_mut = tune_dt(trainx_rank_corr, trainy, dparams, lsparams)


# In[40]:


# train decision tree using best parameters
max_depth = result_mut['best-depth']
min_samples_leaf = result_mut['best-leaf-samples']

# Create a Decision Tree classifier
model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

# train model
model.fit(trainx_rank_mut, trainy)

# make prediction on test data
testy_pred = model.predict(testx_rank_mut)
trainy_pred = model.predict(trainx_rank_mut)

# Calculate AUC, F1-score, F2-score
test_auc_score = roc_auc_score(testy, testy_pred) #AUC
train_auc_score = roc_auc_score(trainy, trainy_pred) #AUC
test_f1 = f1_score(testy, testy_pred) # F1-score
train_f1 = f1_score(trainy, trainy_pred) # F1-score
test_f2 = fbeta_score(testy, testy_pred, beta=2) #F2-score
train_f2 = fbeta_score(trainy,trainy_pred, beta=2) #F2-score

print('Best Max Depth: ', max_depth)
print('Best Min Samples of Leaf: ', min_samples_leaf)
print('-----------------------------')
print('test AUC: ', test_auc_score)
print('test F1-score: ', test_f1)
print('test F2-score: ', test_f2)
print('-----------------------------')
print('train AUC: ', train_auc_score)
print('train F1-score: ', train_f1)
print('train F2-score: ', train_f2)

