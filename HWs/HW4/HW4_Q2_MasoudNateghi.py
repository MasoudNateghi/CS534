#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor


# In[2]:


csv_file_path_train = 'energy_train.csv'
csv_file_path_val   = 'energy_val.csv'
csv_file_path_test  = 'energy_test.csv'

# Read the CSV file into a Pandas DataFrame
df_train = pd.read_csv(csv_file_path_train)
df_val   = pd.read_csv(csv_file_path_val  )
df_train = pd.concat([df_train, df_val], axis=0)
df_train = df_train.reset_index(drop=True)
df_test  = pd.read_csv(csv_file_path_test )


# In[3]:


df_train.head()


# In[4]:


def feature_extraction(df, normalization='z-score'):
    import warnings
    warnings.filterwarnings('ignore')
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract the day of the week and create a new column
    df['DayOfWeek'] = df['date'].dt.day_name()
    
    # Create a new column for 2-hour intervals
    df['TimeInterval'] = df['date'].dt.strftime('%H:%M:%S').str[:2].astype(int) // 2 * 2
    
    # Encode 'TimeInterval' using one-hot encoding
    df['TimeInterval'] = df['TimeInterval'].astype(str)
    
    # import OneHotEncoder
    from sklearn.preprocessing import OneHotEncoder
    
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    
    # Fit the encoder on the categorical column and transform it
    encoded_data = encoder.fit_transform(df[['DayOfWeek', 'TimeInterval']])
    
    # Create a new DataFrame with one-hot encoded columns
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['DayOfWeek', 'TimeInterval']))
    y = np.reshape(df['Appliances'].to_numpy(), (-1, 1))
    
    # Drop the original 'Category' column
    df.drop(columns=['DayOfWeek', 'TimeInterval', 'date', 'Appliances'], inplace=True)
    
    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    X = pd.concat([df, encoded_df], axis=1).to_numpy()
    
    return X, y


# In[5]:


# preprocess data
trainx, trainy = feature_extraction(df_train)
testx, testy = feature_extraction(df_test)

# normaliztion
scaler = StandardScaler()
trainx = scaler.fit_transform(trainx)
# Transform the testing data based on the fitted scaler
testx = scaler.transform(testx)

# flatten y to be 1D numpy array
trainy = trainy.flatten()
testy = testy.flatten()


# In[6]:


def tune_sgtb(x, y, lst_nIter, lst_nu, lst_q, md):
    # create SGTB model
    model = SGTB(md=md)
    
    # Define the hyperparameter grid to search
    param_grid = {
        'nIter': lst_nIter,  
        'q': lst_q, 
        'nu': lst_nu, 
    }
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=2)
    
    # Perform grid search with cross-validation
    grid_search.fit(x, y)
    
    # Get the best hyperparameters
    best_nIter = grid_search.best_params_['nIter']
    best_q = grid_search.best_params_['q']
    best_nu = grid_search.best_params_['nu']
    best_score = grid_search.best_score_
    
    # Access the cross-validation results
    cv_results = grid_search.cv_results_
    
    return {'best-nIter': best_nIter, 'best-nu': best_nu, 'best-q': best_q, 
            'best-RMSE': best_score, 'grid_search': grid_search, 'cv-results':cv_results}


def compute_residual(y_true, y_pred):
    return y_true - y_pred


class SGTB(RegressorMixin):
    def __init__(self, nIter=1, q=1, nu=0.1, md=3):
        self.nIter = nIter # number of boosting iterations
        self.q = q # subsampling rate
        self.nu = nu # shrinkage parameter
        self.md = md # max depth of the tree that is trained at each iteration
        # key is the iteration and the value is the RMSE of the training data. Note that for the 0th iteration,
        # it should return the RMSE for a model that predicts the mean of y and the 1st iteration would return
        # the RMSE for a single tree fit to the initial set of residuals multiplied by the shrinkage rate.
        self.train_dict = {} 

    def get_params(self, deep=True):
        return {"nIter": self.nIter,
                "q": self.q,
                "nu": self.nu,
                "md": self.md}

    # set the parameters
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # train data using gradient boosting
    def fit(self, x, y):
        # to store h_m at m-th iteration
        self.model = []
        # target mean
        f_0 = np.mean(y)
        # add f_0 to models
        self.model.append(f_0)
        # calculate RMSE at 0th iteration
        y_pred = self.model[0] * np.ones(x.shape[0])
        self.train_dict[0] = mean_squared_error(y, y_pred, squared=False)
        
        for m in range(1, self.nIter + 1):
            # size of subsampled dataset
            n = int(x.shape[0] * self.q)
            # indices to be selected from dataset
            idx = np.random.choice(x.shape[0], n, replace=False)
            # retrive subsampled data
            x_sub = x[idx, :]
            y_sub = y[idx]
            # make prediction on x_sub to calculate residuals
            y_sub_pred = self.model[0] * np.ones(n)
            for i in range(1, m):
                # calculate similar to line 6 of Alg1
                y_sub_pred += self.nu * self.model[i].predict(x_sub)
            # calculate residuals
            r_m = compute_residual(y_sub, y_sub_pred)
            # instantiate decision tree regressor model
            h_m = DecisionTreeRegressor(max_depth=self.md)
            # train h_m
            h_m.fit(x_sub, r_m)
            # add h_m to the list of models
            self.model.append(h_m)
            # update self.train_dict
            y_sub_pred += self.nu * self.model[m].predict(x_sub)
            self.train_dict[m] = mean_squared_error(y_sub, y_sub_pred, squared=False)

    def predict(self, x):
        # number of samples
        n_sample = x.shape[0]
        # build the output
        y = self.model[0] * np.ones(n_sample)
        
        for i in range(1, self.nIter):
            # retrieve h_m form list of models
            h_m = self.model[i]
            # predict on data
            y += self.nu * h_m.predict(x)
        
        return y



# In[15]:


lst_nIter = list(range(1, 21))
lst_nu = list(np.arange(0.1, 1.01, 0.05))
lst_q = [1]
md = 3

result = tune_sgtb(trainx, trainy, lst_nIter, lst_nu, lst_q, md)


# In[26]:


get_ipython().run_line_magic('matplotlib', 'notebook')

# Create a 3D plot for scores and parameters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Set integer ticks on the x, y, and z axes
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
X, Y = np.meshgrid(lst_nu, lst_nIter)
mean_test_scores = np.array(-result['cv-results']['mean_test_score']).reshape(len(lst_nIter), -1)

ax.plot_surface(X, Y, mean_test_scores, cmap='viridis')
ax.set_ylabel('nIter')
ax.set_xlabel('nu')
ax.set_zlabel('RMSE')

plt.show()


# In[25]:


# train decision tree using best parameters
nIter = result['best-nIter']
nu = result['best-nu']
q = 1
md = 3

# Create a Decision Tree classifier
model = SGTB(nIter, q, nu, md)

# train model
model.fit(trainx, trainy)

# make prediction on test data
testy_pred = model.predict(testx)
trainy_pred = model.predict(trainx)

print('nIter: ', nIter)
print('nu: ', nu)
print('q: ', q)
print('max depth: ', md)
print('------------------')
print('RMSE train', mean_squared_error(trainy, trainy_pred, squared=False))
print('R2 train', r2_score(trainy, trainy_pred))
print('RMSE test', mean_squared_error(testy, testy_pred, squared=False))
print('R2 test', r2_score(testy, testy_pred))


# In[29]:


lst_nIter = list(range(1, 21))
lst_nu = list(np.arange(0.05, 0.41, 0.05))
lst_q = [0.6, 0.7, 0.8, 0.9, 1]
md = 3

result = tune_sgtb(trainx, trainy, lst_nIter, lst_nu, lst_q, md)


# In[39]:


# train decision tree using best parameters
nIter = result['best-nIter']
nu = result['best-nu']
q = result['best-q']
md = 3

# Create a Decision Tree classifier
model = SGTB(nIter, q, nu, md)

# train model
model.fit(trainx, trainy)

# make prediction on test data
testy_pred = model.predict(testx)
trainy_pred = model.predict(trainx)

print('nIter: ', nIter)
print('nu: ', nu)
print('q: ', q)
print('max depth: ', md)
print('------------------')
print('RMSE train', mean_squared_error(trainy, trainy_pred, squared=False))
print('R2 train', r2_score(trainy, trainy_pred))
print('RMSE test', mean_squared_error(testy, testy_pred, squared=False))
print('R2 test', r2_score(testy, testy_pred))


# In[ ]:




