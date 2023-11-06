#!/usr/bin/env python
# coding: utf-8
# Author: Masoud Nateghi
# 09/19/2023
# CS534 HW1
# PLease run each block of code one by one. 
# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


csv_file_path_train = 'energy_train.csv'
csv_file_path_val   = 'energy_val.csv'
csv_file_path_test  = 'energy_test.csv'

# Read the CSV file into a Pandas DataFrame
df_train = pd.read_csv(csv_file_path_train)
df_val   = pd.read_csv(csv_file_path_val  )
df_test  = pd.read_csv(csv_file_path_test )


# In[3]:


df_train.head()


# In[6]:


def feature_extraction(df, normalization='z-score'):
    """
    As I have introduced categorical data into dataset, I found it less confusing to just implement normalization parts
    here. I have also done normalization and handling nan values in preprocess function just for the purpose of the
    question. But major preprocessing is being done in this single function. 
    """
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
#     df.drop(columns=['date', 'Appliances'], inplace=True)
    # normalization z-score
    # if normalization == 'z-score':
    #     df = (df - df.mean()) / df.std()
    # elif normalization == 'min-max':
    #     df = (df - df.min()) / (df.max() - df.min())
    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    X = pd.concat([df, encoded_df], axis=1).to_numpy()
#     X = df.to_numpy()
    return X, y


# In[7]:


def preprocess_data(trainx, valx, testx):
    
    # Find the indices of rows containing NaN values
    trainx_nan_indices = np.isnan(trainx).any(axis=1)
    valx_nan_indices = np.isnan(valx).any(axis=1)
    testx_nan_indices = np.isnan(testx).any(axis=1)
    # Use boolean indexing to drop rows with NaN values
    trainx = trainx[~trainx_nan_indices]
    valx = valx[~valx_nan_indices]
    testx = testx[~testx_nan_indices]
    
    # Compute mean
    trainx_mean = np.mean(trainx, axis=0)
    
    # Compute standard deviation
    trainx_std_dev = np.std(trainx, axis=0)
    
    # z-score normalization
    trainx = (trainx - trainx_mean) / trainx_std_dev
    valx = (valx - trainx_mean) / trainx_std_dev
    testx = (testx - trainx_mean) / trainx_std_dev
    
    return trainx, valx, testx


# In[8]:


# preprocess data
trainx, trainy = feature_extraction(df_train)
valx , valy = feature_extraction(df_val)
testx, testy = feature_extraction(df_test)
trainx, valx, testx = preprocess_data(trainx, valx, testx)
trainy = trainy.flatten()
valy = valy.flatten()
testy = testy.flatten()


# In[10]:


def eval_linear1(trainx, trainy, valx, valy, testx, testy): 
    
    # Create a linear regression model
    model = LinearRegression()
    
    # Construct training data
    X = trainx
    y = trainy
    
    # Train the model on the training data
    model.fit(X, y)
    
    # Make predictions on the training validation and testing data
    trainy_predicted = model.predict(trainx)
    valy_predicted   = model.predict(valx)
    testy_predicted  = model.predict(testx)
    
    # Calculate RMSE (Root Mean Square Error)
    train_rmse = np.sqrt(mean_squared_error(trainy, trainy_predicted))
    val_rmse   = np.sqrt(mean_squared_error(valy  , valy_predicted  ))
    test_rmse  = np.sqrt(mean_squared_error(testy , testy_predicted ))
    
    # Calculate R-squared (R2)
    train_r2 = r2_score(trainy, trainy_predicted)
    val_r2   = r2_score(valy  , valy_predicted  )
    test_r2  = r2_score(testy , testy_predicted )
    
    return {'train-rmse':train_rmse, 'train-r2':train_r2, 'val-rmse':val_rmse, 'val-r2':val_r2,
            'test-rmse':test_rmse, 'test-r2':test_r2,}
    


# In[11]:


print(eval_linear1(trainx, trainy, valx, valy, testx, testy))


# In[12]:


def eval_linear2(trainx, trainy, valx, valy,testx, testy):
    
    # Create a linear regression model
    model = LinearRegression()
    
    # Construct training data
    X = np.concatenate((trainx, valx))
    y = np.concatenate((trainy, valy))
    
    # Train the model on the training data
    model.fit(X, y)
    
    # Make predictions on the training validation and testing data
    trainy_predicted = model.predict(trainx)
    valy_predicted   = model.predict(valx)
    testy_predicted  = model.predict(testx)
    
    # Calculate RMSE (Root Mean Square Error)
    train_rmse = np.sqrt(mean_squared_error(trainy, trainy_predicted))
    val_rmse   = np.sqrt(mean_squared_error(valy  , valy_predicted  ))
    test_rmse  = np.sqrt(mean_squared_error(testy , testy_predicted ))
    
    # Calculate R-squared (R2)
    train_r2 = r2_score(trainy, trainy_predicted)
    val_r2   = r2_score(valy  , valy_predicted  )
    test_r2  = r2_score(testy , testy_predicted )
    
    return {'train-rmse':train_rmse, 'train-r2':train_r2, 'val-rmse':val_rmse, 'val-r2':val_r2,
            'test-rmse':test_rmse, 'test-r2':test_r2,}
    


# In[13]:


print(eval_linear2(trainx, trainy, valx, valy, testx, testy))


# In[14]:


def eval_ridge1(trainx, trainy, valx, valy, testx, testy, alpha):
    
    # Create a Ridge regression model
    model = Ridge(alpha=alpha)  # alpha (regularization strength)
    
    # Construct training data
    X = trainx
    y = trainy
    
    # Train the model on the training data
    model.fit(X, y)
    
    # Make predictions on the training validation and testing data
    trainy_predicted = model.predict(trainx)
    valy_predicted   = model.predict(valx)
    testy_predicted  = model.predict(testx)
    
    # Calculate RMSE (Root Mean Square Error)
    train_rmse = np.sqrt(mean_squared_error(trainy, trainy_predicted))
    val_rmse   = np.sqrt(mean_squared_error(valy  , valy_predicted  ))
    test_rmse  = np.sqrt(mean_squared_error(testy , testy_predicted ))
    
    # Calculate R-squared (R2)
    train_r2 = r2_score(trainy, trainy_predicted)
    val_r2   = r2_score(valy  , valy_predicted  )
    test_r2  = r2_score(testy , testy_predicted )
    
    return {'train-rmse':train_rmse, 'train-r2':train_r2, 'val-rmse':val_rmse, 'val-r2':val_r2,
            'test-rmse':test_rmse, 'test-r2':test_r2,}
    


# In[15]:


print(eval_ridge1(trainx, trainy, valx, valy, testx, testy, alpha=822))


# In[16]:


def eval_ridge2(trainx, trainy, valx, valy, testx, testy, alpha): 
    
    # Create a Ridge regression model
    model = Ridge(alpha=alpha)  # alpha (regularization strength)
    
    # Construct training data
    X = np.concatenate((trainx, valx))
    y = np.concatenate((trainy, valy))
    
    # Train the model on the training data
    model.fit(X, y)
    
    # Make predictions on the training validation and testing data
    trainy_predicted = model.predict(trainx)
    valy_predicted   = model.predict(valx)
    testy_predicted  = model.predict(testx)
    
    # Calculate RMSE (Root Mean Square Error)
    train_rmse = np.sqrt(mean_squared_error(trainy, trainy_predicted))
    val_rmse   = np.sqrt(mean_squared_error(valy  , valy_predicted  ))
    test_rmse  = np.sqrt(mean_squared_error(testy , testy_predicted ))
    
    # Calculate R-squared (R2)
    train_r2 = r2_score(trainy, trainy_predicted)
    val_r2   = r2_score(valy  , valy_predicted  )
    test_r2  = r2_score(testy , testy_predicted )
    
    return {'train-rmse':train_rmse, 'train-r2':train_r2, 'val-rmse':val_rmse, 'val-r2':val_r2,
            'test-rmse':test_rmse, 'test-r2':test_r2,}
    


# In[17]:


print(eval_ridge2(trainx, trainy, valx, valy, testx, testy, alpha=822))


# In[19]:


def eval_lasso1(trainx, trainy, valx, valy, testx, testy, alpha):
    
    # Create a Ridge regression model
    model = Lasso(alpha=alpha)  # alpha (regularization strength)
    
    # Construct training data
    X = trainx
    y = trainy
    
    # Train the model on the training data
    model.fit(X, y)
    
    # Make predictions on the training validation and testing data
    trainy_predicted = model.predict(trainx)
    valy_predicted   = model.predict(valx)
    testy_predicted  = model.predict(testx)
    
    # Calculate RMSE (Root Mean Square Error)
    train_rmse = np.sqrt(mean_squared_error(trainy, trainy_predicted))
    val_rmse   = np.sqrt(mean_squared_error(valy  , valy_predicted  ))
    test_rmse  = np.sqrt(mean_squared_error(testy , testy_predicted ))
    
    # Calculate R-squared (R2)
    train_r2 = r2_score(trainy, trainy_predicted)
    val_r2   = r2_score(valy  , valy_predicted  )
    test_r2  = r2_score(testy , testy_predicted )
    
    return {'train-rmse':train_rmse, 'train-r2':train_r2, 'val-rmse':val_rmse, 'val-r2':val_r2,
            'test-rmse':test_rmse, 'test-r2':test_r2}
    


# In[20]:


def eval_lasso2(trainx, trainy, valx, valy, testx, testy, alpha):
    
    # Create a Ridge regression model
    model = Lasso(alpha=alpha)  # alpha (regularization strength)
    
    # Construct training data
    X = np.concatenate((trainx, valx))
    y = np.concatenate((trainy, valy))
    
    # Train the model on the training data
    model.fit(X, y)
    
    # Make predictions on the training validation and testing data
    trainy_predicted = model.predict(trainx)
    valy_predicted   = model.predict(valx)
    testy_predicted  = model.predict(testx)
    
    # Calculate RMSE (Root Mean Square Error)
    train_rmse = np.sqrt(mean_squared_error(trainy, trainy_predicted))
    val_rmse   = np.sqrt(mean_squared_error(valy  , valy_predicted  ))
    test_rmse  = np.sqrt(mean_squared_error(testy , testy_predicted ))
    
    # Calculate R-squared (R2)
    train_r2 = r2_score(trainy, trainy_predicted)
    val_r2   = r2_score(valy  , valy_predicted  )
    test_r2  = r2_score(testy , testy_predicted )
    
    return {'train-rmse':train_rmse, 'train-r2':train_r2, 'val-rmse':val_rmse, 'val-r2':val_r2,
            'test-rmse':test_rmse, 'test-r2':test_r2,}
    


# In[21]:


print(eval_lasso2(trainx, trainy, valx, valy, testx, testy, alpha=2.32))


# In[22]:


# create range of alpha 
alpha = np.linspace(0, 1000, 1001)

# assign place holders for optimal values and parameters
best_val, best_alpha = np.inf, 0

# place holder for best_metrics
best_metrics = {}

for i in range(len(alpha)):
    
    # evaluate metrics for given alpha
    metrics = eval_ridge1(trainx, trainy, valx, valy, testx, testy, alpha=alpha[i])
    
    # find optimal alpha corresponded to best val-rmse
    if metrics['val-rmse'] < best_val:
        best_val = metrics['val-rmse']
        best_alpha = alpha[i]
        best_metrics = metrics
    
print(best_val, best_alpha)
print(best_metrics)


# In[24]:


# create range of alpha 
n_points = 51
alpha = np.linspace(0, 1e3, n_points)

# assign place holders for optimal values and parameters
best_val, best_alpha = np.inf, 0


# history of weights for different alpha's
n_features = trainx.shape[1]
beta_hist = np.zeros((n_features + 1, n_points))

for i in range(n_points):
    
    # Create a lasso regression model
    model = Ridge(alpha=alpha[i])
    
    # Train the model on the training data
    model.fit(trainx, trainy)
    
    # Store weights
    beta_hist[:-1, i] = model.coef_
    beta_hist[-1 , i] = model.intercept_
    
    # Make predictions on the training validation and test data
    valy_predicted = model.predict(valx)
    
    # Calculate RMSE (Root Mean Square Error)
    val_rmse   = np.sqrt(mean_squared_error(valy  , valy_predicted  ))
    
    
    # find optimal alpha corresponded to best val-rmse
    if val_rmse < best_val:
        best_val = val_rmse
        best_alpha = alpha[i]
    
print(best_val, best_alpha)


# In[43]:


for i in range(n_features + 1):
    plt.plot(alpha, beta_hist[i, :], 'o-', markersize=1)
    
plt.axvline(x=822, color='red', linestyle='--', label='$\lambda = 822$')
plt.xlabel('$\lambda$', fontsize=16)
plt.ylabel(r'$\beta $', fontsize=16)
plt.title('Coefficient Path Plot for Ridge Regression')
plt.legend()
plt.show()


# In[25]:


# create range of alpha 
alpha = np.linspace(0, 5, 1001)

# assign place holders for optimal values and parameters
best_val, best_alpha = np.inf, 0

# place holder for best_metrics
best_metrics = {}

for i in range(len(alpha)):
    
    # evaluate metrics for given alpha
    metrics = eval_lasso1(trainx, trainy, valx, valy, testx, testy, alpha=alpha[i])
    
    # find optimal alpha corresponded to best val-rmse
    if metrics['val-rmse'] < best_val:
        best_val = metrics['val-rmse']
        best_alpha = alpha[i]
        best_metrics = metrics
        
print(best_metrics)    
print(best_val, best_alpha)


# In[45]:


# create range of alpha 
n_points = 51
alpha = np.linspace(0, 30, n_points)

# assign place holders for optimal values and parameters
best_val, best_alpha = np.inf, 0

# place holder for best_metrics
best_metrics = {}

# history of weights for different alpha's
n_features = trainx.shape[1]
beta_hist = np.zeros((n_features + 1, n_points))

for i in range(n_points):
    
    # Create a lasso regression model
    model = Lasso(alpha=alpha[i])
    
    # Train the model on the training data
    model.fit(trainx, trainy)
    
    # Store weights
    beta_hist[:-1, i] = model.coef_
    beta_hist[-1 , i] = model.intercept_
    
    # Make predictions on the training validation and test data
    valy_predicted = model.predict(valx)
    
    # Calculate RMSE (Root Mean Square Error)
    val_rmse   = np.sqrt(mean_squared_error(valy  , valy_predicted  ))
    
    
    # find optimal alpha corresponded to best val-rmse
    if val_rmse < best_val:
        best_val = val_rmse
        best_alpha = alpha[i]
    
print(best_val, best_alpha)


# In[46]:


for i in range(n_features + 1):
    plt.semilogx(alpha, beta_hist[i, :], 'o-', markersize=2)
    
plt.axvline(x=2.32, color='red', linestyle='--', label='$\lambda = 2.37$')
plt.xlabel('$\log(\lambda)$', fontsize=16)
plt.ylabel(r'$\beta $', fontsize=16)
plt.title('Coefficient Path Plot for Lasso Regression')
plt.legend(loc='lower right')
plt.show()


# In[26]:


def loss(x, y, beta, el, alpha):
    # Clculate OLS term 
    y = y.reshape(-1, 1)
    N = x.shape[0]
    ols_term = (1 / 2) * (np.linalg.norm(y - np.matmul(x, beta)) ** 2) / N # note N!!!
    
    # Calculate regulariztion term
    reg_term = el * (alpha * (np.linalg.norm(beta) ** 2) + (1 - alpha) * np.linalg.norm(beta, ord=1))
    
    return ols_term + reg_term

def grad_step(x, y, beta, el, alpha, eta):
    
    # Calculate Gradient of the convex and differentiable part of loss function 
#     y = y.reshape(-1, 1)
    N = x.shape[0] # Batch size
    grad_g = (-(np.matmul(x.T, y)) + (np.matmul(x.T, np.matmul(x, beta))))/N  + 2 * el * alpha * beta # (note N!!!)
    
    # Proximal function for Proximal Gradient Descent
#     t = el * (1 - alpha) * eta # define parameter t for prox function
    t = el * eta # define parameter t for prox function ## note!!!!!!!!!!!
    p = len(beta) # number of weights
    
    def prox_t(z):
        for i in range(p):
            if z[i] >= t:
                z[i] = z[i] - t
            elif z[i] <= -t:
                z[i] = z[i] + t
            else:
                z[i] = 0
        return z
    
    # Update beta using prox function
    return prox_t(beta - t * grad_g)


# In[27]:


class ElasticNet():
    def __init__(self, el, alpha, eta, batch, epoch):
        self.el=el
        self.alpha=alpha
        self.eta=eta
        self.batch=batch
        self.epoch=epoch

    def coef(self):
        return self.beta

    def train(self, x, y):
        # number of samples, number of features
        n_samples, n_features = x.shape
        
        # Add a column of 1's for bias to dataset
        X = np.concatenate([x, np.ones((n_samples, 1))], axis = 1)
        self.beta = np.random.rand(n_features + 1, 1) # + 1 for bias
        
        # concat X and y to shuffle them the same
        Xy = np.concatenate([X, y.reshape(-1, 1)], axis = 1)
        
        train_hist = {}
        for ep in range(self.epoch):
            
            # Random Shuffling
            Xy = np.random.permutation(Xy)
            X = Xy[:, :-1]
            y = Xy[:, -1:]
            
            # Compute number of batches
            B = n_samples // self.batch
            for b in range(B):
                X_batch = X[b * self.batch:(b + 1) * self.batch, :]
                y_batch = y[b * self.batch:(b + 1) * self.batch, :]
                self.beta = grad_step(X_batch, y_batch, self.beta, self.el, self.alpha, self.eta)
            train_hist[ep+1] = loss(X, y, self.coef(), el=self.el, alpha=self.alpha)
        return train_hist
                                  
    def predict(self, x):
        # Add a column of 1's for bias to dataset
        n_samples = x.shape[0]
        X = np.concatenate([x, np.ones((n_samples, 1))], axis = 1) 
        return np.matmul(X, self.beta)


# In[49]:


# percentage of dataset to be used for step size tuning
trainx_percentage = 0.2 

# number of samples from data set
n_samples = int(trainx_percentage * trainx.shape[0]) 

# rows indices to be selected randomly
random_indices = np.random.choice(trainx.shape[0], n_samples, replace=False) 

# Randomly select rows
trainx_subsample = trainx[random_indices, :]
trainy_subsample = trainy[random_indices]

# range of step size
eta = np.array([1e-11, 1e-10, 1e-9, 1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6])
print(eta)


# In[50]:


# history ridge
hist_ridge = []
for i in range(len(eta)):
    print(i)
    el = 822 # optimal regularization parameter for ridge regression
    epoch = 1000
    # create model
    model = ElasticNet(el=el, alpha=0.5, eta=eta[i], batch=100, epoch=epoch)
    error_hist = model.train(trainx_subsample, trainy_subsample)
    hist_ridge.append(error_hist)


# In[74]:


# Create subplots
plt.figure(figsize=(16, 30))
for i in range(len(eta)):
    plt.subplot(5, 2, i+1)
    plt.plot(hist_ridge[i].keys(), hist_ridge[i].values())
    plt.xlabel('$epoch (number)$', fontsize=16)
    plt.ylabel(r'$f_{o}(x)-Ridge$', fontsize=16)
    plt.title(r'$\eta$ = ' + str(eta[i]) + ', loss = ' + "{:.2f}".format(hist_ridge[i][epoch]))
plt.show()


# In[76]:


for i in np.arange(2, 6):
    plt.plot(hist_ridge[i].keys(), hist_ridge[i].values(), label='$\lambda = $' + str(eta[i]))

plt.xlabel('$epoch (number)$', fontsize=16)
plt.ylabel(r'$f_{o}-Ridge(x) $', fontsize=16)
plt.title('Effect of Different Learning Rates for Ridge Regression')
plt.ylim(1e4, 15e3)
plt.legend()
plt.show()


# In[28]:


def eval_elastic(trainx, trainy, valx, valy, testx, testy, el, alpha, eta, batch, epoch):
    
    # Create a Ridge regression model
    model = ElasticNet(el=el, alpha=alpha, eta=eta, batch=batch, epoch=epoch)
    
    # Construct training data
    X = trainx
    y = trainy
    
    # Train the model on the training data
    model.train(X, y)
    
    # Make predictions on the training validation and testing data
    trainy_predicted = model.predict(trainx)
    valy_predicted   = model.predict(valx)
    testy_predicted  = model.predict(testx)
    
    # Calculate RMSE (Root Mean Square Error)
    train_rmse = np.sqrt(mean_squared_error(trainy, trainy_predicted))
    val_rmse   = np.sqrt(mean_squared_error(valy  , valy_predicted  ))
    test_rmse  = np.sqrt(mean_squared_error(testy , testy_predicted ))
    
    # Calculate R-squared (R2)
    train_r2 = r2_score(trainy, trainy_predicted)
    val_r2   = r2_score(valy  , valy_predicted  )
    test_r2  = r2_score(testy , testy_predicted )
    
    return {'train-rmse':train_rmse, 'train-r2':train_r2, 'val-rmse':val_rmse, 'val-r2':val_r2,
            'test-rmse':test_rmse, 'test-r2':test_r2,}
    


# In[29]:


model = eval_elastic(trainx, trainy, valx, valy, testx, testy, el=822, alpha=0.5, eta=5e-8, batch=200, epoch=1000)
print(model)


# In[55]:


# percentage of dataset to be used for step size tuning
trainx_percentage = 0.2 

# number of samples from data set
n_samples = int(trainx_percentage * trainx.shape[0]) 

# rows indices to be selected randomly
random_indices = np.random.choice(trainx.shape[0], n_samples, replace=False) 

# Randomly select rows
trainx_subsample = trainx[random_indices, :]
trainy_subsample = trainy[random_indices]

# range of step size
eta = np.array([1e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2])
print(eta)


# In[56]:


# history lasso
hist_lasso = []
for i in range(len(eta)):
    
    print(i)
    el = 2.37 # optimal regularization parameter for lasso regression
    
    # create model
    model = ElasticNet(el=el, alpha=0.5, eta=eta[i], batch=100, epoch=1000)
    error_hist = model.train(trainx_subsample, trainy_subsample)
    hist_lasso.append(error_hist)


# In[79]:


# Create subplots
plt.figure(figsize=(16, 36))

for i in range(len(eta)):
    plt.subplot(int(len(eta)/2), 2, i+1)
    plt.plot(hist_lasso[i].keys(), hist_lasso[i].values())
    plt.xlabel('$epoch (number)$', fontsize=16)
    plt.ylabel(r'$f_{o}(x)-Lasso$', fontsize=16)
    plt.title(r'$\eta$ = ' + str(eta[i]) + ', loss = ' + "{:.2f}".format(hist_lasso[i][epoch]))
plt.show()


# In[81]:


for i in np.arange(0, 7):
    plt.plot(hist_ridge[i].keys(), hist_ridge[i].values(), label='$\lambda = $' + str(eta[i]))

plt.xlabel('$epoch (number)$', fontsize=16)
plt.ylabel(r'$f_{o}-Lasso(x) $', fontsize=16)
plt.title('Effect of Different Learning Rates')
plt.ylim(9e3, 25e3)
plt.legend(loc='lower right')
plt.show()


# In[83]:


model = eval_elastic(trainx, trainy, valx, valy, testx, testy, el=2.32, alpha=0.5, eta=5e-3, batch=200, epoch=1000)
print(model)


# In[23]:


alpha = np.linspace(0, 1, 11)
alpha_hist = []
for i in range(len(alpha)):
    print(i)
    alpha_hist.append(eval_elastic(trainx, trainy, valx, valy, testx, testy, el=822, alpha=alpha[i], eta=1e-8, batch=200, epoch=1000))
df_ridge = pd.DataFrame(alpha_hist)
df_ridge['alpha'] = alpha
df_ridge


# In[13]:


alpha = np.linspace(0, 1, 11)
alpha_hist = []
for i in range(len(alpha)):
    print(i)
    alpha_hist.append(eval_elastic(trainx, trainy, valx, valy, testx, testy, el=2.32, alpha=alpha[i], eta=1e-3, batch=200, epoch=1000))
df_lasso = pd.DataFrame(alpha_hist)
df_lasso['alpha'] = alpha
df_lasso


# In[43]:


model1 = ElasticNet(el=822, alpha=0.5, eta=1e-8, batch=200, epoch=1000)
model1_hist = model1.train(trainx, trainy)


# In[44]:


plt.plot(model1_hist.keys(), model1_hist.values())
plt.show()


# In[46]:


model1_hist


# In[41]:


print(model1.coef().T)


# In[45]:


print(eval_elastic(trainx, trainy, valx, valy, testx, testy, el=822, alpha=0.5, eta=1e-8, batch=200, epoch=1000))


# In[29]:


model2 = ElasticNet(el=2.32, alpha=0, eta=1e-3, batch=200, epoch=1000)
model2_hist = model2.train(trainx, trainy)


# In[30]:


plt.plot(model2_hist.keys(), model2_hist.values())
plt.show()


# In[31]:


print(model2.coef().T)

