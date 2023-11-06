# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:42:21 2023

@author: Masoud Nateghi
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
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
            'test-rmse':test_rmse, 'test-r2':test_r2,}
    
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
