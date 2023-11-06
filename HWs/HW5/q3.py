import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA, NMF
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, fbeta_score

def build_logr(train_x, test_x, train_y, test_y):
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

def build_dt(train_x, test_x, train_y, test_y): 
    # Create a Decision Tree classifier
    model = DecisionTreeClassifier()

    # Define the hyperparameter grid to search
    n = 5
    param_grid = {
        'max_depth': list(np.arange(2, n+1)),  
        'min_samples_leaf': list(np.arange(2, n+1))  
    }
    
    # Create a custom scoring function
    scoring = {'auc': 'roc_auc', 'f1': 'f1', 'f2': make_scorer(fbeta_score, beta=2)}
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='auc',
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

def build_rf(train_x, test_x, train_y, test_y): 
    # Create a Decision Tree classifier
    model = RandomForestClassifier()

    # Define the hyperparameter grid to search
    n = 5
    param_grid = {
        'n_estimators': [20],
        'max_depth': list(np.arange(2, n+1)),  
        'min_samples_leaf': list(np.arange(2, n+1))  
    }
    
    # Create a custom scoring function
    scoring = {'auc': 'roc_auc', 'f1': 'f1', 'f2': make_scorer(fbeta_score, beta=2)}
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='auc',
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

def build_svm(train_x, test_x, train_y, test_y): 
    # Create a Decision Tree classifier
    model = SVC()

    # Define the hyperparameter grid to search
    param_grid = {'C': [0.01],
                  'gamma': [0.01],
                  'kernel': ['linear', 'poly'],
                  'degree': [2]}
    
    # Create a custom scoring function
    scoring = {'auc': 'roc_auc', 'f1': 'f1', 'f2': make_scorer(fbeta_score, beta=2)}
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='auc',
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

def build_nn(train_x, test_x, train_y, test_y): 
    # Create a Decision Tree classifier
    model = MLPClassifier(max_iter=100, solver='adam', learning_rate='adaptive')

    # Define the hyperparameter grid to search
    param_grid = {
        'hidden_layer_sizes': [(10, 30, 10), (100, 50)],  
        'activation': ['relu', 'logistic', 'tanh'], 
        'alpha': [0.1, 1, 10], 
    }
    
    # Create a custom scoring function
    scoring = {'auc': 'roc_auc', 'f1': 'f1', 'f2': make_scorer(fbeta_score, beta=2)}
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='auc',
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