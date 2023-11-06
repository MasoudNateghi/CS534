from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

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

    
    return {'best-hidden': best_hidden, 'best-activation': best_activation, 'best-alpha': best_alpha, 
            'best-valid-AUC': best_score, 'grid_search': grid_search}