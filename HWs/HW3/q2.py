from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

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