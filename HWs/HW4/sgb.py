import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


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
    
    return {'best-nIter': best_nIter, 'best-nu': best_nu, 'best-q': best_q, 
            'best-RMSE': best_score, 'grid_search': grid_search}


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
        return self

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

