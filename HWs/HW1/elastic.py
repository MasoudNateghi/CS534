import numpy as np
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
    # y = y.reshape(-1, 1)
    N = x.shape[0] # Batch size
    grad_g = (-(np.matmul(x.T, y)) + (np.matmul(x.T, np.matmul(x, beta)))) / N  + 2 * el * alpha * beta # (note N!!!)
    
    # Proximal function for Proximal Gradient Descent
    t = el * (1 - alpha) * eta # define parameter t for prox function
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
