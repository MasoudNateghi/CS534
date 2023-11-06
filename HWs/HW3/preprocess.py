import numpy as np
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.feature_selection import mutual_info_classif

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

def rank_mutual(x, y):
    # Calculate mutual information between features and the target variable y
    mutual_info = mutual_info_classif(x, y)

    # Get the indices that would sort mutual_info in descending order
    sorted_indices = np.argsort(mutual_info)[::-1]

    return sorted_indices