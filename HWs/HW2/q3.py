import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, CategoricalNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

def generate_train_val(x, y, valsize):
    # number of samples for validation dataset
    n_samples = int(valsize * x.shape[0]) 
    
    # rows indices to be selected randomly
    val_indices = np.random.choice(x.shape[0], n_samples, replace=False) # validation indices
    # retrieve other indices that are not present in val_indices for train_indices 
    all_indices = np.arange(x.shape[0])
    train_indices = np.setdiff1d(all_indices, val_indices) # train indices
    
    # Crete training and validation datasets
    # train
    x_train = x[train_indices, :]
    y_train = y[train_indices]
    
    # validation
    x_val = x[val_indices, :]
    y_val = y[val_indices]
    
    return {'train-x':x_train, 'train-y':y_train, 'val-x':x_val, 'val-y':y_val}

def generate_kfold(x, y, k):
    # assign the first k elements with random_indices for the first fold, and so on. 
    random_indices = np.random.choice(x.shape[0], x.shape[0], replace=False)
    
    # number of samples in each fold
    n_samples = x.shape[0] // k
    
    fold = np.zeros(x.shape[0])
    for i in range(k):
        fold_indices = random_indices[i * n_samples:(i + 1) * n_samples]
        fold[fold_indices] = i
    
    if x.shape[0] % k != 0:
        # remaining samples
        r_samples = x.shape[0] % k
        r_indices = np.random.choice(k, r_samples, replace=False)
        fold[random_indices[-r_samples:]] = r_indices

    return fold

def eval_holdout(x, y, valsize, logistic):
    # split data into train and validation
    datasets = generate_train_val(x, y, valsize)
    x_train = datasets['train-x']
    y_train = datasets['train-y']
    x_val = datasets['val-x']
    y_val = datasets['val-y']
    
    # Train the model on the training data
    logistic.fit(x_train, y_train)
    
    # Make predictions on the test data
    trainy_predict = logistic.predict(x_train)
    valy_predict = logistic.predict(x_val)
    
    # Evaluate the model
    train_accuracy = accuracy_score(y_train, trainy_predict)
    val_accuracy = accuracy_score(y_val, valy_predict)
    
    # Calculate the predicted probabilities for class 1
    trainy_prob = logistic.predict_proba(x_train)[:, 1]
    valy_prob = logistic.predict_proba(x_val)[:, 1]
    
    # Calculate the AUC score
    train_auc_score = roc_auc_score(y_train, trainy_prob)
    val_auc_score = roc_auc_score(y_val, valy_prob)
    
    return {'train-acc':train_accuracy, 'train-auc':train_auc_score, 'val-acc':val_accuracy, 'val-auc':val_auc_score}

def eval_kfold(x, y, k, logistic):
    # do k-fold cross validation
    # assign fold indices
    fold = generate_kfold(x, y, k)
    
    # store metrics of each fold
    metrics = []
    
    for i in range(k):
        # create train and validation datasets
        x_val = x[fold == i, :]
        y_val = y[fold == i]
        x_train = x[fold != i, :]
        y_train = y[fold != i]
        
        # Train the model on the training data
        logistic.fit(x_train, y_train)
        
        # Make predictions on the test data
        trainy_predict = logistic.predict(x_train)
        valy_predict = logistic.predict(x_val)

        # Evaluate the model
        train_accuracy = accuracy_score(y_train, trainy_predict)
        val_accuracy = accuracy_score(y_val, valy_predict)

        # Calculate the predicted probabilities for class 1
        trainy_prob = logistic.predict_proba(x_train)[:, 1]
        valy_prob = logistic.predict_proba(x_val)[:, 1]

        # Calculate the AUC score
        train_auc_score = roc_auc_score(y_train, trainy_prob)
        val_auc_score = roc_auc_score(y_val, valy_prob)
        
        # store metrics
        metrics.append([train_accuracy, train_auc_score, val_accuracy, val_auc_score])
    
    # average metrics over all folds
    train_acc_avg = 0
    train_auc_avg = 0
    val_acc_avg = 0
    val_auc_avg = 0
    for i in range(k):
        train_acc_avg += metrics[i][0] / k
        train_auc_avg += metrics[i][1] / k
        val_acc_avg += metrics[i][2] / k
        val_auc_avg += metrics[i][3] / k
        
    return {'train-acc':train_acc_avg, 'train-auc':train_auc_avg, 'val-acc':val_acc_avg, 'val-auc':val_auc_avg}

def eval_mccv(x, y, valsize, s, logistic):
    # store metrics of each fold
    metrics = []
    
    for i in range(s):
        # split data into train and validation
        datasets = generate_train_val(x, y, valsize)
        x_train = datasets['train-x']
        y_train = datasets['train-y']
        x_val = datasets['val-x']
        y_val = datasets['val-y']
        
        # Train the model on the training data
        logistic.fit(x_train, y_train)

        # Make predictions on the test data
        trainy_predict = logistic.predict(x_train)
        valy_predict = logistic.predict(x_val)

        # Evaluate the model
        train_accuracy = accuracy_score(y_train, trainy_predict)
        val_accuracy = accuracy_score(y_val, valy_predict)

        # Calculate the predicted probabilities for class 1
        trainy_prob = logistic.predict_proba(x_train)[:, 1]
        valy_prob = logistic.predict_proba(x_val)[:, 1]

        # Calculate the AUC score
        train_auc_score = roc_auc_score(y_train, trainy_prob)
        val_auc_score = roc_auc_score(y_val, valy_prob)

        # store metrics
        metrics.append([train_accuracy, train_auc_score, val_accuracy, val_auc_score])
        
    # average metrics over all folds
    train_acc_avg = 0
    train_auc_avg = 0
    val_acc_avg = 0
    val_auc_avg = 0
    for i in range(s):
        train_acc_avg += metrics[i][0] / s
        train_auc_avg += metrics[i][1] / s
        val_acc_avg += metrics[i][2] / s
        val_auc_avg += metrics[i][3] / s
        
    return {'train-acc':train_acc_avg, 'train-auc':train_auc_avg, 'val-acc':val_acc_avg, 'val-auc':val_auc_avg}
    