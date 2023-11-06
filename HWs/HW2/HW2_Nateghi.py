#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, CategoricalNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


# In[2]:


# Load data
data_train = pd.read_csv('spam.train.dat', delimiter=' ', header=None) 
data_test = pd.read_csv('spam.test.dat', delimiter=' ', header=None) 

# Split train data and labels (last column: labels) 
# COnvert dataframes to numpy arrays
trainx = data_train.iloc[:, :-1].to_numpy()
trainy = data_train.iloc[:, -1].to_numpy()
testx = data_test.iloc[:, :-1].to_numpy()
testy = data_test.iloc[:, -1].to_numpy()


# In[6]:


data_train


# In[3]:


def do_nothing(train, test):
    return train, test

def do_std(train, test):
    # Create an instance of the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler to your data and transform the data
    train = scaler.fit_transform(train)
    
    # Transform the test data using the same scaler
    test = scaler.transform(test)
    
    return train, test
    
def do_log(train, test):
    # smoothed natural logarithm transformation
    train = np.log(train + 0.1)
    test = np.log(test + 0.1)
    
    return train, test

def do_bin(train, test):
    # Otherwise Initial dataset changes
    train1 = train.copy()
    test1 = test.copy()
    
    # apply indicator function
    train1[train > 0] = 1
    train1[train <= 0] = 0
    
    test1[test > 0] = 1
    test1[test <= 0] = 0
    
    return train1, test1


# In[4]:


# Apply transformations on train and test datasets
trainx, testx = do_nothing(trainx, testx)
trainx_std, testx_std = do_std(trainx, testx)
trainx_log, testx_log = do_log(trainx, testx)
trainx_bin, testx_bin = do_bin(trainx, testx)


# In[121]:


def eval_nb(trainx, trainy, testx, testy):
    # Choose Gaussian, Multinomial, or Bernoulli Naive Bayes classifier
    # uncomment the one you want to try
    nb_classifier = BernoulliNB()
#     nb_classifier = GaussianNB()
#     nb_classifier = MultinomialNB()
#     nb_classifier = CategoricalNB()
#     nb_classifier = ComplementNB()
    
    # Create and train the classifier
    nb_classifier.fit(trainx, trainy)
    
    # Make predictions on the train and test data
    trainy_predict = nb_classifier.predict(trainx)
    testy_predict = nb_classifier.predict(testx)
    
    # Evaluate the model
    train_accuracy = accuracy_score(trainy, trainy_predict)
    test_accuracy = accuracy_score(testy, testy_predict)
    
    # Calculate the predicted probabilities for class 1
    trainy_prob = nb_classifier.predict_proba(trainx)[:, 1]
    testy_prob = nb_classifier.predict_proba(testx)[:, 1]
    
    # Calculate the AUC score
    train_auc_score = roc_auc_score(trainy, trainy_prob)
    test_auc_score = roc_auc_score(testy, testy_prob)
    
    return {'train-acc':train_accuracy, 'train-auc':train_auc_score, 'test-acc':test_accuracy, 
           'test-auc':test_auc_score, 'test-prob':testy_prob}


# In[9]:


print('Bernoulli Naive Bayes')
print('-----------------------------------------')

# do nothing: raw dataset
print('After do_nothing preprocessing function: ')
print(eval_nb(trainx, trainy, testx, testy))
print('-----------------------------------------')

# z-score normalization
print('After do_std preprocessing function: ')
print(eval_nb(trainx_std, trainy, testx_std, testy))
print('-----------------------------------------')

# logarithm transformation
print('After do_log preprocessing function: ')
print(eval_nb(trainx_log, trainy, testx_log, testy))
print('-----------------------------------------')

# indicator function transformation
print('After do_bin preprocessing function: ')
print(eval_nb(trainx_bin, trainy, testx_bin, testy))
print('-----------------------------------------')


# In[122]:


def eval_lr(trainx, trainy, testx, testy):
    # turn off warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Create the Logistic Regression model
    logistic_regression_model = LogisticRegression()
    
    # Train the model on the training data
    logistic_regression_model.fit(trainx, trainy)
    
    # Make predictions on the test data
    trainy_predict = logistic_regression_model.predict(trainx)
    testy_predict = logistic_regression_model.predict(testx)
    
    # Evaluate the model
    train_accuracy = accuracy_score(trainy, trainy_predict)
    test_accuracy = accuracy_score(testy, testy_predict)
    
    # Calculate the predicted probabilities for class 1
    trainy_prob = logistic_regression_model.predict_proba(trainx)[:, 1]
    testy_prob = logistic_regression_model.predict_proba(testx)[:, 1]
    
    # Calculate the AUC score
    train_auc_score = roc_auc_score(trainy, trainy_prob)
    test_auc_score = roc_auc_score(testy, testy_prob)
    
    return {'train-acc':train_accuracy, 'train-auc':train_auc_score, 'test-acc':test_accuracy, 
           'test-auc':test_auc_score, 'test-prob':testy_prob}


# In[12]:


print('Logistic Regression')
print('-----------------------------------------')
# do nothing: raw dataset
print('After do_nothing preprocessing function: ')
print(eval_lr(trainx, trainy, testx, testy))
print('-----------------------------------------')
# z-score normalization
print('After do_std preprocessing function: ')
print(eval_lr(trainx_std, trainy, testx_std, testy))
print('-----------------------------------------')
# logarithm transformation
print('After do_log preprocessing function: ')
print(eval_lr(trainx_log, trainy, testx_log, testy))
print('-----------------------------------------')
# indicator function transformation
print('After do_bin preprocessing function: ')
print(eval_lr(trainx_bin, trainy, testx_bin, testy))
print('-----------------------------------------')


# In[13]:


def eval_nb_roc(trainx, trainy, testx, testy):
    # Choose Gaussian, Multinomial, or Bernoulli Naive Bayes classifier
    # uncomment the one you want to try
    nb_classifier = BernoulliNB()
#     nb_classifier = GaussianNB()
#     nb_classifier = MultinomialNB()
    
    # Create and train the classifier
    nb_classifier.fit(trainx, trainy)
    
    # Make predictions on the train and test data
    trainy_predict = nb_classifier.predict(trainx)
    testy_predict = nb_classifier.predict(testx)
    
    # Evaluate the model
    train_accuracy = accuracy_score(trainy, trainy_predict)
    test_accuracy = accuracy_score(testy, testy_predict)
    
    # Calculate the predicted probabilities for class 1
    trainy_prob = nb_classifier.predict_proba(trainx)[:, 1]
    testy_prob = nb_classifier.predict_proba(testx)[:, 1]
    
    # Calculate the AUC score
    train_auc_score = roc_auc_score(trainy, trainy_prob)
    test_auc_score = roc_auc_score(testy, testy_prob)
    
    # Calculate fpr and tpr
    fpr, tpr, _ = roc_curve(testy, testy_prob)
    
    return fpr, tpr


# In[14]:


# Compute FPR and TPR for each preprocessing method
# do nothing: raw dataset
fpr, tpr = eval_nb_roc(trainx, trainy, testx, testy)
roc_auc = auc(fpr, tpr)
# z-score normalization
fpr_std, tpr_std = eval_nb_roc(trainx_std, trainy, testx_std, testy)
roc_auc_std = auc(fpr_std, tpr_std)
# logarithm transformation
fpr_log, tpr_log = eval_nb_roc(trainx_log, trainy, testx_log, testy)
roc_auc_log = auc(fpr_log, tpr_log)
# indicator function transformation
fpr_bin, tpr_bin = eval_nb_roc(trainx_bin, trainy, testx_bin, testy)
roc_auc_bin = auc(fpr_bin, tpr_bin)

# Create ROC plot
plt.figure()
plt.plot(fpr, tpr, lw=2, label='do_not (area = %0.3f)' % roc_auc)
plt.plot(fpr_std, tpr_std, lw=2, label='do_std (area = %0.3f)' % roc_auc_std)
plt.plot(fpr_log, tpr_log, lw=2, label='do_log (area = %0.3f)' % roc_auc_log)
plt.plot(fpr_bin, tpr_bin, lw=2, label='do_bin (area = %0.3f)' % roc_auc_bin)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for BernoulliNB')
plt.legend(loc='lower right')
plt.show()


# In[139]:


def eval_Model(trainx, trainy, testx, testy, Model):
    # Choose Gaussian, Multinomial, or Bernoulli Naive Bayes classifier
    # uncomment the one you want to try
    classifier = Model
#     nb_classifier = GaussianNB()
#     nb_classifier = MultinomialNB()
    
    # Create and train the classifier
    classifier.fit(trainx, trainy)
    
    # Make predictions on the train and test data
    trainy_predict = classifier.predict(trainx)
    testy_predict = classifier.predict(testx)
    
    # Evaluate the model
    train_accuracy = accuracy_score(trainy, trainy_predict)
    test_accuracy = accuracy_score(testy, testy_predict)
    
    # Calculate the predicted probabilities for class 1
    trainy_prob = classifier.predict_proba(trainx)[:, 1]
    testy_prob = classifier.predict_proba(testx)[:, 1]
    
    # Calculate the AUC score
    train_auc_score = roc_auc_score(trainy, trainy_prob)
    test_auc_score = roc_auc_score(testy, testy_prob)
    
    # Calculate fpr and tpr
    fpr, tpr, _ = roc_curve(testy, testy_prob)
    
    return fpr, tpr


# In[16]:


# Multinomial do_bin
fpr_bin, tpr_bin = eval_Model(trainx_bin, trainy, testx_bin, testy, MultinomialNB())
roc_auc_bin = auc(fpr_bin, tpr_bin)
# Bernoulli do_std
fpr_std, tpr_std = eval_Model(trainx_std, trainy, testx_std, testy, BernoulliNB())
roc_auc_std = auc(fpr_std, tpr_std)
# Gaussian do_log
fpr_log, tpr_log = eval_Model(trainx_log, trainy, testx_log, testy, GaussianNB())
roc_auc_log = auc(fpr_log, tpr_log)

plt.figure()
plt.plot(fpr_std, tpr_std, lw=2, label='Best Bernoulli (area = %0.3f)' % roc_auc_std)
plt.plot(fpr_log, tpr_log, lw=2, label='Best Gaussian (area = %0.3f)' % roc_auc_log)
plt.plot(fpr_bin, tpr_bin, lw=2, label='Best Multinomial (area = %0.3f)' % roc_auc_bin)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('A comparison Between Best Models of Naive Bayes')
plt.legend(loc='lower right')
plt.show()


# In[17]:


# Compute FPR and TPR for each preprocessing method
# do nothing: raw dataset
fpr, tpr = eval_Model(trainx, trainy, testx, testy, LogisticRegression())
roc_auc = auc(fpr, tpr)
# z-score normalization
fpr_std, tpr_std = eval_Model(trainx_std, trainy, testx_std, testy, LogisticRegression())
roc_auc_std = auc(fpr_std, tpr_std)
# logarithm transformation
fpr_log, tpr_log = eval_Model(trainx_log, trainy, testx_log, testy, LogisticRegression())
roc_auc_log = auc(fpr_log, tpr_log)
# indicator function transformation
fpr_bin, tpr_bin = eval_Model(trainx_bin, trainy, testx_bin, testy, LogisticRegression())
roc_auc_bin = auc(fpr_bin, tpr_bin)

# Create ROC plot
plt.figure()
plt.plot(fpr, tpr, lw=2, label='do_not (area = %0.3f)' % roc_auc)
plt.plot(fpr_std, tpr_std, lw=2, label='do_std (area = %0.3f)' % roc_auc_std)
plt.plot(fpr_log, tpr_log, lw=2, label='do_log (area = %0.3f)' % roc_auc_log)
plt.plot(fpr_bin, tpr_bin, lw=2, label='do_bin (area = %0.3f)' % roc_auc_bin)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for Logistic Regression')
plt.legend(loc='lower right')
plt.show()


# In[18]:


# Logistic Regression do_log
fpr_log, tpr_log = eval_Model(trainx_log, trainy, testx_log, testy, LogisticRegression())
roc_auc_log = auc(fpr_log, tpr_log)
# Multinomial do_bin
fpr_bin, tpr_bin = eval_Model(trainx_bin, trainy, testx_bin, testy, MultinomialNB())
roc_auc_bin = auc(fpr_bin, tpr_bin)

# Create ROC plot
plt.figure()
plt.plot(fpr_log, tpr_log, lw=2, label='Logistic Regression do_log (area = %0.3f)' % roc_auc_log)
plt.plot(fpr_bin, tpr_bin, lw=2, label='Multinomial do_bin (area = %0.3f)' % roc_auc_bin)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('A comparison between best models of naive bayes and logistic regression')
plt.legend(loc='lower right')
plt.show()


# In[71]:


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


# In[77]:


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


# In[73]:


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


# In[74]:


model = LogisticRegression()
eval_holdout(trainx, trainy, 0.2, model)


# In[75]:


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


# In[79]:


model = LogisticRegression()
eval_kfold(trainx_log, trainy, 5, model)


# In[113]:


# create a dataframe to store metrics for each hyperparameter
columns = ['C', 'valsize', 'train-acc', 'train-auc', 'val-acc', 'val-auc']
df = pd.DataFrame(columns=columns)

# create the grid for hyperparameters
C_values = np.linspace(0.01, 1, 19)
# split_ratios = [0.1, 0.15, 0.2, 0.25]
split_ratios = [0.25]

# find best metric
best_val_auc = -1
best_index = 0
i = 0

print("{:<10} {:<10}".format('C', 'valsize'))
for valsize in split_ratios:
    for C in C_values:
        # hyperparameters
        hyperparams = {'valsize':valsize, 'C':C}
        print("{:<10} {:<10}".format(C, valsize))
        
        # instantiate model
        model = LogisticRegression(penalty='l2', solver='liblinear', C=C) # 'l1' for lasso and 'l2' for ridge
        metrics = eval_holdout(trainx_log, trainy, valsize, model)
        if metrics['val-auc'] >= best_val_auc:
            best_val_auc = metrics['val-auc']
            best_index = i
        
        # add new row
        row = {}
        row.update(hyperparams)
        row.update(metrics)
        df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
        
        i += 1


print('best model', best_index)
df


# In[123]:


# create a dataframe to store metrics for each hyperparameter
columns = ['C', 'k', 'train-acc', 'train-auc', 'val-acc', 'val-auc']
df = pd.DataFrame(columns=columns)

# create the grid for hyperparameters
C_values = np.arange(0.1, 5, 0.2)
k_values = [10]

# find best metric
best_val_auc = -1
best_index = 0
i = 0

print("{:<10} {:<10}".format('C', 'k'))
for k in k_values:
    for C in C_values:
        # hyperparameters
        hyperparams = {'k':k, 'C':C}
        print("{:<10} {:<10}".format(C, k))
        
        # instantiate model
        model = LogisticRegression(penalty='l2', solver='liblinear', C=C) # 'l1' for lasso and 'l2' for ridge
        metrics = eval_kfold(trainx_log, trainy, k, model)
        if metrics['val-auc'] >= best_val_auc:
            best_val_auc = metrics['val-auc']
            best_index = i
        
        # add new row
        row = {}
        row.update(hyperparams)
        row.update(metrics)
        df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
        
        i += 1
        


# In[124]:


print(best_index)
df


# In[126]:


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
    


# In[28]:


model = LogisticRegression()
eval_mccv(trainx_log, trainy, 0.2, 10, model)


# In[129]:


# create a dataframe to store metrics for each hyperparameter
columns = ['C', 's', 'valsize', 'train-acc', 'train-auc', 'val-acc', 'val-auc']
df = pd.DataFrame(columns=columns)

# create the grid for hyperparameters
C_values = np.arange(0.01, 0.5, 0.02)
s_values = [10]
split_ratios = [0.1]

# find best metric
best_val_auc = -1
best_index = 0
i = 0

print("{:<10} {:<10} {:<10}".format('C', 's', 'valsize'))
for s in s_values:
    for C in C_values:
        for valsize in split_ratios:
            # hyperparameters
            hyperparams = {'s':s, 'C':C, 'valsize':valsize}
            print("{:<10} {:<10} {:<10}".format(C, s, valsize))

            # instantiate model
            model = LogisticRegression(penalty='l2', solver='liblinear', C=C) # 'l1' for lasso and 'l2' for ridge
            metrics = eval_mccv(trainx_log, trainy, valsize, s, model)
            if metrics['val-auc'] >= best_val_auc:
                best_val_auc = metrics['val-auc']
                best_index = i

            # add new row
            row = {}
            row.update(hyperparams)
            row.update(metrics)
            df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)

            i += 1
        


# In[130]:


print(best_index)
df


# In[140]:


def eval_model(trainx, trainy, testx, testy, model):
    # turn off warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Train the model on the training data
    model.fit(trainx, trainy)
    
    # Make predictions on the test data
    trainy_predict = model.predict(trainx)
    testy_predict = model.predict(testx)
    
    # Evaluate the model
    train_accuracy = accuracy_score(trainy, trainy_predict)
    test_accuracy = accuracy_score(testy, testy_predict)
    
    # Calculate the predicted probabilities for class 1
    trainy_prob = model.predict_proba(trainx)[:, 1]
    testy_prob = model.predict_proba(testx)[:, 1]
    
    # Calculate the AUC score
    train_auc_score = roc_auc_score(trainy, trainy_prob)
    test_auc_score = roc_auc_score(testy, testy_prob)
    
    return {'train-acc':train_accuracy, 'train-auc':train_auc_score, 'test_acc':test_accuracy, 
           'test-auc':test_auc_score}


# In[69]:


model = LogisticRegression()
eval_model(trainx_log, trainy, testx_log, testy, model)


# In[142]:


# eval_holdout

#best_lasso
C = 0.5
model = LogisticRegression(penalty='l1', solver='liblinear', C=C) # 'l1' for lasso and 'l2' for ridge
print(eval_model(trainx_log, trainy, testx_log, testy, model))

# best ridge
C = 0.45 
model = LogisticRegression(penalty='l2', solver='liblinear', C=C) # 'l1' for lasso and 'l2' for ridge
print(eval_model(trainx_log, trainy, testx_log, testy, model))


# In[145]:


# eval_kfold

#best_lasso
C = 6.5
model = LogisticRegression(penalty='l1', solver='liblinear', C=C) # 'l1' for lasso and 'l2' for ridge
print(eval_model(trainx_log, trainy, testx_log, testy, model))

# best ridge
C = 1.1 
model = LogisticRegression(penalty='l2', solver='liblinear', C=C) # 'l1' for lasso and 'l2' for ridge
print(eval_model(trainx_log, trainy, testx_log, testy, model))


# In[146]:


# eval_mccv

#best_lasso
C = 10
model = LogisticRegression(penalty='l1', solver='liblinear', C=C) # 'l1' for lasso and 'l2' for ridge
print(eval_model(trainx_log, trainy, testx_log, testy, model))

# best ridge
C = 0.3 
model = LogisticRegression(penalty='l2', solver='liblinear', C=C) # 'l1' for lasso and 'l2' for ridge
print(eval_model(trainx_log, trainy, testx_log, testy, model))

