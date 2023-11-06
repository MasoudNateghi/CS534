import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, CategoricalNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

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

