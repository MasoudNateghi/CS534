#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score, fbeta_score


# In[2]:


def read_file(filename):
    emails = list()
    label = list()
    with open(filename, 'r') as f:
        for line in f:
            row = line.strip().split()
            label.append(int(row[0]))
            emails.append(row[1:])
    return emails, label

emails, labels = read_file('spamAssassin.data')
labels = [-1 if x == 0 else x for x in labels]


# In[8]:


# split the dataset into training and testing emails
train_mails, test_mails, trainy, testy = train_test_split(emails, labels, test_size=0.15, random_state=42)
train_mails, val_mails, trainy, valy = train_test_split(train_mails, trainy, test_size=0.18, random_state=42)


# In[10]:


print(len(train_mails))
print(len(test_mails))
print(len(val_mails))


# In[14]:


def build_vocab(train, test, minn): 
    # STEP 1
    # create document frequency dictionary
    dfs = {}
    
    # go through each training email
    for i in range(len(train)):
        print(i)
        # go through each words of each email
        for word in train[i]:
            # if the word is not present in documents frequncy then create the word key with value = email index 
            if not word in dfs.keys():
                dfs[word] = [i]
            # if the word is present in documents frequncy then store email index if the email is a new one 
            elif not i in dfs[word]:
                dfs[word].append(i)
                
                
    # create vocab list
    vocab = []
    
    for word, ds in dfs.items():
        # if frequency of the word among documents > minn add to vocab list
        if len(ds) >= minn:
            vocab.append(word)
    print(len(vocab))
    print('-------------------------------------------')
    # STEP 2       
    # create feature matrix for train and test emails
    p = len(vocab) # number of features
    n = len(train) # number of train emails
    m = len(test) # number of test emails
    x_train = np.zeros((n, p))
    x_test = np.zeros((m, p))
    
    # calculate train features
    for i in range(n):
        print(i)
        for j in range(p):
            if vocab[j] in train[i]:
                x_train[i, j] = 1
    
    # calculate test features
    for i in range(m):
        print(i)
        for j in range(p):
            if vocab[j] in test[i]:
                x_test[i, j] = 1
                
    # STEP 3
    feature_map = {}
    for i in range(p):
        feature_map[vocab[i]] = i
        
    return x_train, x_test, feature_map


# In[11]:


class Perceptron():

    def __init__(self, epoch):
        # initialization
        self.epoch = epoch
        return

    def get_weight(self):
        return self.w

    def sample_update(self, x, y):
        # compute activation value
        a = np.dot(self.w, x)
        
        # update weights in case of error
        err = 0
        if a * y <= 0:
            err = 1 # sample error
            self.w += y * x
        return self.w, err

    def train(self, trainx, trainy):
        # number of features
        n_features = trainx.shape[1]
        
        # number of training samples
        n_samples = trainx.shape[0]
        
        # weight initialization
#         self.w = np.zeros(n_features)
        self.w = np.random.rand(n_features)
        
        # track mistakes for each epoch in training process
        mistakes = {}
        
        # if reached maximum number of iterations stop training
        for i in range(self.epoch):
            print(i)
            #track total sample error
            error = 0
            
            for j in range(n_samples):
                # sample update 
                self.w, sample_err = self.sample_update(trainx[j, :], trainy[j])
                
                # update error
                error += sample_err
                
            # record mistake for each epoch
            mistakes[i + 1] = error
            
            # stop training if error = 0
            if error == 0:
                break
        return mistakes

    def predict(self, newx):
        # number of test samples
        m = newx.shape[0]
        
        # compute activation
        a = np.dot(newx, self.w)
        
        # apply sign function
        for i in range(m):
            if a[i] >= 0:
                a[i] = 1
            else:
                a[i] = -1
                
        return a


# In[22]:


len(train_mails)


# In[23]:


# parameters of the algorithm
minn_params = [10, 100, 1000]
epoch_params = np.arange(1, 21)

hist_train_all = []
hist_val_mistakes = []
data_hist = []

for minn in minn_params:
    # build features 
    trainx, valx, vocab_dict = build_vocab(train_mails, val_mails, minn=minn)
    
    # add a coulmn of 1s as bias
    bias_train = np.ones((trainx.shape[0], 1))
    bias_val = np.ones((valx.shape[0], 1))
    trainx = np.hstack((bias_train, trainx))
    valx = np.hstack((bias_val, valx))
    
    for epoch in epoch_params:
        # create perceptron model
        model_perceptron = Perceptron(epoch)
        
        # store error for each epoch
        hist_train = model_perceptron.train(trainx, trainy)
        
        # store error in a list to compare different models later
        hist_train_all.append(hist_train)
        
        # predict validation labels
        valy_predict = model_perceptron.predict(valx)
        
        # calculate number of mistakes on validation data
        val_mistakes = np.count_nonzero(valy_predict * valy == -1)
        
        # store number of validation mistakes in a list to compare models later
        hist_val_mistakes.append(val_mistakes)
        


# In[28]:


"""
I am really sorry if the following lines of code might make no sense due to bad naming of variables. Basically what
it does is to plot errors for a fixed minn (denoted by colors) in the graph and x-axis is the maximum epoch. 
"""

# number of parameters
n_minn = len(minn_params)
n_epoch = len(epoch_params)
color = [['#f7a899', '#ff5733'], ['#a9dfbf', '#5cb85c'], ['#a9d8e6', '#33a1de']]


for i in range(n_minn):
    # history of train mistakes for each minn
    hist_train = hist_train_all[i * n_epoch:(i + 1) * n_epoch]
    
    # history of validation mistakes for each minn
    hist_val_mistakes_minn = hist_val_mistakes[i * n_epoch:(i + 1) * n_epoch]
    
    # history of mistakes at last epoch
    hist_train_mistakes = []
    
    # color of the plot
    colors = color[i]
    
    for j in range(n_epoch):
        # last epoch 
        n = len(hist_train[j])
        
        # store train mistakes in a list
        hist_train_mistakes.append(hist_train[j][n])
    
    plt.plot(epoch_params, hist_train_mistakes, label='train err minn = ' + str(minn_params[i]), color=colors[0])
    plt.plot(epoch_params, hist_val_mistakes_minn, label='val err minn = ' + str(minn_params[i]), color=colors[1])
    
plt.legend()
plt.xticks(np.arange(1, 20 + 1, 1))
plt.xlabel('Maximum epoch')
plt.ylabel('Number of Mistakes')
plt.title('Error for Perceptron Model with hyperparameters epoch and minn')
plt.show()
        
    


# In[58]:


trainx.shape


# In[40]:


# combine train and val
data = train_mails + val_mails
labels = trainy + valy

# build vocab again
trainx, testx, vocab_dict = build_vocab(data, test_mails, minn=100)

# add a coulmn of 1s as bias
bias_train = np.ones((trainx.shape[0], 1))
bias_test = np.ones((testx.shape[0], 1))
trainx = np.hstack((bias_train, trainx))
testx = np.hstack((bias_test, testx))


# In[46]:


# create perceptron model
model_perceptron = Perceptron(20)

# store error for each epoch
hist_train = model_perceptron.train(trainx, labels)


# In[48]:


print('history of training: ', hist_train)
print('-------------------------------')
print('number of mistakes on training set: ', sum(hist_train.values()))


# In[49]:


# evaluate model
# make prediction on test data
testy_pred = model_perceptron.predict(testx)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(testy, testy_pred)
precision = precision_score(testy, testy_pred)
recall = recall_score(testy, testy_pred)
f1 = f1_score(testy, testy_pred)
f2 = fbeta_score(testy, testy_pred, beta = 2)

print('Accuracy: ', accuracy)
print('precision: ', precision)
print('recall: ', recall)
print('F1-score: ', f1)
print('F2-score: ', f2)


# In[50]:


class AvgPerceptron(Perceptron):

    def get_weight(self):
        # number of weights
        n_weight = len(self.w_hist)
        
        # find the average weight
        avg_weight = 0
        for i in range(n_weight):
            avg_weight += self.w_hist[i] / n_weight 
        return avg_weight

    def train(self, trainx, trainy):
        # number of features
        n_features = trainx.shape[1]
        
        # number of training samples
        n_samples = trainx.shape[0]
        
        # weight initialization
#         self.w = np.zeros(n_features)
        self.w = np.random.rand(n_features)
        
        # weight history
        self.w_hist = []
        
        # track mistakes for each epoch in training process
        mistakes = {}
        
        # if reached maximum number of iterations stop training
        for i in range(self.epoch):
            print(i)
            #track total sample error
            error = 0
            
            for j in range(n_samples):
                # sample update 
                self.w, sample_err = self.sample_update(trainx[j, :], trainy[j])
                
                # store weight for averaging
                self.w_hist.append(self.w)
                
                # update error
                error += sample_err
                
            # record mistake for each epoch
            mistakes[i + 1] = error
            
            # stop training if error = 0
            if error == 0:
                break
        return mistakes

    def predict(self, newx):
        # number of test samples
        m = newx.shape[0]
        
        # retrieve learned weights
        weights = self.get_weight()
        
        # compute activation
        a = np.dot(newx, weights)
        
        # apply sign function
        for i in range(m):
            if a[i] >= 0:
                a[i] = 1
            else:
                a[i] = -1
                
        return a


# In[51]:


# parameters of the algorithm
minn_params = [10, 100, 1000]
epoch_params = np.arange(1, 21)

hist_train_all = []
hist_val_mistakes = []
data_hist = []

for minn in minn_params:
    # build features 
    trainx, valx, vocab_dict = build_vocab(train_mails, val_mails, minn=minn)
    
    # add a coulmn of 1s as bias
    bias_train = np.ones((trainx.shape[0], 1))
    bias_val = np.ones((valx.shape[0], 1))
    trainx = np.hstack((bias_train, trainx))
    valx = np.hstack((bias_val, valx))
    
    for epoch in epoch_params:
        # create perceptron model
        model_perceptron = AvgPerceptron(epoch)
        
        # store error for each epoch
        hist_train = model_perceptron.train(trainx, trainy)
        
        # store error in a list to compare different models later
        hist_train_all.append(hist_train)
        
        # predict validation labels
        valy_predict = model_perceptron.predict(valx)
        
        # calculate number of mistakes on validation data
        val_mistakes = np.count_nonzero(valy_predict * valy == -1)
        
        # store number of validation mistakes in a list to compare models later
        hist_val_mistakes.append(val_mistakes)
        


# In[52]:


"""
I am really sorry if the following lines of code might make no sense due to bad naming of variables. Basically what
it does is to plot errors for a fixed minn (denoted by colors) in the graph and x-axis is the maximum epoch. 
"""

# number of parameters
n_minn = len(minn_params)
n_epoch = len(epoch_params)
color = [['#f7a899', '#ff5733'], ['#a9dfbf', '#5cb85c'], ['#a9d8e6', '#33a1de']]


for i in range(n_minn):
    # history of train mistakes for each minn
    hist_train = hist_train_all[i * n_epoch:(i + 1) * n_epoch]
    
    # history of validation mistakes for each minn
    hist_val_mistakes_minn = hist_val_mistakes[i * n_epoch:(i + 1) * n_epoch]
    
    # history of mistakes at last epoch
    hist_train_mistakes = []
    
    # color of the plot
    colors = color[i]
    
    for j in range(n_epoch):
        # last epoch 
        n = len(hist_train[j])
        
        # store train mistakes in a list
        hist_train_mistakes.append(hist_train[j][n])
    
    plt.plot(epoch_params, hist_train_mistakes, label='train err minn = ' + str(minn_params[i]), color=colors[0])
    plt.plot(epoch_params, hist_val_mistakes_minn, label='val err minn = ' + str(minn_params[i]), color=colors[1])
    
plt.legend()
plt.xticks(np.arange(1, 20 + 1, 1))
plt.xlabel('Maximum epoch')
plt.ylabel('Number of Mistakes')
plt.title('Error for Average Perceptron Model with hyperparameters epoch and minn')
plt.show()
        
    


# In[73]:


# combine train and val
data = train_mails + val_mails
labels = trainy + valy

# build vocab again
trainx, testx, vocab_dict = build_vocab(data, test_mails, minn=100)

# add a coulmn of 1s as bias
bias_train = np.ones((trainx.shape[0], 1))
bias_test = np.ones((testx.shape[0], 1))
trainx = np.hstack((bias_train, trainx))
testx = np.hstack((bias_test, testx))


# In[68]:


# create perceptron model
model_perceptron_avg = Perceptron(20)

# store error for each epoch
hist_train = model_perceptron_avg.train(trainx, labels)


# In[69]:


print('history of training: ', hist_train)
print('-------------------------------')
print('number of mistakes on training set: ', sum(hist_train.values()))


# In[70]:


# evaluate model
# make prediction on test data
testy_pred = model_perceptron_avg.predict(testx)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(testy, testy_pred)
precision = precision_score(testy, testy_pred)
recall = recall_score(testy, testy_pred)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(testy, testy_pred)
precision = precision_score(testy, testy_pred)
recall = recall_score(testy, testy_pred)
f1 = f1_score(testy, testy_pred)
f2 = fbeta_score(testy, testy_pred, beta = 2)

print('Accuracy: ', accuracy)
print('precision: ', precision)
print('recall: ', recall)
print('F1-score: ', f1)
print('F2-score: ', f2)


# In[71]:


# turn off warnings
import warnings
warnings.filterwarnings('ignore')

# Get the indices that would sort the array
sorted_indices_positives = np.argsort(model_perceptron.get_weight())[::-1]
sorted_indices_negatives = np.argsort(model_perceptron.get_weight())


# In[72]:


words_pos_weights = sorted_indices_positives[:15]
words_neg_weights = sorted_indices_negatives[:15]

# Reverse lookup to find associated words
associated_words_pos = [key for key, value in vocab_dict.items() if value in words_pos_weights]
associated_words_neg = [key for key, value in vocab_dict.items() if value in words_neg_weights]
print('Words with the most positive weights: ', associated_words_pos)
print('Words with the most negative weights: ', associated_words_neg)

