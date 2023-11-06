import numpy as np


def read_file(filename):
    emails = list()
    label = list()
    with open(filename, 'r') as f:
        for line in f:
            row = line.strip().split()
            label.append(int(row[0]))
            emails.append(row[1:])
    return emails, label


def build_vocab(train, test, minn): 
    # STEP 1
    # create document frequency dictionary
    dfs = {}
    
    # go through each training email
    for i in range(len(train)):
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
    # STEP 2       
    # create feature matrix for train and test emails
    p = len(vocab) # number of features
    n = len(train) # number of train emails
    m = len(test) # number of test emails
    x_train = np.zeros((n, p))
    x_test = np.zeros((m, p))
    
    # calculate train features
    for i in range(n):
        for j in range(p):
            if vocab[j] in train[i]:
                x_train[i, j] = 1
    
    # calculate test features
    for i in range(m):
        for j in range(p):
            if vocab[j] in test[i]:
                x_test[i, j] = 1
                
    # STEP 3
    feature_map = {}
    for i in range(p):
        feature_map[vocab[i]] = i
        
    return x_train, x_test, feature_map


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
        self.w = np.random.rand(n_features)
        
        # track mistakes for each epoch in training process
        mistakes = {}
        
        # if reached maximum number of iterations stop training
        for i in range(self.epoch):
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



