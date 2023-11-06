import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA, NMF
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, fbeta_score

def preprocess(df):
    # supress warnings
    import warnings
    warnings.filterwarnings('ignore')
    # preprocessing
    # encode grade column
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

    # apply the mapping
    df['grade'] = df['grade'].map(grade_mapping)

    # encode emp_length column
    emp_length_mapping = {'< 1 year': 1, '1 year': 2, '2 years': 3, '3 years': 4, '4 years': 5, '5 years': 6,
                          '6 years': 7, '7 years': 8, '8 years': 9, '9 years': 10, '10+ years': 11}
    # apply the mapping
    df['emp_length'] = df['emp_length'].map(emp_length_mapping)

    # fill NaN values of emp_length column
    df['emp_length'].fillna(0, inplace=True)
    
    # for i in range(len(df)):
    #     year = int(df['earliest_cr_line'][i][4:])
    #     if year > 20:
    #         df['earliest_cr_line'][i] = 2018 - (1900 + year)
    #     else:
    #         df['earliest_cr_line'][i] = 2018 - (2000 + year)
    

    # Use one-hot encoding for home_ownership, verification_status, purpose columns
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse=False)

    # Fit the encoder on the categorical column and transform it
    encoded_data = encoder.fit_transform(df[['home_ownership', 'verification_status', 'purpose', 'term']])

    # Create a new DataFrame with one-hot encoded columns
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['home_ownership', 'verification_status', 'purpose', 'term']))
    encoded_df = encoded_df.iloc[:, :-1]
    
    # remove irrelevant features
    df = df.drop(columns=['earliest_cr_line'])

    # remove redundant categorical features
    df = df.drop(columns=['home_ownership', 'verification_status', 'purpose', 'term'])

    # extract lables from dataset and remove 'class' column
    # labels = df['class'].values
    # df = df.drop('class', axis=1)

    # concat encoded dataframe with original dataframe
    # df1 = pd.concat([df, encoded_df], axis=1)
    data = pd.concat([df, encoded_df], axis=1).to_numpy()
    
    return data

def run_pca(train_x, test_x):
    # normaliztion
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    # Transform the testing data based on the fitted scaler
    test_x = scaler.transform(test_x)
    # PCA
    pca = PCA(n_components = 0.95)
    pca.fit(train_x)
    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)
    # number of PCs
    nPC = train_x.shape[1]
    # PCs
    PC = pca.components_
    return (nPC, PC, train_x, test_x)

def run_nmf(train_x, test_x, k):
    nmf = NMF(n_components=k, init='random')
    W_train = nmf.fit_transform(train_x)
    H = nmf.components_
    W_test = nmf.transform(test_x)
    train_x_reconstruct = np.dot(W_train, H)
    error = np.linalg.norm(train_x - train_x_reconstruct)
    return (error, H, W_train, W_test)