#%% import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
# read files
# List of non-standard representations for missing values
non_standard_missing_values = ['?', '--', ' ', 'NA', 'N/A', '-']

# Reading the data from the specified CSV file into a pandas DataFrame using the 'na_values' parameter
df = pd.read_csv('Data/cardio_data_processed.csv', na_values=non_standard_missing_values)

# Drop the excessive columns from the DataFrame
df = df.drop(columns=['id', 'bp_category_encoded'])

# one-hot encoding of categorical variable
df = pd.get_dummies(df, columns=['bp_category'], prefix='bp_category')

# Counting the number of duplicate data samples
duplicate_datasamples_sum = df.duplicated().sum()

# Check if there are duplicate data samples
if duplicate_datasamples_sum:
    print('Number of duplicate data samples in the Dataset : {}'.format(duplicate_datasamples_sum))
    print('________________________________________________________________')

    # Identify the duplicate data samples
    duplicate = df[df.duplicated(keep=False)]
    # Sort the duplicates based on specified columns
    duplicate = duplicate.sort_values(by=['gender', 'height', 'weight'], ascending=False)
    print(duplicate)

    # Remove duplicates, keeping the first occurrence
    df.drop_duplicates(keep='first', inplace=True)

    # Display information about the DataFrame after removing duplicates
    print('________________________________________________________________')
    print('Total data samples after removing duplicates: {} | Variables (Features + Label): {}'.format(df.shape[0], df.shape[1]))

else:
    print('Dataset contains no duplicate data samples')
    print('________________________________________________________________')

    # Display information about the DataFrame
    print('Total data samples: {} | Variables (Features + Label): {}'.format(df.shape[0], df.shape[1]))
    
# Initial Pre-processing for 'bmi', 'height', and 'weight'
# This step is performed to ensure the data adheres to reasonable health-related standards.
df = df[(df['bmi'] >= 9) & (df['bmi'] <= 60) &
        (df['height'] >= 100) & (df['weight'] >= 30) &
        (df['height'] <= 250) & (df['weight'] <= 500)]

# Create a copy of the DataFrame to modify without affecting the original
df_log = df.copy()
df_std = df.copy()

# Apply the logarithm transformation on the numerical features
df_log[['height', 'weight', 'bmi']] = np.log(df_log[['height', 'weight', 'bmi']])

# Apply the StandardScalar transformation on the numerical features
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()
df_std[['height', 'weight', 'bmi']] = scaler.fit_transform(df_std[['height', 'weight', 'bmi']])

# Removing outliers from the log transformed Dataframe
# df_log = df_log[(df_log['age_years'] > df_log['age_years'].quantile(0.005)) & (df_log['age_years'] < df_log['age_years'].quantile(0.995))]
df_log = df_log[(df_log['weight'] > df_log['weight'].quantile(0.005)) & (df_log['weight'] < df_log['weight'].quantile(0.995))]
df_log = df_log[(df_log['height'] > df_log['height'].quantile(0.005)) & (df_log['height'] < df_log['height'].quantile(0.995))]
df_log = df_log[(df_log['bmi'] > df_log['bmi'].quantile(0.005)) & (df_log['bmi'] < df_log['bmi'].quantile(0.995))]
# df_log = df_log[(df_log['ap_lo'] > df_log['ap_lo'].quantile(0.005)) & (df_log['ap_lo'] < df_log['ap_lo'].quantile(0.995))]
# df_log = df_log[(df_log['ap_hi'] > df_log['ap_hi'].quantile(0.005)) & (df_log['ap_hi'] < df_log['ap_hi'].quantile(0.995))]

# Removing outliers from the StandardScaler transformed Dataframe
# df_std = df_std[(df_std['age_years'] > df_std['age_years'].quantile(0.005)) & (df_std['age_years'] < df_std['age_years'].quantile(0.995))]
df_std = df_std[(df_std['weight'] > df_std['weight'].quantile(0.005)) & (df_std['weight'] < df_std['weight'].quantile(0.995))]
df_std = df_std[(df_std['height'] > df_std['height'].quantile(0.005)) & (df_std['height'] < df_std['height'].quantile(0.995))]
df_std = df_std[(df_std['bmi'] > df_std['bmi'].quantile(0.005)) & (df_std['bmi'] < df_std['bmi'].quantile(0.995))]
# df_std = df_std[(df_std['ap_lo'] > df_std['ap_lo'].quantile(0.005)) & (df_std['ap_lo'] < df_std['ap_lo'].quantile(0.995))]
# df_std = df_std[(df_std['ap_hi'] > df_std['ap_hi'].quantile(0.005)) & (df_std['ap_hi'] < df_std['ap_hi'].quantile(0.995))]

# expanding features
# devide data and labels
data = df_log.drop('cardio', axis=1).to_numpy()
labels = df_log['cardio'].to_numpy()
# 3rd order polynomial
poly = PolynomialFeatures(degree=2)
spline = SplineTransformer(degree=2, n_knots=4)
data_poly = poly.fit_transform(data)
data_spline = spline.fit_transform(data)

def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=bool)     
        chromosome[:int(0.3*n_feat)]=False             
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train[:,chromosome],Y_train)         
        predictions = logmodel.predict(X_test[:,chromosome])
        scores.append(accuracy_score(Y_test,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)                                    
    return list(scores[inds][::-1]), list(population[inds,:][::-1]) 


def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    pop_nextgen = pop_after_sel
    for i in range(0,len(pop_after_sel),2):
        new_par = []
        child_1 , child_2 = pop_nextgen[i] , pop_nextgen[i+1]
        new_par = np.concatenate((child_1[:len(child_1)//2],child_2[len(child_1)//2:]))
        pop_nextgen.append(new_par)
    return pop_nextgen


def mutation(pop_after_cross,mutation_rate,n_feat):   
    mutation_range = int(mutation_rate*n_feat)
    pop_next_gen = []
    for n in range(0,len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = [] 
        for i in range(0,mutation_range):
            pos = np.random.randint(0,n_feat-1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]  
        pop_next_gen.append(chromo)
    return pop_next_gen

def generations(df,label,size,n_feat,n_parents,mutation_rate,n_gen,X_train,
                                   X_test, Y_train, Y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        print('Best score in generation',i+1,':',scores[:1])  #2
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate,n_feat)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score

from sklearn.model_selection import train_test_split
def split(df,label):
    X_tr, X_te, Y_tr, Y_te = train_test_split(df, label, test_size=0.15, random_state=42)
    return X_tr, X_te, Y_tr, Y_te

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score

classifiers = ['Logistic']

models = [LogisticRegression(max_iter = 1000)]


def acc_score(df,label):
    Score = pd.DataFrame({"Classifier":classifiers})
    j = 0
    acc = []
    X_train,X_test,Y_train,Y_test = split(df,label)
    for i in models:
        model = i
        model.fit(X_train,Y_train)
        predictions = model.predict(X_test)
        acc.append(accuracy_score(Y_test,predictions))
        j = j+1     
    Score["Accuracy"] = acc
    Score.sort_values(by="Accuracy", ascending=False,inplace = True)
    Score.reset_index(drop=True, inplace=True)
    return Score

def plot(score,x,y,c = "b"):
    gen = [1,2,3,4,5]
    plt.figure(figsize=(6,4))
    ax = sns.pointplot(x=gen, y=score,color = c )
    ax.set(xlabel="Generation", ylabel="Accuracy")
    ax.set(ylim=(x,y))
#%% perform ga   
train = np.hstack((data, data_poly, data_spline))

#%%
logmodel = LogisticRegression()
X_train,X_test, Y_train, Y_test = split(train,labels)
# normalization
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()
# Fit the scaler on the training data and transform both the training and test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
chromo_df,scores=generations(train,labels,size=80,n_feat=train.shape[1],
                                  n_parents=64,mutation_rate=0.20,n_gen=15,
                                  X_train = X_train,X_test = X_test,
                                  Y_train = Y_train,Y_test = Y_test)
#%%
