# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 00:05:47 2020

@author: mateu
"""
#%% imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import statsmodels.api as sm 
#%% data reading 
#link: https://towardsdatascience.com/the-art-of-finding-the-best-features-for-machine-learning-a9074e2ca60d
#df = pd.read_csv('Output/df_ml.csv')
df = pd.read_hdf('Output/ml_hdf.h5', key='match')
# use the pands .corr() function to compute pairwise correlations for the dataframe
df.dropna(inplace=True)

#SAMPLING HERE to reduce computation time
df = df.sample(n=500)

target = "PTS_Result"
corr = df.corr()
corr[target] = abs(corr[target])
corr = corr.sort_values(target, ascending=False)

#prepare lower trainagular matrix (with deleted diagonal)
low_tri_no_diag = np.tril(np.ones(corr.shape).astype(np.bool), k=-1)
corr = corr.where(low_tri_no_diag)
#drop tagret variable form rows and columns
corr = corr.drop(corr.columns[0], axis=1).drop(corr.index[0], axis=0)

corr['MaxCorr'] = corr.abs().max(axis = 1)


'''
return Index (or MultiIndex) of dataframe containing features with highest correlation to target variable
and correlation with other features lower than threshold set
'''
def preselect_non_redundant_features(df, target, corr_threshold):
    corr = df.corr()
    corr[target] = abs(corr[target])
    corr = corr.sort_values(target, ascending=False)

    #prepare lower trainagular matrix (with deleted diagonal)
    low_tri_no_diag = np.tril(np.ones(corr.shape).astype(np.bool), k=-1)
    corr = corr.where(low_tri_no_diag)
    #drop tagret variable form rows and columns
    corr = corr.drop(corr.columns[0], axis=1).drop(corr.index[0], axis=0)

    corr['MaxCorr'] = corr.abs().max(axis = 1)

    pre_selected_features = corr[corr['MaxCorr'] < corr_threshold].index 
    return pre_selected_features
    
#a03 = preselect_non_redundant_features(df, 'PTS_Result', 0.3)
#a05 = preselect_non_redundant_features(df, 'PTS_Result', 0.5)
a = preselect_non_redundant_features(df, 'PTS_Result', 0.5)
#a07 = preselect_non_redundant_features(df, 'PTS_Result', 0.7)

#pick pre-selected columns
df = df.loc[:,a]


y = df.reset_index().dropna()['Won_Result'].values

X_train = df[~df.index.isin(['2019-20'], level = 'season')]
y_train = df[~df.index.isin(['2019-20'], level = 'season')].reset_index().dropna()['Won_Result'].values

X_test = df[df.index.isin(['2019-20'], level = 'season')]
y_test = df[df.index.isin(['2019-20'], level = 'season')].reset_index().dropna()['Won_Result'].values                        







from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pprint as pp
import itertools


#pp.pprint(sorted(classifier.get_params().keys()))


#https://www.tomasbeuzen.com/post/scikit-learn-gridsearch-pipelines/

def create_pipe_grid(func_parameters):
    '''
    Parameters
    ----------
    func_parameters : dict
        dictionary in form of: 
        {sklearn method : {param_name1 : [param_val1, param_val2..], param_name2: [param_val1, param_val2], 'passthrough':True}}
    'passthrough' key is optional. If its set to True 'passthrough' for sklearn Pipeline will be added to specific step
    Returns
    multi_grid: list
        List of arguments (invoked sklearn methods with specific parameters) to be searched through by pipeline
    -------
    '''
    
    multi_grid = []
    for func, parameters_dict in func_parameters.items():
            
        main_list=[]
        passthrough = False
        
        for key in list(parameters_dict.keys()):
            
            if key == 'passthrough':
                if parameters_dict[key] == True: 
                    passthrough = True
                else: pass
        
            else: 
                dicts_list=[]
                for i in range(len(list(parameters_dict[key]))):
                    param_dict = {key:list(parameters_dict[key])[i]}
                    dicts_list.append(param_dict)
                main_list.append(dicts_list)
        
        param_variant = list(itertools.product(*main_list))
        
        
        param_variants = [dict(ChainMap(*param_variant[i])) for i in range(len(param_variant))]
        
        grid = []
        for variant in param_variants:
            grid.append(func(**variant))
       
        #add passthrough keyword if it appeared as argument in dictionary and value was True
        if passthrough == True:
            grid.append('passthrough')
        
        multi_grid.extend(grid)
        
    return(multi_grid)

scaler = [StandardScaler(), 'passthrough']

pca_parameters = {PCA : {'n_components':[5], 'passthrough':True}}
pca = create_pipe_grid(pca_parameters)

selector_parameters = {SelectKBest: {'k':[5,10,15,25], 'score_func':[f_classif, mutual_info_classif], 'passthrough':True}}

selector = create_pipe_grid(selector_parameters)


clf_parameters = {LogisticRegression:{'C':[0.1, 1, 10], 'penalty':['none', 'l2']},
              RandomForestClassifier:{'max_depth':[2,5,10,None]},
              SVC:{'C':[0.1,1,10]},
              AdaBoostClassifier:{'n_estimators':[20,50,100]},
              GaussianProcessClassifier:{'max_iter_predict':[50,100]}
              }

clf = create_pipe_grid(clf_parameters)



''' ZALĄŻEK NOWEJ FUNKCJI'''

results = []
for scaler_ in scaler :
    for selector_ in selector:
            for clf_ in clf:
                pipe = Pipeline(steps = [('scaler', scaler_), ('selector', selector_), ('clf', clf_)]) 
                pipe.fit(X_train, y_train)
                score = pipe.score(X_test, y_test)

                results.append([scaler_, selector_, clf_, score])





''' Function composing pipeline for separate dicts containing parameters'''



    
"""
search_space = [
                {'classifier': [LogisticRegression(solver='lbfgs')],
                 'classifier__C': [0.01, 0.1, 1.0]},
                {'classifier': [RandomForestClassifier(n_estimators=100)],
                 'classifier__max_depth': [5, 10, None]},
                {'classifier': [KNeighborsClassifier()],
                 'classifier__n_neighbors': [3, 7, 11],
                 'classifier__weights': ['uniform', 'distance']}]
"""


estimators = [('selector', SelectKBest()), ('scaler', StandardScaler()), ('clf', SVC())]
pipe = Pipeline(estimators)


grid_search = GridSearchCV(pipe, param_grid, verbose=5)


grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)

cv_results = grid_search.cv_results_
cv_results_df = pd.DataFrame(cv_results).sort_values(by=['rank_test_score'])


test_pipe = Pipeline(
                     [('selector', selector),
                     ('scaler', cv_results_df['param_scaler'].iloc[1]),
                     ('clf', cv_results_df['param_clf'].iloc[1])]
                     )
test_pipe.fit(X_train, y_train)
y_pred_test = test_pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred_test)
print(acc)

print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))


#Ridge Regression
#For classification: chi2, f_classif, mutual_info_classif

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(grid_search.best_params_)


# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = grid_search.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()




####autoML

#df = pd.read_csv('Output/df_ml.csv')
df = pd.read_hdf('Output/ml_hdf.h5', key='match')
# use the pands .corr() function to compute pairwise correlations for the dataframe
df.dropna(inplace=True)

a = preselect_non_redundant_features(df, 'PTS_Result', 0.7)

df = df.loc[:,a]
y = df.reset_index().dropna()['Won_Result'].values

X_train = df[~df.index.isin(['2019-20'], level = 'season')]
y_train = df[~df.index.isin(['2019-20'], level = 'season')].reset_index().dropna()['Won_Result'].values

X_test = df[df.index.isin(['2019-20'], level = 'season')]
y_test = df[df.index.isin(['2019-20'], level = 'season')].reset_index().dropna()['Won_Result'].values                        


# example of tpot for a classification dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier
# define dataset

# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# define search
model = TPOTClassifier(generations=5, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
# perform the search
model.fit(X_train, y_train)
# export the best model
acc = model.score(X_test, y_test)
print(acc)

















