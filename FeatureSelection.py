# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 00:05:47 2020

@author: mateu
"""
#%% imports
import pandas as pd
import time
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
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
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
from collections import ChainMap
from sklearn.model_selection import cross_validate
import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import random

'''

return Index (or MultiIndex) of dataframe containing features with highest correlation to target variable
and correlation with other features lower than threshold set
'''

def seed_everything(seed=21):
    """"
    Seed everything.
    """   
    random.seed(seed)
    np.random.seed(seed)

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
    
def create_pipe_grid(func_parameters, indicator):
    '''
    Parameters
    ----------
    func_parameters : dict, string
        dictionary in form of: 
        {sklearn method : {param_name1 : [param_val1, param_val2..], param_name2: [param_val1, param_val2], 'passthrough':True}}
    'passthrough' key is optional. If its set to True 'passthrough' for sklearn Pipeline will be added to specific step
    Indicator is a string tied to parameters to indicate name of the step, e.g. when multiple classifiers are tested 'clf'
    is a good indicator, 'selector' for selector tests.
    Returns
    multi_grid: list
        List of arguments (invoked sklearn methods with specific parameters) to be searched through by pipeline
    -------
    '''
    
    multi_grid = []
    grid_step_dict = {}
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
        
        grid_step_dict[indicator] = multi_grid
        
    return(grid_step_dict)

def cluster_data(X_train, y_train, X_test, y_test, k=2, pca_components=3):
    
    X_train_sc, X_test_sc = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)
    
    pca = PCA(n_components=pca_components)
    pca_comps_train = pca.fit_transform(X_train_sc)
    pca_comps_test = pca.fit_transform(X_test_sc)
    
    pca_comps_train_df = pd.DataFrame(pca_comps_train)
    pca_comps_test_df = pd.DataFrame(pca_comps_test)
    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pca_comps_train_df)
    cluster_labels_train = kmeans.predict(pca_comps_train_df)
    cluster_labels_test = kmeans.predict(pca_comps_test_df)
    
    cluster_data = {}
    for k in range(k):
        X_train_k = X_train[cluster_labels_train == k]
        y_train_k = y_train[cluster_labels_train == k]
        X_test_k = X_test[cluster_labels_test == k]
        y_test_k = y_test[cluster_labels_test == k]
        
        data_k = [X_train_k, y_train_k, X_test_k, y_test_k]
        cluster_data[k] = data_k
    
    return cluster_data  

def get_non_redundant_df(df, target, corr_threshold):
    features = preselect_non_redundant_features(df, target, corr_threshold)
    output = df.loc[:,features]
    return output

def feature_based_train_test_split(df, target, splitting_feature, test_split_value, 
                                   splitting_feature_as_index=True):
    if splitting_feature_as_index == True:
        X_train = df[~df.index.isin([test_split_value], level = splitting_feature)]
        X_test = df[df.index.isin([test_split_value], level = splitting_feature)]
                                  
        y_train = df[~df.index.isin([test_split_value], level = splitting_feature)].reset_index()[target].values
        y_test = df[df.index.isin([test_split_value], level = splitting_feature)].reset_index()[target].values
                   
    else:
        X_train = df[~df[splitting_feature]==test_split_value]
        X_test = df[df[splitting_feature]==test_split_value]
        
        y_train = df[~df[splitting_feature]==test_split_value].reset_index()[target].values
        y_test = df[df[splitting_feature]==test_split_value].reset_index()[target].values
        
    return X_train, y_train, X_test, y_test

def pipeline_search_report(data_list, data_desc, *kwargs):
    
    steps_lists = [list(arg.values())[0] for arg in kwargs]
    step_names = [list(arg.keys())[0] for arg in kwargs] 
    results = []
    
    X_train = data_list[0]
    y_train = data_list[1]
    X_test = data_list[2]
    y_test = data_list[3]
    
    for pipe_steps in itertools.product(*steps_lists):
        
        time_start = time.time()
        steps = list(zip(step_names, pipe_steps)) 
        pipe = Pipeline(steps)
        pipe.fit(X_train, y_train)
        cv = cross_validate(pipe, X_train, y_train, cv=5)
        time_end = time.time()
        model_duration = time_end - time_start
        
        cv_scores = [np.mean(cv[key]) for key in cv.keys()]
        test_score = pipe.score(X_test, y_test)
        train_size = X_train.shape
        test_size = X_test.shape
        timestamp = datetime.datetime.now()
        if pipe['selector'] != 'passthrough':
            mask = pipe['selector'].get_support()
            selected_features = X_train.columns[mask]
        else:
            selected_features = X_train.columns
        
        results.append([timestamp, model_duration, data_desc, selected_features, *pipe_steps, *cv_scores, test_score, train_size, test_size, pipe_steps])
        
            
    cv_score_types = ['cv_' + k for k in cv.keys()]
    col_names = ['Timestamp', 'Duration', 'Data_description', 'Selected_features', *step_names, *cv_score_types, 'Test_score', 'Train_size', 'Test_size', 'PipeSteps_Tech']
    
    result_df = pd.DataFrame(results, columns=col_names)
    result_df = result_df.sort_values(by=['Test_score'], ascending=False)
    
    return result_df
 
def cluster_pipeline_search_report(clustered_data_dict, data_desc, *kwargs):
    
    #steps_lists = [list(arg.values())[0] for arg in kwargs]
    steps_lists = [list(arg.values())[0] for arg in kwargs]
    step_names = [list(arg.keys())[0] for arg in kwargs] 
    results = []
    for k in clustered_data_dict.keys():
        X_train = clustered_data_dict[k][0]
        y_train = clustered_data_dict[k][1]
        X_test = clustered_data_dict[k][2]
        y_test = clustered_data_dict[k][3]
        
        for pipe_steps in itertools.product(*steps_lists):
            time_start = time.time()
            steps = list(zip(step_names, pipe_steps)) 
            pipe = Pipeline(steps)
            pipe.fit(X_train, y_train)
            cv = cross_validate(pipe, X_train, y_train, cv=5)
            time_end = time.time()
            model_duration = time_end - time_start
        
            cv_scores = [np.mean(cv[key]) for key in cv.keys()]
            test_score = pipe.score(X_test, y_test)
            train_size = X_train.shape
            test_size = X_test.shape
            timestamp = datetime.datetime.now()
            if pipe['selector'] != 'passthrough':
                mask = pipe['selector'].get_support()
                selected_features = X_train.columns[mask]
            else:
                selected_features = X_train.columns
        
            results.append([timestamp, model_duration, data_desc, k, selected_features, *pipe_steps, *cv_scores, test_score, train_size, test_size, pipe_steps])

    cv_score_types = ['cv_' + key for key in cv.keys()]
    col_names = ['Timestamp', 'Duration', 'Data_description', 'Cluster', 'Selected_features', *step_names, *cv_score_types, 'Test_score', 'Train_size', 'Test_size', 'PipeSteps_Tech']
    
    result_df = pd.DataFrame(results, columns=col_names)
    result_df = result_df.sort_values(by=['Test_score'], ascending=False)
    
    return result_df

def write_report(df, description, filepath='Output/Reports/'):
    max_test_score = str(np.round(df['Test_score'].max(),3))
    max_timestamp = str(df['Timestamp'].max().date().strftime("%d%m%Y"))
    path_string = filepath + "PipeReport_" + description + "_" + max_test_score + "_" + max_timestamp + '_' + '.csv'
    df.to_csv(path_string)
    return path_string


seed_everything(42)

###PIPELINE MODELS, PARAMETERS
scaler_parameters = [StandardScaler(), MinMaxScaler(), 'passthrough']
scaler = {'scaler':scaler_parameters}

pca_parameters = {PCA : {'n_components':[3], 'passthrough':True}}
pca = create_pipe_grid(pca_parameters, 'PCA')


selector_parameters = {SelectKBest: {'k':[5,10], 'score_func':[f_classif], 'passthrough':False}}
selector = create_pipe_grid(selector_parameters, 'selector')


clf_parameters = {KNeighborsClassifier:{'n_neighbors':[3,5]},
              LogisticRegression:{'C':[0.1, 1, 5], 'penalty':['none', 'l2']},
              RandomForestClassifier:{'max_depth':[2,5,10]},
              SVC:{'C':[0.1,1]},
              DecisionTreeClassifier:{'max_depth':[5,10,25]},
              #MLPClassifier:{'alpha':[1], 'max_iter':[1000]},
              GaussianNB:{'var_smoothing':[0.000000001]},
              AdaBoostClassifier:{'n_estimators':[20,50,100]},
              GaussianProcessClassifier:{'max_iter_predict':[50,100]},
              QuadraticDiscriminantAnalysis:{'tol':[0.0001,0.001]}
              }

#clf_parameters = {LogisticRegression:{'C':[1, 10], 'penalty':['l2']}}

clf = create_pipe_grid(clf_parameters, 'clf')



###Read Data
df = pd.read_hdf('Data/preprocessed/df_ml_hdf.h5', key='match')

# use the pands .corr() function to compute pairwise correlations for the dataframe
df.dropna(inplace=True)

#df = get_non_redundant_df(df, 'PTS_Result', 0.5) 
       
#SAMPLING HERE to reduce computation time
#df = df.sample(n=1000)

X_train, y_train, X_test, y_test = feature_based_train_test_split(df, 'Won_Result', 'season', '2019-20') 


#CLUSTERING THE DATA
clustered_data = cluster_data(X_train, y_train, X_test, y_test)

pipe_report = pipeline_search_report([X_train, y_train, X_test, y_test], scaler, selector, clf)
write_report(pipe_report, 'Full_Seed_21')


clustered_result_pipe_search = cluster_pipeline_search_report(clustered_data,'Test', scaler, selector, clf)
write_report(clustered_result_pipe_search, 'Clusterized_Seed_21')


#data_description = 'TEST_all'
#filepath='Output/Reports/'
#max_test_score = str(np.round(result_df['Test_score'].max(),2))
#max_timestamp = str(result_df['Timestamp'].max().date().strftime("%d%m%Y"))
#result_df.to_csv(filepath + "PipeReport_Clustered_"+ max_test_score + "_" + max_timestamp + '_' + '.csv')


