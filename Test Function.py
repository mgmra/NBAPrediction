# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 22:36:51 2020

@author: mateu
"""


import itertools
from collections import ChainMap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

test_d = {'C':[1,2,3], 'penalty': ['l1'], 'passthrough': True}
func = LogisticRegression

def create_pipe_grid(func_parameters):
    
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

clf_parameters = {LogisticRegression:{'C':[0.1, 1, 10], 'penalty':['l1', 'l2'], 'passthrough':True},
              RandomForestClassifier:{'max_depth':[2,5,10,None]},
              SVC:{'C':[0.1,1,10]},
              AdaBoostClassifier:{'n_estimators':[20,50,100]},
              GaussianProcessClassifier:{'max_iter_predict':[50,100]}
              }

test_x = create_func_grid(clf_parameters)
