from sklearn.svm import SVC,SVR
import sklearn.pipeline
import numpy as np

from sklearn.linear_model import LogisticRegression

X_train=x_train_images_15_features1
y_train=y_train_images_15_ground_truth1

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn import (cross_validation, feature_selection,
                     linear_model)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV 
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd

import sys
import folds, nested_k_fold_cross_validation, pipeline_builder, old, pipeline_evaluator, trained_pipeline

#Use pyplearnr with different estimators:
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#We create three boolean variables to test three differente approaches:
doRFEVC=True
doSelectKBest=True
doNothing=True

#Then, we create three conditions to test each:
#if doRFEVC==True:

if doNothing==True:
    # Combinatorial pipeline schematic
    feature_count=60
    pipeline_schematic = [
           {'scaler': {
            'min_max': {},
            'standard': {}
                }},
           
           {'estimator': {
                'knn': {
                    'n_neighbors': range(1,31,2)
                },
                'linearsvm': {
                        'sklo': LinearSVC,
                        'C' : [2**-5 , 2**-3 , 2**-1, 2, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
                    },
                'svm' : {
                        'sklo': SVC,
                        'C' : [2**-5 , 2**-3 , 2**-1, 2, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15],
                        'gamma': [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2, 2**3 ]
                        },
                'logistic_regression': {
                    'sklo' : LogisticRegression,
                    'C' : [0.001, 0.01, 0.1, 1, 10]
                },
                'naive_bayes': {
                    'sklo': GaussianNB
                }
        }}
    ]
    
    


#Our goal is:
# Initialize nested k-fold cross-validation object
kfcv = nested_k_fold_cross_validation.NestedKFoldCrossValidation(outer_loop_fold_count=5, 
                                      inner_loop_fold_count=5,
                                      shuffle_seed=3243,
                                      outer_loop_split_seed=45,
                                      inner_loop_split_seeds=[62, 207, 516, 420, 100])

kfcv.fit(X_new_1, y_train, pipeline_schematic=pipeline_schematic, 
         scoring_metric='auc', score_type='median')
