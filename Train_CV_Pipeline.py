from sklearn.svm import SVC,SVR
import sklearn.pipeline
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import randint as sp_randint
from sklearn import naive_bayes

cenas=np.isnan(allfeatures_from_20_images_19dec2017)

allfeatures_from_20_images_19dec2017[cenas==True]=0

X_train, X_test, y_train, y_test = train_test_split(allfeatures_from_20_images_19dec2017, gt, test_size=0.20, random_state=10)

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

X_train = min_max_scaler.fit_transform(X_train);
X_test = min_max_scaler.fit_transform(X_test);

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

#Our goal is:
# Initialize nested k-fold cross-validation object
kfcv = nested_k_fold_cross_validation.NestedKFoldCrossValidation(outer_loop_fold_count=3, 
                                      inner_loop_fold_count=3,
                                      shuffle_seed=3243,
                                      outer_loop_split_seed=45,
                                      inner_loop_split_seeds=[62, 207, 516])

# Combinatorial pipeline schematic
feature_count=63
pipeline_schematic = [
    {'scaler': {
            'min_max': {},
            'standard': {}
        }
    },
    {'transform': {
            'pca': {
                'n_components': [feature_count]
            }
        }         
    },
    {'feature_selection': {
            'select_k_best': {
                'k': range(1, feature_count+1)
            },
        }
    },
       {'estimator': {
            'knn': {
                'n_neighbors': range(1,31),
            },
            'linearsvm': {
                    'sklo': LinearSVC,
                    'C' : [0.1, 1, 10]
                },
            'svm' : {
                    'sklo': SVC,
                    'C' : [0.1, 1, 10],
                    'gamma' : [0.001, 0.01, 0.1, 1, 10]
                    },
            
            'logistic_regression': {
                'random_state': [65]
            },
            'random_forest': {
                'sklo': RandomForestClassifier,
                'max_depth': range(2,6),
                'random_state': [57]
            },
            'naive_bayes': {
                'sklo': GaussianNB
            }
    }}
]

kfcv.fit(X_train, y_train, pipeline_schematic=pipeline_schematic, 
         scoring_metric='auc', score_type='median')