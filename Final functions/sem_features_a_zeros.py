# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 23:30:39 2017

@author: Gabriel
"""

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


# Initialize nested k-fold cross-validation object
kfcv = nested_k_fold_cross_validation.NestedKFoldCrossValidation(outer_loop_fold_count=3, 
                                      inner_loop_fold_count=3,
                                      shuffle_seed=3243,
                                      outer_loop_split_seed=45,
                                      inner_loop_split_seeds=[62, 207, 516],
                                      random_combinations=50,
                                      random_combination_seed=2374)

# Design combinatorial pipeline schematic
feature_count = X_cenas.shape[1]

pipeline_schematic = [           {'scaler': {
            'standard': {}
                }
    },
      {'feature_selection': {
            'select_k_best': {
                'k': [44]
            }
        }
    },
           {'estimator': {
                'knn': {
                    'n_neighbors': range(1,21,2)
                },
                'svm' : {
                        'sklo': SVC,
                        'C' : [2**-5 , 2**-3 , 2**-1, 2, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15],
                        'gamma': [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2, 2**3 ]

        }}
                }
    ]

# Perform nested k-fold cross-validation
kfcv.fit(X_cenas, y_train_images_15_ground_truth1, pipeline_schematic=pipeline_schematic, scoring_metric='auc')

kfcv.plot_best_pipeline_scores()