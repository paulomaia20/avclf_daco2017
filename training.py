# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:21:44 2017

@author: Paulo Maia
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(allfeatures, gt, test_size=0.20, random_state=10)
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train);
X_test = StandardScaler().fit_transform(X_test);

from sklearn import svm
from sklearn.model_selection import GridSearchCV
parameters = [{'kernel': ['rbf'],
               'gamma': [0.001, 0.01, 0.1, 1, 10],
                'C': [0.1, 1, 10, 100]},
              {'kernel': ['linear'], 'C': [1, 10, 100]}]

clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
print("# Tuning hyper-parameters")
clf.fit(X_train, y_train);

print("Best parameters set found on training set:")
print()
print(clf.best_params_)
print()
print("Grid scores on training set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
maxAccuracy_SVM=np.amax(means); 