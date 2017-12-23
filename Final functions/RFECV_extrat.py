import os

import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage import img_as_float
from skimage import color
import retinal_image as ri
import math
from skimage.draw import line_aa
from skimage.morphology import skeletonize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from skimage.measure import regionprops


#TRAIN CLASSIFIER JUST FOR SEEING RESULTS - NOT OPTIMIZED YET !

cenas=np.isnan(x_train_images_15_features1)

x_train_images_15_features1[cenas==True]=0

X_train = StandardScaler().fit_transform(x_train_images_15_features1)
y_train = y_train_images_15_ground_truth1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X_train, y_train)
selector.support_ 
selector.ranking_
X_new_1= X_train[:, selector.support_ == True]




