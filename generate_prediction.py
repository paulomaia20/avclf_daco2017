# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:36:44 2017

@author: Paulo Maia
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage import img_as_float
from skimage import color
import retinal_image as ri

path_to_test_retinal_ims = 'data/test/images/'
test_im_list = os.listdir(path_to_test_retinal_ims)

nr_test_ims = 1
nr_features = 69 # red and green intensity, saturation

im_name = test_im_list[0]
image = ri.retinal_image(im_name, 'test')
coords_vessel_pixels = np.nonzero(image.skeletonWithoutCrossings)
nr_pixels_to_classify = len(coords_vessel_pixels[0])
X = np.zeros([nr_pixels_to_classify, nr_features])

image.load_red_intensity()
image.load_green_intensity()
image.load_blue_intensity()
image.load_saturation()
image.load_hue()
image.load_value()
image.load_local_features()
image.load_distance_to_optic_disk()
image.load_distance_from_image_center()
image.load_compute_line_features()
image.load_magnitude_gradient()


X[:,0] = image.red_intensity[coords_vessel_pixels]
X[:,1] = image.green_intensity[coords_vessel_pixels]
X[:,2] = image.blue_intensity[coords_vessel_pixels]
X[:,3] = image.hue[coords_vessel_pixels]
X[:,4] = image.saturation[coords_vessel_pixels]
X[:,5] = image.value[coords_vessel_pixels]
X[:,6] = image.mean_red_intensity_large[coords_vessel_pixels]
X[:,7] = image.mean_green_intensity_large[coords_vessel_pixels]
X[:,8] = image.mean_blue_intensity_large[coords_vessel_pixels]
X[:,9] = image.mean_hue_large[coords_vessel_pixels]
X[:,10] = image.mean_saturation_large[coords_vessel_pixels]
X[:,11] = image.mean_value_large[coords_vessel_pixels]
X[:,12] = image.mean_red_intensity[coords_vessel_pixels]
X[:,13] = image.mean_green_intensity[coords_vessel_pixels]
X[:,14] = image.mean_blue_intensity[coords_vessel_pixels]
X[:,15] = image.mean_hue[coords_vessel_pixels]
X[:,16] = image.mean_saturation[coords_vessel_pixels]
X[:,17] = image.mean_value[coords_vessel_pixels]
X[:,18] = image.minimum_red_intensity_large[coords_vessel_pixels]
X[:,19] = image.minimum_green_intensity_large[coords_vessel_pixels]
X[:,20] = image.minimum_blue_intensity_large[coords_vessel_pixels]
X[:,21] = image.minimum_hue_large[coords_vessel_pixels]
X[:,22] = image.minimum_saturation_large[coords_vessel_pixels]
X[:,23] = image.minimum_value_large[coords_vessel_pixels]
X[:,24] = image.minimum_red_intensity[coords_vessel_pixels]
X[:,25] = image.minimum_green_intensity[coords_vessel_pixels]
X[:,26] = image.minimum_blue_intensity[coords_vessel_pixels]
X[:,27] = image.minimum_hue[coords_vessel_pixels]
X[:,28] = image.minimum_saturation[coords_vessel_pixels]
X[:,29] = image.minimum_value[coords_vessel_pixels]
X[:,30] = image.maximum_red_intensity_large[coords_vessel_pixels]
X[:,31] = image.maximum_green_intensity_large[coords_vessel_pixels]
X[:,32] = image.maximum_blue_intensity_large[coords_vessel_pixels]
X[:,33] = image.maximum_hue_large[coords_vessel_pixels]
X[:,34] = image.maximum_saturation_large[coords_vessel_pixels]
X[:,35] = image.maximum_value_large[coords_vessel_pixels]
X[:,36] = image.maximum_red_intensity[coords_vessel_pixels]
X[:,37] = image.maximum_green_intensity[coords_vessel_pixels]
X[:,38] = image.maximum_blue_intensity[coords_vessel_pixels]
X[:,39] = image.maximum_hue[coords_vessel_pixels]
X[:,40] = image.maximum_saturation[coords_vessel_pixels]
X[:,41] = image.maximum_value[coords_vessel_pixels] 
X[:,42] = image.std_red_final[coords_vessel_pixels]
X[:,43] = image.std_green_final[coords_vessel_pixels]
X[:,44] = image.std_blue_final[coords_vessel_pixels]
X[:,45] = image.std_hue_final[coords_vessel_pixels]
X[:,46] = image.std_saturation_final[coords_vessel_pixels]
X[:,47] = image.std_value_final[coords_vessel_pixels]
X[:,48] = image.std_red_final_small[coords_vessel_pixels]
X[:,49] = image.std_green_final_small[coords_vessel_pixels]
X[:,50] = image.std_blue_final_small[coords_vessel_pixels]
X[:,51] = image.std_hue_final_small[coords_vessel_pixels]
X[:,52] = image.std_saturation_final_small[coords_vessel_pixels]
X[:,53] = image.std_value_final_small[coords_vessel_pixels]
X[:,54] = image.distance_to_optic_disk[coords_vessel_pixels]
X[:,55] = image.distance_from_image_center[coords_vessel_pixels]
X[:,56] = image.std_image[coords_vessel_pixels]
X[:,57] = image.line_skewness[coords_vessel_pixels]
X[:,58] = image.line_kurtosis[coords_vessel_pixels]
X[:,59] = image.line_mean[coords_vessel_pixels]
#We have to save the value of the gradient of each channel.
#Compute Maximum of each channel
#X[:,60] = np.amax(image.magnitude_gradient[:,0][coords_vessel_pixels])
#X[:,61] = np.amax(image.magnitude_gradient[:,1][coords_vessel_pixels])
#X[:,62] = np.amax(image.magnitude_gradient[:,2][coords_vessel_pixels])
##Compute Minimum of each channel
#X[:,63] = np.amin(image.magnitude_gradient[:,0][coords_vessel_pixels])
#X[:,64] = np.amin(image.magnitude_gradient[:,1][coords_vessel_pixels])
#X[:,65] = np.amin(image.magnitude_gradient[:,2][coords_vessel_pixels])
##Compute Mean of each channel
#X[:,66] = np.mean(image.magnitude_gradient[:,0][coords_vessel_pixels])
#X[:,67] = np.mean(image.magnitude_gradient[:,1][coords_vessel_pixels])
#X[:,68] = np.mean(image.magnitude_gradient[:,2][coords_vessel_pixels])

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


cenas=np.isnan(allfeatures_from_20_images_19dec2017)

allfeatures_from_20_images_19dec2017[cenas==True]=0

Xtrain = StandardScaler().fit_transform(allfeatures_from_20_images_19dec2017[:,0:59])

X_train, X_val, y_train, y_val = train_test_split(Xtrain, gt, test_size=0.33)



from sklearn import svm
gamma = 1 # SVM RBF radius
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
print(knn.score(X_train, y_train))
print(knn.score(X_val, y_val))

##PRINT TEST IMAGE

X_temp=X;

X=X[:,0:59]
X = StandardScaler().fit_transform(X)
    
# use YOUR classifier to predict vessel pixels
probabilities = knn.predict_proba(X)
arteriness, veinness = probabilities[:,0],probabilities[:,1]
artery_predictions = np.zeros_like(image.skeletonWithoutCrossings).astype(float)
vein_predictions = np.zeros_like(image.skeletonWithoutCrossings).astype(float)
artery_predictions[coords_vessel_pixels] = arteriness
vein_predictions[coords_vessel_pixels] = veinness
predictions_image = np.zeros_like(image.image)
predictions_image[:,:,0] = artery_predictions
predictions_image[:,:,2] = vein_predictions

plt.figure()
plt.imshow(predictions_image)
plt.show()