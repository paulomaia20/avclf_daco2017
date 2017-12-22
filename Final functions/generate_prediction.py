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
import math
from skimage.draw import line_aa
from skimage.morphology import skeletonize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from skimage.measure import regionprops


path_to_test_retinal_ims = 'data/test/images/'
test_im_list = os.listdir(path_to_test_retinal_ims)

nr_test_ims = 1
nr_features = 69 # red and green intensity, saturation

#EXTRACT TEST FEATURES

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


#TRAIN CLASSIFIER JUST FOR SEEING RESULTS - NOT OPTIMIZED YET !

cenas=np.isnan(allfeatures_from_20_images_19dec2017)

allfeatures_from_20_images_19dec2017[cenas==True]=0

Xtrain = StandardScaler().fit_transform(allfeatures_from_20_images_19dec2017[:,0:59])
X_train, X_val, y_train, y_val = train_test_split(Xtrain, gt, test_size=0.33)

gamma = 1 # SVM RBF radius


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
print(knn.score(X_val, y_val))

##PRINT TEST IMAGE

X=test_features

X=X[:,0:59]
X = StandardScaler().fit_transform(X)
    
# use YOUR classifier to predict vessel pixels
probabilities = knn.predict_proba(X)
arteriness, veinness = probabilities[:,0],probabilities[:,1]
artery_predictions = np.zeros_like(image.skeletonWithoutCrossings).astype(float)
vein_predictions = np.zeros_like(image.skeletonWithoutCrossings).astype(float)
artery_predictions[coords_vessel_pixels] = arteriness
vein_predictions[coords_vessel_pixels] = veinness

#Put them on RGB image
predictions_image = np.zeros_like(image.image)
predictions_image[:,:,0] = artery_predictions
predictions_image[:,:,2] = vein_predictions

plt.figure()
plt.imshow(predictions_image)
plt.show()

#Go from soft labels to hard labels
hard_predictions_artery = (predictions_image[:,:,0]>0.5).astype(int)
hard_predictions_vein=(predictions_image[:,:,2]>0.5).astype(int)
regions=image.regions

i=0
temp=np.zeros((image.image.shape[0],image.image.shape[1]))
for props in regions:
    i=i+1; 
    #Count number of pixels in hard predictions
    nArteries=np.count_nonzero(hard_predictions_artery[image.labels==i])
    nVeins=np.count_nonzero(hard_predictions_vein[image.labels==i])
    #Find region and coordinates of points in region
    labels_points=(image.labels==i)
    labels_points_indexes=np.nonzero(labels_points)
    rows=(labels_points_indexes)[0]
    cols=(labels_points_indexes)[1]
    #Temporary image is one at label region
    temp[rows,cols]=1
    sum_veins=hard_predictions_vein+temp #places where the classifier is sure we have a vein
    sum_arteries=hard_predictions_artery+temp #places where the classifier is sure we have an artery
    intersection_veins=(sum_veins==2) 
    intersection_arteries=(sum_arteries==2)
    coordinates_intersection_arteries=np.nonzero(intersection_arteries)
    coordinates_intersection_veins=np.nonzero(intersection_veins)

    if (nArteries>nVeins):
        #Find intersection between points which were predicted as veins and current region
        #At intersection region, veins are now arteries 
        predictions_image[coordinates_intersection_veins[0],coordinates_intersection_veins[1],0]=np.mean(predictions_image[coordinates_intersection_arteries[0],coordinates_intersection_arteries[1],0])
        #Predictions at veins are now the mean of all arteries probabilities 
        predictions_image[coordinates_intersection_veins[0],coordinates_intersection_veins[1],0]=1

    else:
        #Find intersection between points which were predicted as veins and current region
        #At intersection region, veins are now arteries 
        predictions_image[coordinates_intersection_arteries[0],coordinates_intersection_arteries[1],2]=np.mean(predictions_image[coordinates_intersection_veins[0],coordinates_intersection_veins[1],0])
        #Predictions at arteries are now the mean of all veins probabilities 
        predictions_image[coordinates_intersection_arteries[0],coordinates_intersection_arteries[1],0]=0
    #Reset temp image, because it will now be zero in regions other than the region of interest
    temp=np.zeros((image.image.shape[0],image.image.shape[1]))

plt.figure(2)
plt.imshow(predictions_image)
plt.show()


#CALCULATE ORIENTATION IMAGE


#Obtain orientations image
orientations_image=np.zeros((image.preprocessed_image.shape[0],image.preprocessed_image.shape[1]))

for kk in range(1,len(image.labels)):
    skeleton_y,skeleton_x=np.nonzero(image.labels==kk)
    for pp in range(len(skeleton_x)):
        if (len(skeleton_x)>8):
            if (pp>=4 and pp<=len(skeleton_x)-4):
                small_segment_x=skeleton_x[pp-3:pp+3]
                small_segment_y=skeleton_y[pp-3:pp+3]
            elif (pp<4):
                small_segment_x=skeleton_x[0:pp+3]
                small_segment_y=skeleton_y[0:pp+3]
            elif (pp>len(skeleton_x)+4):
                small_segment_x=skeleton_x[pp-3:len(skeleton_x)]
                small_segment_y=skeleton_y[pp-3:len(skeleton_x)]
            small_segment_x = small_segment_x - np.mean(small_segment_x)
            small_segment_y = small_segment_y - np.mean(small_segment_y) 
            coords = np.vstack([small_segment_x, small_segment_y])
            cov = np.cov(coords)
            evals, evecs = np.linalg.eig(cov)
            sort_indices = np.argsort(evals)[::-1]
            evec1, evec2 = evecs[:, sort_indices]
            x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
            x_v2, y_v2 = evec2
            theta = np.tanh((x_v1)/(y_v1))
            print(theta)
            orientations_image[skeleton_y[pp],skeleton_x[pp]]=theta
    


tempImg=np.zeros((image.preprocessed_image.shape[0],image.preprocessed_image.shape[1]))
vesselsWithSkeletonPrediction = np.zeros_like(image.image).astype(np.float64)
i=0
predictionsOnSkeleton_indexesArteries=np.nonzero(predictions_image[:,:,0])
predictionsOnSkeleton_indexesVeins=np.nonzero(predictions_image[:,:,2])
vesselsWithSkeletonPrediction[:,:,0]=image.vessels
vesselsWithSkeletonPrediction[:,:,2]=image.vessels
vesselsWithSkeletonPrediction[predictionsOnSkeleton_indexesArteries[0],predictionsOnSkeleton_indexesArteries[1],:]=predictions_image[predictionsOnSkeleton_indexesArteries[0],predictionsOnSkeleton_indexesArteries[1],:]
vesselsWithSkeletonPrediction[predictionsOnSkeleton_indexesVeins[0],predictionsOnSkeleton_indexesVeins[1],:]=predictions_image[predictionsOnSkeleton_indexesVeins[0],predictionsOnSkeleton_indexesVeins[1],:]
for props in regions:
    i=i+1
    labels_points=(image.labels==i)
    labels_points_indexes=np.nonzero(labels_points)
    rows=(labels_points_indexes)[0]
    cols=(labels_points_indexes)[1]
    orientation = props.orientation
    for kk in range(len(rows)-1): #JUST FOR DEBUGGING !
        y0=rows[kk]
        x0=cols[kk]
        start_x=x0 - math.cos(0.5 + orientations_image[y0,x0])*0.25*props.major_axis_length;
        start_y=y0 + math.sin(0.5 + orientations_image[y0,x0]) * 0.25 * props.major_axis_length;
        end_x=x0 + math.cos(0.5 + orientations_image[y0,x0])*0.25*props.major_axis_length;
        end_y=y0 - math.sin(0.5 + orientations_image[y0,x0]) * 0.25 * props.major_axis_length;
        rr, cc, val = line_aa(int(start_x), int(start_y), int(end_x), int(end_y))
        tempImg[rr-3,cc-3]=1; 
        thin_perpendicularlines=skeletonize(tempImg)
       # plt.imshow(image.skeleton)
     #   plt.plot((start_x, end_x), (start_y, end_y), '-r', linewidth=2.5)
      #  plt.show()
        region = thin_perpendicularlines*image.vessels #0s em todos os sitios menos na interseçao
        regionCoordinates = np.nonzero(region)
        vesselsWithSkeletonPrediction[regionCoordinates[0],regionCoordinates[1],:]=vesselsWithSkeletonPrediction[x0,y0,:]
        tempImg=np.zeros((image.preprocessed_image.shape[0],image.preprocessed_image.shape[1]))

plt.imshow(vesselsWithSkeletonPrediction)