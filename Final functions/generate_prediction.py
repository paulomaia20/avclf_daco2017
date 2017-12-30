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
from scipy import ndimage

import math
from skimage.draw import line_aa
from skimage.morphology import skeletonize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from skimage.measure import regionprops

#%%
path_to_train_retinal_ims = 'data/training/images/'
train_im_list = os.listdir(path_to_train_retinal_ims)

nr_test_ims = 1
nr_features = 61

#EXTRACT TEST FEATURES

for pp in range(5):
                #im_name = train_im_list[7] #Wasnt used in classifier 
     image = ri.retinal_image(fitting_img[pp], 'train')
     coords_vessel_pixels = np.nonzero(image.skeletonWithoutCrossings)
     nr_pixels_to_classify = len(coords_vessel_pixels[0])
     X = np.zeros([nr_pixels_to_classify, nr_features])
    
# =============================================================================
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
     image.load_Diameter()
     
     
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
     X[:,60] = image.diameter[coords_vessel_pixels]
     
     np.save(open('test' + str(pp) +'.npy', 'wb'),
            X)
 #=============================================================================
    #%%
    
    #TRAIN CLASSIFIER
    
    
X_train=X_train_15images_5000pts_withoutPreProc_61feat

X_temp=np.zeros((X_train.shape[0],57))
X_temp[:,0:56]=X_train[:,0:56]
X_temp[:,56]=X_train[:,60]
#X_train=x_fraction

#    
#    X_cenas=X_cenas[:,features_select==True]




y_train=y_train_15images_5000pts_withoutPreProc_61feat


X_train = StandardScaler().fit_transform(X_temp);

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

#clf=RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
#    max_depth=None, max_features=9, max_leaf_nodes=None,
#    min_impurity_decrease=0.0, min_impurity_split=None,
#    min_samples_leaf=9, min_samples_split=8,
#    min_weight_fraction_leaf=0.0, n_estimators=15, n_jobs=1,
#    oob_score=False, random_state=None, verbose=0,
#    warm_start=False)

import pickle
clf=pickle.load( open( "nn_best.sav", "rb" ) )

clf.fit(X_train,y_train)
    
#%% Load features
accuracies_skeleton=[];
accuracies_vessel=[];
for pp in range(5):
    
    image = ri.retinal_image(fitting_img[pp], 'train')
    coords_vessel_pixels = np.nonzero(image.skeletonWithoutCrossings)
    nr_pixels_to_classify = len(coords_vessel_pixels[0])
    X = np.zeros([nr_pixels_to_classify, nr_features])
    X=np.load(open('test' + str(pp) +'.npy', 'rb'))
        
        #Selected features:
        

            
    
    X_temp2=np.zeros((X.shape[0],57))
    X_temp2[:,0:56]=X[:,0:56]
    X_temp2[:,56]=X[:,60]
    

    
        
    #%%
    # use YOUR classifier to predict vessel pixels
    
    #X=X_cenas2
    X_cenas2 = StandardScaler().fit_transform(X_temp2);
    
    
    probabilities = clf.predict_proba(X_cenas2)
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
    
    plt.figure()
    plt.imshow(predictions_image)
    plt.show()
      
#%% 

#    #Obtain distance to optic disk
#    distances_image=np.ones((image.image.shape[0],image.image.shape[1]))
#    distances_image[int(image.x_opticdisk),int(image.y_opticdisk)]=0; 
#    #distance transform gives distance from white pixels to black pixels
#    distanceToOpticDisk=ndimage.distance_transform_edt(distances_image)
#    #Obtain coordinates of branch and crossings
#    coordinates=image.coordinates;
#    #Get points which are further than 20%*max(dist)
#    subSampleOD=(distanceToOpticDisk>0.20*np.max(distanceToOpticDisk)).astype(int)
#    #Get same points but only on skeleton
#    subSample_skel=(subSampleOD+image.skeleton.astype(int))==2
#    #subSample_labels=(subSample+(image.labels>0).astype(int))==2
#    
#    
#    tempImg=np.zeros((image.image.shape[0],image.image.shape[1]))
#    
#    tempImg[coordinates[0],coordinates[1]]=1
#    
#    #Select subregion
#    
#    subSample_intersections=(subSampleOD+tempImg.astype(int))==2
#    
#    rows_branch,cols_branch=np.nonzero(subSample_intersections)
#    
#    predictions_temp=np.copy(predictions_image)
#    
#    for jj in range(len(rows_branch)):
#        
#        #Get 3x3 neighbourhood and see which labels are present
#        neighbourhood=image.labels[rows_branch[jj]-5:rows_branch[jj]+5,cols_branch[jj]-5:cols_branch[jj]+5]
#        
#        #Find labels present in neighbourhood
#        labelsInNeighbourhood=np.unique(neighbourhood)
#        #delete label 0
#        labelsInNeighbourhood=np.delete(labelsInNeighbourhood,0)
#        
#        #Find label with maximum distance to optic disk
#        maximum_distances=[]
#        minimum_distances=[]
#        for labelIdx in range(len(labelsInNeighbourhood)):
#            label_pixels=(image.labels==labelsInNeighbourhood[labelIdx])
#            #See distance to optic disk
#            maxDist=np.max(distanceToOpticDisk[label_pixels==True])
#            minDist=np.min(distanceToOpticDisk[label_pixels==True])
#            #Append max,min distances to distances vector, to see which label
#            #has the closest and the most distance pixel to the OD
#            maximum_distances.append(maxDist)
#            minimum_distances.append(minDist)
#            
#        #Find label in neighbourhood with max and mim dist
#        if (len(maximum_distances)!=0 and len(minimum_distances)!=0):
#            closestLabel=labelsInNeighbourhood[np.argmin(minimum_distances)]
#            furthestLabel=labelsInNeighbourhood[np.argmax(maximum_distances)]
#            
#            #Now that we have the index of the label: 
#            #Probability of the most distant label is equal to probability of the closest label
#            
#            #Find region of label 1
#            label_closest_coord=np.nonzero(image.labels==closestLabel)
#            #Find region of label 2
#            label_furthest_coord=np.nonzero(image.labels==furthestLabel)
#        
#            #prediction in furthest label is equal to prediction in closest label
#            predictions_image[label_furthest_coord[0],label_furthest_coord[1],0]=np.mean(predictions_image[label_closest_coord[0],label_closest_coord[1],0])
#            predictions_image[label_furthest_coord[0],label_furthest_coord[1],2]=np.mean(predictions_image[label_closest_coord[0],label_closest_coord[1],2])
#    
#    plt.figure()
#    plt.imshow(predictions_image)    
#    plt.show()
    
#%%
   
#    temp_image=np.copy(predictions_image)
#    #Second pass
#    for jj in range(len(rows_branch)):
#        
#        #Get 3x3 neighbourhood and see which labels are present
#        neighbourhood=image.labels[rows_branch[jj]-10:rows_branch[jj]+10,cols_branch[jj]-10:cols_branch[jj]+10]
#        
#        #Find labels present in neighbourhood
#        labelsInNeighbourhood=np.unique(neighbourhood)
#        #delete label 0
#        labelsInNeighbourhood=np.delete(labelsInNeighbourhood,0)
#        
#        #For each label: find if it's classified as A or V
#        isArtery=[]
#        isVein=[]
#        for labelIdx in range(len(labelsInNeighbourhood)):
#            label_pixels=(image.labels==labelsInNeighbourhood[labelIdx])
#            r,c=np.nonzero(label_pixels)
#            if ( np.mean(predictions_image[r,c,2])>0.5):
#                isVein.append(1)
#                isArtery.append(0)
#            else:
#                isVein.append(0)
#                isArtery.append(1)
#            #Now we can make a correspondence between label ID and Vein/artery
#            #because they have the same size
#            #e.g  labelsInNeighbourhood=[1 3 5]
#            # isVein=[1 0 1]
#            # isArtery = [0 1 0]
#            
#        #See number of labels which are mainly artery or vein   
#        numbArteries=np.count_nonzero(isArtery)
#        numbVeins=np.count_nonzero(isVein)
#        
#        #Calculate mean probability of being artery/vein for each label
#        idx_artery=np.nonzero(isArtery)
#        idx_vein=np.nonzero(isVein)
#        means_artery=np.zeros((len(labelsInNeighbourhood),))
#        means_vein=np.zeros((len(labelsInNeighbourhood),))
#        
#        for aa in range(len(idx_artery[0])):
#            label_artery=labelsInNeighbourhood[idx_artery[0][aa]]
#            label_artery_coord=np.nonzero(image.labels==label_artery)
#            means_artery[aa]=np.mean(temp_image[label_artery_coord[0],label_artery_coord[1],0])
#        for aa in range(len(idx_vein[0])):
#            label_vein=labelsInNeighbourhood[idx_vein[0][aa]]
#            label_vein_coord=np.nonzero(image.labels==label_vein)
#            means_vein[aa]=np.mean(temp_image[label_vein_coord[0],label_vein_coord[1],2])    
#        
#        
#        #Is the number of labels which are mainly arteries > number of labels which are mainly veins?
#        if (numbArteries>numbVeins):
#            for pp in range(len(idx_vein[0])): #Go through all the labels classified as vein
#                #Find pixels for that label
#                label_vein=labelsInNeighbourhood[idx_vein[0][pp]]
#                label_vein_coord=np.nonzero(image.labels==label_vein)
#                #Replace those veins by artery
#                predictions_image[label_vein_coord[0],label_vein_coord[1],2]=np.max(means_artery[idx_artery[0]])
#                predictions_image[label_vein_coord[0],label_vein_coord[1],0]=0
#
#        elif (numbArteries<numbVeins):
#            for kk in range(len(idx_artery[0])): #Go through all the labels classified as artery
#                #Find pixels for that label
#                label_artery=labelsInNeighbourhood[idx_artery[0][kk]]
#                label_artery_coord=np.nonzero(image.labels==label_artery)
#                #Replace those veins by artery
#                predictions_image[label_artery_coord[0],label_artery_coord[1],0]=np.max(means_vein[idx_vein[0]])
#                predictions_image[label_vein_coord[0],label_vein_coord[1],2]=0
#
#
#            
#        
#          
#    
#    plt.figure()
#    plt.imshow(predictions_image)    
#    plt.show()

    #%% 
    #Assess accuracy on skeleton
    #Go from soft labels to hard labels
    hard_predictions_artery = (predictions_image[:,:,0]>0.5).astype(int)
    hard_predictions_vein=(predictions_image[:,:,2]>0.5).astype(int)
    image_artery=image.arteries_skeleton.astype(int)
    image_vein=image.veins_skeleton.astype(int)
    
    perc_correct_art=(hard_predictions_artery+image_artery)==2
    perc_correct_vein=(hard_predictions_vein+image_vein)==2
    
    perc_correct= np.count_nonzero(perc_correct_art) + np.count_nonzero(perc_correct_vein)
    totalpixels=np.count_nonzero(image.skeleton)
    
    acc=perc_correct/totalpixels 
    accuracies_skeleton.append(acc)
    
    print("acc on skeleton:", acc)
    
    #%% 
    #Propagate labels
    
    #1-NN algorithm using position as features
    
    rows,cols=np.nonzero(image.skeleton)
    indexes=range(0,len(rows))
    knn = KNeighborsClassifier(n_neighbors = 1)
    X_new=np.transpose(np.array([rows,cols]))
    knn.fit(X_new, indexes)
    
    #Make image of vessel with prediction on center pixel 
    vesselsWithSkeletonPrediction = np.zeros_like(image.image).astype(np.float64)
    predictionsOnSkeleton_indexesArteries=np.nonzero(predictions_image[:,:,0])
    predictionsOnSkeleton_indexesVeins=np.nonzero(predictions_image[:,:,2])
    vesselsWithSkeletonPrediction[:,:,0]=image.vessels
    vesselsWithSkeletonPrediction[:,:,2]=image.vessels
    vesselsWithSkeletonPrediction[predictionsOnSkeleton_indexesArteries[0],predictionsOnSkeleton_indexesArteries[1],:]=predictions_image[predictionsOnSkeleton_indexesArteries[0],predictionsOnSkeleton_indexesArteries[1],:]
    vesselsWithSkeletonPrediction[predictionsOnSkeleton_indexesVeins[0],predictionsOnSkeleton_indexesVeins[1],:]=predictions_image[predictionsOnSkeleton_indexesVeins[0],predictionsOnSkeleton_indexesVeins[1],:]
    
    #See to which label the new points belong
    #points which belong to the vessel and not to the skeleton
    rows2,cols2=np.nonzero(image.vessels^image.skeleton) 
    X_new2=np.transpose(np.array([rows2,cols2]))
    predicted_class=knn.predict(X_new2)
    
    #Assign them the probability of the nearest neighbour
    for ii in range(len(predicted_class)):
        prediction=X_new[predicted_class[ii]]
        vesselsWithSkeletonPrediction[X_new2[ii,0],X_new2[ii,1],:]=vesselsWithSkeletonPrediction[prediction[0],prediction[1],:]
    
    #%%
    #Assess accuracy on full image
    #Go from soft labels to hard labels
    hard_predictions_artery2 = (vesselsWithSkeletonPrediction[:,:,0]>0.5).astype(int)
    hard_predictions_vein2 =(vesselsWithSkeletonPrediction[:,:,2]>0.5).astype(int)
    image_artery=image.arteries.astype(int)
    image_vein=image.veins.astype(int)
    
    TP=np.count_nonzero((hard_predictions_vein2+image_vein)==2)
    TN=np.count_nonzero((hard_predictions_vein2+image_vein)==0)
    FP=np.count_nonzero((hard_predictions_vein2-image_vein)==-1)
    FN=np.count_nonzero((hard_predictions_vein2-image_vein)==1)
    
    print("TP+TN/Total",(TP+TN)/(TP+TN+FP+FN))

    
    perc_correct_art=(hard_predictions_artery2+image_artery)==2
    perc_correct_vein=(hard_predictions_vein2+image_vein)==2
    
    perc_correct= np.count_nonzero(perc_correct_art) + np.count_nonzero(perc_correct_vein)
    totalpixels=np.count_nonzero(image.vessels)
    
    acc=perc_correct/totalpixels 
    accuracies_vessel.append(acc)
    
    
    #%% Final metrics
    from sklearn import metrics
    
    coords_vessel_pixels = np.nonzero(image.vessels)
    vein_predictions=vesselsWithSkeletonPrediction[:,:,2]
    fpr,tpr,th=metrics.roc_curve(image.veins[coords_vessel_pixels],vein_predictions[coords_vessel_pixels])
    
    #plt.plot(fpr,tpr)
   # plt.show
    print("acc on vessel:", acc)
    print("AUC",metrics.roc_auc_score(image.veins[coords_vessel_pixels],vein_predictions[coords_vessel_pixels]))
    print("acc_score",metrics.accuracy_score(image.veins[coords_vessel_pixels],vein_predictions[coords_vessel_pixels]>0.5))
print("meanacc_vessel",np.mean(accuracies_vessel))
print("meanacc_skel",np.mean(accuracies_skeleton))
