# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:11:34 2017

@author: Paulo Maia
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import retinal_image as ri
from skimage import io, color, img_as_float
import apply_skeleton
import find_interestpoints 
import divideIntoSegments
import detectOpticDisk
import obtainSkeletonWithoutCrossings
from scipy import ndimage
from skimage.measure import regionprops
from skimage.filters import rank
from skimage.filters.rank import maximum, minimum, mean
from skimage.morphology import skeletonize, square, disk


#import find_interestpointsv2 

#Paths
path_to_training_retinal_ims = 'data/training/images/'
path_to_training_retinal_masks = 'data/training/masks/'
path_to_training_retinal_vessels = 'data/training/vessels/'
path_to_training_arteries = 'data/training/arteries/'
path_to_training_veins = 'data/training/veins/'
retinal_im_list = os.listdir(path_to_training_retinal_ims)
nr_retinal_ims = len(retinal_im_list) # number of retinal images of the training set
nr_ims = len(retinal_im_list) # same as above

retinal_image = ri.retinal_image(retinal_im_list[6], 'train')


red_channel=retinal_image.image[:,:,0]
green_channel=retinal_image.image[:,:,1]
blue_channel=retinal_image.image[:,:,2]
hue_channel=color.rgb2hsv(retinal_image.image)[:,:,0]
saturation_channel=color.rgb2hsv(retinal_image.image)[:,:,1]
value_channel=color.rgb2hsv(retinal_image.image)[:,:,2]
mean_red_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
mean_value_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
minimum_red_intensity=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
minimum_value=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
minimum_blue_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
minimum_hue_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
maximum_blue_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
maximum_value_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
maximum_saturation_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
max_labels=np.amax(retinal_image.labels)
distanceTransform=ndimage.distance_transform_edt(retinal_image.vessels)
diameter=distanceTransform * retinal_image.skeletonWithoutCrossings
meanDiameterInRegion=np.zeros(np.max(retinal_image.labels))
i=0
for props in retinal_image.regions:
    i=i+1; 
    meanDiameterInRegion[i-1]=np.mean(diameter[retinal_image.labels==(i)])
for i in range(1, max_labels + 1):
    disk_diameter=meanDiameterInRegion[i-1]
    disk_diameter_large=2*disk_diameter
    labels_points=(retinal_image.labels==i)
    labels_points_indexes=np.nonzero(labels_points)
    labels_points_indexes=list(labels_points_indexes)
    rows=(labels_points_indexes)[0]
    cols=(labels_points_indexes)[1]
    #labels_points_int=labels_points.astype(int)
    #mean_intensity[rows,cols]=mean(img_rgb[labels_points==True], disk(disk_diameter)) 
    #mean_intensity[rows,cols]=mean(red_channel[rows,cols], disk(disk_diameter)) 
    mean_red_intensity_large_iteration=mean(red_channel,disk(disk_diameter_large))
    mean_red_intensity_large[rows,cols]=mean_red_intensity_large_iteration[rows,cols]
    mean_value_large_iteration=mean(value_channel,disk(disk_diameter_large))
    mean_value_large[rows,cols]=mean_value_large_iteration[rows,cols]
    minimum_red_intensity_iteration=minimum(red_channel,disk(disk_diameter))
    minimum_red_intensity[rows,cols]=minimum_red_intensity_iteration[rows,cols]
    minimum_value_iteration=minimum(value_channel,disk(disk_diameter))
    minimum_value[rows,cols]=minimum_value_iteration[rows,cols]
    minimum_blue_intensity_large_iteration=minimum(blue_channel,disk(disk_diameter_large))
    minimum_blue_intensity_large[rows,cols]=minimum_blue_intensity_large_iteration[rows,cols]
    minimum_hue_large_iteration=minimum(hue_channel,disk(disk_diameter_large))
    minimum_hue_large[rows,cols]=minimum_hue_large_iteration[rows,cols]
    maximum_blue_intensity_large_iteration=maximum(blue_channel,disk(disk_diameter_large))
    maximum_blue_intensity_large[rows,cols]=maximum_blue_intensity_large_iteration[rows,cols]
    maximum_saturation_large_iteration=maximum(saturation_channel,disk(disk_diameter_large))
    maximum_saturation_large[rows,cols]=maximum_saturation_large_iteration[rows,cols]
    maximum_value_large_iteration=maximum(value_channel,disk(disk_diameter_large))
    maximum_value_large[rows,cols]=maximum_value_large_iteration[rows,cols]
    #print(mean_intensity)
    print(i, ':',disk_diameter)