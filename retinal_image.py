import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, img_as_float
import apply_skeleton
import find_interestpoints 
import divideIntoSegments
import detectOpticDisk
import obtainSkeletonWithoutCrossings
from scipy import ndimage
from skimage.restoration import denoise_nl_means
from skimage.measure import regionprops
from skimage.filters import rank
from skimage.filters.rank import maximum, minimum, mean
from skimage.morphology import skeletonize, square, disk
import apply_homomorphic_filtering
from scipy.stats import skew
from scipy.stats import skewtest
from scipy.stats import kurtosis
import cv2
import math
from skimage.draw import line_aa
from skimage.measure import shannon_entropy
from skimage.feature import greycomatrix
from skimage.feature import greycoprops

path_to_training_retinal_ims = 'data/training/images/'
path_to_training_retinal_masks = 'data/training/masks/'
path_to_training_retinal_vessels = 'data/training/vessels/'
path_to_training_arteries = 'data/training/arteries/'
path_to_training_veins = 'data/training/veins/'

# FEATURE EXTRACTION

def compute_saturation(retinal_image):
    '''
    This function expects a retinal_image object
    1) retinal image 
    2) Field Of View mask 
    3) binary vessel map
    4) binary artery map
    5) binary vein map
    You may want to use (part of) them, or not
    '''
    return color.rgb2hsv(retinal_image.image)[:,:,1]

def compute_hue(retinal_image):
    '''
    This function expects a retinal_image object
    1) retinal image 
    2) Field Of View mask 
    3) binary vessel map
    4) binary artery map
    5) binary vein map
    You may want to use (part of) them, or not
    '''
    return color.rgb2hsv(retinal_image.image)[:,:,0]

def compute_value(retinal_image):
    '''
    This function expects a retinal_image object
    1) retinal image 
    2) Field Of View mask 
    3) binary vessel map
    4) binary artery map
    5) binary vein map
    You may want to use (part of) them, or not
    '''
    return color.rgb2hsv(retinal_image.image)[:,:,2]

def compute_red_intensity(retinal_image):
    '''
    This function expects a retinal_image object.
    '''
    return retinal_image.image[:,:,0]

def compute_green_intensity(retinal_image):
    '''
    This function expects a retinal_image object.
    '''
    
    return retinal_image.image[:,:,1]

def compute_blue_intensity(retinal_image):
    '''
    This function expects a retinal_image object.
    '''
    
    return retinal_image.image[:,:,2]
 
def compute_distance_to_optic_disk(retinal_image):
    distances_image=np.ones((retinal_image.image.shape[0],retinal_image.image.shape[1]))
    distances_image[int(retinal_image.y_opticdisk),int(retinal_image.x_opticdisk)]=0; 
    #distance transform gives distance from white pixels to black pixels
    distanceToOpticDisk=ndimage.distance_transform_edt(distances_image)
    return distanceToOpticDisk

def compute_distance_from_image_center(retinal_image):
    distances_image=np.ones((retinal_image.image.shape[0],retinal_image.image.shape[1]))
    distances_image[int(retinal_image.image.image.shape[1]/2),int(retinal_image.image.shape[0]/2)]=0; 
    #distance transform gives distance from white pixels to black pixels
    distanceFromImageCenter=ndimage.distance_transform_edt(distances_image)
    return distanceFromImageCenter
  
def compute_local_features(retinal_image):
    red_channel=retinal_image.preprocessed_image[:,:,0]
    green_channel=retinal_image.preprocessed_image[:,:,1]
    blue_channel=retinal_image.preprocessed_image[:,:,2]
    hue_channel=color.rgb2hsv(retinal_image.preprocessed_image)[:,:,0]
    saturation_channel=color.rgb2hsv(retinal_image.preprocessed_image)[:,:,1]
    value_channel=color.rgb2hsv(retinal_image.preprocessed_image)[:,:,2]
    #mean- large
    mean_red_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_blue_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_green_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_hue_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_saturation_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_value_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    #mean- small
    mean_red_intensity=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_green_intensity=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_blue_intensity=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_hue=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_saturation=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_value=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    #minimum- large
    minimum_red_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    minimum_green_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    minimum_blue_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    minimum_hue_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    minimum_saturation_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    minimum_value_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))  
    #minimum- small
    minimum_red_intensity=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    minimum_green_intensity=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    minimum_blue_intensity=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    minimum_hue=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    minimum_saturation=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    minimum_value=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    #maximum- large
    maximum_red_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    maximum_green_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    maximum_blue_intensity_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    maximum_hue_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    maximum_saturation_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    maximum_value_large=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    #maximum- small
    maximum_red_intensity=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    maximum_green_intensity=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    maximum_blue_intensity=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    maximum_hue=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    maximum_saturation=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    maximum_value=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    # std- large
    mean_red_intensity_large1 = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_red_intensity_large_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_green_intensity_large1=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_green_intensity_large_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_blue_intensity_large1=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_blue_intensity_large_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_hue_large1=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_hue_large_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_saturation_large1=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_saturation_large_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_value_large1=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_value_large_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    # std- small
    mean_red_intensity_1 = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_red_intensity_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_green_intensity_1=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_green_intensity_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_blue_intensity_1=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_blue_intensity_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_hue_1=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_hue_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_saturation_1=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_saturation_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_value_1=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    mean_value_potency=np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    
    #GLCM Features Large Diameter
    glcm_image_entropy_large = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_large = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_contrast_large = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_dissimilarity_large = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_homogeneity_large = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_energy_large = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_correlation_large = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_ASM_large = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    
    #GLCM Features Small Diameter
    glcm_image_entropy_small = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_small = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_contrast_small = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_dissimilarity_small = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_homogeneity_small = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_energy_small = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_correlation_small = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    glcm_image_ASM_small = np.zeros((retinal_image.labels.shape[0], retinal_image.labels.shape[1]))
    
    
    max_labels=np.amax(retinal_image.labels)
    distanceTransform=ndimage.distance_transform_edt(retinal_image.vessels)
    diameter=distanceTransform * retinal_image.skeletonWithoutCrossings
    meanDiameterInRegion=np.zeros(np.max(retinal_image.labels))

    for i in range(1, max_labels + 1):
        meanDiameterInRegion[i-1]=np.mean(diameter[retinal_image.labels==(i)])
        disk_diameter=meanDiameterInRegion[i-1]
#        disk_diameter=2; 
        disk_diameter_large=2*disk_diameter
        labels_points=(retinal_image.labels==i)
        labels_points_indexes=np.nonzero(labels_points)
        labels_points_indexes=list(labels_points_indexes)
        rows=(labels_points_indexes)[0]
        cols=(labels_points_indexes)[1]
        #labels_points_int=labels_points.astype(int)
        #mean_intensity[rows,cols]=mean(img_rgb[labels_points==True], disk(disk_diameter)) 
        #mean_intensity[rows,cols]=mean(red_channel[rows,cols], disk(disk_diameter)) 
        #mean- large
        mean_red_intensity_large_iteration=mean(red_channel,disk(disk_diameter_large))
        mean_red_intensity_large[rows,cols]=mean_red_intensity_large_iteration[rows,cols]
        mean_green_intensity_large_iteration=mean(green_channel,disk(disk_diameter_large))
        mean_green_intensity_large[rows,cols]=mean_green_intensity_large_iteration[rows,cols]
        mean_blue_intensity_large_iteration=mean(blue_channel,disk(disk_diameter_large))
        mean_blue_intensity_large[rows,cols]=mean_blue_intensity_large_iteration[rows,cols]
        mean_hue_large_iteration=mean(hue_channel,disk(disk_diameter_large))
        mean_hue_large[rows,cols]=mean_hue_large_iteration[rows,cols]
        mean_saturation_large_iteration=mean(saturation_channel,disk(disk_diameter_large))
        mean_saturation_large[rows,cols]=mean_saturation_large_iteration[rows,cols]
        mean_value_large_iteration=mean(value_channel,disk(disk_diameter_large))
        mean_value_large[rows,cols]=mean_value_large_iteration[rows,cols]
        #mean- small
        mean_red_intensity_iteration=mean(red_channel,disk(disk_diameter))
        mean_red_intensity[rows,cols]=mean_red_intensity_iteration[rows,cols]
        mean_green_intensity_iteration=mean(green_channel,disk(disk_diameter))
        mean_green_intensity[rows,cols]=mean_green_intensity_iteration[rows,cols]
        mean_blue_intensity_iteration=mean(blue_channel,disk(disk_diameter))
        mean_blue_intensity[rows,cols]=mean_blue_intensity_iteration[rows,cols]
        mean_hue_iteration=mean(hue_channel,disk(disk_diameter))
        mean_hue[rows,cols]=mean_hue_iteration[rows,cols]
        mean_saturation_iteration=mean(saturation_channel,disk(disk_diameter))
        mean_saturation[rows,cols]=mean_saturation_iteration[rows,cols]
        mean_value_iteration=mean(value_channel,disk(disk_diameter))
        mean_value[rows,cols]=mean_value_iteration[rows,cols]
        #minimum- large
        minimum_red_intensity_iteration=minimum(red_channel,disk(disk_diameter))
        minimum_red_intensity[rows,cols]=minimum_red_intensity_iteration[rows,cols]
        minimum_green_intensity_iteration=minimum(green_channel,disk(disk_diameter))
        minimum_green_intensity[rows,cols]=minimum_green_intensity_iteration[rows,cols]
        minimum_blue_intensity_iteration=minimum(blue_channel,disk(disk_diameter))
        minimum_blue_intensity[rows,cols]=minimum_blue_intensity_iteration[rows,cols]
        minimum_hue_iteration=minimum(hue_channel,disk(disk_diameter))
        minimum_hue[rows,cols]=minimum_hue_iteration[rows,cols]
        minimum_saturation_iteration=minimum(saturation_channel,disk(disk_diameter))
        minimum_saturation[rows,cols]=minimum_saturation_iteration[rows,cols]
        minimum_value_iteration=minimum(value_channel,disk(disk_diameter))
        minimum_value[rows,cols]=minimum_value_iteration[rows,cols]
        #minimum- small
        minimum_red_intensity_large_iteration=minimum(red_channel,disk(disk_diameter_large))
        minimum_red_intensity_large[rows,cols]=minimum_red_intensity_large_iteration[rows,cols]
        minimum_green_intensity_large_iteration=minimum(green_channel,disk(disk_diameter_large))
        minimum_green_intensity_large[rows,cols]=minimum_green_intensity_large_iteration[rows,cols]
        minimum_blue_intensity_large_iteration=minimum(blue_channel,disk(disk_diameter_large))
        minimum_blue_intensity_large[rows,cols]=minimum_blue_intensity_large_iteration[rows,cols]
        minimum_hue_large_iteration=minimum(hue_channel,disk(disk_diameter_large))
        minimum_hue_large[rows,cols]=minimum_hue_large_iteration[rows,cols]
        minimum_saturation_large_iteration=minimum(saturation_channel,disk(disk_diameter_large))
        minimum_saturation_large[rows,cols]=minimum_saturation_large_iteration[rows,cols]
        minimum_value_large_iteration=minimum(value_channel,disk(disk_diameter_large))
        minimum_value_large[rows,cols]=minimum_value_large_iteration[rows,cols]
        #maximum- large
        maximum_red_intensity_large_iteration=maximum(red_channel,disk(disk_diameter_large))
        maximum_red_intensity_large[rows,cols]=maximum_red_intensity_large_iteration[rows,cols]
        maximum_green_intensity_large_iteration=maximum(green_channel,disk(disk_diameter_large))
        maximum_green_intensity_large[rows,cols]=maximum_green_intensity_large_iteration[rows,cols]
        maximum_blue_intensity_large_iteration=maximum(blue_channel,disk(disk_diameter_large))
        maximum_blue_intensity_large[rows,cols]=maximum_blue_intensity_large_iteration[rows,cols]
        maximum_hue_large_iteration=maximum(hue_channel,disk(disk_diameter_large))
        maximum_hue_large[rows,cols]=maximum_hue_large_iteration[rows,cols]
        maximum_saturation_large_iteration=maximum(saturation_channel,disk(disk_diameter_large))
        maximum_saturation_large[rows,cols]=maximum_saturation_large_iteration[rows,cols]
        maximum_value_large_iteration=maximum(value_channel,disk(disk_diameter_large))
        maximum_value_large[rows,cols]=maximum_value_large_iteration[rows,cols]
        #maximum- small
        maximum_red_intensity_iteration=maximum(red_channel,disk(disk_diameter))
        maximum_red_intensity[rows,cols]=maximum_red_intensity_iteration[rows,cols]
        maximum_green_intensity_iteration=maximum(green_channel,disk(disk_diameter))
        maximum_green_intensity[rows,cols]=maximum_green_intensity_iteration[rows,cols]
        maximum_blue_intensity_iteration=maximum(blue_channel,disk(disk_diameter))
        maximum_blue_intensity[rows,cols]=maximum_blue_intensity_iteration[rows,cols]
        maximum_hue_iteration=maximum(hue_channel,disk(disk_diameter))
        maximum_hue[rows,cols]=maximum_hue_iteration[rows,cols]
        maximum_saturation_iteration=maximum(saturation_channel,disk(disk_diameter))
        maximum_saturation[rows,cols]=maximum_saturation_iteration[rows,cols]
        maximum_value_iteration=maximum(value_channel,disk(disk_diameter))
        maximum_value[rows,cols]=maximum_value_iteration[rows,cols]
        #std-large
        #std red
        mean_red_intensity_large_iteration1=mean(red_channel ** 2,disk(disk_diameter_large))
        mean_red_intensity_large1[rows,cols]=mean_red_intensity_large_iteration1[rows,cols] 
        mean_red_intensity_large_potency_iteration = mean(red_channel,disk(disk_diameter_large))
        mean_red_intensity_large_potency[rows,cols] =  mean_red_intensity_large_potency_iteration[rows,cols] ** 2  
        std_red = mean_red_intensity_large_potency-mean_red_intensity_large1
        std_red = np.abs(std_red)
        std_red_final = np.sqrt(std_red)
        #std green
        mean_green_intensity_large_iteration1=mean(green_channel ** 2,disk(disk_diameter_large))
        mean_green_intensity_large1[rows,cols]=mean_green_intensity_large_iteration1[rows,cols] 
        mean_green_intensity_large_potency_iteration = mean(green_channel,disk(disk_diameter_large))
        mean_green_intensity_large_potency[rows,cols] =  mean_green_intensity_large_potency_iteration[rows,cols] ** 2  
        std_green = mean_green_intensity_large_potency-mean_green_intensity_large1
        std_green = np.abs(std_green)
        std_green_final = np.sqrt(std_green)
        #std Blue 
        mean_blue_intensity_large_iteration1=mean(blue_channel ** 2,disk(disk_diameter_large))
        mean_blue_intensity_large1[rows,cols]=mean_blue_intensity_large_iteration1[rows,cols] 
        mean_blue_intensity_large_potency_iteration = mean(blue_channel,disk(disk_diameter_large))
        mean_blue_intensity_large_potency[rows,cols] =  mean_blue_intensity_large_potency_iteration[rows,cols] ** 2 
        std_blue =mean_blue_intensity_large_potency - mean_blue_intensity_large1
        std_blue =np.abs(std_blue)
        std_blue_final = np.sqrt(std_blue)
        #std hue
        mean_hue_large_iteration1=mean(hue_channel ** 2,disk(disk_diameter_large))
        mean_hue_large1[rows,cols]=mean_hue_large_iteration1[rows,cols] 
        mean_hue_large_potency_iteration = mean(hue_channel,disk(disk_diameter_large))
        mean_hue_large_potency[rows,cols] =  mean_hue_large_potency_iteration[rows,cols] ** 2
        std_hue =mean_hue_large_potency-mean_hue_large1
        std_hue = np.abs(std_hue)
        std_hue_final = np.sqrt(std_hue)
        #std saturation
        mean_saturation_large_iteration1=mean(saturation_channel ** 2,disk(disk_diameter_large))
        mean_saturation_large1[rows,cols]=mean_saturation_large_iteration1[rows,cols] 
        mean_saturation_large_potency_iteration = mean(saturation_channel,disk(disk_diameter_large))
        mean_saturation_large_potency[rows,cols] =  mean_saturation_large_potency_iteration[rows,cols] ** 2
        std_saturation =mean_saturation_large_potency-mean_saturation_large1
        std_saturation = np.abs(std_saturation)
        std_saturation_final = np.sqrt(std_saturation)
        #std Value
        mean_value_large_iteration1=mean(value_channel ** 2,disk(disk_diameter_large))
        mean_value_large1[rows,cols]=mean_value_large_iteration1[rows,cols] 
        mean_value_large_potency_iteration = mean(value_channel,disk(disk_diameter_large))
        mean_value_large_potency[rows,cols] =  mean_value_large_potency_iteration[rows,cols] ** 2
        std_value =mean_value_large_potency-mean_value_large1
        std_value = np.abs(std_value)
        std_value_final = np.sqrt(std_value)
        #std-small
        #std red
        mean_red_intensity_iteration1=mean(red_channel ** 2,disk(disk_diameter))
        mean_red_intensity_1[rows,cols]=mean_red_intensity_iteration1[rows,cols] 
        mean_red_intensity_potency_iteration = mean(red_channel,disk(disk_diameter))
        mean_red_intensity_potency[rows,cols] =  mean_red_intensity_potency_iteration[rows,cols] ** 2  
        std_red_small = mean_red_intensity_potency-mean_red_intensity_1
        std_red_small = np.abs(std_red_small)
        std_red_final_small = np.sqrt(std_red_small)
        #std green
        mean_green_intensity_iteration1=mean(green_channel ** 2,disk(disk_diameter))
        mean_green_intensity_1[rows,cols]=mean_green_intensity_iteration1[rows,cols] 
        mean_green_intensity_potency_iteration = mean(green_channel,disk(disk_diameter))
        mean_green_intensity_potency[rows,cols] =  mean_green_intensity_potency_iteration[rows,cols] ** 2  
        std_green_small = mean_green_intensity_potency-mean_green_intensity_1
        std_green_small = np.abs(std_green_small)
        std_green_final_small = np.sqrt(std_green_small)
        #std Blue 
        mean_blue_intensity_iteration1=mean(blue_channel ** 2,disk(disk_diameter))
        mean_blue_intensity_1[rows,cols]=mean_blue_intensity_iteration1[rows,cols] 
        mean_blue_intensity_potency_iteration = mean(blue_channel,disk(disk_diameter))
        mean_blue_intensity_potency[rows,cols] =  mean_blue_intensity_potency_iteration[rows,cols] ** 2 
        std_blue_small =mean_blue_intensity_potency - mean_blue_intensity_1
        std_blue_small =np.abs(std_blue_small)
        std_blue_final_small = np.sqrt(std_blue_small)
        #std hue
        mean_hue_iteration1=mean(hue_channel ** 2,disk(disk_diameter))
        mean_hue_1[rows,cols]=mean_hue_iteration1[rows,cols] 
        mean_hue_potency_iteration = mean(hue_channel,disk(disk_diameter))
        mean_hue_potency[rows,cols] =  mean_hue_potency_iteration[rows,cols] ** 2
        std_hue_small =mean_hue_potency-mean_hue_1
        std_hue_small = np.abs(std_hue_small)
        std_hue_final_small = np.sqrt(std_hue_small)
        #std saturation
        mean_saturation_iteration1=mean(saturation_channel ** 2,disk(disk_diameter))
        mean_saturation_1[rows,cols]=mean_saturation_iteration1[rows,cols] 
        mean_saturation_potency_iteration = mean(saturation_channel,disk(disk_diameter))
        mean_saturation_potency[rows,cols] =  mean_saturation_potency_iteration[rows,cols] ** 2
        std_saturation_small =mean_saturation_potency-mean_saturation_1
        std_saturation_small = np.abs(std_saturation_small)
        std_saturation_final_small = np.sqrt(std_saturation_small)
        #std Value
        mean_value_iteration1=mean(value_channel ** 2,disk(disk_diameter))
        mean_value_1[rows,cols]=mean_value_iteration1[rows,cols] 
        mean_value_potency_iteration = mean(value_channel,disk(disk_diameter))
        mean_value_potency[rows,cols] =  mean_value_potency_iteration[rows,cols] ** 2
        std_value_small =mean_value_potency-mean_value_1
        std_value_small = np.abs(std_value_small)
        std_value_final_small = np.sqrt(std_value_small)
        
        #GLCM Features Large Diameter
        glcm_image_entropy_large_iteration = shannon_entropy(retinal_image.preprocessed_image, disk(disk_diameter_large))
        glcm_image_entropy_large[rows,cols] = glcm_image_entropy_large_iteration[rows,cols]
        #This creates the GLCM local matrix which is arg of the functions under:
        glcm_image_large[rows,cols] = 
        glcm_image_contrast_large[rows,cols] = 
        glcm_image_dissimilarity_large[rows,cols] = 
        glcm_image_homogeneity_large[rows,cols] = 
        glcm_image_energy_large[rows,cols] = 
        glcm_image_correlation_large[rows,cols] = 
        glcm_image_ASM_large[rows,cols] = 
        
        #GLCM Features Small Diameter
        glcm_image_entropy_small[rows,cols] = 
        glcm_image_small[rows,cols] = 
        glcm_image_contrast_small[rows,cols] = 
        glcm_image_dissimilarity_small[rows,cols] = 
        glcm_image_homogeneity_small[rows,cols] = 
        glcm_image_energy_small[rows,cols] = 
        glcm_image_correlation_small[rows,cols] = 
        glcm_image_ASM_small[rows,cols] = 
       
        #print(mean_intensity)
        print(i, ':',disk_diameter)
    return mean_red_intensity_large, mean_green_intensity_large, mean_blue_intensity_large, mean_hue_large, mean_saturation_large, mean_value_large, mean_red_intensity, mean_green_intensity, mean_blue_intensity, mean_hue, mean_saturation, mean_value, minimum_red_intensity_large, minimum_green_intensity_large, minimum_blue_intensity_large, minimum_hue_large, minimum_saturation_large, minimum_value_large, minimum_red_intensity, minimum_green_intensity, minimum_blue_intensity, minimum_hue, minimum_saturation, minimum_value, maximum_red_intensity_large, maximum_green_intensity_large, maximum_blue_intensity_large, maximum_hue_large, maximum_saturation_large, maximum_value_large, maximum_red_intensity, maximum_green_intensity, maximum_blue_intensity, maximum_hue, maximum_saturation, maximum_value, std_red_final, std_green_final, std_blue_final, std_hue_final, std_saturation_final, std_value_final, std_red_final_small, std_green_final_small, std_blue_final_small, std_hue_final_small, std_saturation_final_small, std_value_final_small
    
def compute_line_features(retinal_image):
    line_mean = np.zeros((retinal_image.preprocessed_image.shape[0], retinal_image.preprocessed_image.shape[1]))
    line_kurtosis = np.zeros((retinal_image.preprocessed_image.shape[0], retinal_image.preprocessed_image.shape[1]))
    line_skewness = np.zeros((retinal_image.preprocessed_image.shape[0], retinal_image.preprocessed_image.shape[1]))
    std_image=np.zeros((retinal_image.preprocessed_image.shape[0], retinal_image.preprocessed_image.shape[1]))
    tempImg=np.zeros((retinal_image.preprocessed_image.shape[0],retinal_image.preprocessed_image.shape[1]))
    orientations_image=np.zeros((retinal_image.preprocessed_image.shape[0],retinal_image.preprocessed_image.shape[1]))

    i=0
    for props in retinal_image.regions:
        i=i+1
        y0, x0 = props.centroid
        orientation = props.orientation
        orientations_image[retinal_image.labels==i]=orientation;
        x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
        x3 = x0 + math.cos(math.pi/2 + orientation)*0.5*props.major_axis_length;
        y3 = y0 - math.sin(math.pi/2 + orientation) * 0.5 * props.major_axis_length;
        start_x=x0 - math.cos(math.pi/2 + orientation)*0.25*props.major_axis_length;
        start_y=y0 + math.sin(math.pi/2 + orientation) * 0.25 * props.major_axis_length;
        end_x=x0 + math.cos(math.pi/2 + orientation)*0.25*props.major_axis_length;
        end_y=y0 - math.sin(math.pi/2 + orientation) * 0.25 * props.major_axis_length;
        rr, cc, val = line_aa(int(start_x), int(start_y), int(end_x), int(end_y))
        tempImg[rr, cc]=1
        thin_perpendicularlines=skeletonize(tempImg)
        region = thin_perpendicularlines*retinal_image.vessels #0s em todos os sitios menos na interseçao
        
        #Line Standard Deviation
        if math.isnan(np.std(retinal_image.preprocessed_image[region==True])):
            std_image[retinal_image.labels==i] = 0
        else:
            std_image[retinal_image.labels==i] = np.std(retinal_image.preprocessed_image[region==True])
           
        #Line Skewness
        line_skewness[retinal_image.labels==i] = skew(retinal_image.preprocessed_image[region==True])
        
        #Line Kurtosis
        line_kurtosis[retinal_image.labels==i] = kurtosis(retinal_image.preprocessed_image[region==True])
        
        #Line Mean
        line_mean[retinal_image.labels==i] = np.mean(retinal_image.preprocessed_image[region==True])
        
    
    return std_image, line_skewness, line_kurtosis, line_mean
    


#Function MagnitudeGradient computes Gradient of the Sobel Filters. 
#Params: img (image to be loaded), kernel (size of the kernel [1, 3, 5, 7])
def magnitude_gradient(retinal_image):
    # Load the image in grayscale
    img = retinal_image.preprocessed_image.astype(np.float64)
    kernel=1
    #Apply Sobel Filter (this function has a variable Kernel Size)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernel)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernel)
    #Get the norm or absolute value
    dsobelx = np.absolute(sobelx)
    dsobely = np.absolute(sobely)
    #Calculate the magnitude of the gradient
    sm = cv2.magnitude(dsobelx, dsobely)
    return sm

class retinal_image:   
    def __init__(self, name, train_or_test):
        self.name = name
        if train_or_test == 'train':
            path_im = path_to_training_retinal_ims
            path_mask = path_to_training_retinal_masks
            path_vessels = path_to_training_retinal_vessels
            path_arteries = path_to_training_arteries
            path_veins = path_to_training_veins
        elif train_or_test == 'test':
            path_im = path_to_training_retinal_ims
            path_mask = path_to_training_retinal_masks
            path_vessels = path_to_training_retinal_vessels
            path_arteries = path_to_training_arteries
            path_veins = path_to_training_veins
        else:
            print('Invalid mode')
        denoised_img = denoise_nl_means(img_as_float(io.imread(path_im+name)), h=0.01, multichannel=True)
        self.image=img_as_float(io.imread(path_im+name))
        self.mask = io.imread(path_mask+name[:-4]+'_mask.gif', dtype=bool)
        self.preprocessed_image = apply_homomorphic_filtering.apply_homomorphic_filtering(self.mask,denoised_img,0)
        self.vessels = io.imread(path_vessels+name, dtype=bool) #[:-4]+'.png'
        self.arteries = io.imread(path_arteries+name, dtype=bool) #[:-4]+'.png'
        self.veins = io.imread(path_veins+name, dtype=bool) #[:-4]+'.png'
        self.skeleton = apply_skeleton.apply_skeleton(self.vessels,0)
        self.x_opticdisk,self.y_opticdisk=detectOpticDisk.detectOpticDisk((self.image)[:,:,1],0)
        self.coordinates=find_interestpoints.find_interestpoints(self.skeleton,0)
        self.labels=divideIntoSegments.divideIntoSegments(self.skeleton, self.coordinates,0)
        self.skeletonWithoutCrossings=obtainSkeletonWithoutCrossings.obtainSkeletonWithoutCrossings(self.skeleton,self.coordinates)
        self.regions=regionprops(self.labels)
        self.veins_skeleton=self.skeletonWithoutCrossings*self.veins; 
        self.arteries_skeleton=self.skeletonWithoutCrossings*self.arteries; 
        
        
        # AVAILABLE FEATURES: These are place-holders for features that you may want to compute out of each image
        self.saturation = None
        self.hue = None
        self.value = None
        self.red_intensity = None
        self.green_intensity = None
        self.blue_intensity = None
        self.mean_red_intensity_large = None
        self.mean_green_intensity_large = None
        self.mean_blue_intensity_large = None
        self.mean_hue_large = None
        self.mean_saturation_large = None
        self.mean_value_large = None
        self.mean_red_intensity = None
        self.mean_green_intensity = None
        self.mean_blue_intensity = None
        self.mean_hue = None
        self.mean_saturation = None
        self.mean_value = None
        self.minimum_red_intensity_large = None
        self.minimum_green_intensity_large = None
        self.minimum_blue_intensity_large = None
        self.minimum_hue_large = None
        self.minimum_saturation_large = None
        self.minimum_value_large = None
        self.minimum_red_intensity = None
        self.minimum_green_intensity = None
        self.minimum_blue_intensity = None
        self.minimum_hue = None
        self.minimum_saturation = None
        self.minimum_value = None
        self.maximum_red_intensity_large = None
        self.maximum_green_intensity_large = None
        self.maximum_blue_intensity_large = None
        self.maximum_hue_large = None
        self.maximum_saturation_large = None
        self.maximum_value_large = None
        self.maximum_red_intensity = None
        self.maximum_green_intensity = None
        self.maximum_blue_intensity = None
        self.maximum_hue = None
        self.maximum_saturation = None
        self.maximum_value = None
        self.std_red_final = None
        self.std_green_final = None
        self.std_blue_final = None
        self.std_hue_final = None
        self.std_saturation_final = None
        self.std_value_final = None
        self.std_red_final_small = None
        self.std_green_final_small = None
        self.std_blue_final_small = None
        self.std_hue_final_small = None
        self.std_saturation_final_small = None
        self.std_value_final_small = None
        self.distance_to_optic_disk = None
        self.distance_from_image_center = None
        self.std_image = None
        self.line_skewness = None
        self.line_kurtosis = None
        self.line_mean = None
        self.magnitude_gradient = None
        
    # The retinal_image object knows how to compute these features. 
    # It does that by calling to the functions defined in the previous cells    
   
    def load_hue(self):
        self.hue = compute_hue(self)
        
    def load_saturation(self):
        self.saturation = compute_saturation(self)
        
    def load_value(self):
        self.value = compute_value(self)
    
    def load_red_intensity(self):
        self.red_intensity = compute_red_intensity(self)       
        
    def load_green_intensity(self):
        self.green_intensity = compute_green_intensity(self)  
        
    def load_blue_intensity(self):
        self.blue_intensity = compute_blue_intensity(self) 
        
    def load_local_features(self):
        self.mean_red_intensity_large, self.mean_green_intensity_large, self.mean_blue_intensity_large, self.mean_hue_large, self.mean_saturation_large, self.mean_value_large, self.mean_red_intensity, self.mean_green_intensity, self.mean_blue_intensity, self.mean_hue, self.mean_saturation, self.mean_value, self.minimum_red_intensity_large, self.minimum_green_intensity_large, self.minimum_blue_intensity_large, self.minimum_hue_large, self.minimum_saturation_large, self.minimum_value_large, self.minimum_red_intensity, self.minimum_green_intensity, self.minimum_blue_intensity, self.minimum_hue, self.minimum_saturation, self.minimum_value, self.maximum_red_intensity_large, self.maximum_green_intensity_large, self.maximum_blue_intensity_large, self.maximum_hue_large, self.maximum_saturation_large, self.maximum_value_large, self.maximum_red_intensity, self.maximum_green_intensity, self.maximum_blue_intensity, self.maximum_hue, self.maximum_saturation, self.maximum_value, self.std_red_final, self.std_green_final, self.std_blue_final, self.std_hue_final, self.std_saturation_final, self.std_value_final,  self.std_red_final_small, self.std_green_final_small, self.std_blue_final_small, self.std_hue_final_small, self.std_saturation_final_small, self.std_value_final_small   = compute_local_features(self)
    
    def load_distance_to_optic_disk(self):
        self.distance_to_optic_disk = compute_distance_to_optic_disk(self)
        
    def load_distance_from_image_center(self):   
        self.distance_from_image_center = compute_distance_from_image_center(self)
    
    def load_compute_line_features(self):
        self.std_image, self.line_skewness, self.line_kurtosis, self.line_mean = compute_line_features(self)

    def load_magnitude_gradient(self):
        self.magnitude_gradient = magnitude_gradient(self)