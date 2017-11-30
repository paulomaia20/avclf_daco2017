import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, img_as_float
import apply_skeleton
import find_interestpoints 
import divideIntoSegments
import detectOpticDisk
from scipy import ndimage
from skimage.measure import regionprops
from skimage.filters import rank
from skimage.filters.rank import maximum, minimum, mean
from skimage.morphology import skeletonize, square, disk

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
  
def compute_local_features(retinal_image):
    red_channel=retinal_image.image[:,:,0]
    green_channel=retinal_image.image[:,:,1]
    blue_channel=retinal_image.image[:,:,2]
    hue_channel=color.rgb2hsv(retinal_image.image)[:,:,0]
    saturation_channel=color.rgb2hsv(retinal_image.image)[:,:,1]
    value_channel=color.rgb2hsv(retinal_image.image)[:,:,2]
    skeleton_pixels=np.nonzero(retinal_image.skeleton)
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
    diameter=distanceTransform * retinal_image.skeleton
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
    return mean_red_intensity_large, mean_value_large, minimum_red_intensity, minimum_value, minimum_blue_intensity_large, minimum_hue_large, maximum_blue_intensity_large, maximum_saturation_large, maximum_value_large
    
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
            
        self.image = img_as_float(io.imread(path_im+name))
        self.mask = io.imread(path_mask+name[:-4]+'_mask.gif', dtype=bool)
        self.vessels = io.imread(path_vessels+name, dtype=bool) #[:-4]+'.png'
        self.arteries = io.imread(path_arteries+name, dtype=bool) #[:-4]+'.png'
        self.veins = io.imread(path_veins+name, dtype=bool) #[:-4]+'.png'
        self.skeleton = apply_skeleton.apply_skeleton(self.vessels,0)
        self.veins_skeleton=self.skeleton*self.veins; 
        self.arteries_skeleton=self.skeleton*self.arteries; 
        self.x_opticdisk,self.y_opticdisk=detectOpticDisk.detectOpticDisk((self.image)[:,:,1],0)
        self.coordinates=find_interestpoints.find_interestpoints(self.skeleton,0)
        self.labels=divideIntoSegments.divideIntoSegments(self.skeleton, self.coordinates,0)
        self.regions=regionprops(self.labels)
        
        # AVAILABLE FEATURES: These are place-holders for features that you may want to compute out of each image
        self.saturation = None
        self.hue = None
        self.value = None
        self.red_intensity = None
        self.green_intensity = None
        self.blue_intensity = None
        self.mean_red_intensity_large = None
        self.mean_value_large = None
        self.minimum_red_intensity = None
        self.minimum_value = None
        self.minimum_blue_intensity_large = None
        self.minimum_hue_large = None
        self.maximum_blue_intensity_large = None
        self.maximum_saturation_large = None
        self.maximum_value_large = None
        self.distance_to_optic_disk = None
        
    # The retinal_image object knows how to compute these features. 
    # It does that by calling to the functions defined in the previous cells    
    def load_saturation(self):
        # this calls an external function. If the attribute has not been initialized above, this will crash.
        self.saturation = compute_saturation(self)
        
    def load_hue(self):
        self.hue = compute_hue(self)
        
    def load_value(self):
        self.value = compute_value(self)
    
    def load_red_intensity(self):
        self.red_intensity = compute_red_intensity(self)       
        
    def load_green_intensity(self):
        self.green_intensity = compute_green_intensity(self)  
        
    def load_blue_intensity(self):
        self.blue_intensity = compute_blue_intensity(self) 
        
    def load_local_features(self):
        self.mean_red_intensity_large, self.mean_value_large, self.minimum_red_intensity, self.minimum_value, self.minimum_blue_intensity_large, self.minimum_hue_large, self.maximum_blue_intensity_large, self.maximum_saturation_large, self.maximum_value_large = compute_local_features(self)
    
    def load_distance_to_optic_disk(self):
        self.distance_to_optic_disk = compute_distance_to_optic_disk(self)
