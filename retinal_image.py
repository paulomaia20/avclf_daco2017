import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, img_as_float

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
        
        # AVAILABLE FEATURES: These are place-holders for features that you may want to compute out of each image
        self.saturation = None
        self.hue = None
        self.value = None
        self.red_intensity = None
        self.green_intensity = None
        self.blue_intensity = None
        
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