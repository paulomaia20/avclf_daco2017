import os

import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage import img_as_float
from skimage import color
import retinal_image as ri


def extractFeatures(diameter):
    path_to_training_retinal_ims = 'data/training/images/'
    retinal_im_list = os.listdir(path_to_training_retinal_ims)
    nr_ims = len(retinal_im_list) # same as above
    nr_features = 6 # red intensity and saturation, ... 
    
    # the number of samples depends on the number of vessel pixels we select (WHICH NUMBER SHOULD WE USE??)
    nr_artery_samples_per_image = 30
    nr_vein_samples_per_image = 30
    # nr of pixel samples is the sum of these
    nr_samples_per_image =  nr_artery_samples_per_image + nr_vein_samples_per_image
    # total number of examples is number of samples per image multiplied by nr of images
    nr_examples = nr_ims *nr_samples_per_image
    
    # pre-allocate training examples matrix
    X = np.zeros([nr_examples, nr_features])
    # pre-allocate ground-truth vector
    y = np.zeros([nr_examples,])
    
    for i in range(nr_ims):
        im_name = retinal_im_list[i]
        image = ri.retinal_image(im_name, 'train')
    
        # Load features
        image.load_red_intensity()
        image.load_green_intensity()
        image.load_blue_intensity()
        image.load_saturation()
        image.load_hue()
        image.load_value()
        
        # extract samples from arteries
        arteries_samples_red_intensity = image.red_intensity[image.arteries == True]
        # extract samples from veins
        veins_samples_red_intensity = image.red_intensity[image.veins == True]
        
        # extract samples from arteries
        arteries_samples_green_intensity = image.green_intensity[image.arteries == True]
        # extract samples from veins
        veins_samples_green_intensity = image.green_intensity[image.veins == True]
    
        # extract samples from arteries
        arteries_samples_blue_intensity = image.blue_intensity[image.arteries == True]
        # extract samples from veins
        veins_samples_blue_intensity = image.blue_intensity[image.veins == True]
    
        # extract samples from arteries
        arteries_samples_saturation = image.saturation[image.arteries == True]
        # extract samples from veins
        veins_samples_saturation = image.saturation[image.veins == True]
        
        # extract samples from arteries
        arteries_samples_hue = image.hue[image.arteries == True]
        # extract samples from veins
        veins_samples_hue = image.hue[image.veins == True]
        
        # extract samples from arteries
        arteries_samples_value = image.value[image.arteries == True]
        # extract samples from veins
        veins_samples_value = image.value[image.veins == True]
    
        # randomly choose which artery examples to use, 1st column for red, 2nd for green, 3rd for blue, 4th for hue, 5th for saturation and 6th for value
        random_sample = np.random.randint(len(arteries_samples_red_intensity), size=nr_artery_samples_per_image)
    
        # first 20 examples are arteries
        X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,0] = arteries_samples_red_intensity[random_sample]
        X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,1] = arteries_samples_green_intensity[random_sample]
        X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,2] = arteries_samples_blue_intensity[random_sample]
        X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,3] = arteries_samples_hue[random_sample]
        X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,4] = arteries_samples_saturation[random_sample]
        X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,5] = arteries_samples_value[random_sample]
    
        # arteries are the negative class
        y[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image] = np.zeros(nr_artery_samples_per_image,)  
        
        # randomly choose which vein examples to use                                          
        random_sample = np.random.randint(len(veins_samples_red_intensity), size=nr_vein_samples_per_image)
        
        # second twenty examples are veins
        X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,0] = veins_samples_red_intensity[random_sample]
        X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,1] = veins_samples_green_intensity[random_sample]
        X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,2] = veins_samples_blue_intensity[random_sample]
        X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,3] = veins_samples_hue[random_sample]
        X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,4] = veins_samples_saturation[random_sample]
        X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,5] = veins_samples_value[random_sample]
        
        # veins are the positive class
        y[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image] = np.ones(nr_vein_samples_per_image,)  
        
        print(i+1, '/', nr_ims)