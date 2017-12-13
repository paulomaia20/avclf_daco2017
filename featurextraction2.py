import os

import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage import img_as_float
from skimage import color
import retinal_image as ri

path_to_training_retinal_ims = 'data/training/images/'
retinal_im_list = os.listdir(path_to_training_retinal_ims)
nr_ims = len(retinal_im_list) # same as above
nr_features = 20 # red intensity and saturation, ... 

# the number of samples depends on the number of vessel pixels we select (WHICH NUMBER SHOULD WE USE??)
nr_artery_samples_per_image = 300
nr_vein_samples_per_image = 300
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
    image.load_local_features()
    image.load_distance_to_optic_disk()
    image.load_distance_from_image_center()
    
    # extract samples from arteries
    arteries_samples_red_intensity = image.red_intensity[image.arteries == True]
    # extract samples from veins
    veins_samples_red_intensity = image.red_intensity[image.veins == True]
        
     # extract samples from arteries
    arteries_samples_std_blue_final= image.std_blue_final[image.arteries == True]
    # extract samples from veins
    veins_samples_std_blue_final = image.std_blue_final[image.veins == True]
    
     # extract samples from arteries
    arteries_samples_std_value_final = image.std_value_final[image.arteries == True]
    # extract samples from veins
    veins_samples_std_value_final = image.std_value_final[image.veins == True]
    
     # extract samples from arteries
    arteries_samples_std_red_final = image.std_red_final[image.arteries == True]
    # extract samples from veins
    veins_samples_std_red_final = image.std_red_final[image.veins == True]
    
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
    
    # extract samples from arteries
    arteries_samples_distance_optic_disk = image.distance_to_optic_disk[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_distance_optic_disk = image.distance_to_optic_disk[image.veins_skeleton == True]
    
    # extract samples from arteries
    arteries_mean_red_intensity_large = image.mean_red_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_mean_red_intensity_large = image.mean_red_intensity_large[image.veins_skeleton == True]

    # extract samples from arteries
    arteries_samples_mean_value_large = image.mean_value_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_mean_value_large = image.mean_value_large[image.veins_skeleton == True]
    
    # extract samples from arteries
    arteries_samples_minimum_red_intensity = image.minimum_red_intensity[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_red_intensity = image.minimum_red_intensity[image.veins_skeleton == True]
    
    # extract samples from arteries
    arteries_samples_minimum_value = image.minimum_value[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_value = image.minimum_value[image.veins_skeleton == True]
    
    # extract samples from arteries
    arteries_samples_minimum_blue_intensity_large = image.minimum_blue_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_blue_intensity_large = image.minimum_blue_intensity_large[image.veins_skeleton == True]
    
      
    # extract samples from arteries
    arteries_samples_minimum_hue_large = image.minimum_hue_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_hue_large = image.minimum_hue_large[image.veins_skeleton == True]
      
    # extract samples from arteries
    arteries_samples_maximum_blue_intensity_large = image.maximum_blue_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_blue_intensity_large = image.maximum_blue_intensity_large[image.veins_skeleton == True]
       
    # extract samples from arteries
    arteries_samples_maximum_saturation_large = image.maximum_saturation_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_saturation_large = image.maximum_saturation_large[image.veins_skeleton == True]
    
    # extract samples from arteries
    arteries_samples_maximum_value_large = image.maximum_value_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_value_large = image.maximum_value_large[image.veins_skeleton == True]
    
    #extract samples from arteries
    arteries_samples_distance_from_image_center = image.distance_from_image_center[image.arteries_skeleton == True]
    #extract samples from veins
    veins_samples_distance_from_image_center= image.distance_from_image_center[image.veins_skeleton == True]

    # randomly choose which artery examples to use, 1st column for red, 2nd for green, 3rd for blue, 4th for hue, 5th for saturation and 6th for value
    random_sample = np.random.randint(len(arteries_samples_maximum_value_large), size=nr_artery_samples_per_image)

    # first 20 examples are arteries
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,0] = arteries_samples_distance_optic_disk[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,1] = arteries_mean_red_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,2] = arteries_samples_mean_value_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,3] = arteries_samples_minimum_red_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,4] = arteries_samples_minimum_value[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,5] = arteries_samples_minimum_blue_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,6] = arteries_samples_minimum_hue_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,7] = arteries_samples_maximum_blue_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,8] = arteries_samples_maximum_saturation_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,9] = arteries_samples_maximum_value_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,10] = arteries_samples_distance_from_image_center[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,11] = arteries_samples_red_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,12] = arteries_samples_green_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,13] = arteries_samples_blue_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,14] = arteries_samples_hue[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,15] = arteries_samples_saturation[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,16] = arteries_samples_value[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,17] = arteries_samples_std_blue_final[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,18] = arteries_samples_std_value_final[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,19] = arteries_samples_std_red_final[random_sample]
       
    # arteries are the negative class
    y[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image] = np.zeros(nr_artery_samples_per_image,)  
    
    # randomly choose which vein examples to use                                          
    random_sample = np.random.randint(len(veins_samples_maximum_value_large), size=nr_vein_samples_per_image)
    
    # second twenty examples are veins
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,0] = veins_samples_distance_optic_disk[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,1] = veins_mean_red_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,2] = veins_samples_mean_value_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,3] = veins_samples_minimum_red_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,4] = veins_samples_minimum_value[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,5] = veins_samples_minimum_blue_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,6] = veins_samples_minimum_hue_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,7] = veins_samples_maximum_blue_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,8] = veins_samples_maximum_saturation_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,9] = veins_samples_maximum_value_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,10] = arteries_samples_distance_from_image_center[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,11] = veins_samples_red_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,12] = veins_samples_green_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,13] = veins_samples_blue_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,14] = veins_samples_hue[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,15] = veins_samples_saturation[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,16] = veins_samples_value[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,17] = veins_samples_std_blue_final[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,18] = veins_samples_std_value_final[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,19] = veins_samples_std_red_final[random_sample]

    # veins are the positive class
    y[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image] = np.ones(nr_vein_samples_per_image,)  
    
    print(i+1, '/', nr_ims)