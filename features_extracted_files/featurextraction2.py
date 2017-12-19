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
nr_features = 63 # red intensity and saturation, ... 

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
    image.load_compute_line_features()
    image.load_magnitude_gradient()
    
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
    arteries_samples_hue = image.hue[image.arteries == True]
    # extract samples from veins
    veins_samples_hue = image.hue[image.veins == True]
    
    # extract samples from arteries
    arteries_samples_saturation = image.saturation[image.arteries == True]
    # extract samples from veins
    veins_samples_saturation = image.saturation[image.veins == True]
           
    # extract samples from arteries
    arteries_samples_value = image.value[image.arteries == True]
    # extract samples from veins
    veins_samples_value = image.value[image.veins == True]
    
    
    #mean-large
    # extract samples from arteries
    arteries_mean_red_intensity_large = image.mean_red_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_mean_red_intensity_large = image.mean_red_intensity_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_mean_green_intensity_large = image.mean_green_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_mean_green_intensity_large = image.mean_green_intensity_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_mean_blue_intensity_large = image.mean_blue_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_mean_blue_intensity_large = image.mean_blue_intensity_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_mean_hue_large = image.mean_hue_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_mean_hue_large = image.mean_hue_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_mean_saturation_large = image.mean_saturation_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_mean_saturation_large = image.mean_saturation_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_mean_value_large = image.mean_value_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_mean_value_large = image.mean_value_large[image.veins_skeleton == True]
    
    #mean-small
    # extract samples from arteries
    arteries_mean_red_intensity = image.mean_red_intensity[image.arteries_skeleton == True]
    # extract samples from veins
    veins_mean_red_intensity = image.mean_red_intensity[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_mean_green_intensity = image.mean_green_intensity[image.arteries_skeleton == True]
    # extract samples from veins
    veins_mean_green_intensity = image.mean_green_intensity[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_mean_blue_intensity = image.mean_blue_intensity[image.arteries_skeleton == True]
    # extract samples from veins
    veins_mean_blue_intensity = image.mean_blue_intensity[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_mean_hue = image.mean_hue[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_mean_hue = image.mean_hue[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_mean_saturation = image.mean_saturation[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_mean_saturation = image.mean_saturation[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_mean_value = image.mean_value[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_mean_value = image.mean_value[image.veins_skeleton == True]
    
    #minimum- large
    # extract samples from arteries
    arteries_samples_minimum_red_intensity_large = image.minimum_red_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_red_intensity_large = image.minimum_red_intensity_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_minimum_green_intensity_large = image.minimum_green_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_green_intensity_large = image.minimum_green_intensity_large[image.veins_skeleton == True]
     # extract samples from arteries
    arteries_samples_minimum_blue_intensity_large = image.minimum_blue_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_blue_intensity_large = image.minimum_blue_intensity_large[image.veins_skeleton == True]  
    # extract samples from arteries
    arteries_samples_minimum_hue_large = image.minimum_hue_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_hue_large = image.minimum_hue_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_minimum_saturation_large = image.minimum_saturation_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_saturation_large = image.minimum_saturation_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_minimum_value_large = image.minimum_value_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_value_large = image.minimum_value_large[image.veins_skeleton == True]
    
    #minimum- small
    # extract samples from arteries
    arteries_samples_minimum_red_intensity = image.minimum_red_intensity[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_red_intensity = image.minimum_red_intensity[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_minimum_green_intensity = image.minimum_green_intensity[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_green_intensity = image.minimum_green_intensity[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_minimum_blue_intensity = image.minimum_blue_intensity[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_blue_intensity = image.minimum_blue_intensity[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_minimum_hue = image.minimum_hue[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_hue = image.minimum_hue[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_minimum_saturation = image.minimum_saturation[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_saturation = image.minimum_saturation[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_minimum_value = image.minimum_value[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_minimum_value = image.minimum_value[image.veins_skeleton == True]
    
    #maximum- large
    # extract samples from arteries
    arteries_samples_maximum_red_intensity_large = image.maximum_red_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_red_intensity_large = image.maximum_red_intensity_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_maximum_green_intensity_large = image.maximum_green_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_green_intensity_large = image.maximum_green_intensity_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_maximum_blue_intensity_large = image.maximum_blue_intensity_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_blue_intensity_large = image.maximum_blue_intensity_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_maximum_hue_large = image.maximum_hue_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_hue_large = image.maximum_hue_large[image.veins_skeleton == True]  
    # extract samples from arteries
    arteries_samples_maximum_saturation_large = image.maximum_saturation_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_saturation_large = image.maximum_saturation_large[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_maximum_value_large = image.maximum_value_large[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_value_large = image.maximum_value_large[image.veins_skeleton == True]
    
    #maximum- small
    # extract samples from arteries
    arteries_samples_maximum_red_intensity = image.maximum_red_intensity[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_red_intensity = image.maximum_red_intensity[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_maximum_green_intensity = image.maximum_green_intensity[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_green_intensity = image.maximum_green_intensity[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_maximum_blue_intensity = image.maximum_blue_intensity[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_blue_intensity = image.maximum_blue_intensity[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_maximum_hue = image.maximum_hue[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_hue = image.maximum_hue[image.veins_skeleton == True]  
    # extract samples from arteries
    arteries_samples_maximum_saturation = image.maximum_saturation[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_saturation = image.maximum_saturation[image.veins_skeleton == True]
    # extract samples from arteries
    arteries_samples_maximum_value = image.maximum_value[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_maximum_value = image.maximum_value[image.veins_skeleton == True]
    
    #std-large
    # extract samples from arteries
    arteries_samples_std_red_final = image.std_red_final[image.arteries == True]
    # extract samples from veins
    veins_samples_std_red_final = image.std_red_final[image.veins == True]
    # extract samples from arteries
    arteries_samples_std_green_final= image.std_green_final[image.arteries == True]
    # extract samples from veins
    veins_samples_std_green_final = image.std_green_final[image.veins == True]    
    # extract samples from arteries
    arteries_samples_std_blue_final= image.std_blue_final[image.arteries == True]
    # extract samples from veins
    veins_samples_std_blue_final = image.std_blue_final[image.veins == True]
    # extract samples from arteries
    arteries_samples_std_hue_final = image.std_hue_final[image.arteries == True]
    # extract samples from veins
    veins_samples_std_hue_final = image.std_hue_final[image.veins == True]
    # extract samples from arteries
    arteries_samples_std_saturation_final = image.std_saturation_final[image.arteries == True]
    # extract samples from veins
    veins_samples_std_saturation_final = image.std_saturation_final[image.veins == True]
    # extract samples from arteries
    arteries_samples_std_value_final = image.std_value_final[image.arteries == True]
    # extract samples from veins
    veins_samples_std_value_final = image.std_value_final[image.veins == True]
    
    #std-small
    # extract samples from arteries
    arteries_samples_std_red_final_small = image.std_red_final_small[image.arteries == True]
    # extract samples from veins
    veins_samples_std_red_final_small = image.std_red_final_small[image.veins == True]
    # extract samples from arteries
    arteries_samples_std_green_final_small= image.std_green_final_small[image.arteries == True]
    # extract samples from veins
    veins_samples_std_green_final_small = image.std_green_final_small[image.veins == True]    
    # extract samples from arteries
    arteries_samples_std_blue_final_small= image.std_blue_final_small[image.arteries == True]
    # extract samples from veins
    veins_samples_std_blue_final_small = image.std_blue_final_small[image.veins == True]
    # extract samples from arteries
    arteries_samples_std_hue_final_small = image.std_hue_final_small[image.arteries == True]
    # extract samples from veins
    veins_samples_std_hue_final_small = image.std_hue_final_small[image.veins == True]
    # extract samples from arteries
    arteries_samples_std_saturation_final_small = image.std_saturation_final_small[image.arteries == True]
    # extract samples from veins
    veins_samples_std_saturation_final_small = image.std_saturation_final_small[image.veins == True]
    # extract samples from arteries
    arteries_samples_std_value_final_small = image.std_value_final_small[image.arteries == True]
    # extract samples from veins
    veins_samples_std_value_final_small = image.std_value_final_small[image.veins == True]
    
    
    
    # extract samples from arteries
    arteries_samples_distance_optic_disk = image.distance_to_optic_disk[image.arteries_skeleton == True]
    # extract samples from veins
    veins_samples_distance_optic_disk = image.distance_to_optic_disk[image.veins_skeleton == True]
    
    #extract samples from arteries
    arteries_samples_distance_from_image_center = image.distance_from_image_center[image.arteries_skeleton == True]
    #extract samples from veins
    veins_samples_distance_from_image_center= image.distance_from_image_center[image.veins_skeleton == True]
    
    #Line Feautures
    #Standard Deviation
    #extract samples from arteries
    arteries_samples_std_image = image.std_image[image.arteries_skeleton == True]
    #extract samples from veins
    veins_samples_std_image = image.std_image[image.veins_skeleton == True]
    #Skewness
    #extract samples from arteries
    arteries_samples_line_skewness = image.line_skewness[image.arteries_skeleton == True]
    #extract samples from veins
    veins_samples_line_skewness = image.line_skewness[image.veins_skeleton == True]
    #Kurtosis
    #extract samples from arteries
    arteries_samples_line_kurtosis = image.line_kurtosis[image.arteries_skeleton == True]
    #extract samples from veins
    veins_samples_line_kurtosis = image.line_kurtosis[image.veins_skeleton == True]
    #Mean
    #extract samples from arteries
    arteries_samples_line_mean = image.line_mean[image.arteries_skeleton == True]
    #extract samples from veins
    veins_samples_line_mean = image.line_mean[image.veins_skeleton == True]
    
    #Magnitude Gradient
    # extract samples from arteries
    arteries_samples_magnitude_gradient = image.magnitude_gradient[image.arteries == True]
    # extract samples from veins
    veins_samples_magnitude_gradient = image.magnitude_gradient[image.veins == True]
    
    #    #GLCM Features Large Diameter
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_entropy_large = image.glcm_image_entropy_large[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_entropy_large = image.glcm_image_entropy_large[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_contrast_large = image.glcm_image_contrast_large[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_contrast_large = image.glcm_image_contrast_large[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_dissimilarity_large = image.glcm_image_dissimilarity_large[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_dissimilarity_large = image.glcm_image_dissimilarity_large[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_homogeneity_large = image.glcm_image_homogeneity_large[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_homogeneity_large = image.glcm_image_homogeneity_large[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_energy_large = image.glcm_image_energy_large[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_energy_large = image.glcm_image_energy_large[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_correlation_large = image.glcm_image_correlation_large[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_correlation_large = image.glcm_image_correlation_large[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_ASM_large = image.glcm_image_ASM_large[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_ASM_large = image.glcm_image_ASM_large[image.veins == True]
    
    #    #GLCM Features Small Diameter
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_entropy_small = image.glcm_image_entropy_small[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_entropy_small = image.glcm_image_entropy_small[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_contrast_small = image.glcm_image_contrast_small[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_contrast_small = image.glcm_image_contrast_small[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_dissimilarity_small = image.glcm_image_dissimilarity_small[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_dissimilarity_small = image.glcm_image_dissimilarity_small[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_homogeneity_small = image.glcm_image_homogeneity_small[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_homogeneity_small = image.glcm_image_homogeneity_small[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_energy_small = image.glcm_image_energy_small[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_energy_small = image.glcm_image_energy_small[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_correlation_small = image.glcm_image_correlation_small[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_correlation_small = image.glcm_image_correlation_small[image.veins == True]
    #    # extract samples from arteries
    #    arteries_samples_glcm_image_ASM_small = image.glcm_image_ASM_small[image.arteries == True]
    #    # extract samples from veins
    #    veins_samples_glcm_image_ASM_small = image.glcm_image_ASM_small[image.veins == True]
    
    
    # randomly choose which artery examples to use, 1st column for red, 2nd for green, 3rd for blue, 4th for hue, 5th for saturation and 6th for value
    random_sample = np.random.randint(len(arteries_samples_maximum_value_large), size=nr_artery_samples_per_image)
    
    # first 75 examples are arteries
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,0] = arteries_samples_red_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,1] = arteries_samples_green_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,2] = arteries_samples_blue_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,3] = arteries_samples_hue[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,4] = arteries_samples_saturation[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,5] = arteries_samples_value[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,6] = arteries_mean_red_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,7] = arteries_mean_green_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,8] = arteries_mean_blue_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,9] = arteries_samples_mean_hue_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,10] = arteries_samples_mean_saturation_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,11] = arteries_samples_mean_value_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,12] = arteries_mean_red_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,13] = arteries_mean_green_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,14] = arteries_mean_blue_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,15] = arteries_samples_mean_hue[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,16] = arteries_samples_mean_saturation[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,17] = arteries_samples_mean_value[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,18] = arteries_samples_minimum_red_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,19] = arteries_samples_minimum_green_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,20] = arteries_samples_minimum_blue_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,21] = arteries_samples_minimum_hue_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,22] = arteries_samples_minimum_saturation_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,23] = arteries_samples_minimum_value_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,24] = arteries_samples_minimum_red_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,25] = arteries_samples_minimum_green_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,26] = arteries_samples_minimum_blue_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,27] = arteries_samples_minimum_hue[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,28] = arteries_samples_minimum_saturation[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,29] = arteries_samples_minimum_value[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,30] = arteries_samples_maximum_red_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,31] = arteries_samples_maximum_green_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,32] = arteries_samples_maximum_blue_intensity_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,33] = arteries_samples_maximum_hue_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,34] = arteries_samples_maximum_saturation_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,35] = arteries_samples_maximum_value_large[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,36] = arteries_samples_maximum_red_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,37] = arteries_samples_maximum_green_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,38] = arteries_samples_maximum_blue_intensity[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,39] = arteries_samples_maximum_hue[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,40] = arteries_samples_maximum_saturation[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,41] = arteries_samples_maximum_value[random_sample] 
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,42] = arteries_samples_std_red_final[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,43] = arteries_samples_std_green_final[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,44] = arteries_samples_std_blue_final[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,45] = arteries_samples_std_hue_final[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,46] = arteries_samples_std_saturation_final[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,47] = arteries_samples_std_value_final[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,48] = arteries_samples_std_red_final_small[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,49] = arteries_samples_std_green_final_small[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,50] = arteries_samples_std_blue_final_small[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,51] = arteries_samples_std_hue_final_small[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,52] = arteries_samples_std_saturation_final_small[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,53] = arteries_samples_std_value_final_small[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,54] = arteries_samples_distance_optic_disk[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,55] = arteries_samples_distance_from_image_center[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,56] = arteries_samples_std_image[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,57] = arteries_samples_line_skewness[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,58] = arteries_samples_line_kurtosis[random_sample]
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,59] = arteries_samples_line_mean[random_sample]
    #We have to save the value of the gradient of each channel.
    #Compute Maximum of each channel
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,60] = np.amax(arteries_samples_magnitude_gradient[:,0][random_sample])
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,61] = np.amax(arteries_samples_magnitude_gradient[:,1][random_sample])
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,62] = np.amax(arteries_samples_magnitude_gradient[:,2][random_sample])
    #Compute Minimum of each channel
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,63] = np.amin(arteries_samples_magnitude_gradient[:,0][random_sample])
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,64] = np.amin(arteries_samples_magnitude_gradient[:,1][random_sample])
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,65] = np.amin(arteries_samples_magnitude_gradient[:,2][random_sample])
    #Compute Mean of each channel
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,66] = np.mean(arteries_samples_magnitude_gradient[:,0][random_sample])
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,67] = np.mean(arteries_samples_magnitude_gradient[:,1][random_sample])
    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,68] = np.mean(arteries_samples_magnitude_gradient[:,2][random_sample])
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,61] = arteries_samples_glcm_image_entropy_large[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,62] = arteries_samples_glcm_image_contrast_large[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,63] = arteries_samples_glcm_image_dissimilarity_large[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,64] = arteries_samples_glcm_image_homogeneity_large[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,65] = arteries_samples_glcm_image_energy_large[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,66] = arteries_samples_glcm_image_correlation_large[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,67] = arteries_samples_glcm_image_ASM_large[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,68] = arteries_samples_glcm_image_contrast_small[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,69] = arteries_samples_glcm_image_dissimilarity_small[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,70] = arteries_samples_glcm_image_homogeneity_small[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,71] = arteries_samples_glcm_image_energy_small[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,72] = arteries_samples_glcm_image_correlation_small[random_sample]
    #    X[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image,73] = arteries_samples_glcm_image_ASM_small[random_sample]
    # arteries are the negative class
    y[i*nr_samples_per_image:i*nr_samples_per_image+nr_artery_samples_per_image] = np.zeros(nr_artery_samples_per_image,)  
    
    # randomly choose which vein examples to use                                          
    random_sample = np.random.randint(len(veins_samples_maximum_value_large), size=nr_vein_samples_per_image)
    
    # second 75 examples are veins
       
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,0] = veins_samples_red_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,1] = veins_samples_green_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,2] = veins_samples_blue_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,3] = veins_samples_hue[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,4] = veins_samples_saturation[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,5] = veins_samples_value[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,6] = veins_mean_red_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,7] = veins_mean_green_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,8] = veins_mean_blue_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,9] = veins_samples_mean_hue_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,10] = veins_samples_mean_saturation_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,11] = veins_samples_mean_value_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,12] = veins_mean_red_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,13] = veins_mean_green_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,14] = veins_mean_blue_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,15] = veins_samples_mean_hue[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,16] = veins_samples_mean_saturation[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,17] = veins_samples_mean_value[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,18] = veins_samples_minimum_red_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,19] = veins_samples_minimum_green_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,20] = veins_samples_minimum_blue_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,21] = veins_samples_minimum_hue_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,22] = veins_samples_minimum_saturation_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,23] = veins_samples_minimum_value_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,24] = veins_samples_minimum_red_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,25] = veins_samples_minimum_green_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,26] = veins_samples_minimum_blue_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,27] = veins_samples_minimum_hue[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,28] = veins_samples_minimum_saturation[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,29] = veins_samples_minimum_value[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,30] = veins_samples_maximum_red_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,31] = veins_samples_maximum_green_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,32] = veins_samples_maximum_blue_intensity_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,33] = veins_samples_maximum_hue_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,34] = veins_samples_maximum_saturation_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,35] = veins_samples_maximum_value_large[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,36] = veins_samples_maximum_red_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,37] = veins_samples_maximum_green_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,38] = veins_samples_maximum_blue_intensity[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,39] = veins_samples_maximum_hue[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,40] = veins_samples_maximum_saturation[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,41] = veins_samples_maximum_value[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,42] = veins_samples_std_red_final[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,43] = veins_samples_std_green_final[random_sample]   
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,44] = veins_samples_std_blue_final[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,45] = veins_samples_std_hue_final[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,46] = veins_samples_std_saturation_final[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,47] = veins_samples_std_value_final[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,48] = veins_samples_std_red_final_small[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,49] = veins_samples_std_green_final_small[random_sample]   
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,50] = veins_samples_std_blue_final_small[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,51] = veins_samples_std_hue_final_small[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,52] = veins_samples_std_saturation_final_small[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,53] = veins_samples_std_value_final_small[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,54] = veins_samples_distance_optic_disk[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,55] = veins_samples_distance_from_image_center[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,56] = veins_samples_std_image[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,57] = veins_samples_line_skewness[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,58] = veins_samples_line_kurtosis[random_sample]
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,59] = veins_samples_line_mean[random_sample]
    #We have to save the value of the gradient of each channel.
    #Compute Maximum of each channel
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,60] = np.amax(veins_samples_magnitude_gradient[:,0][random_sample])
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,61] = np.amax(veins_samples_magnitude_gradient[:,1][random_sample])
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,62] = np.amax(veins_samples_magnitude_gradient[:,2][random_sample])
    #Compute Minimum of each channel
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,63] = np.amin(veins_samples_magnitude_gradient[:,0][random_sample])
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,64] = np.amin(veins_samples_magnitude_gradient[:,1][random_sample])
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,65] = np.amin(veins_samples_magnitude_gradient[:,2][random_sample])
    #Compute Mean of each channel
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,66] = np.mean(veins_samples_magnitude_gradient[:,0][random_sample])
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,67] = np.mean(veins_samples_magnitude_gradient[:,1][random_sample])
    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,68] = np.mean(veins_samples_magnitude_gradient[:,2][random_sample])
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,61] = veins_samples_glcm_image_entropy_large[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,62] = veins_samples_glcm_image_contrast_large[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,63] = veins_samples_glcm_image_dissimilarity_large[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,64] = veins_samples_glcm_image_homogeneity_large[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,65] = veins_samples_glcm_image_energy_large[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,66] = veins_samples_glcm_image_correlation_large[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,67] = veins_samples_glcm_image_ASM_large[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,68] = veins_samples_glcm_image_contrast_small[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,69] = veins_samples_glcm_image_dissimilarity_small[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,70] = veins_samples_glcm_image_homogeneity_small[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,71] = veins_samples_glcm_image_energy_small[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,72] = veins_samples_glcm_image_correlation_small[random_sample]
    #    X[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image,73] = veins_samples_glcm_image_ASM_small[random_sample]    
    # veins are the positive class
    y[i*nr_samples_per_image+nr_vein_samples_per_image:i*nr_samples_per_image+nr_samples_per_image] = np.ones(nr_vein_samples_per_image,)  
    
    print(i+1, '/', nr_ims)