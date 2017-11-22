import os
import matplotlib.pyplot as plt
import numpy as np
import retinal_image as ri
import apply_skeleton
import find_interestpoints 
import divideIntoSegments
import detectOpticDisk
import apply_homomorphic_filtering

#Paths
path_to_training_retinal_ims = 'data/training/images/'
path_to_training_retinal_masks = 'data/training/masks/'
path_to_training_retinal_vessels = 'data/training/vessels/'
path_to_training_arteries = 'data/training/arteries/'
path_to_training_veins = 'data/training/veins/'
retinal_im_list = os.listdir(path_to_training_retinal_ims)

#Open images
image_object = ri.retinal_image(retinal_im_list[39-21], 'train')
img_rgb=image_object.image; 
image_vessels=image_object.vessels

#Do you want to plot?
plotFlag=1

# perform skeletonization
skeleton=apply_skeleton.apply_skeleton(image_vessels,plotFlag)

#Find interest points
coordinates=find_interestpoints.find_interestpoints(skeleton,plotFlag)

#Divide into segments
divideIntoSegments.divideIntoSegments(skeleton, coordinates,plotFlag)

#Homomorphic filtering to reduce 
img_rgb=apply_homomorphic_filtering.apply_homomorphic_filtering(image_object,img_rgb,plotFlag)

#Detect optic disk
detectOpticDisk.detectOpticDisk(img_rgb,plotFlag)



