# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:50:43 2017

@author: Gabriel
"""

from skimage.restoration import denoise_nl_means
import os
import matplotlib.pyplot as plt
import numpy as np
import retinal_image as ri


path_to_training_retinal_ims = 'data/training/images/'
path_to_training_retinal_masks = 'data/training/masks/'
path_to_training_retinal_vessels = 'data/training/vessels/'
path_to_training_arteries = 'data/training/arteries/'
path_to_training_veins = 'data/training/veins/'
retinal_im_list = os.listdir(path_to_training_retinal_ims)
nr_retinal_ims = len(retinal_im_list) # number of retinal images of the training set
nr_ims = len(retinal_im_list) # same as above

#Open images
#for kk in range(19):
image_object = ri.retinal_image(retinal_im_list[6], 'train')
img_rgb=image_object.image;
plt.figure (1)
plt.imshow(img_rgb)
plt.show()
plt.figure (2)
plt.imshow(denoise_nl_means(img_rgb,h=0.01, multichannel=True))
