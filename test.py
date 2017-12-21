import os
import numpy as np
import random 

path_to_training_retinal_ims = 'data/training/images/'
retinal_im_list = os.listdir(path_to_training_retinal_ims)
nr_ims = len(retinal_im_list) # same as above
np.random.seed(15)
randomT=random.sample(retinal_im_list,15)
randomT.sort()
fitting_img=[]
for i in range (nr_ims):
    if retinal_im_list[i] in randomT:
        fit_img = randomT
    else:
        fitting_img.append(retinal_im_list[i])
        