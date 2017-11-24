import os
import matplotlib.pyplot as plt
import numpy as np
import retinal_image as ri
import apply_skeleton
import find_interestpoints 
import divideIntoSegments
import detectOpticDisk
import apply_homomorphic_filtering
import cv2
from skimage.morphology import skeletonize


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
labels = divideIntoSegments.divideIntoSegments(skeleton, coordinates,plotFlag)

#Homomorphic filtering to reduce 
img_rgb=apply_homomorphic_filtering.apply_homomorphic_filtering(image_object,img_rgb,plotFlag)

#Detect optic disk
img_gray=cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

detectOpticDisk.detectOpticDisk(img_gray,plotFlag)

from skimage.measure import regionprops

regions=regionprops(labels)
import math

orientations_image=np.zeros((img_rgb.shape[0],img_rgb.shape[1]))
i=0
from skimage.draw import line_aa

from scipy import ndimage
distanceTransform=ndimage.distance_transform_edt(image_vessels)
diameter=distanceTransform * skeleton
meanDiameterInRegion=np.zeros(np.max(labels))

for props in regions:
    i=i+1; 
    y0, x0 = props.centroid
    orientation = props.orientation
    orientations_image[labels==i]=orientation; 
    x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
    x3 = x0 + math.cos(math.pi/2 + orientation)*0.5*props.major_axis_length;
    y3 = y0 - math.sin(math.pi/2 + orientation) * 0.5 * props.major_axis_length; 
    start_x=x0 - math.cos(math.pi/2 + orientation)*0.25*props.major_axis_length;
    start_y=y0 + math.sin(math.pi/2 + orientation) * 0.25 * props.major_axis_length; 
    end_x=x0 + math.cos(math.pi/2 + orientation)*0.25*props.major_axis_length;
    end_y=y0 - math.sin(math.pi/2 + orientation) * 0.25 * props.major_axis_length;
    tempImg=np.zeros((img_rgb.shape[0],img_rgb.shape[1]))
    rr, cc, val = line_aa(int(start_x), int(start_y), int(end_x), int(end_y))
    tempImg[cc,rr]=1; 
    thin_perpendicularlines=skeletonize(tempImg)
    coordinates=np.nonzero(thin_perpendicularlines)
    plt.imshow(image_vessels)
    plt.plot((start_x, end_x), (start_y, end_y), '-r', linewidth=2.5)
    plt.scatter(x=rr,y=cc,c='b',s=20,marker='x') 
    plt.scatter(x=coordinates[1],y=coordinates[0],c='g',s=20,marker='o')
    plt.show()
    meanDiameterInRegion[i-1]=np.mean(diameter[labels==(i-1)])

   
    

    

    


#imag_asd = np.uint8(img_rgb*255)
#cv2.imwrite('newImage.png',imag_asd)


#from skimage.restoration import inpaint
#from scipy.ndimage import binary_dilation, generate_binary_structure
#from skimage import color
#
#struct1 = generate_binary_structure(2, 2)
#mask1 = binary_dilation(image_vessels, structure=struct1).astype(np.float)
# #Defect image over the same region in each color channel
#
#image_result = inpaint.inpaint_biharmonic(image_object.image, mask1, multichannel=True)
#
#fig, axes = plt.subplots(ncols=2, nrows=2)
#ax = axes.ravel()
#
#ax[0].set_title('Original image')
#ax[0].imshow(image_object.image)
#
#ax[1].set_title('Mask')
#ax[1].imshow(mask1, cmap=plt.cm.gray)
#
#ax[3].set_title('Inpainted image')
#ax[3].imshow(image_result)
#
#for a in ax:
#    a.axis('off')
#
#
#fig.tight_layout()
#plt.show()

