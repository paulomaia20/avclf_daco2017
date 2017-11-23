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
vessel_keypoints=[];
discretized_lines=[]; 
discretized_singleline=[]; 
from skimage.draw import line_aa

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
   
    plt.imshow(skeleton)
    plt.plot((start_x, end_x), (start_y, end_y), '-r', linewidth=2.5)
    #plt.plot((x0, x1), (y0, y1), '-g', linewidth=2.5)
   # rr, cc, val = line_aa(int(start_x), int(start_y), int(end_x), int(end_y))
    #plt.scatter(x=rr,y=cc,c='b',s=20,marker='x')   
    #discretized_singleline=[i, rr,cc]

    line=np.transpose([i,start_x, start_y, end_x,end_y])
    vessel_keypoints.append(line)
    

plt.show()

np_vesselkeypoints=np.array(vessel_keypoints)
#Organized as follows: label , x0, y0 , x3 , y3. How to discretize ?

#newImg=np.zeros((img_rgb.shape[0],img_rgb.shape[1]))
diameter=np.zeros(np_vesselkeypoints[:,0].shape[0])
for pp in range(np_vesselkeypoints.shape[0]):
    tempImg=np.zeros((img_rgb.shape[0],img_rgb.shape[1]))
    label=np_vesselkeypoints[pp,0]
    start_x=np_vesselkeypoints[pp,1]
    start_y=np_vesselkeypoints[pp,2]
    end_x=np_vesselkeypoints[pp,3]
    end_y=np_vesselkeypoints[pp,4]
    rr, cc, val = line_aa(int(start_x), int(start_y), int(end_x), int(end_y))
    #plt.scatter(x=rr,y=cc,c='b',s=20,marker='x')   
    #discretized_singleline=[i, rr,cc]
    tempImg[cc,rr]=1; 
   # newImg[cc,rr]=1;
    thin_perpendicularlines=skeletonize(tempImg)
    diameter[int(label)]=np.count_nonzero(np.bitwise_and(image_vessels,thin_perpendicularlines)) 
    #Diametro para label=label 

    


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

