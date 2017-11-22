# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:42:58 2017

@author: Paulo Maia
"""
from skimage.color import label2rgb
from skimage import measure
import matplotlib.pyplot as plt



def divideIntoSegments(skeleton, coordinates, plotFlag):
    
    skeleton[coordinates[0],coordinates[1]]=0; 
    skeleton[coordinates[0]+1,coordinates[1]]=0; 
    skeleton[coordinates[0]-1,coordinates[1]]=0; 
    skeleton[coordinates[0],coordinates[1]-1]=0; 
    skeleton[coordinates[0],coordinates[1]+1]=0; 
    
    all_labels = measure.label(skeleton)
    image_label_overlay = label2rgb(all_labels,image=None,alpha=0.3,bg_label=0,bg_color=(0,0,0))

    if(plotFlag==1):
        plt.figure(5)
        plt.imshow(image_label_overlay)
        plt.scatter(x=coordinates[1],y=coordinates[0],c='r',s=20,marker='x')
        plt.show() 