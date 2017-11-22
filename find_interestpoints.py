# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:30:09 2017

@author: Paulo Maia
"""

from scipy.ndimage import binary_hit_or_miss
import numpy as np
import matplotlib.pyplot as plt


def find_interestpoints(skeleton,plotFlag):
    B1 = np.array([[0, 1, 0], 
                       [1, 1, 1], 
                       [0, 1, 0]])
    B2 = np.array([[1, 0, 1], 
                       [0, 1, 0], 
                       [1, 0, 1]])
    B3=np.array([[0, 1, 0], 
                     [1, 1, 1], 
                     [0, 0, 0]])
    
    B4=np.array([[1, 0, 1], 
                     [0, 1, 0],
                     [1, 0, 0]])  
    
    B5=np.array([[0, 1, 0], 
                     [1, 1, 0],
                     [0, 1, 0]])
    
    B6=np.array([[1, 0, 0],
                     [0, 1, 0],
                     [1, 0, 1]])
    
    B7=np.array([[0, 0, 0],
                     [1, 1, 1],
                     [0, 1, 0]])
    
    B8=np.array([[0, 0, 1], 
                     [0, 1, 0],
                     [1, 0, 1]])
    
    B9=np.array([[0, 1, 0],
                    [0, 1, 1],
                     [0, 1, 0]])
    
    B10=np.array([[1, 0, 1],
                     [0, 1, 0],
                     [0, 0, 1]])
    B11=np.array([[1, 0, 1], 
                     [0, 1, 0], 
                     [0, 1, 0]])
    
    B12=np.array([[0, 1, 0], 
                     [1, 1, 0], 
                     [0, 0, 1]])
    
    B13=np.array([[1, 0, 0], 
                     [0, 1, 1], 
                     [1, 0, 0]])
    
    B14=np.array([[0, 0, 1], 
                     [1, 1, 0], 
                     [0, 1, 0]])
    
    B15=np.array([[0, 1, 0], 
                     [0, 1, 0], 
                     [1, 0, 1]])
    B16=np.rot90(B14)
    B17 = np.rot90(B15)
    B18 = np.rot90(B16)
    
    IMhit_mis=(binary_hit_or_miss(skeleton,B1)+binary_hit_or_miss(skeleton,B2)+binary_hit_or_miss(skeleton,B3)+binary_hit_or_miss(skeleton,B4)+binary_hit_or_miss(skeleton,B5)+binary_hit_or_miss(skeleton,B6)+binary_hit_or_miss(skeleton,B7)+binary_hit_or_miss(skeleton,B8)+binary_hit_or_miss(skeleton,B9)+binary_hit_or_miss(skeleton,B10)+binary_hit_or_miss(skeleton,B11)+binary_hit_or_miss(skeleton,B12)+binary_hit_or_miss(skeleton,B13)+binary_hit_or_miss(skeleton,B14)+binary_hit_or_miss(skeleton,B15)+binary_hit_or_miss(skeleton,B16)+binary_hit_or_miss(skeleton,B17)+binary_hit_or_miss(skeleton,B18))
    
    IMhit_mis_bin=IMhit_mis.astype(np.int); 
    coordinates=np.nonzero(IMhit_mis_bin)
    
    
    if (plotFlag==1):
        plt.figure(3)
        plt.imshow(skeleton)
        plt.scatter(x=coordinates[1],y=coordinates[0],c='r',s=20,marker='x')
        plt.show() 
    return coordinates
        
     