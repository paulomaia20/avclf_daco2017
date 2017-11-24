# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:15:16 2017

@author: Paulo Maia
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:30:09 2017

@author: Paulo Maia
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters

def find_interestpointsv2(skeleton,plotFlag):
   
    
    kernel=np.array([[0, 1, 0], 
                     [1, 0, 1], 
                     [0, 1, 0]])
    
    
    neighbours=scipy.ndimage.filters.convolve(skeleton.astype(np.int),kernel)
    plt.imshow(neighbours)
    interest_pts=neighbours>1; 
    coordinates=np.nonzero(interest_pts)
    
    
    if (plotFlag==1):
        plt.figure(3)
        plt.imshow(skeleton)
        plt.scatter(x=coordinates[1],y=coordinates[0],c='r',s=20,marker='x')
        plt.show() 
    return coordinates
        
     