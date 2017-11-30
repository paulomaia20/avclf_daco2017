# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:18:35 2017

@author: Paulo Maia
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:16:51 2017

@author: Paulo Maia
"""

def obtainSkeletonWithoutCrossings(skeleton, coordinates):
    
    
    #substituir na divideintosegments ? tava a dar merda...
    
    skeleton[coordinates[0],coordinates[1]]=0; 
    skeleton[coordinates[0]+1,coordinates[1]]=0; 
    skeleton[coordinates[0]-1,coordinates[1]]=0; 
    skeleton[coordinates[0],coordinates[1]-1]=0; 
    skeleton[coordinates[0],coordinates[1]+1]=0; 
    skeletonWithoutCrossings=skeleton; 

    return skeletonWithoutCrossings; 
