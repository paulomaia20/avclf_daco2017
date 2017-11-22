# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:14:04 2017

@author: Paulo Maia
"""
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

def apply_skeleton(image_vessels,plotFlag):
    
    skeleton = skeletonize(image_vessels)
    
    #TODO - MOSTRAR SKELETON EM CIMA DA IMAGEM ORIGINAL 
        
    if(plotFlag==1): 
        #display results
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                                 sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        
        ax = axes.ravel()
        
        ax[0].imshow(image_vessels, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('original', fontsize=20)
        
        ax[1].imshow(skeleton, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('skeleton', fontsize=20)
        
        fig.tight_layout()
        plt.show()
    return skeleton