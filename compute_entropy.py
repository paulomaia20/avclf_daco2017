# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:32:11 2017

@author: tiago
"""

#Calculate the Entropy
from skimage.measure import shannon_entropy
from skimage.feature import greycomatrix
from skimage.feature import greycoprops

def compute_glcm_features(retinal_image):
    #Compute Entropy
    glcm_image_entropy = shannon_entropy()
    
    #Compute GLCM (Gray Level Co-occurrence Matrix)
    glcm_image = greycomatrix()
    
    #Compute GLCM Features
    #Contrast
    glcm_image_contrast = greycoprops(glcm_image, 'contrast')
    
    #Dissimilarity
    glcm_image_dissimilarity = greycoprops(glcm_image, 'dissimilarity')
    
    #Homogeneity
    glcm_image_homogeneity = greycoprops(glcm_image, 'homogeneity')
    
    #Energy
    glcm_image_energy = greycoprops(glcm_image, 'energy')
    
    #Correlation
    glcm_image_correlation = greycoprops(glcm_image, 'correlation')
    
    #ASM
    glcm_image_ASM = greycoprops(glcm_image, 'ASM')
    