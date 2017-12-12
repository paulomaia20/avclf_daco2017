# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:00:17 2017

@author: Paulo Maia
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:54:21 2017

@author: Paulo Maia
"""

import numpy as np
import cv2


def apply_homomorphic_filtering(mask, img_rgb, plotFlag):
    img_rgb = np.float32(img_rgb)
    
    rows,cols,dim = img_rgb.shape
    
    rh, rl, cutoff = 0.95,0.25,0.01 #possivelmente será necessário modificar os parametros 
    
    imgHSV = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(imgHSV)
    indices=(mask==0)
    indices=indices.astype(np.int)
    H[indices==1]=0
    S[indices==1]=0
    V[indices==1]=0
    
#    cv2.imshow('H', H)
#    
#    cv2.imshow('S', S)
#    
#    cv2.imshow('V', V)
    
    
    V_log = np.log(V+0.01)
    V_fft = np.fft.fft2(V_log)
    V_fft_shift = np.fft.fftshift(V_fft)

     
    
    DX = cols/cutoff
    G = np.ones((rows,cols))
    for i in range(rows):
        for j in range(cols):
            G[i][j]=((rh-rl)*(1-np.exp(-((i-rows/2)**2+(j-cols/2)**2)/(2*DX**2))))+rl
    
    result_filter_V = G * V_fft_shift
    result_interm_V = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter_V)))
    result_V = np.exp(result_interm_V)
    result1 = np.dstack((H,S,result_V))
    result1 = np.float32(result1)
    imgRGB = cv2.cvtColor(result1, cv2.COLOR_HSV2RGB)
    
    if (plotFlag==1):    
        cv2.imshow('Homomorphic Filtered RGB Image', imgRGB)

    return imgRGB