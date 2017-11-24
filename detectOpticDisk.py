# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:49:21 2017

@author: Paulo Maia
"""
from skimage.color import rgb2gray

from skimage.feature import blob_dog
from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt


def detectOpticDisk(img_gray,plotFlag):
    
    img_gray=gaussian(img_gray,sigma=5)
    
    blobs_dog = blob_dog(img_gray, max_sigma=30, threshold=.8)
    
    blobs = [blobs_dog]
    colors = ['red']
    titles = ['Difference of Gaussian']
    sequence = zip(blobs, colors, titles)
    
    meancolorintensity=[]; 
    for blobs, cor, title in sequence:
        fig, ax = plt.subplots(1, 1)
        ax.set_title(title)
        ax.imshow(img_gray, interpolation='nearest',cmap='gray')
        for blob in blobs:
            y, x, r = blob
            print("y",y,"x",x,"r",r)
            c = plt.Circle((x, y), r, color=cor, linewidth=1, fill=False)
            ax.add_patch(c)
            area=np.mean(img_gray[int(y-round(r)):int(y+round(r)),int(x-round(r)):int(x+round(r))]) #ver se estao trocados
            print(area)
            meancolorintensity.append(area)
    print(meancolorintensity)
    
    if(plotFlag==1):
        plt.imshow(img_gray,cmap='gray')
        maxIndex=np.argmax(meancolorintensity); 
        
        plt.scatter(x=blobs[maxIndex,1],y=blobs[maxIndex,0],c='r',s=20,marker='x')
        plt.imshow(img_gray,cmap='gray')
        
        plt.show()