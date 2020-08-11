# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:14:58 2020

@author: serdarhelli
"""
########Creating MASK#############
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

for i in range (1,112):
    path=str(i)+'.png'
    img=plt.imread("/images/"+path)
    img_seg=plt.imread("/image_seg/"+path)
    
    canal_1=img_seg[:,:,0]
    canal_2=img_seg[:,:,1]
    canal_3=img_seg[:,:,2]
    canal_4=img_seg[:,:,3]
    
    
    
    
    
    mask_2=(canal_2!=img)*1 
    mask_3=(canal_3!=img)*1
    mask_1=(canal_1!=img)*1
    
    mask=mask_1+mask_2+mask_3
    
    
    
    
    print(mask.shape)
    plt.imsave("/masks/"+path,mask,cmap='gray')



