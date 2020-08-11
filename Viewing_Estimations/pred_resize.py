# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 00:39:13 2020

@author: serdarhelli
"""

####PRED_RESIZE####
import numpy as np
from PIL import Image
import os 
import matplotlib.pyplot as plt

a=np.load("predictions.npy")
sizes=np.load("sizes.npy")

path="/Imagesfornames"
dirs=sorted(os.listdir(path),key=len)

fold=0
b=fold*11

plt.imshow(a[2,:,:,0])

if fold!=0:
    for i in range (1,11):  
        img2=np.reshape(a[0,:,:,0]*255,(512,512))
        img=np.reshape(a[i,:,:,0]*255,(512,512))
        img=np.uint8(img)
        img2=np.uint8(img)
        x=int((sizes[((10-fold)*10)+i-2,0]))
        y=int((sizes[((10-fold)*10)+i-2,1]))
        x1=int((sizes[10-fold-1,0]))
        y1=int((sizes[10-fold-1,1]))
        img=((Image.fromarray(img))).resize((x,y),Image.ANTIALIAS)
        img2=((Image.fromarray(img2))).resize((x1,y1),Image.ANTIALIAS)
        plt.imsave("pred_image"+dirs[((10-fold)*10)+i-2],img,cmap='gray')
        plt.imsave("pred_image"+dirs[10-fold-1],img2,cmap='gray')

else:
    for i in range (0,11): 
        img=np.reshape(a[i,:,:,0]*255,(512,512))
        img=(np.uint8(img))
        x=int((sizes[100+i,0]))
        y=int((sizes[100+i,1]))
        (((Image.fromarray(img))).resize((x,y),Image.ANTIALIAS)).save("pred_image"+dirs[111-11+i],"png")

   