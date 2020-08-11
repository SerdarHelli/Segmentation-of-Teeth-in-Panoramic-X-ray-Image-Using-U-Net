# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 21:29:01 2020

@author: serdarhelli
"""

###APPLY OUTPUT MASKS###
import matplotlib.pyplot as plt
import cv2  
import os
import numpy as np

#which png
a=16

image =plt.imread("Images/"+np.str(a)+".png",0)
mask1=plt.imread("masks/"+np.str(a)+".png",0)
RED = (0, 0, 255)  # opencv uses BGR not RGB
_, mask = cv2.threshold(mask1[:,:,0], thresh=255/2, maxval=255, type=cv2.THRESH_BINARY)
cnts,hieararch=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

img = cv2.drawContours(image, cnts, -1, RED, 2)

cv2.imwrite("file mask1"+np.str(a)+".png",img)


