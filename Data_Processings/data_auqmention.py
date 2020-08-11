# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:37:46 2020

@author: serdarhelli
"""
#####DATA_AUGMENTION#####
from PIL import Image
import os 
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
path="x/"
dirs=sorted(os.listdir(path),key=len)
img_x_train=np.asarray(Image.open("/x/1.png"))
img_x_train=np.flipud(np.fliplr(img_x_train))
img_x_train=random_noise(img_x_train,mode='s&p',clip=True)
img_x_train1=np.asarray(Image.open("x/2.png"))
img_x_train1=np.flipud(np.fliplr(img_x_train1))
img_x_train1=random_noise(img_x_train1,mode='s&p',clip=True)

x_train=np.concatenate((img_x_train,img_x_train1),axis=None)

for i in range (2,len(dirs)):
    img=np.asarray(Image.open(path+dirs[i]))
    img2=np.flipud(np.fliplr(img))
    img2=random_noise(img2,mode='s&p',clip=True)
    
    x_train=np.concatenate((x_train,img2),axis=None)
    
    


x_train=np.reshape(x_train,(111,512,512,1))

path2="/y/"
dirs2=sorted(os.listdir(path2),key=len)

img_y_train=np.asarray(Image.open("/y/1.png"))
img_y_train=np.flipud(np.fliplr(img_y_train))
img_y_train=random_noise(img_y_train,mode='s&p',clip=True)
img_y_train1=np.asarray(Image.open("/y/2.png"))
img_y_train1=np.flipud(np.fliplr(img_y_train1))
img_y_train1=random_noise(img_y_train1,mode='s&p',clip=True)

y_train=np.concatenate((img_y_train[:,:,0],img_y_train1[:,:,0]),axis=None)
for j in range (2,len(dirs2)):
    img1=np.asarray(Image.open(path2+dirs2[j]))
    img3=np.flipud(np.fliplr(img1))
    y_train=np.concatenate((y_train,img3[:,:,0]),axis=None)


y_train=np.reshape(y_train,(111,512,512,1))



np.save("/x_augmention.npy",x_train)
np.save("/y_augmention.npy",y_train)
