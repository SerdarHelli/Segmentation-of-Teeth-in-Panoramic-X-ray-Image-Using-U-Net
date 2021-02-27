# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:37:46 2020

@author: serdarhelli
"""
from PIL import Image
import os 
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

##### It is manuel preparation. Also , you can use TensorFlow library for these#########


##These applications are applied all of images, do not take "train" word  seriously
## We will separete train-test in model.py

## I choosed 512,512,1 because I resized images before, be carefull !!
col=512
row=512
channels=1

###### X  Images      ##########
###### X  Images  Path    ##########
path="/x/"
dirs=sorted(os.listdir(path),key=len)

##first image reading for concat and I applied augmention for its
img_x_train=np.asarray(Image.open("C:/Users/sserd/Desktop/TEZ2/x/1.png"))
img_x_train=(np.fliplr(img_x_train))
img_x_train=random_noise(img_x_train,mode='s&p',clip=True)

##second image reading for concat and I applied augmention for its
img_x_train1=(Image.open("C:/Users/sserd/Desktop/TEZ2/x/2.png"))

## you can examine different tecniques on libraries lik skimage , numpy, tensorflow, opencv
img_x_train1=(np.fliplr(img_x_train1))
img_x_train1=random_noise(img_x_train1,mode='s&p',clip=True)

x_train=np.concatenate((img_x_train,img_x_train1),axis=None)

## in loop 
for i in range (2,len(dirs)):
    image=np.asarray(Image.open("C:/Users/sserd/Desktop/TEZ2/x/"+dirs[i]))
    img=np.asarray(Image.open(path+dirs[i]))
    img2=np.fliplr(img)
    img2=random_noise(img2,mode='s&p',clip=True)
    
    x_train=np.concatenate((x_train,img2),axis=None)
    
x_train=np.reshape(x_train,(len(dirs),col,row,channels))

########################################################################
######     Y Images     ##########

###### Y  Images  Path    ##########

#### In data augmention, the tecniques like rotations,flips must be applied to y. not including noise 
path2="C:/Users/sserd/Desktop/TEZ2/y/"
dirs2=sorted(os.listdir(path2),key=len)

##first image reading for concat and I applied augmention for its
img_y_train=np.asarray(Image.open("C:/Users/sserd/Desktop/TEZ2/y/1.png"))
img_y_train=(np.fliplr(img_y_train))

##second image reading for concat and I applied augmention for its
img_y_train1=np.asarray(Image.open("C:/Users/sserd/Desktop/TEZ2/y/2.png"))
img_y_train1=(np.fliplr(img_y_train1))

y_train=np.concatenate((img_y_train[:,:,0],img_y_train1[:,:,0]),axis=None)

## in loop 
for j in range (2,len(dirs2)):
    img1=np.asarray(Image.open(path2+dirs2[j]))
    ## you can examine different tecniques on libraries lik skimage , numpy, tensorflow, opencv

    img3=(np.fliplr(img1))
    y_train=np.concatenate((y_train,img3[:,:,0]),axis=None)


y_train=np.reshape(y_train,(len(dirs2),col,row,channels))


########################################################################
## Saving these vectorized inputs with path
np.save("/x_augmention.npy",x_train)
np.save("/y_augmention.npy",y_train)
