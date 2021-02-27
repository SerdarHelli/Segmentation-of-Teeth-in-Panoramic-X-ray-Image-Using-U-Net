# -*- coding: utf-8 -*-
"""
@author: serdarhelli
"""
from PIL import Image
import os 
import numpy as np


##### It is manuel preparation. Also , you can use TensorFlow library for these#########



##These applications are applied all of images, do not take "train" word  seriously
## We will separete train-test in model.py


## I choosed 512,512,1 because I resized images before, be carefull !!
col=512
row=512
channels=1
###### X Images      ##########

###### X  Images  Path    ##########

path="/X/"
dirs=os.listdir(path)


##first image reading for concat
img_x_train=np.asarray(Image.open(path+dirs[0]))
##second image reading for concat
img_x_train1=np.asarray(Image.open(path+dirs[1]))

x_train=np.concatenate((img_x_train,img_x_train1),axis=None)
## in loop 
for i in range (2,len(dirs)):
    img=np.asarray(Image.open(path+dirs[i]))
    x_train=np.concatenate((x_train,img),axis=None)


x_train=np.reshape(x_train,(len(dirs),col,row,channels))


########################################################################

######     Y Images     ##########

###### Y  Images  Path    ##########
path2="/y/"
dirs2=os.listdir(path2)


##first image reading for concat
img_y_train=np.asarray(Image.open(path2+dirs2[0]))
##second image reading for concat
img_y_train1=np.asarray(Image.open(path2+dirs2[1]))

y_train=np.concatenate((img_y_train[:,:,0],img_y_train1[:,:,0]),axis=None)
## in loop 
for j in range (2,len(dirs2)):
    img1=np.asarray(Image.open(path2+dirs2[j]))
    y_train=np.concatenate((y_train,img1[:,:,0]),axis=None)

y_train=np.reshape(y_train,(len(dirs2),col,row,channels))

########################################################################
## Saving these vectorized inputs with path
np.save("/x_train.npy",x_train)
np.save("/y_train.npy",y_train)