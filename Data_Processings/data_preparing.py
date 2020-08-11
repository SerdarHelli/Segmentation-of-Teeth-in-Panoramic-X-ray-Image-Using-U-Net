# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:11:38 2020

@author: serdarhelli
"""

##### DATA Preparing#########
from PIL import Image
import os 
import numpy as np
import matplotlib.pyplot as plt


###### x_train       ##########

path="data_x_train"
dirs=sorted(os.listdir(path),key=len)

img_x_train=np.asarray(Image.open("data_x_train/1.png"))
img_x_train1=np.asarray(Image.open("data_x_train/2.png"))
x_train=np.concatenate((img_x_train,img_x_train1),axis=None)
for i in range (2,len(dirs)):
    img=np.asarray(Image.open(path+dirs[i]))
    x_train=np.concatenate((x_train,img),axis=None)


x_train=np.reshape(x_train,(100,512,512,1))

######      y_train    ##########
path2="data_y_train"
dirs2=sorted(os.listdir(path2),key=len)

img_y_train=np.asarray(Image.open("data_y_train/1.png"))
img_y_train1=np.asarray(Image.open("data_y_train/2.png"))

y_train=np.concatenate((img_y_train[:,:,0],img_y_train1[:,:,0]),axis=None)
for j in range (2,len(dirs2)):
    img1=np.asarray(Image.open(path2+dirs2[j]))
    y_train=np.concatenate((y_train,img1[:,:,0]),axis=None)


y_train=np.reshape(y_train,(100,512,512,1))

######      x_test    ##########

path3="data_x_test/"
dirs3=sorted(os.listdir(path3),key=len)

img_x_test=np.asarray(Image.open("data_x_test//101.png"))
img_x_test1=np.asarray(Image.open("data_x_test//102.png"))
x_test=np.concatenate((img_x_test,img_x_test1),axis=None)
for k in range (2,len(dirs3)):
    img2=np.asarray(Image.open(path3+dirs3[k]))
    x_test=np.concatenate((x_test,img2),axis=None)
    
x_test=np.reshape(x_test,(11,512,512,1))
    
    
######      y_test    ##########
    
path4="data_y_test/"
dirs4=sorted(os.listdir(path4),key=len)

img_y_test=np.asarray(Image.open("data_y_test/101.png"))
img_y_test1=np.asarray(Image.open("data_y_test/102.png"))

y_test=np.concatenate((img_y_test[:,:,0],img_y_test1[:,:,0]),axis=None)
for l in range (2,len(dirs4)):
    img3=np.asarray(Image.open(path4+dirs4[l]))
    y_test=np.concatenate((y_test,img3[:,:,0]),axis=None)
    
y_test=np.reshape(y_test,(11,512,512,1))
np.save("x_test.npy",x_test)
np.save("x_train.npy",x_train)
np.save("y_test.npy",y_test)
np.save("y_train.npy",y_train)