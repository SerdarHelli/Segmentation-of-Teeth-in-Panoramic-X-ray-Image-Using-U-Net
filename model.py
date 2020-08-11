# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:42:46 2020

@author: serdarhelli
"""
#### MODEL ###
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization,concatenate,Conv2DTranspose,Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.losses
import matplotlib.pyplot as plt
import numpy as np



x1=np.load("/x_train.npy")
x2=np.load("/x_test.npy")
y1=np.load("/y_train1.npy")
y2=np.load("/y_test1.npy")
x_augmention=np.load("/x_augmention.npy")
y_augmention=np.load("y_augmention.npy")
def split_test(x,fold,test_value):
    split=len(x)-(test_value*fold)
    split2=split-test_value
    x_test=np.copy(x[split2:split,:,:,:])
    x_test=np.reshape(x_test,(test_value,512,512,1))
    return np.uint8(x_test)
def split_train(x,x_augmention,fold,test_value):
    split=len(x)-(test_value*fold)
    split2=split-test_value
    x_train1=np.copy(x[:split2,:,:,:])
    x_train2=np.copy(x[split:,:,:,:])
    x_train3=np.concatenate((x_train1,x_train2),axis=None)
    x_augmention1=np.copy(x_augmention[:split2,:,:,:])
    x_augmention2=np.copy(x_augmention[split:,:,:,:])
    x_augmention3=np.concatenate((x_augmention1,x_augmention2),axis=None)    
    x_train=np.concatenate((x_train3,x_augmention3),axis=None)
    x_train=np.reshape(x_train,((len(x)+len(x_augmention))-(2*test_value),512,512,1))
    return np.uint8(x_train)
def split_train_noaug(x,fold,test_value):
    split=len(x)-(test_value*fold)
    split2=split-test_value
    x_train1=np.copy(x[:split2,:,:,:])
    x_train2=np.copy(x[split:,:,:])
    x_train=np.concatenate((x_train1,x_train2),axis=None)
    x_train=np.reshape(x_train,(len(x)-test_value,512,512,1))
    return np.uint8(x_train)



fold_value=3
y_test=split_test(y,fold_value,11)
x_test=split_test(x,fold_value,11)
y_train=split_train(y,y_augmention,fold_value,11)
x_train=split_train(x,x_augmention,fold_value,11)

x_test=x_test/255
y_test=y_test/255
x_train=x_train/255
y_train=y_train/255


inputs=Input(shape=(512, 512,1 ))

conv1 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
d1=Dropout(0.1)(conv1)
conv2 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d1)
b=BatchNormalization()(conv2)

pool1 = MaxPooling2D(pool_size=(2, 2))(b)
conv3 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
d2=Dropout(0.2)(conv3)
conv4 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d2)
b1=BatchNormalization()(conv4)

pool2 = MaxPooling2D(pool_size=(2, 2))(b1)
conv5 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
d3=Dropout(0.3)(conv5)
conv6 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d3)
b2=BatchNormalization()(conv6)

pool3 = MaxPooling2D(pool_size=(2, 2))(b2)
conv7 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
d4=Dropout(0.4)(conv7)
conv8 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d4)
b3=BatchNormalization()(conv8)

pool4 = MaxPooling2D(pool_size=(2, 2))(b3)
conv9 = Conv2D(512,(3,3),activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
d5=Dropout(0.5)(conv9)
conv10 = Conv2D(512,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d5)
b4=BatchNormalization()(conv10)


conv11 = Conv2DTranspose(512,(4,4), activation = 'relu', padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(conv10)
x= concatenate([conv11,conv8])
conv12 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
d6=Dropout(0.4)(conv12)
conv13 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d6)
b5=BatchNormalization()(conv13)


conv14 = Conv2DTranspose(256,(4,4), activation = 'relu', padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(b5)
x1=concatenate([conv14,conv6])
conv15 = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
d7=Dropout(0.3)(conv15)
conv16 = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d7)
b6=BatchNormalization()(conv16)

conv17 = Conv2DTranspose(128,(4,4), activation = 'relu', padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(b6)
x2=concatenate([conv17,conv4])
conv18 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x2)
d8=Dropout(0.2)(conv18)
conv19 = Conv2D(64,(3,3) ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d8)
b7=BatchNormalization()(conv19)

conv20 = Conv2DTranspose(64,(4,4), activation = 'relu', padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(b7)
x3=concatenate([conv20,conv2])
conv21 = Conv2D(32,(3,3) ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x3)
d9=Dropout(0.1)(conv21)
conv22 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d9)

outputs = Conv2D(1,(1,1), activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv22)
model2 = Model( inputs = inputs, outputs = outputs)


model2.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


model2.summary()