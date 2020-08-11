# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 15:44:39 2020

@author: serdarhelli
"""

#####THRESH VALIDATION####
import os
import numpy as np
import matplotlib.pyplot as plt
a=np.load("x_test.npy")
b=np.load("x_train.npy")
c=np.load("y_train1.npy")
d=np.load("y_test1.npy")




x=np.concatenate((b,a),axis=None)
x=np.reshape(x,(111,512,512,1))
y=np.concatenate((c,d),axis=None)
y=np.reshape(y,(111,512,512,1))
def split_test(x,fold,test_value):
    split=len(x)-(test_value*fold)
    split2=split-test_value
    x_test=np.copy(x[split2:split,:,:,:])
    x_test=np.reshape(x_test,(test_value,512,512,1))
    return np.uint8(x_test)
path="file /Predictions_noaug/"
dirs=sorted(os.listdir(path),key=len)
sensitivity=np.zeros([10])
specificity=np.zeros([10])
pos_pred_val=np.zeros([10])
neg_pred_val=np.zeros([10])
DSC=np.zeros([10])
J=np.zeros([10])

c=np.zeros([9])
for i in range (0,9):
    
    y_pred=np.load("predictions9.npy")
    y_pred=(np.reshape(y_pred,(11*512*512)))
    c[i]=(i+1)/10
    y_pred=(y_pred>c[i])*1
    y_true=split_test(y,9,11)
    y_true=(np.reshape(y_true,(11*512*512)))/255
    TP=0
    TN=0
    FP=0
    FN=0               
    for j in range (11*512*512):
        if y_true[j]==1 and y_pred[j]==1:
            TP+=1
        if y_true[j]==0 and y_pred[j]==0:
            TN+=1
        if y_true[j]==1 and y_pred[j]==0:
            FP+=1
        if y_true[j]==0 and y_pred[j]==1:
            FN+=1       
    sensitivity[i]  = TP / (TP+FN)
    specificity[i]  = TN / (TN+FP)
    pos_pred_val[i] = TP/ (TP+FP)
    neg_pred_val[i] = TN/ (TN+FN)    
    DSC[i]=TP*2/((2*TP+FN+FP))
    J[i]=DSC[i]/(2-DSC[i])
print(sum(DSC)/10,"Dice SCORE")
print(sum(J)/10,"Jacard Score")
print(sum(sensitivity)/10,"sensitivity")
print(sum(specificity)/10,"specificity")
print(sum(pos_pred_val)/10,"positive prediction value")
print(sum(neg_pred_val)/10,"negative prediction value")

plt.figure(figsize=(9,5))

plt.plot(c,DSC[0:9],"r-",marker="*",label="Dice Score")
plt.plot(c,J[0:9],"b-",marker="o",label="Jacard Score")
plt.xlabel('Thresh Value ')
plt.ylabel('Scores')


plt.grid()
plt.legend()


plt.savefig("scores.png",facecolor="#f0f9e8",dpi=600,quality=95)






