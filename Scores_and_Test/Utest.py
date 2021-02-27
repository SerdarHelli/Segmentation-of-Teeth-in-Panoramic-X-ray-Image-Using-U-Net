# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 11:52:45 2020

@author: serdarhelli
"""

from numpy.random import seed
from scipy.stats import mannwhitneyu
import numpy as np
import pandas as pd
import os
seed(1)
d = os.path.dirname(os.getcwd())

data0 =np.loadtxt(d+"/Scores_and_Test/normal.txt")
data1 =np.loadtxt(d+"/Scores_and_Test/non_process.txt")
data2 = np.loadtxt(d+"/Scores_and_Test/post_processli.txt")
df=pd.read_excel(d+"/Scores_and_Test/dsc.xlsx")
e1=np.asarray(df["E1"])
e1=np.reshape(e1,(10,1))
e2=np.asarray(df["E2"])
e2=np.reshape(e2,(10,1))
e3=np.asarray(df["E3"])
e3=np.reshape(e3,(10,1))
e4=np.asarray(df["E4"])
e4=np.reshape(e4,(10,1))
e5=np.asarray(df["E5"])
e5=np.reshape(e5,(10,1))
import numpy as np



error1=np.zeros([len(data1)])
error2=np.zeros([len(data1)])

for i in range(len(data1)):
    error1[i]=np.abs(data1[i]-data0[i])/data0[i]
    error2[i]=np.abs(data2[i]-data0[i])/data0[i]
print("Tooth Count Error")

print("Non-Process Error :",np.sum(error1)/len(error1))
print("Post-Process Error :",np.sum(error2)/len(error2))


print("----------------------------------------------------------")

print("Non-Process Error vs Post-Process Error")





for z in range(0,10):
    print("Fold-",9-z)
    split=z*11
    split2=(z*11)+11
    print("Non-Process Error :",np.sum(error1[split:split2])/len(error1[split:split2]),"Std: ",np.std(error1[split:split2]))
    print("Post-Process Error :",np.sum(error2[split:split2])/len(error2[split:split2]),"Std: ",np.std(error2[split:split2]))


print("---------------------------------------------------------")


for j in range(0,10):
    split=j*11
    split2=(j*11)+11
    print("Fold-",9-j)
    stat, p = mannwhitneyu(error1[split:split2],error2[split:split2])
    print('Statistics=%.5f, p=%.5f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Same distribution (fail to reject H0)')
    else:
    	print('Different distribution (reject H0)')
    
    
print("----------------------------------------------------------")


stat, p = mannwhitneyu(error1[split:split2],error2[split:split2])
print('Statistics=%.5f, p=%.5f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')




print('E1 vs E2')

stat1, p1 = mannwhitneyu(e1,e2)
print('Statistics=%.5f, p=%.5f' % (stat1, p1))
# interpret
alpha1 = 0.05
if p1 > alpha1:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')

print('E2 vs E5')

stat2, p2 = mannwhitneyu(e2,e5)
print('Statistics=%.5f, p=%.5f' % (stat2, p2))
# interpret
alpha2 = 0.05
if p2 > alpha2:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
    
print('E2 vs E3')

stat3, p3 = mannwhitneyu(e2,e3)
print('Statistics=%.5f, p=%.5f' % (stat3, p3))
# interpret
alpha3 = 0.05
if p3 > alpha3:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')
    
print('E2 vs E4')
    
stat4, p4 = mannwhitneyu(e2,e4)
print('Statistics=%.5f, p=%.5f' % (stat4, p4))
# interpret
alpha4 = 0.05
if p4 > alpha4:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')


print('E3 vs E4')
    
stat5, p5 = mannwhitneyu(e3,e4)
print('Statistics=%.5f, p=%.5f' % (stat5, p5))
# interpret
alpha5 = 0.05
if p5 > alpha5:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')










    
    
