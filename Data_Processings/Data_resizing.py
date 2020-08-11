# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:36:38 2020

@author: serdarhelli
"""

##########DATA RESIZING########
from PIL import Image
import os 

path="resize_x/images/"
dirs=sorted(os.listdir(path),key=len)


for i in range (0,len(dirs)):
    img=Image.open(path+dirs[i])
    (img.resize((512,512),Image.ANTIALIAS)).save(path+dirs[i],"png", optimize=True)
    


path2="resize_y/masks/"   
dirs2=sorted(os.listdir(path2),key=len)





for j in range (0,len(dirs2)):
    img2=np.asarray(Image.open(path2+dirs2[j]))
    (img2.resize((512,512),Image.ANTIALIAS)).save(path2+dirs2[j],"png", optimize=True)
    




