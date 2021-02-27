# -*- coding: utf-8 -*-
"""
@author: serdarhelli
"""

from PIL import Image
import os 

### Your numbers for resizing images : colunm ,rows...
col=512
row=512

### Path all of original Images
path="/X/"
### Path saving images  for X
path_save1="SavePATH"

dirs=os.listdir(path)

for i in range (0,len(dirs)):
    img=Image.open(path+dirs[i])
    (img.resize((col,row),Image.ANTIALIAS)).save(path_save1+dirs[i],"png", optimize=True)
    

### Path all of segmented Images
path2="/masks/"   
dirs2=os.listdir(path2)
### Path saving images  for MASKs
path_save2="SavePATH"
for j in range (0,len(dirs2)):
    img2=Image.open(path2+dirs2[j])
    (img2.resize((col,row),Image.ANTIALIAS)).save(path_save2+dirs2[j],"png", optimize=True)
    


    