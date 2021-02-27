# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:03:25 2020

@author: serdarhelli
"""

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from imutils import perspective
from imutils import contours
import os
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
# Load in image, convert to gray scale, and Otsu's threshold
kernel1 =( np.ones((5,5), dtype=np.float32))
blur_radius=0.5
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                             [-1,-1,-1]])

#path3 Path your original images
path3="/Images/"
## path2 Path Predicted Images which was applied CCA
path2="/SAVE_PATH/"
## path Path Predicted Images which has original size
path="/Predicted_Images_PATH"


count1=0
dirs=sorted(os.listdir(path),key=len)

for i in range  (0,len(dirs)):
    ## if the names of predicted and original images are same 
    ## you can use for pairing images 
    ## if not, predicted and original images must be paired together
    ## for example, you have image which name is 1.png
    ## and your predicted image name is 1.png 
    image = cv2.imread(path+dirs[i])
    image2 =cv2.imread(path3+dirs[i])

    image=cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1,iterations=3 )
    

    image = cv2.filter2D(image, -1, kernel_sharpening)
    image=cv2.erode(image,kernel1,iterations =2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    labels=cv2.connectedComponents(thresh,connectivity=8)[1]       
    plt.imshow(thresh)
    total_area = 0
    a=np.unique(labels)
    area=np.zeros(len(a))
    count2=0
    for label in a:
        if label == 0:
            continue
    
        # Create a mask
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255

        # Find contours and determine contour area
        cnts,hieararch = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        c_area = cv2.contourArea(cnts)
        # threshhold for tooth count
        if c_area>2000:
            count2+=1
        
        (x,y),radius = cv2.minEnclosingCircle(cnts)
        rect = cv2.minAreaRect(cnts)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")    
        box = perspective.order_points(box)
        color1 = (list(np.random.choice(range(150), size=3)))  
        color =[int(color1[0]), int(color1[1]), int(color1[2])]  
        cv2.drawContours(image2,[box.astype("int")],0,color,2)
        (tl,tr,br,bl)=box
        
        (tltrX,tltrY)=midpoint(tl,tr)
        (blbrX,blbrY)=midpoint(bl,br)
    	# compute the midpoint between the top-left and top-right points,
    	# followed by the midpoint between the top-righ and bottom-right
        (tlblX,tlblY)=midpoint(tl,bl)
        (trbrX,trbrY)=midpoint(tr,br)
    	# draw the midpoints on the image
        cv2.circle(image2, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(image2, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(image2, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(image2, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        cv2.line(image2, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),color, 2)
        cv2.line(image2, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),color, 2)
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        
        
        pixelsPerMetric=1
        dimA = dA * pixelsPerMetric
        dimB = dB *pixelsPerMetric
        cv2.putText(image2, "{:.1f}pixel".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, color, 2)
        cv2.putText(image2, "{:.1f}pixel".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, color, 2)
        cv2.putText(image2, "{:.1f}".format(label),(int(tltrX - 35), int(tltrY - 5)), cv2.FONT_HERSHEY_SIMPLEX,0.65, color, 2)
    print(".Image",dirs[i])
    print(".Tooth",count2)  
    cv2.imwrite(path2+dirs[i],image2)
    cv2.waitKey()
