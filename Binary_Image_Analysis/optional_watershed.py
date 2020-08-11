# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:53:24 2020

@author: serdarhelli
"""


import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import matplotlib.pyplot as plt
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
# Load in image, convert to gray scale, and Otsu's threshold
kernel =( np.ones((3,3), dtype=np.float32))

image = cv2.imread('pred_image')
image=cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

erosion = cv2.erode(thresh,kernel,iterations =5)
gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel)



# Compute Euclidean distance from every binary pixel
# to the nearest zero pixel then find peaks
distance_map = ndimage.distance_transform_edt(erosion)
distance_map = ndimage.maximum_filter(distance_map, size=15, mode='constant')
local_max = peak_local_max(distance_map, indices=False, min_distance=40, labels=thresh)

# Perform connected component analysis then apply Watershed
markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
labels = watershed(-distance_map, markers, mask=thresh)


# Iterate through unique labels
total_area = 0
a=np.unique(labels)
area=np.zeros(len(a))
for label in a:
    if label == 0:
        continue

    # Create a mask
    mask = np.zeros(thresh.shape, dtype="uint8")
    mask[labels == label] = 255

    # Find contours and determine contour area
    cnts,hieararch = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] 
    (x,y),radius = cv2.minEnclosingCircle(cnts)
    rect = cv2.minAreaRect(cnts)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="int")    
    box = perspective.order_points(box)
    color1 = (list(np.random.choice(range(256), size=3)))  
    color =[int(color1[0]), int(color1[1]), int(color1[2])]  
    cv2.drawContours(image,[box.astype("int")],0,color,2)
    (tl,tr,br,bl)=box
    
    (tltrX,tltrY)=midpoint(tl,tr)
    (blbrX,blbrY)=midpoint(bl,br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
    (tlblX,tlblY)=midpoint(tl,bl)
    (trbrX,trbrY)=midpoint(tr,br)
	# draw the midpoints on the image
    cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),color, 2)
    cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),color, 2)
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
    
    ##your pixel size of your x_ray image
    pixelsPerMetric=0.096
    dimA = dA * pixelsPerMetric
    dimB = dB *pixelsPerMetric
    cv2.putText(image, "{:.1f}mm".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, color, 2)
    cv2.putText(image, "{:.1f}mm".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, color, 2)







cv2.imwrite('water_pred_image',image)
cv2.waitKey()
