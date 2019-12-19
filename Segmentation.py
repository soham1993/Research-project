# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.transform import hough_ellipse
from scipy import ndimage
from skimage import data, color

from skimage import data
#from skimage.filters import threshold_multiotsu
from PIL import Image

from fcmeans import FCM
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from astropy.convolution import MexicanHat2DKernel
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu, threshold_local

import argparse
import imutils
from skimage import util 


import skfuzzy as fuzz

from PIL import Image
"""applying histogram stretching to the image"""
img=cv2.imread('D:/PAP/para02.BMP')
row,col,n=img.shape
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  

"""extracting three rgb images"""
r1=img[:,:,0]
g1=img[:,:,1]
b1=img[:,:,2]

"""application of contrast stretching"""
rnew1=(r1-r1.flatten().min())
rnew=rnew1/(r1.flatten().max()-r1.flatten().min())*255
gnew1=(g1-g1.flatten().min())
gnew=gnew1/(g1.flatten().max()-g1.flatten().min())*255
bnew1=(b1-b1.flatten().min())
bnew=bnew1/(b1.flatten().max()-b1.flatten().min())*255
#img1=cv2.merge((rnew, gnew, bnew))
img1 = np.empty((row, col, 3), dtype=np.uint8)
img1[:,:,0]=rnew
img1[:,:,1]=gnew
img1[:,:,2]=bnew

plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.hist(img1.flatten(),256,[0,256], color = 'b')
plt.xlim([0,256])
plt.show()

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(img1,cmap = 'gray')
plt.title('contrast stretching'), plt.xticks([]), plt.yticks([])


"""applying CLAHE to the image- data preprocessing"""
 

r=img1[:,:,0].astype('uint8')
g=img1[:,:,1].astype('uint8')
b=img1[:,:,2].astype('uint8')

clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8,8))
clahe_output_1 = clahe.apply(r)    #applying histogram equalization to equalize the intensity 
clahe_output_2 = clahe.apply(g)
clahe_output_3 = clahe.apply(b)

clahe_output=np.empty((row, col, 3), dtype=np.uint8)
clahe_output[:,:,0]=clahe_output_1
clahe_output[:,:,1]=clahe_output_2
clahe_output[:,:,2]=clahe_output_3


plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.hist(clahe_output.flatten(),256,[0,256], color = 'b')
plt.xlim([0,256])
plt.show()
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(clahe_output,cmap = 'gray')
plt.title('CLAHE'), plt.xticks([]), plt.yticks([])

#cv2.imwrite("D:/PAP/CLAHE.jpg", clahe_output)


"""Grayscale extraction"""
clahe_output_g=cv2.cvtColor(clahe_output, cv2.COLOR_BGR2GRAY)

"""Applying SOBEL edge detection technique"""
sobelx = cv2.Sobel(clahe_output_g,cv2.CV_32F,1,0,ksize=3) 
sobely=  cv2.Sobel(clahe_output_g,cv2.CV_32F,0,1,ksize=3) 
sobel=sobelx+sobely


plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobel,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.show()




plt.subplot(2,1,1), plt.imshow(sobel,cmap = 'gray')
plt.title('Original Noisy Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2), plt.hist(sobel.ravel(), 256)
plt.title('Histogram'), plt.xticks([]), plt.yticks([])
 
plt.show()

"""apply canny edge detector"""

canny=cv2.Canny(clahe_output_g,10,20)
cv2.imwrite("D:/PAP/canny_edge_detector.jpg", canny)



""" Apply Hough transform on the blurred image-eliptical/circular"""


"""cimg=clahe_output_g.astype('uint8')
circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,10,param1=30,param2=35, minRadius = 1, maxRadius = 30)

# Draw circles that are detected. 
circles = np.uint16(np.around(circles))
  
for i in circles[0, :]: 
        cv2.circle(cimg,(i[0],i[1]),i[2],(255,0,0),5)
    # draw the center of the circle
        #cv2.circle(cimg,(i[0],i[1]),2,(255,0,0),3)
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()"""


"""Applying otsu's thresh Convert the circle parameters a, b and r to integers. 
    
  
old method"""

"""applying watershed algorithm
D = ndimage.distance_transform_edt(fuzzy_img)
localMax = peak_local_max(D,labels=fuzzy_img)

markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=fuzzy_img)
cv2.imwrite("D:/PAP/watershed.jpg", labels)"""

"""calculating the threshold"""

#thr1,thr2=sobel.max()-1,sobel.max()+1
#ret,th1  = cv2.threshold(clahe_output_g,160,255,cv2.THRESH_BINARY)
sob=clahe_output_g.astype('uint8')
ret1,th1 = cv2.threshold(sob,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("D:/PAP/otsu.jpg", th1)  

   
            
"""implementing connected components
th=th1.astype('uint8')
ret, labels = cv2.connectedComponents(th)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()

imshow_components(labels)"""

    # Plot assigned clusters, for each data point in training set
"""cluster_membership = np.argmax(u, axis=0)
for j in range(2):
        ax.plot(a['component_1'][cluster_membership == j],
                a['component_2'][cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')



cv2.imwrite("D:/PAP/fuzzy.jpg", fuzzy_img)"""


#Applying watershed algorithm"""

# noise removal
# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(th1)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=th1)
 
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=th1)
cv2.imwrite("D:/PAP/watershed.jpg", labels)
"""total_area = 0
for label in np.unique(labels):
    if label == 0:
        continue

    # Create a mask
    mask = np.zeros(sobel.shape, dtype="uint8")
    mask[labels == label] = 255

    # Find contours and determine contour area
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    total_area += area
    cv2.drawContours(img, [c], -1, (36,0,12), 4)

print(total_area)
cv2.imshow('image', img)
cv2.waitKey()"""

"""Binarizing the image """

lab=labels.astype('uint8')
ret3,th3  = cv2.threshold(lab,7,1,cv2.THRESH_BINARY)
cv2.imwrite("D:/PAP/watershed_bin.jpg", th3)

  
inv = np.zeros((row, col), dtype=np.uint8)
for i in range(0,row):
    for j in range(0,col):
        if th3[i,j]==0:
            inv[i,j]=1
           # print(inv[i,j])
        elif th3[i,j]==1:
            inv[i,j]=0


cv2.imwrite("D:/PAP/inverted.jpg",inv )



"""superimposing """
Final_output = np.empty((row, col, 3), dtype=np.uint8)
added_image_r=np.empty((row, col), dtype=np.uint8)
added_image_g=np.empty((row, col), dtype=np.uint8)
added_image_b=np.empty((row, col), dtype=np.uint8)

for i in range(0,row):
    for j in range(0,col):
        added_image_r[i][j]=inv[i][j]*r1[i][j]
        added_image_g[i][j]=inv[i][j]*g1[i][j]
        added_image_b[i][j]=inv[i][j]*b1[i][j]
        

Final_output[:,:,0]=added_image_r
Final_output[:,:,1]=added_image_g
Final_output[:,:,2]=added_image_b

cv2.imshow('image',Final_output )
cv2.waitKey()

cv2.imwrite("D:/PAP/superimposed.jpg",Final_output )

# noise removal
"""kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(th1,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L1,3)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)



 Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
cv2.imwrite("D:/PAP/watershed.jpg", markers)"""