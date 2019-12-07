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
"""applying histogram stretching to the image"""
img=cv2.imread('D:/PAP/carcinoma_in_situ/149143370-149143378-002.BMP')
"""extracting three rgb images"""
r1=img[:,:,0]
g1=img[:,:,1]
b1=img[:,:,2]

"""application of contrast stretching
rnew1=(r1-r1.flatten().min())
rnew=rnew1/(r1.flatten().max()-r1.flatten().min())*255
gnew1=(g1-g1.flatten().min())
gnew=gnew1/(g1.flatten().max()-g1.flatten().min())*255
bnew1=(b1-b1.flatten().min())
bnew=bnew1/(b1.flatten().max()-b1.flatten().min())*255
img1=cv2.merge((rnew, gnew, bnew))

plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.hist(img1.flatten(),256,[0,256], color = 'b')
plt.xlim([0,256])
plt.show()

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(img1,cmap = 'gray')
plt.title('contrast stretching'), plt.xticks([]), plt.yticks([])"""


"""applying CLAHE to the image- data preprocessing"""
 #converts it to gray scale image 
"""extracting three rgb images"""

r=img1[:,:,0].astype('uint8')
g=img1[:,:,1].astype('uint8')
b=img1[:,:,2].astype('uint8')

clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8,8))
clahe_output_1 = clahe.apply(r)    #applying histogram equalization to equalize the intensity 
clahe_output_2 = clahe.apply(g)
clahe_output_3 = clahe.apply(b)

clahe_output=cv2.merge((r, g, b))


plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.hist(clahe_output.flatten(),256,[0,256], color = 'b')
plt.xlim([0,256])
plt.show()
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(clahe_output,cmap = 'gray')
plt.title('CLAHE'), plt.xticks([]), plt.yticks([])

cv2.imwrite("D:/PAP/CLAHE.jpg", clahe_output)


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

sobel_rgb=cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)



cv2.imwrite("D:/PAP/sobel.jpg", sobel_rgb)






""" Apply Hough transform on the blurred image-eliptical/circular"""


cimg=sobel.astype('uint8')

circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=30, minRadius = 1, maxRadius = 40)

# Draw circles that are detected. 
circles = np.uint16(np.around(circles))
  
    # Convert the circle parameters a, b and r to integers. 
    
  
for i in circles[0, :]: 
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""Applying otsu's threshold method"""
otsu=sobel.astype('uint8')
ret3,th3 = cv2.threshold(otsu,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("D:/PAP/otsu.jpg", th3)

"""applying watershed algorithm
D = ndimage.distance_transform_edt(fuzzy_img)
localMax = peak_local_max(D,labels=fuzzy_img)

markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=fuzzy_img)
cv2.imwrite("D:/PAP/watershed.jpg", labels)"""

"""Applying Fuzzy C means algorithm"""

"""shape=th3.shape
t=np.array(th3.flatten())
cntr, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(
        th3, 2, 2, error=0.005, maxiter=10000, init=None)
im=[]
for pix in u.T:
       
        im.append(cntr[np.argmax(pix)])
        
fuzzy_img = np.reshape(im,shape).astype(np.uint8)

cv2.imwrite("D:/PAP/fuzzy.jpg", fuzzy_img)"""

"""Applying watershed algorithm"""

# noise removal
# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(th3)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=th3)
 
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=th3)
cv2.imwrite("D:/PAP/watershed1.jpg", labels)
