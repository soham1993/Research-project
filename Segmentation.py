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

"""application of contrast stretching"""
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
plt.title('contrast stretching'), plt.xticks([]), plt.yticks([])


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

detected_circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=30, minRadius = 1, maxRadius = 40)

# Draw circles that are detected. 
if detected_circles is  not None: 
  
    # Convert the circle parameters a, b and r to integers. 
    detected_circles = np.uint16(np.around(detected_circles)) 
  
    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
  
        # Draw the circumference of the circle. 
        cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
  
        # Draw a small circle (of radius 1) to show the center. 
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
        cv2.imshow("Detected Circle",img) 
        cv2.waitKey(0)
        print(pt[2])
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

shape=th3.shape
t=np.array(th3.flatten())
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        th3, 2, 2, error=0.005, maxiter=10000, init=None)
im=[]
for pix in u.T:
       
        im.append(cntr[np.argmax(pix)])
        
fuzzy_img = np.reshape(im,shape).astype(np.uint8)

cv2.imwrite("D:/PAP/fuzzy.jpg", fuzzy_img)

"""Applying watershed algorithm"""

# noise removal
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(fuzzy_img,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv2.imwrite("D:/PAP/watershed.jpg", markers)



