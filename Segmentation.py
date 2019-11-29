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

"""applying CLAHE to the image- data preprocessing"""
img=cv2.imread('D:/PAP/carcinoma_in_situ/149143370-149143378-002.BMP') #converts it to gray scale image 
"""extracting three rgb images"""

r=img[:,:,0]
g=img[:,:,1]
b=img[:,:,2]

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
"""
D = ndimage.distance_transform_edt(th3)
localMax = peak_local_max(D, indices=False, min_distance=1,
	labels=th3)

markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=th3)
cv2.imwrite("D:/PAP/watershed.jpg", labels)


 
