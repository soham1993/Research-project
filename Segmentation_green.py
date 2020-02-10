# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 18:49:53 2020

@author: Soham Dutta
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

#from fcmeans import FCM
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
#import imutils
from skimage import util 

from sklearn.mixture import GaussianMixture 
from os import listdir
from os.path import isfile, join
import os 


#import skfuzzy as fuzz

from PIL import Image
import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import color, io
path='C:/Users/Soham Dutta/Desktop/Pictures/2-01-2019/40X'

for i in os.listdir('C:/Users/Soham Dutta/Desktop/Pictures/2-01-2019/40X'):
    
    
    img=cv2.imread('C:/Users/Soham Dutta/Desktop/Pictures/2-01-2019/40X/'+str(i))
    row,col,n=img.shape
    
    r1=img[:,:,0]
    g1=img[:,:,1]
    b1=img[:,:,2]
    #cv2.imwrite("D:/PAP/green_channel.jpg", g1)
 
    print(i)
    n=i[:-4]
    print(n)
    
    
    sobelx = cv2.Sobel(g1,cv2.CV_32F,1,0,ksize=3) 
    sobely=  cv2.Sobel(g1,cv2.CV_32F,0,1,ksize=3) 
    sobel=sobelx+sobely
    #cv2.imwrite("D:/PAP/Sobel_Green.jpg", sobel)
    
    canny=cv2.Canny(g1,1,2)
    #cv2.imwrite("D:/PAP/canny_edge_detector_green.jpg", canny)
    
    """Binarizing"""
    
    ret1,th1 = cv2.threshold(g1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imwrite("D:/PAP/otsu_green.jpg", th1) 
    """phi = np.random.randn(2748,3584) # initial value for phi F = ... # some function dt = 1
    dt = 100
    F=1"""
    
    #dphi = {}
    #dphi_t = {}
    #dphi_norm = {}"""
    
    """for i in range(3):
        dphi = np.gradient(phi)
        dphi_norm = np.sqrt(np.sum(np.square(dphi), axis=0))
        phi = phi + dt * F * dphi_norm
    
        # plot the zero level curve of phi     
        plt.contour(phi, 0)
        plt.show()"""
    
    
    
    
    def grad(x):
        return np.array(np.gradient(x))
    
    
    def norm(x, axis=0):
        return np.sqrt(np.sum(np.square(x), axis=axis))
    
    
    def stopping_fun(x):
        return 1. / (1. + norm(grad(x))**2)
    
    
    
    
    # Smooth the image to reduce noise and separation between noise and edge becomes clear img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)
    """img = io.imread('D:/PAP/2d_40X.png')
    img = color.rgb2gray(img)
    img = img - np.mean(img)"""
    
    
    
    img_smooth = scipy.ndimage.filters.gaussian_filter(th1,sigma=1)
    F = stopping_fun(img_smooth)
    """Fnew=F.astype('uint8')
    added_image=np.empty((row, col), dtype=np.uint8)
    for i in range(0,row):
        for j in range(0,col):
            
            added_image[i][j]=th1[i][j]*F[i][j]
    
    circles = cv2.HoughCircles(F,cv2.HOUGH_GRADIENT,1,10,param1=5,param2=10, minRadius = 1, maxRadius = 30)
    
    # Draw circles that are detected. 
    
      
    for i in circles[0, :]: 
            cv2.circle(Fnew,(i[0],i[1]),i[2],(255,0,0),5)
        # draw the center of the circle
            #cv2.circle(cimg,(i[0],i[1]),2,(255,0,0),3)
    cv2.imshow('detected circles',Fnew)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    
    """def default_phi(x):
        # Initialize surface phi at the border (5px from the border) of the image     # i.e. 1 outside the curve, and -1 inside the curve     phi = np.ones(x.shape[:2])
        phi = np.ones(x.shape[:2])
        phi[5:-5, 5:-5] = -1.
        return phi"""
    
    
    """dt = 1
    n_iter=5
    dphi = {}
    dphi_t = {}
    dphi_norm = {}"""
    """for i in range(n_iter):
        dphi = grad(phi)
        dphi_norm = norm(dphi)
    
        dphi_t = F * dphi_norm
    
        phi = phi + dt * dphi_t"""
    
    """for i in range(10):
        dphi = grad(phi)
        dphi_norm = norm(dphi)
    
        dphi_t = F * dphi_norm
    
        phi = phi + dt * dphi_t"""
    
    """watershed"""
    """D = ndimage.distance_transform_edt(th1)
    localMax = peak_local_max(D, indices=False, min_distance=20,
    	labels=th1)
     
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=th1)
    cv2.imwrite("D:/PAP/watershed_green.jpg", labels)
    
    
    
    
    
    lab=labels.astype('uint8')
    
    plt.hist(lab.flatten(),256)
    plt.gca().set(title='Instensity', ylabel='Frequency');
     
    plt.show()
    
    ret3,th3  = cv2.threshold(lab,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    
    
      
    inv = np.zeros((row, col), dtype=np.uint8)
    for i in range(0,row):
        for j in range(0,col):
            if th3[i,j]==0:
                inv[i,j]=255
               # print(inv[i,j])
            elif th3[i,j]==1:
                inv[i,j]=0"""
    Final_output = np.empty((row, col, 3), dtype=np.uint8)
    added_image_r=np.empty((row, col), dtype=np.uint8)
    added_image_g=np.empty((row, col), dtype=np.uint8)
    added_image_b=np.empty((row, col), dtype=np.uint8)
    
    for i in range(0,row):
        for j in range(0,col):
            added_image_r[i][j]=F[i][j]*r1[i][j]
            added_image_g[i][j]=F[i][j]*g1[i][j]
            added_image_b[i][j]=F[i][j]*b1[i][j]
    
    kernel = np.ones(shape=(3, 3),
                     dtype=np.uint8)
    
    
    Final_output[:,:,0]=added_image_r
    Final_output[:,:,1]=added_image_g
    Final_output[:,:,2]=added_image_b
    #image_2= cv2.dilate(Final_output,kernel=kernel,iterations=5)
    #image_3= cv2.erode(Final_output,kernel=kernel,iterations=2)
    
    cv2.imwrite('C:/Users/Soham Dutta/Desktop/Pictures/2-01-2019/40X/'+n+"_output.jpg",Final_output )
    
    #cv2.imwrite("D:/PAP/superimposed_green_dilated.jpg",image_2 )
    
    #cv2.imwrite("D:/PAP/superimposed_green_eroded.jpg",image_3 )
    
    """im=th1.reshape(-1,1)
    gmm = GaussianMixture(n_components = 2, covariance_type='tied') 
    
    gmm.fit(im)
    lab_1 = gmm.predict(im) 
    lab_1=lab_1.reshape(row,col)
    
    kernel = np.ones((5,5),np.uint8)
    lab_2=lab_1.astype('uint8')
    
    erosion = cv2.erode(lab_2,kernel,iterations = 1)
    
    inv1=np.zeros((row, col), dtype=np.uint8)
    for i in range(0,row):
        for j in range(0,col):
            if erosion[i,j]==0:
                inv1[i,j]=0
               # print(inv[i,j])
            elif erosion[i,j]==1:
                inv1[i,j]=255"""
                
    """Applying watershed"""
    """D = ndimage.distance_transform_edt(inv1)
    localMax = peak_local_max(D, indices=False, min_distance=20,
    	labels=inv1)
     
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels_1 = watershed(-D, markers, mask=inv1)
    cv2.imwrite("D:/PAP/inverse_watershed_green.jpg",labels_1 )
    
    Final_output_1 = np.empty((row, col, 3), dtype=np.uint8)
    added_image_r_1=np.empty((row, col), dtype=np.uint8)
    added_image_g_1=np.empty((row, col), dtype=np.uint8)
    added_image_b_1=np.empty((row, col), dtype=np.uint8)
    
    
                
    for i in range(0,row):
        for j in range(0,col):
            
            added_image_r_1[i][j]=inv1[i][j]*r1[i][j]
            added_image_g_1[i][j]=inv1[i][j]*g1[i][j]
            added_image_b_1[i][j]=inv1[i][j]*b1[i][j]
    
          
    
    Final_output_1[:,:,0]=added_image_r_1
    Final_output_1[:,:,1]=added_image_g_1
    Final_output_1[:,:,2]=added_image_b_1
    
    cv2.imwrite("D:/PAP/final_output_GMM_green.jpg",Final_output_1 )
    cv2.imwrite("D:/PAP/inverted_image.jpg",inv1 )"""
    
    """def default_phi(x):
        # Initialize surface phi at the border (5px from the border) of the image     # i.e. 1 outside the curve, and -1 inside the curve    
        phi = np.ones(x.shape[:2])
        return phi
    
    def curvature(f):
        fy, fx = grad(f)
        norm = np.sqrt(fx*2 + fy*2)
        Nx = fx / (norm + 1e-8)
        Ny = fy / (norm + 1e-8)
        return div(Nx, Ny)
    
    
    def div(fx, fy):
        fyy, fyx = grad(fy)
        fxy, fxx = grad(fx)
        return fxx + fyy
    
    
    def dot(x, y, axis=0):
        return np.sum(x * y, axis=axis)
    
    
    
    def grad(x):
        return np.array(np.gradient(x))
    
    
    def norm(x, axis=0):
        return np.sqrt(np.sum(np.square(x), axis=axis))
    
    
    def stopping_fun(x):
        return 1. / (1. + norm(grad(x))**2)
    
    
    img = io.imread('D:/PAP/2d_40X.png')
    img = color.rgb2gray(img)
    img = img - np.mean(img)
    sigma=.5
    # Smooth the image to reduce noise and separation between noise and edge becomes clear 
    img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)
    
    F = stopping_fun(img_smooth)
    
    
    v = 1.
    dt = 1.
    alpha =100
    g = stopping_fun(img_smooth)
    dg = grad(g)
    dphi = {}
    dphi_t = {}
    phi = default_phi(img) 
    
    for i in range(1):
        dphi = grad(phi)
        dphi_norm = norm(dphi)
        kappa = curvature(phi)
        smoothing = g * kappa * dphi_norm
        balloon = g * dphi_norm * v
        attachment = dot(dphi, dg)
    
        dphi_t = smoothing + balloon + attachment
    
        phi = phi + dt * dphi_t
    """
