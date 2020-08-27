from PIL import Image
from PIL import ImageOps
from numpy import pi, mgrid, exp, square, zeros, ravel, dot, uint8
from itertools import product
import math
import numpy as np
import random
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color

#Loading and saving images
Q3image = Image.open('pic1.jpg')
Q3_1image=Image.open('pic2.jpg')
Q3_2image=Image.open('pic3.jpg')
Q3image = Q3image.resize((1024,768),Image.ANTIALIAS)
Q3_1image = Q3_1image.resize((1024,768),Image.ANTIALIAS)
Q3_2image = Q3_2image.resize((1024,768),Image.ANTIALIAS)
Q3image.save('face 01 u6734495.jpg')
Q3_1image.save('face 02 u6734495.jpg')
Q3_2image.save('face 03 u6734495.jpg')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axes[0].imshow(Q3image)
axes[0].set_title('Picture 1',fontsize=16)
axes[1].imshow(Q3_1image)
axes[1].set_title('Picture 2',fontsize=16)
axes[2].imshow(Q3_2image)
axes[2].set_title('Picture 3',fontsize=16)

#REizing the image 
Q33image=Image.open('face 01 u6734495.jpg')
Q33image = Q33image.resize((768,512),Image.ANTIALIAS)
np.array(Q33image).shape

#Splitting RGB image into 3 grayscale channels separately
red_image = Q33image.copy()
#Rchannel
rchannel=np.array(red_image)
rc=rchannel[:,:,0]
#Bchannel
blue_image=Q33image.copy()
bchannel=np.array(blue_image)
bc=bchannel[:,:,2]
#Gchannel
green_image=Q33image.copy()
gchannel=np.array(green_image)
gc=gchannel[:,:,1]


#Plotting
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
axes[0].imshow(Q33image)
axes[0].set_title("Original Image ",fontsize=16)
axes[1].imshow(rc)

axes[1].set_title(" R-Channel ",fontsize=16)
axes[2].imshow(gc)
axes[2].set_title(" G-Channel ",fontsize=16)
axes[3].imshow(bc)
axes[3].set_title(" B-Channel ",fontsize=16)

#Histogram
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
axes[0].hist(rc.flatten(),bins=256)
axes[0].set_title(" R-Channel Histogram ",fontsize=16)
axes[1].hist(gc.flatten(),bins=256)
axes[1].set_title(" G-Channel Histogram ",fontsize=16)
axes[2].hist(bc.flatten(),bins=256)
axes[2].set_title(" B-Channel Histogram ",fontsize=16)

#Equilization of the separate greyscale images
Original= ImageOps.equalize(Q33image, mask = None)
R= Image.fromarray(rc)
G= Image.fromarray(gc)
B= Image.fromarray(bc)
R.save('RImage.jpg')
G.save('GImage.jpg')
B.save('BImage.jpg')
R_eq= ImageOps.equalize(R, mask = None)
G_eq= ImageOps.equalize(G, mask = None) 
B_eq= ImageOps.equalize(B, mask = None)
eq=cv2.merge((np.array(R_eq),np.array(G_eq),np.array(B_eq)))
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
axes[0].imshow(Original)
axes[0].set_title(" Original Image Equilization",fontsize=10)
axes[1].imshow(R_eq)
axes[1].set_title(" R-Channel Equilization ",fontsize=10)
axes[2].imshow(G_eq)
axes[2].set_title(" G-Channel Equilization ",fontsize=10)
axes[3].imshow(B_eq)
axes[3].set_title(" B-Channel Equilization ",fontsize=10)
# saving the images
eqr=Image.fromarray(np.array(R_eq))
eqg=Image.fromarray(np.array(G_eq))
eqb=Image.fromarray(np.array(B_eq))
eq0=Image.fromarray(np.array(Original))
eqr.save('r_equ.jpg')
eqg.save('g_equ.jpg')
eqb.save('b_equ.jpg')
eq0.save('equ.jpg')

#plotting after equilization
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axes[0].hist(np.array(R_eq).flatten(),bins=256)
axes[0].set_title('Req Histogram',fontsize=16)
axes[1].hist(np.array(B_eq).flatten(),bins=256)
axes[1].set_title('Beq Histogram',fontsize=16)
axes[2].hist(np.array(G_eq).flatten(),bins=256)
axes[2].set_title('Geq Histogram',fontsize=16)
print("A")

plt.show()
