from PIL import Image
from PIL import ImageOps
from numpy import pi, mgrid, exp, square, zeros, ravel, dot, uint8
import math
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from skimage import io, color
		

#Loading a Greyscale image 
GRAYimg= Image.open('Lenna.png')
GRAYimg=GRAYimg.convert('L')
#Inverting the image 
GRAYimg_inv=ImageOps.invert(GRAYimg)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
axes[0].imshow(GRAYimg)
axes[0].set_title("Greyscale Image",fontsize=16)
axes[1].imshow(GRAYimg_inv)
axes[1].set_title("Negative Image ",fontsize=16)
GRAYimg.save('Q2_1.jpg')
GRAYimg_inv.save('Q2_1Ans.jpg')

#Flipping the image
row,col= np.array(GRAYimg).shape
GRAYimg=np.array(GRAYimg)
GRAYimg_flip = GRAYimg[row-1: :-1]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
axes[0].imshow(GRAYimg)
axes[0].set_title("Original Image ",fontsize=16)
axes[1].imshow(GRAYimg_flip)
axes[1].set_title("Flipped Image ",fontsize=16)
flip=Image.fromarray(GRAYimg_flip)
flip.save('Flip_image.jpg')


#Loading Original RGB Image
RGBimageo = Image.open('Lenna.png')

# Reading the image as an array
RGBimage=np.array(RGBimageo)


#Swapping R and B channel Inputs
RGBimage_BGR=RGBimage[:,:,::-1] 
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
axes[0].imshow(RGBimage)
axes[0].set_title("Original Image ",fontsize=16)
axes[1].imshow(RGBimage_BGR)
axes[1].set_title("Swapping R and B Channels ",fontsize=16)
RBswap=Image.fromarray(RGBimage_BGR)
RBswap.save('Swap_RB.jpg')

#inverteting the image and saving
row1,col1,x1 = RGBimage.shape
RGBflip = RGBimage[row1-1: :-1, :]
im = Image.fromarray(RGBflip)
im.save("RGBflip.jpg")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
axes[0].imshow(RGBimage)
axes[0].set_title("Original Image ",fontsize=16)
axes[1].imshow(RGBflip)
axes[1].set_title("Flipped Image ",fontsize=16)
flip_RGB=Image.fromarray(RGBflip)
flip_RGB.save('RGB_flip.jpg')

#RGBinverted=np.array('flipfig.jpg')
RGBinverted = Image.open('RGBflip.jpg')

RGBinv=np.array(RGBinverted)
#Averaging the Original and invereted image

f_avg=np.average([RGBimage,RGBinv],axis=0)
f_avg=f_avg.astype(int)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
axes[0].imshow(RGBimage)
axes[0].set_title("Original Image ",fontsize=16)
axes[1].imshow(RGBinv)
axes[1].set_title("Flipped Image ",fontsize=16)
axes[2].imshow(f_avg.astype(np.uint8))
axes[2].set_title("Averaged Image ",fontsize=16)
Averaged_Image= Image.fromarray(f_avg.astype(np.uint8))
Averaged_Image.save('Averaged.jpg')

#Task 5
GRAYimg_arr=np.array(GRAYimg)
r_noise=np.zeros((GRAYimg_arr.shape))
for i in range(GRAYimg_arr.shape[0]):
    for j in range(GRAYimg_arr.shape[1]):
        r_noise[i][j]=GRAYimg_arr[i][j]+np.random.randint(0,255)          
output=np.clip(r_noise,0,255)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.imshow(output)
ax.set_title("Random Pixel Added Image",fontsize=16)
random_noise= Image.fromarray(output.astype(uint8))
random_noise.save('Random_noise.jpg')

plt.show()
