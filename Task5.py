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
from Task4 import apply_gaussian

#Implementing our own  Sobel Filter
def my_sobel_filter(img):
    sobel_outcome = np.copy(img)
    g_outcome=np.zeros((img.shape))
    size = sobel_outcome.shape

    #Creating kernels for sobel filter
    F1=np.array([[-1 ,0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]])
    F2=F1.T
    for i in range(size[0] - 2):
        for j in range(size[1] - 2):
            gx=np.sum(img[i:i+3,j:j+3]*np.flip(np.flip(F1, 1), 0) ) # 180 degree rotated
            gy=np.sum(img[i:i+3,j:j+3]*np.flip(np.flip(F2, 1), 0) ) # 180 degree rotated
            #Computing the sum of square root and putting it to our new array 
            sobel_outcome[i][j] = np.sqrt(gx**2 + gy**2)
            #Clipping the outcome to have min of 0 and max of 255
            sobel_outcome = np.clip(sobel_outcome,0,255)
#     plt.imshow(g_outcome)
#     plt.savefig('sobel_y.png')
    return sobel_outcome
smoothed_image = apply_gaussian()
image=Image.open('Q4_crop.jpg')
image_gray = image.convert('L')
image_gray=np.array(image_gray)
image1=Image.open('gblur.jpg')
gblur=np.array(image1)

sobimg=my_sobel_filter(smoothed_image)
sobimg_=my_sobel_filter(image_gray)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
axes[0].imshow(gblur)
axes[0].set_title(" Image with noise ",fontsize=16)
axes[2].imshow(sobimg)
axes[2].set_title(" Applying Sobel after Gaussian Blur ",fontsize=10)
axes[1].imshow(sobimg_)
axes[1].set_title(" Applying Sobel without Gaussian Blur ",fontsize=10)


#Testing on other face images
Q4im=Image.open('face 01 u6734495.jpg').convert('L')
Q4im_arr=np.array(Q4im)
q=Q4im_arr[0:450:,300:700]
q=Image.fromarray(q)
a=my_sobel_filter(np.array(q))
Q4im1=Image.open('face 02 u6734495.jpg').convert('L')
Q4im1_arr=np.array(Q4im1)
b=Q4im1_arr[0:450:,300:700]
b=Image.fromarray(b)
h=my_sobel_filter(np.array(b))

# plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
axes[0].imshow(a)
axes[0].set_title("Sobel Testing on 2nd image ",fontsize=16)
axes[1].imshow(h)
axes[1].set_title("Sobel Testing on 3rd image ",fontsize=16)



#Inbuilt Sobel
blur=Image.open('Gaussian_blur_output.jpg')
sobelx = cv2.Sobel(np.array(blur),cv2.CV_64F,1,0,ksize=3)  # x
sobely = cv2.Sobel(np.array(blur),cv2.CV_64F,0,1,ksize=3)  # y
G=np.sqrt(sobelx**2 + sobely**2)
#fig, ax = plt.subplots()
#ax.imshow(G)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
axes[0].imshow(sobimg)
axes[0].set_title("My sobel ",fontsize=16)
axes[1].imshow(G)
axes[1].set_title("Inbuilt Sobel ",fontsize=16)



plt.show()
