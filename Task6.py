
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


#Backward Mapping
def my_rotation(angle):
    img=Image.open('pic3.jpg').convert('L')
    img=img.resize((512,512),Image.ANTIALIAS)
    img_arr=np.array(img)
    x= int((img_arr.shape[0]-1)/2)
    y= int((img_arr.shape[1]-1)/2)
    B=np.zeros((img_arr.shape))
    cos=np.cos(math.radians(angle))
    sin=np.sin(math.radians(angle))
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            xout=int((i-x)*cos + (j-y)*sin + x)
            yout=int((j-y)*cos - (i-x)*sin + y)
            if xout < 512 and  xout> 0  and yout < 512 and yout >0:
                B[i,j]=img_arr[xout,yout]
    return B
rot=my_rotation(45)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.imshow(rot)
ax.set_title("Backward Mapping")
#Forward Mapping

def my_rotation(angle):
    img=Image.open('pic3.jpg').convert('L')
    img=img.resize((512,512),Image.ANTIALIAS)
    img_arr=np.array(img)
    x= int((img_arr.shape[0]-1)/2)
    y= int((img_arr.shape[1]-1)/2)
    B=np.zeros((img_arr.shape))
    cos=np.cos(math.radians(angle))
    sin=np.sin(math.radians(angle))
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            xout=int((i-x)*cos + (j-y)*sin + x)
            yout=int((j-y)*cos - (i-x)*sin + y)
            if xout < 512 and  xout> 0  and yout < 512 and yout >0:
                B[xout,yout]=img_arr[i,j]
    return B
rot1=my_rotation(45)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.imshow(rot1)
ax.set_title("Forward Mapping")
#interpolation methods

image=Image.open('num_resized.jpg')
#ima=cv2.resize(np.array(image),(50,50))
image_nearest= cv2.resize(np.array(image),(256,256),interpolation = cv2.INTER_NEAREST)
image_linear= cv2.resize(np.array(image),(256,256),interpolation = cv2.INTER_LINEAR)
image_cubic= cv2.resize(np.array(image),(256,256),interpolation = cv2.INTER_CUBIC)
#im=Image.fromarray(ima)
#im.save('num_resized.jpg')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
axes[0].imshow(image_nearest)
axes[0].set_title("Nearest Neigbour ",fontsize=16)
axes[1].imshow(image_linear)
axes[1].set_title("Linear Interpolation ",fontsize=16)
axes[2].imshow(image_cubic)
axes[2].set_title("Cubic Interpolation ",fontsize=16)


plt.show()
