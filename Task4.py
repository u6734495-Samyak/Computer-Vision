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



#creating a gaussian random noise
def add_noise(img):
    noise = np.zeros(img.shape, dtype=np.uint8)
    noise= np.random.normal(mean,std,Q42image.shape)
    new_img = img + noise
    return new_img



#Gaussian Blur implementation
def my_Gauss_filter(k_size,sigma):
    gaussian_filter = np.zeros((k_size, k_size), np.float32)
    m = k_size//2 
    n = k_size//2
    sum=0.0
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            #applying the formula for 2-D Gaussian
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2
            sum+=gaussian_filter[x+m,y+n]
    gaussian_filter/=sum
    return gaussian_filter
def apply_gaussian():
    img = np.array(Image.open('gblur.jpg'))
    img_out = img.copy()
    gaussian_filter=my_Gauss_filter(5,5)
    gaussian_filter =np.flip(np.flip(gaussian_filter, 1), 0) # rotating the filter 180 degress for convolution
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height-4): # avoiding corners
        for j in range(width-4):
            img_out[i][j]=np.sum(img[i:i+5,j:j+5]*gaussian_filter)
            
    return img_out


if __name__=='__main__':

#cropping the facial part of the picture

	Q4image=Image.open('pic2.jpg')
	Q4image_arr=np.array(Q4image)
	crop=Q4image_arr[0:2000,1200:3200,:]
	Q4_crop=Image.fromarray(crop)
	Q4_crop=Q4_crop.resize((256,256),Image.ANTIALIAS)
	Q4_crop.save('Q4_crop.jpg')


	image=Image.open('Q4_crop.jpg')
	image_gray = image.convert('L')

	image_gray.save('gray_image.jpg')
	image_gray=np.array(image_gray)
	#Plotting
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
	axes[0].imshow(Q4_crop)
	axes[0].set_title(" Cropped Image ",fontsize=16)
	axes[1].imshow(image_gray)
	axes[1].set_title(" Croppped Image Greyscale ",fontsize=16)

	#using python's normal method to create a random gaussain with a spec mean and covariance
	mean=0
	std=15
	Q42image=np.array(image_gray)
	u=add_noise(image_gray)
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
	axes[0].imshow(Q42image)
	axes[0].set_title(" Before Noise",fontsize=16)
	axes[1].imshow(u)
	axes[1].set_title(" After Noise",fontsize=16)
	u = u.astype(np.uint8)
	gblur=Image.fromarray(u)
	gblur.save('gblur.jpg')

	# plotting histogram for before and after adding
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
	axes[0].hist(Q42image.flatten(),bins=256)
	axes[0].set_title("Histrogram Before Noise",fontsize=16)
	axes[1].hist(u.flatten(),bins=256)
	axes[1].set_title(" Histogram After Noise",fontsize=16)
	fig.savefig('Hist Noise.png')
	print("Histograms")


	smoothed_image = apply_gaussian()
	sm=Image.fromarray(smoothed_image)
	sm.save('gaussian output_3.jpg')
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
	axes[0].imshow(gblur)
	axes[0].set_title(" After Noise",fontsize=16)
	axes[1].imshow(smoothed_image)
	axes[1].set_title(" Applying Gaussian Blur ",fontsize=16)

	#Checking with inbuilt Gaussian Blur
	blur  = cv2.GaussianBlur(np.array(gblur),(5,5),1)
	Cimage= Image.fromarray(blur)
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
	axes[0].imshow(gblur)
	axes[0].set_title(" Noise Image",fontsize=16)
	axes[1].imshow(blur)
	axes[1].set_title(" Inbuilt Gaussain",fontsize=16)

	plt.show()
