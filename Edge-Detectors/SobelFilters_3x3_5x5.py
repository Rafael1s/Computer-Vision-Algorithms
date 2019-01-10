# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 07:53:03 2018

@author: Rafael Stekolshchik
"""
''' Import resources and display image'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

# Read in the image
image = mpimg.imread('images/curved_lane.jpg')

plt.imshow(image)

''' Convert the image to grayscale'''

# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#plt.set_title('Curved Lane')
plt.imshow(gray, cmap='gray')

'''
TODO: Create a custom kernel
Below, you've been given one common type of edge detection filter:
a Sobel operator.
 
It's up to you to create a Sobel x operator and apply it to the given image.

For a challenge, see if you can put the image through a series of filters:
first one that blurs the image (takes an average of pixels), 
and then one that detects the edges.'''

# Create a custom kernel

# 3x3 array for edge detection
sobel_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])

## TODO: Create and apply a Sobel x operator

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])


dec_filter = np.array([[-0.5, -1, -0.5],
                       [0, 0, 0],
                       [0.5, 1, 0.5]])

sobel_y_5x5 = np.array([[2, 2, 4, 2, 2],
                        [1, 1, 2, 1, 1], 
                        [0, 0, 0, 0, 0], 
                        [-1, -1, -2, -1, -1],
                        [-2,- 2, -4, -2, -2]])
    
# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_x = cv2.filter2D(gray, -1, sobel_x)
filtered_image_y = cv2.filter2D(gray, -1, sobel_y)
filtered_image_dec = cv2.filter2D(gray, -1, dec_filter)
filtered_image_y_5x5 = cv2.filter2D(gray, -1, sobel_y_5x5)

image = mpimg.imread('images/birds.jpg')
gray_birds = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
filtered_image_y_birds = cv2.filter2D(gray_birds, -1, sobel_y)
filtered_image_y_5x5_birds = cv2.filter2D(gray_birds, -1, sobel_y_5x5)
#plt.imshow(filtered_image, cmap='gray')

f, axes = plt.subplots(2, 3, figsize=(20,10))

axes[0,0].set_title('Sobel X 3x3')
axes[0,0].imshow(filtered_image_x, cmap='gray')

axes[0,1].set_title('Sobel Y 3x3')
axes[0,1].imshow(filtered_image_y, cmap='gray')

axes[1,0].set_title('Decimal filter 3x3')
axes[1,0].imshow(filtered_image_dec)

axes[1,1].set_title('Sobel y 5x5')
axes[1,1].imshow(filtered_image_y_5x5, cmap='gray')

axes[0,2].set_title('Sobel_y 3x3 birds')
axes[0,2].imshow(filtered_image_y_birds, cmap='gray')

axes[1,2].set_title('Sobel_y_5x5 Birds')
axes[1,2].imshow(filtered_image_y_5x5_birds, cmap='gray')


