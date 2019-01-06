# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 21:51:54 2018

@author: Rafael Stekolshchik
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read in the images
image = cv2.imread('images/city_hall.jpg')

image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

#plt.imshow(image_copy)

gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

#plt.imshow(gray, cmap = 'gray')

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])


filtered_image_x = cv2.filter2D(gray, -1, sobel_x)

filtered_image_y = cv2.filter2D(gray, -1, sobel_y)

''' more white edges : '''
retval, binary_image_x = cv2.threshold(filtered_image_x, 100, 255, cv2.THRESH_BINARY)
retval, binary_image_y_50 = cv2.threshold(filtered_image_y, 50, 255, cv2.THRESH_BINARY)

f, axes = plt.subplots(3, 2, figsize=(20,25))

axes[0,0].set_title('City Hall')
axes[0,0].imshow(image)

axes[0,1].set_title('City Hall')
axes[0,1].imshow(gray, cmap = 'gray')

axes[1,0].set_title('Sobel X Edges')
axes[1,0].imshow(filtered_image_x, cmap='gray')

axes[1,1].set_title('Sobel Y Edges')
axes[1,1].imshow(filtered_image_y, cmap='gray')

axes[2,0].set_title('Sobel X Edges, Threshold 100-255')
axes[2,0].imshow(binary_image_x, cmap = 'gray')

axes[2,1].set_title('Sobel Y Edges, Threshold 50-255')
axes[2,1].imshow(binary_image_y_50, cmap = 'gray')
