# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 00:06:25 2019

@author: Rafael Stekolshchik
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("images/family.jpg")
image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#image_copy = np.copy(image)
plt.imshow(image_copy)

## convert to hsv
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

## mask of green (36,0,0) ~ (70, 255,255)
hsv_copy = np.copy(hsv)
mask_green = cv2.inRange(hsv_copy, (36, 0, 0), (70, 255,255))

#masked_image_01[labels_reshape == 1] = [0, 255, 0]

#mask_green_copy = np.copy(mask_green)
## mask o yellow (15,0,0) ~ (36, 255, 255)
hsv_copy = np.copy(hsv)
mask_yellow = cv2.inRange(hsv_copy, (15,0,0), (36, 255, 255))

print('shape: ', mask_yellow.shape)

## final mask and masked
mask_green_yellow = cv2.bitwise_or(mask_green, mask_yellow)
# Bitwise-AND mask and original image

# define range of blue color in HSV
# Threshold the HSV image to get only blue colors
hsv_copy = np.copy(hsv)
mask_blue = cv2.inRange(hsv_copy, (100, 0, 0), (130, 255, 255))


f, axes = plt.subplots(4, 2, figsize=(20,30))

target = cv2.bitwise_and(image_copy, image_copy, mask=mask_green_yellow)
axes[0,0].set_title('Mask Green Yellow')
axes[0,0].imshow(mask_green_yellow, 'gray') 
axes[0,1].set_title('Masked Green Yellow')
axes[0,1].imshow(target) 

target_green = cv2.bitwise_and(image_copy, image_copy, mask=mask_green)
axes[1,0].set_title('Mask Green')
axes[1,0].imshow(mask_green, 'gray') 
#axes[1,0].imshow(mask_green_copy) 
axes[1,1].set_title('Masked Green')
axes[1,1].imshow(target_green) 

target_yellow = cv2.bitwise_and(image_copy, image_copy, mask=mask_yellow)
axes[2,0].set_title('Mask Yellow')
axes[2,0].imshow(mask_yellow, 'gray') 
axes[2,1].set_title('Masked Yellow')
axes[2,1].imshow(target_yellow) 

target_blue = cv2.bitwise_and(image_copy, image_copy, mask= mask_blue)
axes[3,0].set_title('Mask Blue')
axes[3,0].imshow(mask_blue, 'gray') 
axes[3,1].set_title('Masked Blue')
axes[3,1].imshow(target_blue) 

