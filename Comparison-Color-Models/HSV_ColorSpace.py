# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 00:27:47 2018

@author: Rafael Stekolshchik
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


# Read in the image
image = cv2.imread('images/water_balloons.jpg')
# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)

# RGB channels
r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]

#f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))

f, axes = plt.subplots(3, 3, figsize=(25,15))

axes[0,0].set_title('Red')
axes[0,0].imshow(r, cmap='gray')

axes[0,1].set_title('Green')
axes[0,1].imshow(g, cmap='gray')

axes[0,2].set_title('Blue')
axes[0,2].imshow(b, cmap='gray')

# Convert from RGB to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]


axes[1,0].set_title('Hue')
axes[1,0].imshow(h, cmap='gray')

axes[1,1].set_title('Saturation')
axes[1,1].imshow(s, cmap='gray')

axes[1,2].set_title('Value')
axes[1,2].imshow(v, cmap='gray')

#----------------


# Define our color selection criteria in RGB values
lower_pink = np.array([180,0,100]) 
upper_pink = np.array([255,255,230])

# Define the masked area in RGB space
mask_rgb = cv2.inRange(image, lower_pink, upper_pink)

# mask the image
masked_image = np.copy(image)
masked_image[mask_rgb==0] = [0,0,0]

# Vizualize the mask
#plt.imshow(masked_image)

axes[2,0].set_title('Pink Masked Image')
axes[2,0].imshow(masked_image)

# Now try HSV!
# Define our color selection criteria in HSV values
#lower_hue = np.array([160,0,0]) 
upper_hue = np.array([180,255,255])

lower_hue = np.array([0,0,0]) 
upper_hue = np.array([140,255,255])

# Define the masked area in HSV space
mask_hsv = cv2.inRange(hsv, lower_hue, upper_hue)

# mask the image
masked_image = np.copy(image)
masked_image[mask_hsv==0] = [0,0,0]

# Vizualize the mask
axes[2,1].set_title('Hue 0-140')
axes[2,1].imshow(masked_image)
#axes[2,1].imshow(image)

#---------------------------------
# OpenCV halves the H values to fit the range [0,255]
# H value instead of being in range [0, 360], is in range [0, 180]. 
# S and V are still in range [0, 255].
lower_hue_2 = np.array([140,0,0]) 
upper_hue_2 = np.array([180,255,255])


# Define the masked area in HSV space
mask_hue_2 = cv2.inRange(hsv, lower_hue_2, upper_hue_2)

# mask the image
masked_image = np.copy(image)
masked_image[mask_hue_2==0] = [0,0,0]

# Vizualize the mask
axes[2,2].set_title('Hue 140-255')
axes[2,2].imshow(masked_image)

