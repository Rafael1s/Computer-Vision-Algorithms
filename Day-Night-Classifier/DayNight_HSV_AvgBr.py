# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:03:34 2018

@author: Rafael Stekolshchik
"""

import cv2 # computer vision library
import helpers

import numpy as np
import matplotlib.pyplot as plt

''' 
1. Show HSV channels for one day image'
2. Testing average brightness levels'''


# Image data directories
image_dir_training = "day_night_images/training/"
image_dir_test = "day_night_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(image_dir_training)

# Standardize all training images
''' see day_night2 to create own function standardize'''
STANDARDIZED_LIST = helpers.standardize(IMAGE_LIST)

# Find the average Value or brightness of an image
def avg_brightness(rgb_image):
    
    # Convert image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Add up all the pixel values in the V channel
    sum_brightness = np.sum(hsv[:,:,2])
    
    ## TODO: Calculate the average brightness using the area of the image
    # and the sum calculated above
    # avg = 0
    
    area = rgb_image.shape[0] * rgb_image.shape[1]
    avg = sum_brightness/area
    
    return avg


# Display a standardized image and its label
# Select an image by index
image_num = 0
selected_image = STANDARDIZED_LIST[image_num][0]
selected_label = STANDARDIZED_LIST[image_num][1]

# Display image and data about it
plt.imshow(selected_image)
print("Shape: "+str(selected_image.shape))
print("Label [1 = day, 0 = night]: " + str(selected_label))

# Convert and image to HSV colorspace
# Visualize the individual color channels

image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]


# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

''' 
# Look at a number of different day and night images and think about 
# what average brightness value separates the two types of images

# As an example, a "night" image is loaded in and its avg brightness is displayed'''
image_num = 140
test_im = STANDARDIZED_LIST[image_num][0]

avg = avg_brightness(test_im)
br_night = "= {:6.2f}".format(avg)
ax5.set_title('Night, Brigtness: ' +  str(br_night))
ax5.imshow(test_im)

print('Avg brightness for image ', image_num, br_night)

numb_day_im = 0
numb_night_im = 0
avr_days = 0
avr_nigths = 0
for i in range(len(STANDARDIZED_LIST)):
    curr_im = STANDARDIZED_LIST[i][0] 
    label = STANDARDIZED_LIST[i][1]
    avg_br = avg_brightness(curr_im)
    if (label == 1):
        numb_day_im += 1
        avr_days += avg_br
    else:
        numb_night_im +=1
        avr_nigths += avg_br

day_avr_br = avr_days/numb_day_im
night_avr_br = avr_nigths/numb_night_im
    
print('Day Images: ', numb_day_im, ', Night Images: ', numb_night_im)    
print('Day Avg Br: ', "{:6.2f}".format(day_avr_br), 
      ', Night Avg Br: ', "{:6.2f}".format(night_avr_br))    
    




