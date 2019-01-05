# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:48:54 2018

@author: Rafael Stekolshchik
"""
''' 
We'd like to build a classifier that can accurately label 
these images as day or night, and that relies
on finding distinguishing features between the two types of images!

Note: All images come from the AMOS dataset (Archive of Many Outdoor Scenes).
'''

''' 1. Day Night Threshold
    2. Blue channel Threshold ''' 

import cv2 # computer vision library
import helpers

import numpy as np
import matplotlib.pyplot as plt

'''Training and Testing Data
The 92 day/night images are separated into training and testing datasets.

52 of these images are training images, for you to use as you create a classifier.
40 are test images, which will be used to test the accuracy of your classifier.
First, we set some variables to keep track of some where our images are stored'''

'''First, we set some variables to keep track of some where our images are stored:'''
image_dir_training = "day_night_images/training/"
image_dir_test = "day_night_images/test/"

'''Load the datasets
These first few lines of code will load the training day/night images 
and store all of them in a variable, IMAGE_LIST. 
This list contains the images and their associated label ("day" or "night").
For example, the first image-label pair 
in IMAGE_LIST can be accessed by index:  IMAGE_LIST[0][:].
'''
''' 
Using the load_dataset function in helpers.py
Load training data'''
IMAGE_LIST = helpers.load_dataset(image_dir_training)

'''
Construct a STANDARDIZED_LIST of input images and output labels.
This function takes in a list of image-label pairs and outputs
 a standardized list of resized images and numerical labels
'''
# Standardize all training images
STANDARDIZED_LIST = helpers.standardize(IMAGE_LIST)

'''
Visualize the standardized data
Display a standardized image from STANDARDIZED_LIST.
'''

# Display a standardized image and its label

# Select an image by index
image_num = 0
selected_image = STANDARDIZED_LIST[image_num][0]
selected_label = STANDARDIZED_LIST[image_num][1]

# Display image and data about it
plt.imshow(selected_image)
print("Shape: "+str(selected_image.shape))
print("Label [1 = day, 0 = night]: " + str(selected_label))


'''Feature Extraction
Create a feature that represents the brightness in an image. 
We'll be extracting the average brightness using HSV colorspace.
 Specifically, we'll use the V channel (a measure of brightness),
 add up the pixel values in the V channel, then divide that sum 
 by the area of the image to get the average Value of the image.

Find the average brightness using the V channel
This function takes in a standardized RGB image and returns
 a feature (a single value) that represent the average level of 
 brightness in the image. We'll use this value to classify
 the image as day or night.
'''

# Find the average Value or brightness of an image
def avg_brightness(rgb_image):
    # Convert image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Add up all the pixel values in the V channel
    sum_brightness = np.sum(hsv[:,:,2])
    area = 600*1100.0  # pixels
    
    # find the avg
    avg = sum_brightness/area
    
    
    return avg

def blue_estimate(rgb_image):
    blue_comp = rgb_image[:,:,2]
    blue_sum_brightness = np.sum(blue_comp)
    area = 600*1100.0  # pixels
    avg_blue = blue_sum_brightness/area
    return avg_blue

# Testing average brightness levels
# Look at a number of different day and night images and think about 
# what average brightness value separates the two types of images

# As an example, a "night" image is loaded in and its avg brightness is displayed
image_num = 19
test_im = STANDARDIZED_LIST[image_num][0]

avg = avg_brightness(test_im)
avg_blue = blue_estimate(test_im)
print('Avg brightness: ' + str(avg), ', Blue Avg brightness: ', str(avg_blue))
plt.imshow(test_im)

numb_day_im = 0
numb_night_im = 0
avr_days = 0
avr_nigths = 0
avr_blue_days = 0
avr_blue_nights = 0
for i in range(len(STANDARDIZED_LIST)):
    curr_im = STANDARDIZED_LIST[i][0] 
    label = STANDARDIZED_LIST[i][1]
    avg_br = avg_brightness(curr_im)
    blue_avg = blue_estimate(curr_im)
    if (label == 1):
        numb_day_im += 1
        avr_days += avg_br
        avr_blue_days += blue_avg
    else:
        numb_night_im +=1
        avr_nigths += avg_br
        avr_blue_nights += blue_avg

day_avr_br = avr_days/numb_day_im
night_avr_br = avr_nigths/numb_night_im
      
avr_blue_days = avr_blue_days/numb_day_im
avr_blue_nights = avr_blue_nights/numb_night_im      
    

print('Day Images: ', numb_day_im, ', Night Images: ', numb_night_im)    
print('Day Avg Br: ', "{:6.2f}".format(day_avr_br), 
      ', Night Avg Br: ', "{:6.2f}".format(night_avr_br))
print('Day Blue Avg Br', "{:6.2f}".format(avr_blue_days), 
      'Night Blue Aver Br: ', "{:6.2f}".format(avr_blue_nights))      

threshold = (day_avr_br + night_avr_br)/2
blue_threshold = (avr_blue_days + avr_blue_nights)/2 

print('Day-Night Threshold: ', "{:6.2f}".format(threshold), 
      ', Blue_threshold: ', "{:6.2f}".format(blue_threshold))



