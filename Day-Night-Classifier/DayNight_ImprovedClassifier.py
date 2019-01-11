# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:22:27 2018

@author: Rafael Stekolshchi
"""

import cv2 # computer vision library
import helpers

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%matplotlib inline
# Image data directories
image_dir_training = "day_night_images/training/"
image_dir_test = "day_night_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(image_dir_training)

# Standardize all training images
STANDARDIZED_LIST = helpers.standardize(IMAGE_LIST)

# Display a standardized image and its label

# Select an image by index
image_num = 0
selected_image = STANDARDIZED_LIST[image_num][0]
selected_label = STANDARDIZED_LIST[image_num][1]

# Display image and data about it
plt.imshow(selected_image)
print("Shape: "+str(selected_image.shape))
print("Label [1 = day, 0 = night]: " + str(selected_label))

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
print('Avg brightness: ' + str(avg), ', Ble Avg brightness: ', str(avg_blue))
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
    
print('numb_day_im: ', numb_day_im, ', day_avr_br: ', day_avr_br)    
print('numb_night_im: ', numb_night_im, ', night_avr_br: ', night_avr_br)    
print('Day.Aver.Br: ', day_avr_br, 'Night.Aver.Br: ', night_avr_br)    
print('Day.Blue.Avg.Br', avr_blue_days, 'Night.Blue.Aver.Br: ', avr_blue_nights)      

threshold = (day_avr_br + night_avr_br)/2
blue_threshold = (avr_blue_days + avr_blue_nights)/2 

print('threshold: ', threshold, ', blue_threshold: ', blue_threshold)

''' result Avg brightness: 119.6223 , Ble Avg brightness:  70.4688515152
numb_day_im:  120 , night_avr_br:  69.2008922727
Day.Aver.Br:  137.377906881 Night.Aver.Br:  69.2008922727
Day.Blue.Avg.Br 126.714323851 Night.Blue.Aver.Br:  40.8556655051
threshold:  103.289399577 , blue_threshold:  83.784994678'''

# This function should take in RGB image input
def estimate_label(rgb_image):
    
    # TO-DO: Extract average brightness feature from an RGB image 
    #avg = None
        
    avg = avg_brightness(rgb_image)        
    # Use the avg brightness feature to predict a label (0, 1)
    predicted_label = 0
    # TO-DO: Try out different threshold values to see what works best!
    # threshold = 0
    if(avg > threshold):
        # if the average brightness is above the threshold value, we classify it as "day"
        predicted_label = 1
    # else, the predicted_label can stay 0 (it is predicted to be "night")
    
    return predicted_label

import random

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(image_dir_test)

# Standardize the test data
STANDARDIZED_TEST_LIST = helpers.standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Constructs a list of misclassified images given a list of test images and their labels
def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels

# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

''' result Accuracy: 0.925
Number of misclassified images = 12 out of 160'''

# Visualize misclassified example(s)
num = 0
test_mis_im = MISCLASSIFIED[num][0]
label = 1 - MISCLASSIFIED[num][1]
## TODO: Display an image in the `MISCLASSIFIED` list 
## TODO: Print out its predicted label - 
## to see what the image *was* incorrectly classified as
nmis = len(MISCLASSIFIED)
print('n missclass: ', nmis)

#h = test_mis_im[0]
#w = test_mis_im[1]
fig = plt.figure(figsize=(50, 50))  # width, height in inches

for i in range(nmis):
    curr_im = MISCLASSIFIED[i][0] 
    sub = fig.add_subplot(nmis, 1, i + 1)
    sub.imshow(curr_im, interpolation='nearest')

# Additional feature

all_reclassified = 0

for i in range(len(MISCLASSIFIED)):
    curr_im = MISCLASSIFIED[i][0]
    label = MISCLASSIFIED[i][1]
    blue_comp = curr_im[:,:,2]
    area = 600*1100.0
    sum_brightness = np.sum(blue_comp)
    avg = avg_brightness(curr_im)
    avg_blue = sum_brightness/area
    print('avg brighness: ', avg, 
          'avg_blue : ', avg_blue, ', True.Label: ', label)
    if (avg > threshold and avg_blue < blue_threshold):
         ## i.e. classified as Day
         all_reclassified += 1
         print('reclassified as Night Image')   
    if (avg < threshold and avg_blue > blue_threshold):
         ## i.e. classified as Night
         all_reclassified += 1
         print('reclassified as Day Image')      

print(' Reclassified ', all_reclassified, ' from ', nmis, ' misclassified images')         
         
    
    

