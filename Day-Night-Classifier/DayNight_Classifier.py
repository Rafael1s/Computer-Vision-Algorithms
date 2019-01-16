# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 12:01:34 2018

@author: Rafael Stekolshchik
"""

import cv2 # computer vision library
import helpers

import numpy as np
import matplotlib.pyplot as plt

''' 1. Classify day and night images
    2. Visualize the misclassified imagess
''' 


'''Training and Testing Data
The 228 day/night images are separated into training and testing datasets.

160 of these images are training images. They are used to create a classifier.
128 are test images used to test the accuracy of the classifier.'''

# Image data directories
image_dir_training = "day_night_images/training/"
image_dir_test = "day_night_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(image_dir_training)

print('training images: ', len(image_dir_training))

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

# Testing average brightness levels
# Look at a number of different day and night images and think about 
# what average brightness value separates the two types of images

# As an example, a "night" image is loaded in and its avg brightness is displayed
image_num = 15
test_im = STANDARDIZED_LIST[image_num][0]

avg = avg_brightness(test_im)
print('Avg brightness: ', "{:6.2f}".format(avg))
plt.imshow(test_im)

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


separate = (day_avr_br + night_avr_br)/2

# This function should take in RGB image input
def estimate_label(rgb_image):
    
    ## TODO: extract average brightness feature from an RGB image 
    # Use the avg brightness feature to predict a label (0, 1)
    predicted_label = 0
    
    avg_br = avg_brightness(rgb_image)
    
    if (avg_br > separate):
        predicted_label = 1
        
    return predicted_label    

numb_true = 0
numb_false = 0

for i in range(len(STANDARDIZED_LIST)):
    curr_im = STANDARDIZED_LIST[i][0] 
    label = STANDARDIZED_LIST[i][1]
    #avg_br = avg_brightness(curr_im)
    predicted = estimate_label(curr_im)
    if(label == predicted):
           numb_true += 1
    else:
           numb_false +=1 

print('Correctly classified: ', numb_true, ', Not-Correct: ', numb_false)

## Test out your code by calling the above function and seeing 
# how some of your training data is classified
TEST_IMAGE_LIST = helpers.load_dataset(image_dir_test)
TEST_STANDARDIZED_LIST = helpers.standardize(TEST_IMAGE_LIST)

numb_true = 0
numb_false = 0

for i in range(len(TEST_STANDARDIZED_LIST)):
    curr_im = TEST_STANDARDIZED_LIST[i][0] 
    label = TEST_STANDARDIZED_LIST[i][1]
    predicted = estimate_label(curr_im)
    if(label == predicted):
           numb_true += 1
    else:
           numb_false +=1 

print('Test correctly classified: ', numb_true, ', Test-Not-Correct: ', numb_false)
n =  numb_true + numb_false
print('Accuracy, checked on test: ', numb_true/n)


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


'''Visualize the misclassified images'''
# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(TEST_STANDARDIZED_LIST)

# Accuracy calculations
total = len(TEST_STANDARDIZED_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))


''' Visualize the misclassified images'''
# Visualize misclassified example(s)
num = 0
test_mis_im = MISCLASSIFIED[num][0]
label = 1 - MISCLASSIFIED[num][1]
## TODO: Display an image in the `MISCLASSIFIED` list 
## TODO: Print out its predicted label - 
## to see what the image *was* incorrectly classified as
nmis = len(MISCLASSIFIED)
print('n missclass: ', nmis)


if (nmis == 10):
  fig, axes = plt.subplots(5, 2, figsize=(20,20))
  for i in range(0, 10):
      row = i//2
      col = i%2
      #print('get i: ', i, ', row: ', row, ', col: ', col)
      im_0 = MISCLASSIFIED[i][0] 
      axes[row, col].set_title('Misclassified '+ str(i))
      axes[row, col].imshow(im_0)
      #axes[row,1].imshow(im_1)   
else:    
  fig = plt.figure(figsize=(50, 50))  # width, height in inches
  for i in range(nmis):
      curr_im = MISCLASSIFIED[i][0] 
      sub = fig.add_subplot(nmis, 1, i + 1)
      sub.imshow(curr_im, interpolation='nearest')
   
    
    
