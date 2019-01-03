# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 12:49:12 2018

@author: Rafael Stekolshchik
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2      

# load in color image for face detection

##image = cv2.imread('images/raised_hand1.jpg')

image = cv2.imread('images/thumbs_up_down.jpg')


image_copy = np.copy(image)

# convert to RBG
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

#plt.figure(figsize=(20,10))
#plt.imshow(image_copy)

gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)  

#plt.imshow(gray, cmap='gray')

retval, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

#plt.imshow(binary, cmap='gray')


rerval, contours, hierarchy = \
      cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image_copy2 = np.copy(image_copy)

''' -1 means all the contours '''
''' 200 - thumb up (left hand)'''
contours_image = cv2.drawContours(image_copy2, contours, -1, (0,255,0), 3)

#plt.imshow(contours_image) #, cmap='gray')


## TODO: Complete this function so that 
## it returns the orientations of a list of contours
## The list should be in the same order as the contours
## i.e. the first angle should be the orientation of the first contour
def orientations(contours):
    """
    Orientation 
    :param contours: a list of contours
    :return: angles, the orientations of the contours
    """    
    
    # Create an empty list to store the angles in
    # Tip: Use angles.append(value) to add values to this list
    angles = []
    
    print('numb.of.contours: ', len(contours))
    for i in range(len(contours)):
        # Fit an ellipse to a contour and extract the angle from that ellipse
        #print('type.of.contour: ', type(contours[i]))
        #print('numb.of.elem.in.countour: ', len(contours[i]))
        #if (len(contours[i])) >= 50:
           print(' contour.number: ', i, 'numb.of.elem.in.countour: ', len(contours[i]))
           (x,y), (MA,ma), angle = cv2.fitEllipse(contours[i])
           angles.append(angle)
    
    return angles

# ---------------------------------------------------------- #
# Print out the orientation values
angles = orientations(contours)
numb_max = np.argmax(angles)
print('max angle in contour: ', numb_max)
print('Angles of each contour (in degrees): ' + str(angles))

## TODO: Complete this function so that
## it returns a new, cropped version of the original image
def left_hand_crop(image, selected_contour):
    """
    Left hand crop 
    :param image: the original image
    :param selected_contour: the contour that will be used for cropping
    :return: cropped_image, the cropped image around the left hand
    """
    
    ## TODO: Detect the bounding rectangle of the left hand contour
    # Find the bounding rectangle of a selected contour
    x,y,w,h = cv2.boundingRect(selected_contour)
    
    print('x, y, w, h: ', x, y, w, h)
    
    # Draw the bounding rectangle as a purple box
    box_image = cv2.rectangle(image, (x,y), (x+w,y+h), (200,0,200),2)
    
    #plt.imshow(box_image, cmap='gray')
    ## TODO: Crop the image using the dimensions of the bounding rectangle
    # Make a copy of the image to crop
    # Crop using the dimensions of the bounding rectangle (x, y, w, h)
    cropped_image = image[y: y + h, x: x + w] 
        
    return box_image, cropped_image


## TODO: Select the left hand contour from the list
## Replace this value
selected_contour = contours[numb_max] # None

# If you've selected a contour
if(selected_contour is not None):
    # Call the crop function with that contour passed in as a parameter
    box_image, cropped_image = left_hand_crop(contours_image, selected_contour)
        
    fig=plt.figure(figsize=(15, 10))
        
    fig.add_subplot(2, 2, 1)
    plt.title('Two hands')
    plt.imshow(image_copy)
    
    fig.add_subplot(2, 2, 2)
    plt.title('Boxed one hand')    
    plt.imshow(box_image)
    
    fig.add_subplot(2, 2, 3)
    plt.title('Left hand with green contour')
    plt.imshow(cropped_image)
    
    plt.show()
    
