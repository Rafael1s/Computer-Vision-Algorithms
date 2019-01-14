# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 19:41:23 2019

@author: Rafael Stekolshchik
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# %matplotlib inline
# Read in the image
image = cv2.imread('images/round_farms.jpg')

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

image_Blur = cv2.GaussianBlur(gray, (3, 3), 0)



''' param1 – First method-specific parameter. In case of CV_HOUGH_GRADIENT , 
it is the higher threshold of the two passed to the Canny() edge detector 
(the lower one is twice smaller).
   param2 – Second method-specific parameter. In case of CV_HOUGH_GRADIENT , 
it is the accumulator threshold for the circle centers at the detection stage. 
The smaller it is, the more false circles may be detected.
Circles,  orresponding to the larger accumulator values, will be returned first.
So, as you can see, internally the HoughCircles function calls the Canny edge detector, 
this means that you can use a gray image in the function, instead of their contours.
Now reduce the param1 to 30 and param2 to 15 and see the results in the code that follows:
'''

## TODO: use HoughCircles to detect circles
# right now there are too many, large circles being detected
# try changing the value of maxRadius, minRadius, and minDist
def draw_small_circles(circles_im, circles):
    circles = np.uint16(np.around(circles))
    # draw each one
    for i in circles[0,:]:
       x, y, Rad = i[0],i[1],i[2]
       ''' Draw Green Circles'''
       cv2.circle(circles_im,(x,y),Rad,(0,255,0),2)
       ''' Draw the center of the circle'''
       cv2.circle(circles_im,(x,y),2,(0,0,255),3) 
       
    return  circles_im      

def imageWithCircles(image, image_Blur, minDist, param1, param2, minRadius, maxRadius):
# for drawing circles on
    
#    circles_im = np.copy(image)
    c_im = np.copy(image)
    c_tmp = np.copy(image_Blur)
    circles = cv2.HoughCircles(c_tmp, cv2.HOUGH_GRADIENT, 1, 
                           minDist=minDist,
                           param1=param1,
                           param2=param2,
                           minRadius=minRadius,
                           maxRadius=maxRadius)

    # convert circles into expected type
    circles = np.uint16(np.around(circles))
    c_im_ret = draw_small_circles(c_im, circles)
    return  c_im_ret      


f, axes = plt.subplots(3, 3, figsize=(25,15))

c_im_00 = imageWithCircles(image, image_Blur, minDist=45, param1=70, param2=11, \
                           minRadius=20, maxRadius=40) 
axes[0,0].set_title('Min.R = 20, Max.R = 40, minDist = 45')
axes[0,0].imshow(c_im_00)

print('Circles shape: ', c_im_00.shape)


circles_im = np.copy(image)
c_tmp = np.copy(image_Blur)
c_im_01 = imageWithCircles(image, image_Blur, minDist=65, param1=70, param2=11, \
                           minRadius=20, maxRadius=40) 
axes[0,1].set_title('Min.R = 20, Max.R = 40, minDist = 65')
axes[0,1].imshow(c_im_01)



circles_im = np.copy(image)
c_im_02 = imageWithCircles(circles_im, image_Blur, minDist=75, param1=70, param2=11,\
                           minRadius=20, maxRadius=40) 
axes[0,2].set_title('Min.R = 20, Max.R = 40, minDisT = 75')
axes[0,2].imshow(c_im_02)



c_im_10 = imageWithCircles(image, image_Blur, minDist=85, param1=70, param2=11, \
                           minRadius=20, maxRadius=40) 
axes[1,0].set_title('Min.R = 20, Max.R = 40, minDist = 85')
axes[1,0].imshow(c_im_10)



c_im_11 = imageWithCircles(image, image_Blur, minDist=85, param1=70, param2=11, \
                           minRadius=20, maxRadius=30) 
axes[1,1].set_title('Min.R = 20, Max.R = 30, minDist = 85')
axes[1,1].imshow(c_im_11)



c_im_12 = imageWithCircles(image, image_Blur, minDist=85, param1=70, param2=11, \
                           minRadius=25, maxRadius=30) 
axes[1,2].set_title('Min.R = 25, Max.R = 30, minDist = 85')
axes[1,2].imshow(c_im_12)



c_im_20 = imageWithCircles(image, image_Blur, minDist=65, param1=70, param2=11, \
                           minRadius=20, maxRadius=30) 
axes[2,0].set_title('Min.R = 20, Max.R = 30, minDist = 85')
axes[2,0].imshow(c_im_20)



c_im_21 = imageWithCircles(image, image_Blur, minDist=65, param1=70, param2=11, \
                           minRadius=25, maxRadius=40) 
axes[2,1].set_title('Min.R = 25, Max.R = 40, minDist = 85')
axes[2,1].imshow(c_im_21)



c_im_22 = imageWithCircles(image, image_Blur, minDist=60, param1=70, param2=11, \
                           minRadius=30, maxRadius=40) 
axes[2,2].set_title('Min.R = 30, Max.R = 40, minDist = 60')
axes[2,2].imshow(c_im_22)



