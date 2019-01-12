# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:35:29 2018

@author: Rafael Stekolshchik
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2      

image = cv2.imread('images/monarch.jpg')
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
#plt.imshow(image_copy)

#gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)  

''' Example: A.reshape(-1, 28*28)
    reshape A so that its second dimension has a size of 28*28 and
    calculate the correct size of the first dimension'''

''' separate: bg - green, prange and black of batterflay'''
''' Reshape into 2D array of pixels and 3 color values RGB'''
''' it shoul be  m x 3 dimension, where m - number of pixels, 3 - numb.of colors'''

pixel_vals = image_copy.reshape((-1,3)) 

'''convert to float'''
pixel_vals = np.float32(pixel_vals)

''' input:  m x 3 array pixel_vals'''
''' None - no labels, sto p criteria, 10 steps '''
''' previous val k = 2'''
k = 6 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            
retval, labels, centers = \
   cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
   

f, axes = plt.subplots(4, 3, figsize=(25,15))

axes[0,0].set_title('Original')
axes[0,0].imshow(image_copy)

''' convert back into 8-bit image data values '''
''' flatten : Return a copy of the array collapsed into one dimension'''
centers = np.uint8(centers)   
segmented_data = centers[labels.flatten()]  
segmented_data = segmented_data.reshape(image_copy.shape)   

print('image shape: ', image.shape, ' image len: ',len(image_copy))
print('centers: ', centers)
labels_fl = labels.flatten()
print('labels.shape: ', labels.shape, ' type: ', type(labels), ', len: ', len(labels) )
print('labels flatten,shape: ', labels_fl.shape, ' type: ', type(labels_fl), ', len: ', len(labels_fl))

axes[0,1].set_title('Segmented')
axes[0,1].imshow(segmented_data)

''' vizualize one segment only, where label == 1 '''
''' label means what claster belong every pixel : 0 ,1'''
''' if we have l clasters, then labels are 0,1,2,...k-1'''
labels_reshape = labels.reshape(image_copy.shape[0], image_copy.shape[1])
axes[0,2].set_title('Reshaped, labels = 0')
axes[0,2].imshow(labels_reshape==0, cmap='gray')

labels_reshape = labels.reshape(image_copy.shape[0], image_copy.shape[1])
axes[1,0].set_title('Reshaped, labels = 1')
axes[1,0].imshow(labels_reshape==1, cmap='gray')

labels_reshape = labels.reshape(image_copy.shape[0], image_copy.shape[1])
axes[1,1].set_title('Reshaped, labels = 2')
axes[1,1].imshow(labels_reshape==2, cmap='gray')

labels_reshape = labels.reshape(image_copy.shape[0], image_copy.shape[1])
axes[1,2].set_title('Reshaped, labels = 3')
axes[1,2].imshow(labels_reshape==3, cmap='gray')

labels_reshape = labels.reshape(image_copy.shape[0], image_copy.shape[1])
axes[2,0].set_title('Reshaped, labels = 4')
axes[2,0].imshow(labels_reshape==4, cmap='gray')

labels_reshape = labels.reshape(image_copy.shape[0], image_copy.shape[1])
axes[2,1].set_title('Reshaped, labels = 5')
axes[2,1].imshow(labels_reshape==5, cmap='gray')

''' Put black on the area with label == 1'''
''' Mask image  segment'''
masked_image = np.copy(image_copy)
masked_image[labels_reshape==0] = [0,0,0]
axes[2,2].set_title('Masked, labels = 0')
axes[2,2].imshow(masked_image)

masked_image_1 = np.copy(image_copy)
masked_image_1[labels_reshape==1] = [0,0,0]
axes[3,0].set_title('Masked, labels = 1')
axes[3,0].imshow(masked_image_1)


masked_image_2 = np.copy(image_copy)
masked_image_2[labels_reshape==2] = [0,0,0]
axes[3,1].set_title('Masked, labels = 2')
axes[3,1].imshow(masked_image_2)


masked_image_3 = np.copy(image_copy)
masked_image_3[labels_reshape==4] = [0,0,0]
axes[3,2].set_title('Masked, labels = 3')
axes[3,2].imshow(masked_image_3)
