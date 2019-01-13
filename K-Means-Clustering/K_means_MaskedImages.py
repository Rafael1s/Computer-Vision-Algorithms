# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 12:06:05 2018

@author: Rafael Stekolshchik
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

#%matplotlib inline

# Read in the image
## TODO: Check out the images directory to see other images you can work with
# And select one!
image = cv2.imread('images/monarch.jpg')

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)

# Reshape image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)

# define stopping criteria
# you can change the number of max iterations for faster convergence!
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

## TODO: Select a value for k
# then perform k-means clustering
k = 6
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])

plt.imshow(segmented_image)

## TODO: Visualize one segment, try to find which is the leaves, background, etc!
plt.imshow(labels_reshape==0, cmap='gray')

# mask an image segment by cluster

f, axes = plt.subplots(3, 3, figsize=(25,15))

masked_image_00 = np.copy(image)
masked_image_00[labels_reshape == 0] = [0, 255, 0]
axes[0,0].set_title('Cluster 0 => Green')
axes[0,0].imshow(masked_image_00)

masked_image_01 = np.copy(image)
masked_image_01[labels_reshape == 1] = [0, 255, 0]
axes[0,1].set_title('Cluster 1 => Green')
axes[0,1].imshow(masked_image_01)

masked_image_02 = np.copy(image)
masked_image_02[labels_reshape == 1] = [0, 255, 255]
axes[0,2].set_title('Cluster 1 => Cyan (Blue + Green)')
axes[0,2].imshow(masked_image_02)

masked_image_10 = np.copy(image)
masked_image_10[labels_reshape == 2] = [0, 0, 255]
axes[1,0].set_title('Cluster 2 => Blue')
axes[1,0].imshow(masked_image_10)

masked_image_11 = np.copy(image)
masked_image_11[labels_reshape == 2] = [255, 0, 255]
axes[1,1].set_title('Cluster 2 => Magenta (Red + Blue)')
axes[1,1].imshow(masked_image_11)

masked_image_12 = np.copy(image)
masked_image_12[labels_reshape == 3] = [255, 0, 255]
axes[1,2].set_title('Cluster 2 => Magenta')
axes[1,2].imshow(masked_image_12)

masked_image_20 = np.copy(image)
masked_image_20[labels_reshape == 3] = [255, 0, 0]
axes[2,0].set_title('Cluster 3 => Red')
axes[2,0].imshow(masked_image_20)

masked_image_21 = np.copy(image)
masked_image_21[labels_reshape == 4] = [255, 0, 0]
axes[2,1].set_title('Cluster 4 => Red')
axes[2,1].imshow(masked_image_21)

masked_image_22 = np.copy(image)
masked_image_22[labels_reshape == 5] = [255, 255, 0]
axes[2,2].set_title('Cluster 5 => Yellow (Red + Green)')
axes[2,2].imshow(masked_image_22)
