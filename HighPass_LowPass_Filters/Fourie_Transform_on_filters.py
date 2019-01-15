# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 26 10:35:59 2018

@author: Rafael Stekolshchik
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

#%matplotlib inline

# Define gaussian, sobel, and laplacian (edge) filters
'''
Areas of white or light gray, allow that part of the frequency spectrum through! 
Areas of black mean that part of the spectrum is blocked out of the image.

Recall that the low frequencies in the frequency spectrum are 
at the center of the frequency transform image, and high frequencies are 
at the edges. You should see that the Gaussian filter allows 
only low-pass frequencies through, which is the center 
of the frequency transformed image. 
The sobel filters block out frequencies of a certain orientation 
and a laplace (all edge, regardless of orientation) filter, 
should block out low-frequencies!'''


gaussian = (1/9)*np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])

sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])

# laplacian, edge filter
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])

filters = [gaussian, sobel_x, sobel_y, laplacian]
filter_name = ['gaussian','sobel_x', \
                'sobel_y', 'laplacian']


# perform a fast fourier transform on each filter
# and create a scaled, frequency transform image
f_filters = [np.fft.fft2(x) for x in filters]
fshift = [np.fft.fftshift(y) for y in f_filters]
frequency_tx = [np.log(np.abs(z)+1) for z in fshift]

# display 4 filters
for i in range(len(filters)):
    plt.subplot(2,2,i+1),plt.imshow(frequency_tx[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

plt.show()

#-----------------------------------------

image = cv2.imread('images\city_hall.jpg')
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# perform a fast fourier transform and create a scaled, frequency transform image
def ft_image(image):
    '''This function takes in a normalized, grayscale image
       and returns a frequency spectrum transform of that image. '''
    # normalize the image
    norm_image = image/255.0
    f = np.fft.fft2(norm_image)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20*np.log(np.abs(fshift))
    
    return frequency_tx
#plt.imshow(image_copy)

# convert to grayscale
image = cv2.imread('images/city_hall.jpg')
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
f_gray = ft_image(gray)

laplacian_image = cv2.filter2D(gray, -1, laplacian)
f_image_lapl = ft_image(laplacian_image)

gaussian_image = cv2.filter2D(gray, -1, gaussian)
f_image_gauss = ft_image(gaussian_image)

f, axes = plt.subplots(2, 3, figsize=(20,10))

axes[0,0].set_title('city_hall:')
axes[0,0].imshow(gray, cmap='gray')

axes[0,1].set_title('laplasian_city_hall:')
axes[0,1].imshow(laplacian_image, cmap='gray')

axes[0,2].set_title('gaussian_city_hall:')
axes[0,2].imshow(gaussian_image, cmap='gray')

axes[1,0].set_title('FFT of city_hall:')
axes[1,0].imshow(f_gray, cmap='gray')

axes[1,1].set_title('FFT of laplasian_city_hall:')
axes[1,1].imshow(f_image_lapl, cmap='gray')

axes[1,2].set_title('FFT of gaussian_city_hall:')
axes[1,2].imshow(f_image_gauss, cmap='gray')


