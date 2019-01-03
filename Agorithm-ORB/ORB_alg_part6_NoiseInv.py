# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 21:28:13 2018

@author: Rafael Stekolshchik
"""

'''Noise Invariance
The ORB algorithm is also noise invariant. This means that it is able 
to detect objects in images, even if the images have some degree of noise.
 To see this, we will now use our Brute-Force matcher to match points 
 between the training image and a query image that has a lot of noise.'''
 
import cv2
import matplotlib.pyplot as plt

# Set the default figure size
plt.rcParams['figure.figsize'] = [14.0, 7.0]

# Load the training image
image1 = cv2.imread('./images/face.jpeg')

# Load the noisy, gray scale query image. 
image2 = cv2.imread('./images/faceRN5.png')

# Convert the query image to gray scale
query_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Convert the training image to gray scale
training_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Display the images
plt.subplot(121)
plt.imshow(training_gray, cmap = 'gray')
plt.title('Gray Scale Training Image')
plt.subplot(122)
plt.imshow(query_gray, cmap = 'gray')
plt.title('Query Image')
plt.show()

# Set the parameters of the ORB algorithm by specifying the maximum number of keypoints to locate and
# the pyramid decimation ratio
orb = cv2.ORB_create(1000, 1.3)

# Find the keypoints in the gray scale training and query images and compute their ORB descriptor.
# The None parameter is needed to indicate that we are not using a mask in either case. 
keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

# Create a Brute Force Matcher object. We set crossCheck to True so that the BFMatcher will only return consistent
# pairs. Such technique usually produces best results with minimal number of outliers when there are enough matches.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Perform the matching between the ORB descriptors of the training image and the query image
matches = bf.match(descriptors_train, descriptors_query)

# The matches with shorter distance are the ones we want. So, we sort the matches according to distance
matches = sorted(matches, key = lambda x : x.distance)

# Connect the keypoints in the training image with their best matching keypoints in the query image.
# The best matches correspond to the first elements in the sorted matches list, since they are the ones
# with the shorter distance. We draw the first 100 mathces and use flags = 2 to plot the matching keypoints
# without size or orientation.
result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:100], query_gray, flags = 2)

# we display the image
plt.title('Best Matching Points')
plt.imshow(result)
plt.show()

# Print the number of keypoints detected in the training image
print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))

# Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))

# Print total number of matching points between the training and query images
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))

'''In the above example, again we see that the number of keypoints 
detected in both images is very similar, and that even though the query image 
is has a lot of noise, our Brute-Force matcher can still match about 63% 
of the keypoints found. Also, notice that most of the matching keypoints
 are close to particular facial features, such as the eyes, nose,
 and mouth. In addition, we can see that there are a few features that
 don’t quite match up, but may have been chosen because of similar patterns 
 of intensity in that area of the image. We will also like to point out that
 in this case we used a pyramid decimation ratio of 1.3, instead of 
 the of value of 2.0 we used in the previous examples, because 
 in this particular case, it produces better results.
'''

'''Result: Number of Keypoints Detected In The Training Image:  896
Number of Keypoints Detected In The Query Image:  926

Number of Matching Keypoints Between The Training and Query Images:  557'''

