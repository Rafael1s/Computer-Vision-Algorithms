# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:11:02 2018

@author: Rafael Stekolshchik
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('images/phone.jpg')
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
plt.imshow(image_copy)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap = 'gray')

print('image shapes: ', image_copy.shape[0], image_copy.shape[1])

lower = 50
upper  = 100

edges = cv2.Canny(gray, lower, upper)

print('numb.of.edges: ', len(edges))

one_edge = edges[0]
print('type.of.one.edge: ', type(one_edge))
print('len.of.edge: ', len(one_edge))


rho = 1
theta = np.pi/180.
threshold = 60
min_line_len = 100 # 50
max_line_gap = 5

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)

line_image = np.copy(image_copy)

print('num.of.lines: ', len(lines))

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0),5 )

plt.subplot(131)
plt.imshow(image_copy)
plt.title('Phone')

plt.subplot(132)
plt.imshow(edges, cmap = 'gray')
plt.title('Only edges')

plt.subplot(133)
plt.imshow(line_image)
plt.title('Edges on the Image')
