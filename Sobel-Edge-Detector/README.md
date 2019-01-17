## Sobel filter as Derivative

The Sobel filter is very commonly used in edge detection and 
in finding patterns in intensity in an image.
Applying a Sobel filter to an image is a way of taking (an approximation) 
of the derivative of the image in the x or y direction, separately. 

## Thresholds for Sobel filters

In SobelFilters_3x3_5x5.py we apply 4 filters:

  * Sobel_x 3x3
  * Sobel_y 3x3
  * Decimal filter 3x3
  * Sobel_y 5x5
 
These filters operate as it is shown in the [Sobel filters image](https://github.com/Rafael1s/Computer-Vision-Udacity/blob/master/Sobel-Edge-Detector/Sobel_filters.png).
 
In SobelFilters_with_Thresholds.py we apply  Sobel_x and Sobel_y. 
There we clearly see the pictures with different edges for [different thresholds](https://github.com/Rafael1s/Computer-Vision-Udacity/blob/master/Sobel-Edge-Detector/Sobel_X_Y_Edges.png).
 
The Sobel operator (sometimes called the Sobel-Feldman operator, or Sobel filter) 
represents a rather inaccurate approximation of the image gradient, but is still 
of sufficient quality to be of practical use in many applications.   
