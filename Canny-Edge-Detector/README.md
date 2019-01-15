
## Intensity Gradient and Magnitude

G_x, G_y are the intensity gradients in directions x and y,
they are calculated by one of edge detector operators 
(such as Roberts, Prewitt, or Sobel). Maghnitude G is calculates 
as sqrt(Gx\*G_x + G_y\*G_y).

## Steps of Canny Detector

The Process of Canny edge detection algorithm can be broken down to 5 different steps:

1. Apply Gaussian filter to smooth the image in order to remove the noise.
2. Find the intensity gradients of the image.
3. Apply non-maximum suppression to get rid of spurious response to edge detection.
4. Apply double threshold to determine potential edges.
5. Track edge by hysteresis: Finalize the detection of edges by suppressing. 
   all the other edges that are weak and not connected to strong edges.
   
## Hysteresis

Hysteresis is the double thresholding. Let T_min (resp. T_max) be the low (resp. high)
threshold. 
1. Any edge with magnitude G < T_min is considered as weak.
2. Any edge with magnitude G > T_max is considered as a strong edge.
3. The edge with magnitude G from the interval (T_min, T_max) will be taken
   only if it is connected with another strong edge. Thus this algorithm 
   takes into account the history of intensivity gradient filtering.   
    
The reason for hysteresis is for ripple filtering. 
In digitally generated signals, this isn't a concern, but for real-world signals, 
they may fluctuate around a level.   

## Examples

   * Canny_edge_det.py.  
   
   The [result](https://github.com/Rafael1s/Computer-Vision-Udacity/blob/master/Canny-Edge-Detector/wide_and_tight_edges.png) contains one image with wide edges, and another one with tight.
   
   * Canny_edge_det_Brain.py
   
   See the [result image](https://github.com/Rafael1s/Computer-Vision-Udacity/blob/master/Canny-Edge-Detector/brain_two_images.png). 
