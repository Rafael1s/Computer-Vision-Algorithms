
# K-means
 
Hyperparameters of the clustering algorithm are distance function 
and the number of expected clusters k.
In [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) 
the number of clusters k must be specified ahead. 
Algorithm generates k clusters.

# Reshaping

For the OpenCV k-means function,
the 2D image array should be transformed into an MxN feature vector, where M is
the number of pixels and N is the dimension (number of channels):
 
 pixel_vals = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
 
The same can be obtained as follows:

 pixel_vals = image.reshape(-1,3) 
 
In both cases, if the shape of image is  (2000, 3008, 3) , 
the shape of pixel_vals is  (6016000, 3).

image.reshape(-1,3) means, reshape image so that its second dimension 
has a size of 3 (number of channels) and calculate the correct size of the 
first dimension.

# Examples of clusters

In the [first example](https://github.com/Rafael1s/Computer-Vision-Udacity/blob/master/K-Means-Clustering/result_kmeans.png) all clusters are demonstarated in the gray cmap, and first 4 clusters in the color map.
In the [second example](https://github.com/Rafael1s/Computer-Vision-Udacity/blob/master/K-Means-Clustering/result_masked_by_Clusters.png) different clusters are colored by different colors. In both examples the number of clusters k is specified to 6. 

