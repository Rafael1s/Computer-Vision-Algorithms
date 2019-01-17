
## Visualize 4 filtered outputs of a convolutional layer

1.ConvLayerVizualization.ipynb

This notebook demonstrates a neural network with a single convolutional layer 
with four filters. Instantiate the model and set the weights

weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)

We visualize the [output of each filter](https://github.com/Rafael1s/Computer-Vision-Udacity/blob/master/FashionMNIST-CNN/output_of_each_filters.png).

## Visualize the output of a maxpooling layer in a CNN

2.PoolVizualization.ipynb

Initialize a convolutional layer so that it contains all 4 created filters.
Then add a maxpooling layer, 
[documented here](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html), 
with a kernel size of (4x4) so you can really see that the image resolution has been 
reduced after this step!

We visualize the [output of pooling layer](https://github.com/Rafael1s/Computer-Vision-Udacity/blob/master/FashionMNIST-CNN/output_of_pooling_layer.png).

## Load and transform the data

3.LoadAndVizualizFashionMNIST.ipynb

Train data, number of images:  60000.
Turning all our images into Tensor's for training a neural network.
Define a transform to read the data in as a tensor

data_transform = transforms.ToTensor()
rain_data = FashionMNIST(root='./data', train=True,
                        download=False, transform=data_transform)

Important: *Tensors are similar to numpy arrays, but can also be used
 on a GPU to accelerate computing.*


## Train a CNN to classify images from the Fashion-MNIST database

4.ClassificationFashionMNIST.ipynb

In this cell, we load in both _training_ and _test_ datasets from the 
[FashionMNIST class](https://github.com/zalandoresearch/fashion-mnist).

Define the network architecture. 

### A note on output size

For any convolutional layer, the output feature maps will have the specified depth
(a depth of 10 for 10 filters in a convolutional layer) and the dimensions 
of the produced feature maps (width/height) can be computed as the 
_input image_ width/height, W, minus the filter size, F, divided by the stride, 
S, all + 1. The equation looks like: `output_dim = (W-F)/S + 1`, 
for an assumed padding size of 0. You can find a derivation of this formula, 
[here](http://cs231n.github.io/convolutional-networks/#conv).
