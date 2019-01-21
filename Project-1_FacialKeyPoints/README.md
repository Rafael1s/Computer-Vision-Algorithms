## Define the Convolutional Neural Network
  
Some notes on the notebook 2-DefineNetworkArch.ipynb

In the function transforms.Compose order of input parameters matters: Rescale should be before RandomCrop.

In this esample, the convolutional neural network (CNN) contains 4 convolutional layers, 4 maxpooling layers

and 3 fully-connected layers. Function _visualize_output_ dipalyes 10 faces with facial keyponts.
