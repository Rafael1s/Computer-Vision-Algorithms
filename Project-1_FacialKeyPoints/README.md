## Define the Convolutional Neural Network
  
Some notes on the notebook 2-DefineNetworkArch.ipynb

In the function transforms.Compose order of input parameters matters: Rescale should be before RandomCrop.

In this example, the convolutional neural network (CNN) contains 4 convolutional layers, 4 maxpooling layers

and 3 fully-connected layers. Function _visualize_output_ dipalyes 10 faces with facial keyponts.

The criterion _MSELoss_ gave better results than _SmoothL1Loss_ , _L1Loss_ _SoftMarginLoss_  and _CTCLoss_.

The criterion _CrossEntropyLoss_ was not tested. 

The optimizer _ASGD_ was a little better than _Adagrad_, _Adam_, and _SGD_.

The _learnin rate_ parameter lr = 0.01 was quite good, and almost did not yield to lr = 0.001.


