## Define the Convolutional Neural Network
  
Notes on the notebook 2-DefineNetworkArch.ipynb.

In the function transforms.Compose order of input parameters matters: Rescale should be before RandomCrop.

In this example, the convolutional neural network (CNN) contains 4 convolutional layers, 4 maxpooling layers

and 3 fully-connected layers. Function _visualize_output_ dipalyes 10 faces with facial keyponts.

The criterion _MSELoss_ gave better results than _SmoothL1Loss_ , _L1Loss_ _SoftMarginLoss_  and _CTCLoss_.

The criterion _CrossEntropyLoss_ was not tested. 

The optimizer _ASGD_ was a little better than _Adagrad_, _Adam_, and _SGD_.

The _learnin rate_ parameter lr = 0.01 was quite good, and almost did not yield to lr = 0.001.

## Face and Facial Keypoint detection

Notes on the notebook 3-Facial Keypoint Detection, Complete Pipeline.

We use _class Net_ from _models.py_, where the architecture of CNN is defined as follows:
  
**_First convolution layer, 1 input channel, 12 output channels/feature maps, 4x4 square convolution kernel_**
  
   _#input: (224,224,1), output tensor dimension: (221,221,12) since (224 - 4)/1 + 1 = 221_
  
**_self.conv1 = nn.Conv2d(1, 12, 4)_**
       
   _#input: (221,221,12), output tensor dimension: (110, 110, 12) since 221/2 = 110, rounded down_
  
**_First maxpool layer_**
   
**_self.pool1 = nn.MaxPool2d(2, 2)_**
       
**_Second convoluation layer,  24 output channels/feature maps, 3x3 square convolution kernel_**
    
  _#input: (110,110,12), output tensor dimension: (108,108,24) since (110 - 3)/1 + 1 = 108_
  
**_self.conv2 = nn.Conv2d(12, 24, 3)_**
  
**_Second maxpool layer_**    
  
  _#input: (108,108,24), output tensor dimension: (54, 54, 24) since 108/2 = 54_
      
**_self.pool2 = nn.MaxPool2d(2, 2)_**
        
**_Third convolution layer_**

  _#input: (54,54,24), output tensor dimension: (53,53,48) since (54 - 2)/1 + 1 = 53_
        
**_self.conv3 = nn.Conv2d(24, 48, 2)_**
        
  _#input: (53,53,48), output tensor dimension: (26,26,48), since 53/2 = 26, rounded down_
        
**_Third maxpool layer_**

**self.pool3 = nn.MaxPool2d(2, 2)**

## Credit  

Most of the code is based on the Udacity code.   
