# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 19:17:55 2018

@author: user
"""

## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
                
        # maxpool layer
        # pool with kernel_size=2, stride=2
        #self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 32 inputs, 20 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11
        # the output tensor will have dimensions: (20, 11, 11)
        # after another pool layer this becomes (20, 5, 5); 5.5 is rounded down
        #self.conv2 = nn.Conv2d(32, 64, 5)

        #input: (224,224,1), output tensor dimension: (221,221,32) since (224 - 4)/1 + 1 = 221
        #self.conv1 = nn.Conv2d(1, 32, 4)
        #input: (221,221,32), output tensor dimension: (110, 110, 32) since 221/2 = 110, rounded down
        #self.pool1 = nn.MaxPool2d(2, 2)
        #input: (110,110,32), output tensor dimension: (108,108,64) since (110 - 3)/1 + 1 = 108
        #self.conv2 = nn.Conv2d(32, 64, 3)
        #input: (108,108,64), output tensor dimension: (54, 54, 64) since 108/2 = 54
        #self.pool2 = nn.MaxPool2d(2, 2)
        #input: (54,54,64), output tensor dimension: (53,53,128) since (54 - 2)/1 + 1 = 53
        #self.conv3 = nn.Conv2d(64, 128, 2)
        #input: (53,53,128), output tensor dimension: (26,26,128), since 53/2 = 26, rounded down
        #self.pool3 = nn.MaxPool2d(2, 2)
        #input: (26,26,128), output tensor dimension: (26,26,256), since (26 - 1)/1 + 1 = 26
        #self.conv4 = nn.Conv2d(128, 256, 1)        
        #input: (26,26,256), output tensor dimension: (13,13,256)
        #self.pool4 = nn.MaxPool2d(2, 2)        
        #fc1_inp = 43264 # = 13*13*256 #        
        
        #fc1_inp =  86528 # = 26*26*128         
        #fc1_inp =  186624 ## 54*54*64        
        #self.fc1 = nn.Linear(fc1_inp, 10000)
        #self.fc2 = nn.Linear(10000, 136)
        
        #input: (224,224,1), output tensor dimension: (221,221,10) since (224 - 4)/1 + 1 = 221
        #self.conv1 = nn.Conv2d(1, 10, 4)

        #input: (221,221,10), output tensor dimension: (110, 110, 10) since 221/2 = 110, rounded down
        #self.pool1 = nn.MaxPool2d(2, 2)
        
        #input: (110,110,10), output tensor dimension: (108,108,20) since (110 - 3)/1 + 1 = 108
        #self.conv2 = nn.Conv2d(10, 20, 3)

        #input: (108,108,20), output tensor dimension: (54, 54, 20) since 108/2 = 54
        #self.pool2 = nn.MaxPool2d(2, 2)
        
        #input: (54,54,20), output tensor dimension: (53,53,40) since (54 - 2)/1 + 1 = 53
        #self.conv3 = nn.Conv2d(20, 40, 2)
        
        #input: (53,53,40), output tensor dimension: (26,26,40, since 53/2 = 26, rounded down
        #self.pool3 = nn.MaxPool2d(2, 2)
        
        #input: (26,26,40), output tensor dimension: (26,26,80), since (26 - 1)/1 + 1 = 26
        #self.conv4 = nn.Conv2d(40, 80, 1)        
        
        #input: (26,26,80), output tensor dimension: (13,13,80)
        #self.pool4 = nn.MaxPool2d(2, 2)        
        #fc1_inp = 13520 # = 13*13*80 #        


        #input: (224,224,1), output tensor dimension: (221,221,12) since (224 - 4)/1 + 1 = 221
        self.conv1 = nn.Conv2d(1, 12, 4)

        #input: (221,221,12), output tensor dimension: (110, 110, 12) since 221/2 = 110, rounded down
        self.pool1 = nn.MaxPool2d(2, 2)
        
        #input: (110,110,12), output tensor dimension: (108,108,24) since (110 - 3)/1 + 1 = 108
        self.conv2 = nn.Conv2d(12, 24, 3)

        #input: (108,108,24), output tensor dimension: (54, 54, 24) since 108/2 = 54
        self.pool2 = nn.MaxPool2d(2, 2)
        
        #input: (54,54,24), output tensor dimension: (53,53,48) since (54 - 2)/1 + 1 = 53
        self.conv3 = nn.Conv2d(24, 48, 2)
        
        #input: (53,53,48), output tensor dimension: (26,26,48), since 53/2 = 26, rounded down
        self.pool3 = nn.MaxPool2d(2, 2)
        
        #input: (26,26,48), output tensor dimension: (26,26,96), since (26 - 1)/1 + 1 = 26
        self.conv4 = nn.Conv2d(48, 96, 1)        
        
        #input: (26,26,96), output tensor dimension: (13,13,96)
        self.pool4 = nn.MaxPool2d(2, 2)        
        fc1_inp = 16224 # = 13*13*96 #        
           
        self.fc1 = nn.Linear(fc1_inp, 10000)
        self.fc2 = nn.Linear(10000, 1500)
        self.fc3 = nn.Linear(1500, 136)

        # Accelerating Deep Network Training
        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(96)

        #self.bn1 = nn.BatchNorm2d(10)
        #self.bn2 = nn.BatchNorm2d(20)
        #self.bn3 = nn.BatchNorm2d(40)
        #self.bn4 = nn.BatchNorm2d(80)

        #self.bn1 = nn.BatchNorm2d(32)
        #self.bn2 = nn.BatchNorm2d(64)
        #self.bn3 = nn.BatchNorm2d(128)
        #self.bn4 = nn.BatchNorm2d(256)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
                

        # Flatten and continue with dense layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.dropout(self.pool2(F.relu(self.conv2(x))), p=0.1)
        x = F.dropout(self.pool3(F.relu(self.conv3(x))), p=0.2)
        x = F.dropout(self.pool4(F.relu(self.conv4(x))), p=0.3)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        x = F.dropout(x, p=0.2)
        x = self.fc3(x)
        

        # a modified x, having gone through all the layers of your model, should be returned
        return x
