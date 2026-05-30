import torch
import torch.nn as nn
from torch.utils.data import Dataset #get rid of no such file in directly from Image.open(imageU)
import os

class ConvNeuralNet(nn.Module):


    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels = 3, out_channels=32, kernal_size = 3)
        self.conv_layer3 = nn.Conv2d(in_channels = 32, out_channels=32, kernal_size = 3)
        #create layers
        self.max_pool1 = nn.MaxPool2d(kernal_size = 2, stride = 2)

        #out goes to next in
        self.conv_layer3 = nn.Conv2d(in_channels = 32, out_channels =64, kernal_size = 3)
        self.conv_layer4 = nn.Conv2d(in_channels = 64, out_channels= 64, kernal_size = 3)
        self.max_pool2 = nn.MaxPool2d(kernal_size = 2, stride = 2)

        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU() #USE FUNCTION RELU
        self.fc2 = nn.Linear(128, num_classes) #scale to classes
    

    #actual progression of data across layers
    def forward(self, x):
        x = self.conv_layer1( x)
        x = self.conv_layer2( x)
        x=self.max_pool1(x)

        x = self.conv_layer3( x)
        x = self.conv_layer4( x)
        x=self.max_pool2(x)

        x=x.reshape(x.size(0), -1) #FLATTEN FUNCTION

        x=self.fc1(x)
        x=self.relue1(x)
        x = self.fc2(x)
        return x
