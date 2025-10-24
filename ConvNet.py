import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing

        self.fc1 = nn.Linear(784, 100)  # First fully connected layer
        self.fc2 = nn.Linear(100, 10)

        #STEP 2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=5, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1_conv = nn.Linear(40 * 4 * 4, 100)
        self.fc2_conv = nn.Linear(100, 10)

        # STEP 4: Additional fully connected layer (640 -> 100 -> 100 -> 10)
        self.fc1_step4 = nn.Linear(40 * 4 * 4, 100)  # First FC: 640 -> 100
        self.fc2_step4 = nn.Linear(100, 100)          # Second FC: 100 -> 100 (NEW!)
        self.fc3_step4 = nn.Linear(100, 10)           # Output FC: 100 -> 10

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # returan  fcl
        # Delete line return NotImplementedError() once method is implemented.

        # Step 1: Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
        X = X.view(X.size(0), -1)  # Flatten all dimensions except batch
        
        # Step 2: Pass through first fully connected layer
        X = self.fc1(X)  # Shape: (batch_size, 100)
        
        # Step 3: Apply Sigmoid activation
        X = torch.sigmoid(X)
        
        # Step 4: Pass through output layer (no activation here - will use CrossEntropyLoss)
        X = self.fc2(X)
        return X

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # First convolutional layer
        X = self.conv1(X)           # Shape: (batch, 40, 24, 24)
        X = torch.sigmoid(X)        # Sigmoid activation
        X = self.pool(X)            # Shape: (batch, 40, 12, 12)
        
        # Second convolutional layer
        X = self.conv2(X)           # Shape: (batch, 40, 8, 8)
        X = torch.sigmoid(X)        # Sigmoid activation
        X = self.pool(X)            # Shape: (batch, 40, 4, 4)
        
        # Flatten for fully connected layer
        X = X.view(X.size(0), -1)   # Shape: (batch, 640)
        
        # Fully connected layers
        X = self.fc1_conv(X)        # Shape: (batch, 100)
        X = torch.sigmoid(X)        # Sigmoid activation
        X = self.fc2_conv(X)        # Shape: (batch, 10)
        
        return X

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        X = self.conv1(X)
        X = F.relu(X)              # ReLU instead of Sigmoid
        X = self.pool(X)
        
        # Conv layer 2
        X = self.conv2(X)
        X = F.relu(X)              # ReLU instead of Sigmoid
        X = self.pool(X)
        
        # Flatten
        X = X.view(X.size(0), -1)
        
        # Fully connected layers
        X = self.fc1_conv(X)
        X = F.relu(X)              # ReLU instead of Sigmoid
        X = self.fc2_conv(X)
        
        return X

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # Conv layer 1
        X = self.conv1(X)
        X = F.relu(X)
        X = self.pool(X)
        
        # Conv layer 2
        X = self.conv2(X)
        X = F.relu(X)
        X = self.pool(X)
        
        # Flatten
        X = X.view(X.size(0), -1)
        
        # First fully connected layer
        X = self.fc1_step4(X)       # 640 -> 100
        X = F.relu(X)
        
        # Second fully connected layer (NEW in Step 4!)
        X = self.fc2_step4(X)       # 100 -> 100
        X = F.relu(X)
        
        # Output layer
        X = self.fc3_step4(X)       # 100 -> 10
        
        return X

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()
    
    
