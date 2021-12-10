""" Model classes defined here! """

from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

class BestNN(torch.nn.Module):
    # take hyperparameters from the command line args!
    def __init__(self, n1_channels, n1_kernel, n2_channels, n2_kernel, pool1,
                 n3_channels, n3_kernel, n4_channels, n4_kernel, pool2, linear_features):
        super(BestNN, self).__init__()
        self.sequence1 = nn.Sequential(
            nn.Conv2d(1, n1_channels, n1_kernel), #243 by 320 image input already flattend
            nn.BatchNorm2d(n1_channels),
            nn.ReLU(),
            nn.Conv2d(n1_channels, n2_channels, n2_kernel),
            nn.BatchNorm2d(n2_channels),
            # nn.ReLU(),
            # nn.MaxPool2d(1), #get max from each channel
            # nn.Conv2d(n2_channels, n3_channels, n3_kernel), #243 by 320 image input already flattend
            # nn.BatchNorm2d(n3_channels),
            # nn.ReLU(),
            # nn.Conv2d(n3_channels, n4_channels, n4_kernel),
            # nn.BatchNorm2d(n4_channels),
            # nn.ReLU(),
            # nn.MaxPool2d(pool2), #get max from each channel
            # nn.Conv2d(n4_channels, 10, n1_kernel),
            # nn.BatchNorm2d(10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.reshape([x.size(0), 1, 243, 320]) #reshape to 2d images
        x = self.sequence1(x)
        self.sequence2 = nn.Sequential(
            nn.MaxPool2d(x.size(2))
            )
        x = self.sequence2(x)
        x = x.reshape(x.size(0), -1) #resize to 1 vector with length of 10 for each example in batch 
        return x

                    

                    



            
