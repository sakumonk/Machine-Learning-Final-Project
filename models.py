""" Model classes defined here! """

from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(28*28, hidden_dim), #28 by 28 image input already flattend
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10), #output 10 features
            nn.ReLU()
        )

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        return self.sequence(x)

class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(1, n1_chan, n1_kern), #28 by 28 image input already flattend
            nn.ReLU(),
            nn.Conv2d(n1_chan, 10, n2_kern),
            nn.ReLU(),
            nn.MaxPool2d(14) #get max from each channel
        )

    def forward(self, x):
        x = x.reshape([x.size(0), 1, 28, 28]) #reshape to 2d images
        x = self.sequence(x)
        x = x.reshape(x.size(0), -1) #resize to 1 vector with length of 10 for each example in batch 
        return x

class BestNN(torch.nn.Module):
    # take hyperparameters from the command line args!
    def __init__(self, n1_channels, n1_kernel, n2_channels, n2_kernel, pool1,
                 n3_channels, n3_kernel, n4_channels, n4_kernel, pool2, linear_features):
        super(BestNN, self).__init__()
        self.sequence1 = nn.Sequential(
            nn.Conv2d(1, n1_channels, n1_kernel), #28 by 28 image input already flattend
            nn.BatchNorm2d(n1_channels),
            nn.ReLU(),
            nn.Conv2d(n1_channels, n2_channels, n2_kernel),
            nn.BatchNorm2d(n2_channels),
            nn.ReLU(),
            nn.MaxPool2d(1), #get max from each channel
            nn.Conv2d(n2_channels, n3_channels, n3_kernel), #28 by 28 image input already flattend
            nn.BatchNorm2d(n3_channels),
            nn.ReLU(),
            nn.Conv2d(n3_channels, n4_channels, n4_kernel),
            nn.BatchNorm2d(n4_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool2), #get max from each channel
            nn.Conv2d(n4_channels, 10, n1_kernel),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.reshape([x.size(0), 1, 28, 28]) #reshape to 2d images
        x = self.random_erase(x)

        x = self.sequence1(x)
        self.sequence2 = nn.Sequential(
            nn.MaxPool2d(x.size(2))
            )
        x = self.sequence2(x)
        x = x.reshape(x.size(0), -1) #resize to 1 vector with length of 10 for each example in batch 
        return x
    
    def random_erase(self,x): #use only for 28 * 28 grayscale images
        p = 0.5 #hyperparameter threshold for random erase
        r1 = 0.3 
        r2 = 1 / r1 #r1 and r2 are range for rectangle aspect ratio
        sl = 0.02 
        sh = 0.4 #sl and sh are range for the area range 
        stop = 1
        for i in range(0,x.size(0)):
            random = torch.rand(1).item()
            if ( random >= p):
                while(stop == 1):
                    stop = 1
                    Se = (sl + torch.rand(1).item() * (sh - sl)) * ( 28 * 28 ) #random number in range of sl to sh
                    re = r1 + torch.rand(1).item() * (r2 - r1)
                    He = sqrt(Se * re)
                    We = sqrt(Se/re)
                    xe = torch.rand(1).item() * 28
                    ye = torch.rand(1).item() * 28
                    xe = int(xe)
                    We = int(We)
                    ye = int(ye)
                    He = int(He)
                    if ( xe + We <= 27 and He + ye <= 27):
                        
                        x[i, 0, ye:(ye+He) , xe:(xe+We) ] =  (torch.rand(He, We) * 255).int()
                        stop = 0 #break from loop
            
        return x

                    

                    



            
