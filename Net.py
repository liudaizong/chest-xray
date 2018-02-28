from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io, transform
import pandas as pd
import torchvision
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import os
import cv2

class Net(nn.Module):
    def __init__(self , model):
        super(Net, self).__init__()
        self.removed = list(model.children())[:-2]
        self.resnet_layer = nn.Sequential(*self.removed)
        #self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
                
        #self.transion_layer = nn.Conv2d(2048, 2048, kernel_size=2, stride=1) #8*8
        #self.transion_layer = nn.Conv2d(2048, 2048, kernel_size=2, stride=2) #16*16
        self.transion_layer = nn.Conv2d(2048, 2048, kernel_size=1, stride=1) #32*32
        
        self.pool_layer = nn.MaxPool2d(32)  
        self.Linear_layer = nn.Linear(2048, 8)
        
    def forward(self, x):
        x = self.resnet_layer(x)
        # x = F.relu(self.transion_layer(x)) 
        x = self.transion_layer(x)
        # print(x.data.shape)
        #x = self.pool_layer(x)
        #print(x.data.shape)     
        x = self.LSE_Pool(x)
        x = x.view(x.size(0), -1) 
        
        x = self.Linear_layer(x)        
        return x
        
    def LSE_Pool(self, input):
        batch_size = input.size()[0]
        depth = input.size()[1]
        s_size = input.size()[2]
        total_loc = input.size()[2] * input.size()[3]
        r_hyper = 10
        
        #avoid overflow/underflow
        max1, _ = torch.max(input, dim=3)  # 4*2048*32
        max_val, _ = torch.max(max1, dim=2)  #4*2048
        #max_val_input_size = max_val_input_size * torch.ones_like(input)
        
        max_stack = torch.stack([max_val] * s_size, dim=2)#4*2048*32
        max_val_input_size = torch.stack([max_stack] * s_size, dim=3)#4*2048*32*32
        input_exp = torch.exp(r_hyper * (input - max_val_input_size))
        
        sum_exp1 = torch.sum(input_exp, dim =3)#4*2048*32
        sum_exp2 = torch.sum(sum_exp1, dim=2)#4*2048
        output = max_val + (1.0 / r_hyper) * torch.log((1.0 / total_loc) * sum_exp2)
        #print(output.size())
        #output = output.view(batch_size, depth, 1,1)
        return  output
