from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
#from skimage import io, transform
import pandas as pd
import torchvision
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import os
import cv2
import torch

import warnings

from Data_helper import ChestDataset
from Net import Net
import Func_helper as F_helper

warnings.filterwarnings("ignore")

use_gpu = torch.cuda.is_available()

transform1 = transforms.Compose([#transforms.Resize(256),
                                # transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

chest_train = ChestDataset(root = '/home/sfchen/chest/images',
                           txt_file = '/home/sfchen/chest/label/train_val_list.txt',
                           labels_file = '/home/sfchen/chest/label/train_val.csv',
                           transform = transform1,
                           train = True)
                           
train_loader = DataLoader(chest_train, batch_size=2, shuffle=True, num_workers=4)    

resnet = models.resnet50(pretrained=False)
resnet.load_state_dict(torch.load('resnet50-19c8e357.pth'))
chest_models = Net(resnet)


optimizer = optim.SGD(chest_models.parameters(), lr=0.001, momentum=0.9)

# training process
for epoch in range(20):
    print('Epoch {}'.format(epoch))
    print('-'*10)    
    # chest_models.train(True)
    running_loss = 0.0
    running_corrects = 0
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels,_ = data
        weight = F_helper.get_weight(labels)
        if  (use_gpu):
            chest_models.cuda()
            weight = weight.cuda()
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
            
        optimizer.zero_grad()        
        outputs = chest_models(inputs)

        criterion = nn.BCEWithLogitsLoss(weight=weight)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if not(i % 200 == 199):
          print('[%d, %5d] loss: %.3f' % (epoch + 1, i +1, running_loss /200))
          running_loss =0.0          
    torch.save(chest_models.state_dict(), str(epoch)+'_norm_model.pkl')    
print('Hello World')
#save model
torch.save(chest_models.state_dict(), 'Final.pkl')



               
