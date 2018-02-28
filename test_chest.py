from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from skimage import io, transform
import pandas as pd
import torchvision
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import os
import cv2
import csv

import warnings

from Data_helper import ChestDataset
from Net import Net
import Func_helper as F_helper

warnings.filterwarnings("ignore")

use_gpu = torch.cuda.is_available()

transform1 = transforms.Compose([#transforms.Resize(256),
                                #transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                #transforms.Normalize([0.485, 0.456, 0.406],
                                #     [0.229, 0.224, 0.225])
                                ])

chest_test = ChestDataset(root='/home/sfchen/chest/images',
                           txt_file='/home/sfchen/chest/label/test_list.txt',
                           labels_file='/home/sfchen/chest/label/test_l.csv',
                           transform=transform1,
                           train=False)
test_loader = DataLoader(chest_test, batch_size=2, shuffle=False, num_workers=4)
#net model
resnet = models.resnet50(pretrained=False)
chest_models = Net(resnet)

#load model
chest_models.load_state_dict(torch.load('9_model.pkl'))

correct = 0
totel = 0

# test process
disease_class = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                 'Pneumothorax']
style = ['r-', 'g-', 'b-', 'y-', 'r--', 'g--', 'b--', 'y--']

disease_num = len(disease_class)
threshold_num = 100
threshold_value = np.linspace(start=0., stop=1.0, num=threshold_num)

totel_num = 0
accracy_array = np.zeros((threshold_num, disease_num))
precision_array = np.zeros((threshold_num, disease_num))
TP_array = np.zeros((threshold_num, disease_num))
TN_array = np.zeros((threshold_num, disease_num))
FN_array = np.zeros((threshold_num, disease_num))
FP_array = np.zeros((threshold_num, disease_num))

scoreFile = open('test_score_norm.csv', 'w')
fileHeader =  ['Image Index'] + disease_class
writer = csv.writer(scoreFile)
writer.writerow(fileHeader)

for i, data in enumerate(test_loader, 0):
    print(i)
    inputs, labels, img_name = data
    if use_gpu:
        chest_models = chest_models.cuda()
        inputs = inputs.cuda()
    outputs = chest_models(Variable(inputs))
    outputs = outputs.data
    if use_gpu:
        outputs = outputs.cpu()
    
    img_name = list(img_name)
    outputs = np.ndarray.tolist(outputs.numpy())
    
    for j in range(len(img_name)):
      item = [img_name[j]] + outputs[j]
      writer.writerow(item)
    
    #if i == 5:
    #  break
    
scoreFile.close()


  
