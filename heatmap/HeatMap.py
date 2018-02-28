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
import time
import csv

import warnings

warnings.filterwarnings("ignore")

use_gpu = torch.cuda.is_available()


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.removed = list(model.children())[:-2]
        self.resnet_layer = nn.Sequential(*self.removed)
        # self.resnet_layer = nn.Sequential(*list(model.children())[:-2])

        # self.transion_layer = nn.Conv2d(2048, 2048, kernel_size=2, stride=1) #8*8
        # self.transion_layer = nn.Conv2d(2048, 2048, kernel_size=2, stride=2) #16*16
        self.transion_layer = nn.Conv2d(2048, 2048, kernel_size=1, stride=1)  # 32*32

        self.pool_layer = nn.MaxPool2d(32)
        self.Linear_layer = nn.Linear(2048, 8)

    def forward(self, x):
        x = self.resnet_layer(x)
        # x = F.relu(self.transion_layer(x))
        x = self.transion_layer(x)
        activation = x
        # print(x.data.shape)
        # x = self.pool_layer(x)
        # print(x.data.shape)
        x = self.LSE_Pool(x)
        x = x.view(x.size(0), -1)

        x = self.Linear_layer(x)
        return x, activation, self.Linear_layer.weight

    def LSE_Pool(self, input):
        batch_size = input.size()[0]
        depth = input.size()[1]
        s_size = input.size()[2]
        total_loc = input.size()[2] * input.size()[3]
        r_hyper = 10

        # avoid overflow/underflow
        max1, _ = torch.max(input, dim=3)  # 4*2048*32
        max_val, _ = torch.max(max1, dim=2)  # 4*2048
        # max_val_input_size = max_val_input_size * torch.ones_like(input)

        max_stack = torch.stack([max_val] * s_size, dim=2)  # 4*2048*32
        max_val_input_size = torch.stack([max_stack] * s_size, dim=3)  # 4*2048*32*32
        input_exp = torch.exp(r_hyper * (input - max_val_input_size))

        sum_exp1 = torch.sum(input_exp, dim=3)  # 4*2048*32
        sum_exp2 = torch.sum(sum_exp1, dim=2)  # 4*2048
        output = max_val + (1.0 / r_hyper) * torch.log((1.0 / total_loc) * sum_exp2)
        # print(output.size())
        # output = output.view(batch_size, depth, 1,1)
        return output


class ChestDataset(Dataset):
    def __init__(self, root, txt_file, BBox_file, transform=None, train=True):
        with open(txt_file, 'r') as f:
            data_list = f.readlines()
            data_list = [i.split('\n')[0] for i in data_list]

        self.imgs = [os.path.join(root, img) for img in data_list]
        self.BBox = pd.read_csv(BBox_file)

        self.transform = transform

    def __getitem__(self, index):
        # print(index)
        imgs_path = self.imgs[index]
        # labels = self.labels.ix[index, 1:].as_matrix().astype('float32')
        data = cv2.imread(imgs_path)
        find_label = self.BBox.ix[index][1]
        if self.transform:
            data = self.transform(data)
        return data, imgs_path.split('/')[-1], find_label

    def __len__(self):
        return len(self.imgs)


chest_test = ChestDataset(root='/home/sfchen/chest/images',
                          txt_file='/home/sfchen/chest/code/Heatmap/heatmap.txt',
                          BBox_file='/home/sfchen/chest/code/Heatmap/BBox_List_2017.csv',
                          transform=transforms.ToTensor(),
                          train=False)

test_loader = DataLoader(chest_test, batch_size=1, shuffle=False, num_workers=1)

resnet = models.resnet50(pretrained=False)
chest_models = Net(resnet)
# load model
chest_models.load_state_dict(torch.load('/home/sfchen/chest/code/11_model.pkl'))

chest_models.eval()

# Prepare output directory for models and summaries
# =======================================================
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "figure"))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

disease_class = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia',
                 'Pneumothorax']
                 
row = []

fileHeader = ['Image Index', 'Finding Label']
f = open('map_data.csv', 'w')
writer = csv.writer(f)
for y in range(32):
  fileHeader += ['x_' + str(x) + ' y_' + str(y) for x in range(32)]
writer.writerow(fileHeader)                
for i, data in enumerate(test_loader, 0):
    print(i)
    inputs, img_name, find_label = data
    if   use_gpu:
        chest_models = chest_models.cuda()
        inputs = inputs.cuda()
    outputs, activation, weight = chest_models(Variable(inputs))
    outputs = outputs.data
    activation = activation.data
    weight = weight.data
    if   use_gpu:
        outputs = outputs.cpu()
        activation = activation.cpu()
        weight = weight.cpu()

    act = activation.numpy()

    act = act.reshape(act.shape[1], act.shape[2], act.shape[2])
    wt = weight.numpy()

    act = act.transpose(1, 0, 2).transpose(0, 2, 1)  # 32*32*2048
    wt = wt.transpose(1, 0)  # 2048*8

    map = np.dot(act, wt)  # 32*32*8



    find_label_index = disease_class.index(find_label[0])
    row_i = [img_name[0], find_label[0]]
    row_data = list(map[:,:,find_label_index].reshape(map[:,:,find_label_index].size))
    
    print(type(row)) 
    #row.append(row_i + row_data)# use too much memory
    writer.writerow(row_i + row_data)
    #if i == 2:
        #break

f.close()
