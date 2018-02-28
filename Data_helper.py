import os
import cv2
from torch.utils import data
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

class ChestDataset(data.Dataset):
    def __init__(self, root, txt_file, labels_file, transform = None, train = True):
        with open(txt_file, 'r') as f:
            data_list = f.readlines()
            data_list = [i.split('\n')[0] for i in data_list]

        self.imgs = [os.path.join(root, img) for img in data_list]
        self.labels = pd.read_csv(labels_file)
        self.transform = transform
    
    def __getitem__(self, index):
        #print(index)
        imgs_path = self.imgs[index]
        labels = self.labels.ix[index, 1:].as_matrix().astype('float32')
        data = Image.open(imgs_path).convert('RGB')
        #data = cv2.imread(imgs_path)
        #data = cv2.resize(data, (224,224))
        if self.transform:
            data = self.transform(data)
        return data, labels, imgs_path.split('/')[-1]
        
    def __len__(self):
        return len(self.imgs)
