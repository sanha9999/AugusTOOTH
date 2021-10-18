import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', transform = None):
        super().__init__()

        self.img_dir = "D:\Desktop\project\AugusTOOTH\project\data\img_data"
        self.json_dir = "D:\Desktop\project\AugusTOOTH\project\data\json_data"

        self.transform = transform
        self.to_tensor = ToTensor()

        if mode == 'train':
            self.path = os.path.join(self.img_dir, 'train')
            self.json_path = os.path.join(self.json_dir, 'train')
        elif mode =='val':
            self.path = os.path.join(self.img_dir, 'validation')
            self.json_path = os.path.join(self.json_dir, 'validation')
        else:
            self.path = os.path.join(self.img_dir, 'test')
            self.json_path = os.path.join(self.json_dir, 'test')
        
        self.filenames = os.listdir(self.path)

    def __getitem__(self, idx):
        label_img = self.label_image(idx)
        img_path = os.path.join(self.path, self.filenames[idx])
        img = np.asarray_chkfinite(Image.open(img_path)) 

        if img.dtype == np.uint8:
            img = img / 255.0
        if label_img.dtype == np.uint8:
            label_img = label_img / 255.0
        
        input = {'data' : img, 'label' : label_img}
        if self.transform:
            input = self.transform(input)

        input = self.to_tensor(input)    
        return input

    
    def __len__(self):
        return len(self.filenames)

    def label_image(self, idx):
        img_path = os.path.join(self.path, self.filenames[idx])
        img = np.asarray_chkfinite(Image.open(img_path))
        with open(os.path.join(self.json_path, self.filenames[idx]), 'r') as f:
            json_data = json.load(f)
        
        img2 = img.copy()
        for i in range(0,len(json_data['annotations'])):
            point = np.array(json_data['annotations'][i]['points'])
            label_image = cv2.polylines(img, [point], True, (0, 0, 0))
            label_image = cv2.fillPoly(label_image, [point], (0, 50, 150))
        cv2.addWeighted(img, 0.5, img2, 0.5, 0, img2)    
        
        return img2
class ToTensor(object):
    def __call__(self, input):
        for key, value in input.items():
            value = value.transpose((2,0,1)).astype(np.float32)
            input[key] = torch.from_numpy(value)

        return input