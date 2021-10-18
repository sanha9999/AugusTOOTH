import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from PIL import Image

'''
dataset에서 모든 data 불러오기 -> main.py에서 train test split -> 
dataloader -> train
'''

class Dataset(Dataset):
    def __init__(self, transform = None):
        super().__init__()

        self.img_dir = "이미지경로"
        self.filenames = os.listdir(self.img_dir)

        self.transform = transform

    def __getitem__(self, index):
        label = MakeLabel(self.filenames, index)
        img_path = os.path.join(self.img_dir, self.filenames[index])
        img = np.asarray_chkfinite(Image.open(img_path))

        if img.dtype == np.uint8:
            img = img / 255.0
        
        input = (img, label)

        if self.transform:
            input = self.transform(input)
        
        return input
        

    def __len__(self):
        return len(self.filenames)

    

def MakeLabel(filenames, index):
        input_img = filenames[index]
        if "O" in input_img:
            label = 1
        else : label = 0

        return label
