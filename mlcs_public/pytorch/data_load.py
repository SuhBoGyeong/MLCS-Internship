import numpy as np 
import pandas as pd 

import os 
from glob import glob
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms 
import torch 
from bs4 import BeautifulSoup 
import cv2 

imgs_dir=list(sorted(glob('../train/*.bmp')))
labels_dir=('train.txt')

class dataset(Dataset) :
    def __init__(self, imgs, labels):
        self.imgs=imgs 
        self.labels=labels
        self.transform=transform.Compose([transforms.ToTensor()])
        self.device=torch.device('cuda' if torch.cuda.is_available else 'cpu')

    def __len__(self) :
        return len(self.imgs)

    def __getitem__(self, index):
        x=cv2.imread(self.imgs[index])
        x=self.transform(x).to(self.device)
        y=dict()
        with open(labels_dir, 'r') as f:
            lines=f.readlines()
            print(lines)


D=dataset(imgs_dir, labels_dir)
