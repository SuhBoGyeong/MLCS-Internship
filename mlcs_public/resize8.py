import os, glob
import numpy as np
import pandas as pd 
import sys 
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import json, csv


images_path='train'
path_to_images=os.path.join(images_path, '*.bmp')
images_data=glob.glob(path_to_images)

for images in images_data:
    img=cv2.imread(images)
    '''name=os.path.basename(images)
    img=cv2.resize(img, dsize=(412,192),interpolation=cv2.INTER_LINEAR)
    #print(name)
    
    cv2.imwrite(name, img)'''
    print(img.shape)
