import tensorflow as tf 
#import tensorflow_hub as tensorflow_hub
import matplotlib.pyplot as plt 
import tempfile 
import numpy as np 
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps 
import time 
import os, glob
from six.moves.urllib.request import urlopen 
from six import BytesIO 
from prepare import *
import pandas as pd 
from matplotlib import patches 

'''def draw_bbox(image, region,thickness=4):
    draw=ImageDraw.Draw(image)
    im_width,im_height=image.size
    xmin=region[0]
    ymin=region[1]
    xmax=region[2]
    ymax=region[3]

    (left,right,top,bottom)=(xmin*im_width, xmax*im_width,
                                ymin*im_height, ymax*im_height)
    draw.line([(left,top),(left,bottom),(right,bottom),(right,top),(left,top)],width=thickness)
    return image


idx=0
for image in images_data:
    image=cv2.imread(image,0)
    image=cv2.resize(img, dsize=(0,0), fx=1/8,fy=1/8,interpolation=cv2.INTER_LINEAR)

    for i in range (number_of_regions[idx]):
        draw_bbox(image,region[idx])
        idx+=1'''

#(images_path, xmin, ymin, xmax, ymax,class_name)
data=pd.DataFrame()
data['format']=train['image_names']


for i in range(data.shape[0]):
    data['format'][i]='train_images/'+data['format'][i]

for i in range(data.shape[0]):
    data['format'][i]=data['format'][i]+','+str(train['xmin'][i])+','+str(train['ymin'][i])+','\
        +str(train['xmax'][i])+',' str(train['ymax'][i])+','+train['cell_type'][i]

data.to_csv('train.csv',header=None,index=None,sep=' ')


