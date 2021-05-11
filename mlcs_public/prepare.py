import os, glob
import numpy as np
import pandas as pd 
import sys 
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import json, csv

#image_width=3292
#image_height=1536
#(412, 192)
k=30


images_path='./sample/keras-frcnn/train'
path_to_images=os.path.join(images_path, '*.bmp')
images_data=glob.glob(path_to_images)
image_number=np.empty(261, dtype=int)
number_of_regions=np.empty(261, dtype=int)
regions_id=np.empty(261, dtype=int)
region=np.empty((261,4), dtype='int')

#cropped image saved. (background removal)
'''for images in  images_data:
    img=cv2.imread(images)
    img=img[k:192-k, k:412-k]
    name=images.split('/')[4]
    #print(name)
    cv2.imwrite('train_modified/{}'.format(name),img)'''

'''with open('detected.csv','r') as csvfile:
    reader=csv.reader(csvfile, delimiter=',', quotechar='|')
    headers=next(reader)

    for row in reader:

        #0:image number, 1: file_size, 3: region_count 4:region_id
        #5: region_shape_attributes
        image_number[i]=row[0].split('.')[0]
        number_of_regions[i]=row[3]
        regions_id[i]=row[4]

        if number_of_regions[i]>0:
            x=float(row[6].split(':')[1])
            y=float(row[7].split(':')[1])
            width=float(row[8].split(':')[1])
            height=float(row[9].split(':')[1].split('}')[0])

            x_min=x
            y_min=y
            x_max=x_min+width
            y_max=y_min+height

            #bounding box information. top-left, bottm-right
            region[i]=[x_min/8, y_min/8, x_max/8, y_max/8]
            region[i]

        else:
            region[i]=[0,0,0,0]  
        print(image_number[i], number_of_regions[i], regions_id[i], region[i])
        i+=1
'''
f=open('./train.txt','r')
lines=f.readlines()
image_number=np.empty(261,dtype=int)
region=np.empty((261,4),dtype=int)
idx=0
for line in lines:
    #mod[0]==path name
    mod=line.split('/')
    mod1=mod[1].split(',')
    mod1[1]=int(mod1[1])-k
    mod1[2]=int(mod1[2])-k
    mod1[3]=int(mod1[3])-k
    mod1[4]=int(mod1[4])-k
    region[idx]=[mod1[1],mod1[2],mod1[3],mod1[4]]
    image_number[idx]=mod[1].split('.')[0]

    idx+=1
f.close()


f= open('train_modified.csv', 'w')
writer=csv.writer(f)
for i in range(261):
    #print(image_number[i])
    new_path=os.path.join('train_modified',str(image_number[i])+'.bmp')
    writer.writerow([new_path, int(region[i][0]),int(region[i][1]),int(region[i][2]),int(region[i][3]),'prosthesis'])
    img=cv2.imread(new_path)
    cv2.rectangle(img,(region[i][0], region[i][1]), (region[i][2], region[i][3]), (0,0,255),2)
    cv2.imshow('a',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''for i in range(261):
    cv2.imread()
test=cv2.imread('./train_modified/49.bmp')
cv2.rectangle(test,(new_path[2], new_path[3]), (new_path[4], new_path[5]), (0,0,255),2)
cv2.imshow('a',test)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

