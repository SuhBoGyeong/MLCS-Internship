import numpy as np 
import os, glob 
import cv2 
import tensorflow as tf 
import math 
from tqdm import tqdm 
from PIL import Image, ImageDraw 

resized_image_width=3292/8
resized_image_height=1536/8

rpn_kernel_size=3
subsampled_ratio=8
anchor_sizes=[128,256,512]
anchor_aspect_ratio=[[1,1],[1/math.sqrt(2),math.sqrt(2)],[math.sqrt(2),1/math.sqrt(2)]]
num_anchors_in_box=len(anchor_sizes)*len(anchor_aspect_ratio)
neg_threshold=0.3
pos_threshold=0.7
amchor_sampling_amound=128

def generate_anchors(rpn_kernel_size=rpn_kernel_size, subsampled_ratio=subsampled_ratio,\
                    anchor_sizes=anchor_sizes, anchor_aspect_ratio=anchor_aspect_ratio):
    list_of_anchors=[]
    anchor_booleans=[] #To keep track of an anchor'sstatus. 

    starting_center=divmod(rpn_kernel_size, 2)[0]
    anchor_center=[starting_center-1, starting_center]
    subsampled_height=resized_image_height/subsampled_ratio
    subsampled_width=resized_image_width/subsampled_ratio

    while (anchor_center !=[subsampled_width-(1+starting_center), subsampled_height-(1+starting_center)]):
        #==!=[26,26]
        anchor_center[0]+=1 #increment x-axis
        #if sliding window reached last center, increase y-axis
        if anchor_center[0]>subsampled_width-(1+starting_center):
            anchor_center[1]+=1
            anchor_center[0]=starting_center
        #anchors are referenced to the original image. 
        #Therefore, multiply downsampling ratio to obtain input image's center
        anchor_center_on_image=[anchor_center[0]*subsampled_ratio, anchor_center[1]*subsampled_ratio]

        '''for size in anchor_sizes:
            for a_ratio in anchor_aspect_ratio:
                #[x,y,w,h]
                anchor_info=[anchor_center_on_image[0], anchor_center_on_image[1], size*a_ratio[0], size*a_ratio[1]]
                #check whether anchor crosses the boundary of the image or not
                if (anchor_info[0]-anchor_info[2]/2<0 or anchor_info[0]+anchor_info[2]/2>resized_image_width or
                    anchor_info[1]-anchor_info[3]/2<0 or anchor_info[1]+anchor_info[3]/2>resized_image_height):
                    anchor_booleans.append([0.0])
                else:
                    anchor_booleans.append([1.0])
                list_of_anchors.append(anchor_info)
    return list_of_anchors, anchor_booleans'''

generate_anchors()