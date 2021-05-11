import numpy as np 
import os, glob 
import cv2 

def clahe(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2=clahe.apply(gray)
    img2=cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)

    return img2