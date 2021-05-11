import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import glob

path='lower_jaws/*.bmp'
f = open("lower_jaws/low.txt",'w')
for datas in glob.glob(path):
    name=os.path.basename(datas)+'\n'
    f.write(name)
f.close()
