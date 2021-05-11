import numpy as np 
import os, glob
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2

images_path='train_modified'
path_to_images=os.path.join(images_path, '*.bmp')
images_data=glob.glob(path_to_images)
idx=len(images_data)
r=0
g=0
b=0
for images in images_data:
    r_tmp=[]
    g_tmp=[]
    b_tmp=[]
    name=images.split('/')[1]
    img=mpimg.imread(images)
    ##clahe
    '''gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8,8))
    img=clahe.apply(gray)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    cv2.imshow('a',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break'''
    
    #cv2.imwrite('./train/{}'.format(name),img)

    for x in img:
        for y in x:
            r_tmp.append(y[0])
            g_tmp.append(y[1])
            b_tmp.append(y[2])
        
    r_avg=sum(r_tmp)/len(r_tmp)
    g_avg=sum(g_tmp)/len(g_tmp)
    b_avg=sum(b_tmp)/len(b_tmp)

    r+=r_avg
    g+=g_avg
    b+=b_avg

r/=idx
g/=idx
b/=idx 
print(r, g, b)


