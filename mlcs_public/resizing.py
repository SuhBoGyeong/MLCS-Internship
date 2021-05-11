import numpy as np
import os, glob
import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

#Those 4 constants were pretained from 116 datas for convenience. 
max_width=3138
max_height=1380
min_width=2600
min_height=1000

#resize ratio2is
r=0.3

images_path='DentalPanoramicXrays/Images'
segmentation1_path='DentalPanoramicXrays/Segmentation1'
segmentation2_path='DentalPanoramicXrays/Segmentation2'

path_to_images=os.path.join(images_path, '*.png')
path_to_segmentation1=os.path.join(segmentation1_path, '*.png')
path_to_segmentation2=os.path.join(segmentation2_path, '*.png')
images_data=glob.glob(path_to_images)
segmentation1_data=glob.glob(path_to_segmentation1)
segmentation2_data=glob.glob(path_to_segmentation2)


'''cv2.imshow('ex', sample)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#Procedure for earning the max, min width, height data
'''width=np.empty(len(images_data))
height=np.empty(len(images_data))
for idx, img in enumerate(images_data):
    img_array=cv2.imread(img,0)
    h,w=img_array.shape
    ratio=float(w/h)
    width[idx]=w
    height[idx]=h
    
print(max(width), min(width))
print(max(height), min(height))'''

array1=np.empty(((len(images_data), int(r*max_height), int(r*max_width))))
array2=np.empty(((len(segmentation1_data), int(r*max_height), int(r*max_width))))
array3=np.empty(((len(segmentation2_data), int(r*max_height), int(r*max_width))))

def resize_crop():
    for idx, img in enumerate(images_data):
        img_array=cv2.imread(img,0)
        resized_img_array=cv2.resize(img_array,(0,0),fx=r,fy=r,interpolation=cv2.INTER_AREA)
        new_min_height=int(min_height*r)
        new_min_width=int(min_width*r)
        cropped_img=resized_img_array[80:10+new_min_height,10:10+new_min_width]

        #imshow
        '''cv2.namedWindow('cropped',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('cropped',new_min_width,new_min_height)
        cv2.imshow('cropped',cropped_img)
        cv2.moveWindow('cropped',0,0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

def resize_add():
    #new_img_array=np.empty(((len(images_data), int(r*max_height), int(r*max_width))))
    for idx, img in enumerate(images_data):
        img_array=cv2.imread(img,0)
        original=img_array
        after_clahe1=clahe(img_array)
        after_clahe2=clahe2(img_array)

        img_array=clahe(img_array)


        resized_img_array=cv2.resize(img_array,(0,0),fx=r,fy=r,interpolation=cv2.INTER_AREA)
        h,w=resized_img_array.shape

        w_d=int(r*max_width-int(w)) 
        h_d=int(r*max_height-int(h))
        a=np.zeros((h_d,int(w)))
        b=np.zeros((int(r*max_height),w_d))
        c=np.concatenate((resized_img_array,a),axis=0)
        final=np.concatenate((c,b),axis=1)

        new_image=Image.fromarray(final.astype('uint8'),'L')
        final_image=np.array(new_image)
        array1[idx]=final_image

        

        '''b=Image.fromarray(array1[idx].astype('uint8'),'L')
        c=np.array(b)'''
        
        #imshow
        #To see the resized images, erase the below line. 
        final_image=np.hstack((original,after_clahe1,after_clahe2))

        cv2.namedWindow('added',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('added',int(r*max_width),int(r*max_height))
        cv2.imshow('added',final_image)
        cv2.moveWindow('added',0,0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        '''plt.subplot(1,2,1), plt.imshow(original,'gray')
        plt.subplot(1,2,2), plt.imshow(final_image,'gray')
        plt.show()
        plt.waitkey(0)
        plt.close()'''

        


'''cv2.imshow('sample',resized_img_array[0])
cv2.waitKey(0)
cv2.destroyAllWindows()'''

def clahe(img):
    '''plt.hist(gray_image.flatten(),256,[0,256],color='r')
    plt.xlim([0,256])
    plt.legend(('histogram(before)',), loc='upper left')
    plt.show()'''
    
    #gridsize=w,h
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image2=clahe.apply(img)
    '''plt.hist(image2.flatten(),256,[0,256],color='r')
    plt.xlim([0,256])
    plt.legend(('histogram(after)',), loc= '378,ailmpwupper left')
    plt.show()'''
    return image2

def clahe2(img):
    '''plt.hist(gray_image.flatten(),256,[0,256],color='r')
    plt.xlim([0,256])
    plt.legend(('histogram(before)',), loc='upper left')
    plt.show()'''

    clahe=cv2.createCLAHE(clipLimit=10, tileGridSize=(8,8))
    image2=clahe.apply(img)
    '''plt.hist(image2.flatten(),256,[0,256],color='r')
    plt.xlim([0,256])
    plt.legend(('histogram(after)',), loc= '378,ailmpwupper left')
    plt.show()'''
    return image2

#resize_crop()
resize_add()   



