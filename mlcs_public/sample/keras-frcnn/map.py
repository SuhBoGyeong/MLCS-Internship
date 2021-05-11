from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers

import matplotlib.pyplot as plt 
from sklearn.metrics import average_precision_score, precision_recall_curve 
import os, glob

def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)
'''def ap_per_class(tp, conf, pred_cls, target_cls):
    #sort by objectness
    i=np.argsort(-conf)
    tp, conf, pred_cls=tp[i], conf[i], pred_cls[i]

    #find unique classes
    unique_classes=np.unique(np.concatenate((pred_cls, target_cls), 0))

    #create precision-recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls ==c) #number of ground truth objects
        n_p = sum(i) #number of predicted objects

        if (n_p== 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            #accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            #recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / n_gt + 1e-16)

            #precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            #Ap from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)

def compute_ap(recall, precision):
    #correct AP calculation
    #first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    #compute the precision envelope
    for i in range(mpre.size -1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    #to calculate area under PR curve, look for points
    #where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    #and sum(\Delta recall)*prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

   '''
name_det=[]
name_gt=[]

truths=[]
scores=[]
prob=[]

images_path='train_modified'
path_to_images=os.path.join(images_path, '*.bmp')
images_data=glob.glob(path_to_images)

#since train_modified.txt(gt_box coordinates) and bbox_test.txt(bounding box coordinates) have lists with different order,
#match each line to find whether the information belongs to the same image. 
#if found, append to detected_box, true_box info. 

for images in images_data:
    detected_box=[]
    true_box=[]
    with open('bbox_test.txt', 'r') as f:
        lines=f.readlines()
        for line in lines:
            line=line.split(',')
            if line[0]==images:
                #print(images)
                name_det.append(images)
                detected_box.append([int(line[1]), int(line[2]), int(line[3]), int(line[4])]) #x1, y1, x2, y2
                prob.append(float(line[6].split('\n')[0]))
    f.close()

    with open('train_modified.txt','r') as f2:
        lines=f2.readlines()
        for line in lines:
            line=line.split(',')
            if line[0]==images:
                name_gt.append(images)
                true_box.append([int(line[1]), int(line[2]), int(line[3]), int(line[4])])
    f2.close()

    for i in range(len(detected_box)):
        for j in range(len(true_box)):
            is_true = int(iou(detected_box[i], true_box[j])>0.5)
            #To exclude other boxes for different prosthesis(near prosthesis)
            if iou(detected_box[i], true_box[j])>0.5:
                #True positive
                truths.append(is_true)
                scores.append(prob[i])
            elif iou(detected_box[i], true_box[j])>0.1:
                #False positive
                truths.append(is_true)
                scores.append(prob[i])      
            else:
                #True negative
                pass          
    ##incomplete map calculating, because it does not include false negative case which is not detecting GT box. 
print(len(scores))
truths=np.array(truths)
scores=np.array(scores)

ap=average_precision_score(truths, scores)
print('Average Precision: {}'.format(ap))
precision, recall, _=precision_recall_curve(truths, scores)
plt.plot(recall, precision, label='({0:0.2f})'.format(ap), lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.ylim([0.7, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc='lower left')
plt.show()
#plt.savefig('prcurve.png')


