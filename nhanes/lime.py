import lime 
import lime.lime_tabular 

import pandas as pd 
import numpy as np 
#import lightgbm as lgb 

#for converting textual categories to integer labels
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 

import sys,os
import numpy as np 
import tensorflow as tf 
from tensorflow.keras                        import backend as K
from tensorflow.python.ops        import gen_nn_ops
from tensorflow.keras.applications.vgg16     import VGG16
from tensorflow.keras.applications.vgg19     import VGG19

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool1D, Dropout, Conv1D
from tensorflow.keras.layers import BatchNormalization, Activation
from numpy.random import seed 
import random
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler


opt1=Adam(lr=0.001)
opt2=RMSprop(lr=0.001)
opt3=SGD(lr=0.001)
#Test on class '0'
features=np.load('./0_column.npy')
f_n=len(features)


encoder=LabelEncoder()
encoder.fit(features)
feature_encoded=encoder.transform(features)
#print(risk_encoded)
#quit()

x_train=np.load('./dataset/0_x_train.npy')
x_val=np.load('./dataset/0_x_val.npy')
x_test=np.load('./dataset/0_x_test.npy')
y_train=np.load('./dataset/0_y_train.npy')
y_val=np.load('./dataset/0_y_val.npy')
y_test=np.load('./dataset/0_y_test.npy')


data_number=x_train.shape[0]
#data_number=10

'''model=Sequential()
#5891,153
print(x_train.shape)
#model.add(Conv1D(filters=64, kernel_size=3, input_shape=(153,1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))
#model.add(Flatten())
#model.add(Dense(3))'''


model=Sequential()
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt2, metrics=['acc'])
#history=model.fit(x_train,y_train, epochs=100, batch_size=30, verbose=1)
#print('the mse value is : ', model.evaluate(x_train, y_train))
history=model.fit(x_train,y_train, epochs=100, batch_size=30, verbose=1)
print('the mse value is : ', model.evaluate(x_train, y_train))    




preds=model.predict_proba(x_train,batch_size=None, verbose=1)
preds_label=model.predict_classes(x_train)

#print(preds.shape)
#print(preds)

'''i=0
for label in preds_label:
    print(preds[i], preds_label[i])
    if int(np.argmax(preds[i]))!=int(preds_label[i]):
        print('aaaaaaaaaaaaa')
    i+=1'''


explainer=lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=features,
                            class_names=['0','1','2'])
I=[]
for i in range(data_number):
    exp=explainer.explain_instance(x_train[i],model.predict_proba, num_features=f_n)

    #plt.show(exp.as_pyplot_figure())
    a=exp.as_list()
    I.append(a)
    print('#: ',i)
I=np.array(I)
#print(I.shape)
print(I)
#quit()
    
#mean=np.empty((f_n,1))
total_pos=np.zeros(f_n)
total_neg=np.zeros(f_n)
total=np.zeros(f_n)
total_abs=np.zeros(f_n)
'''for j in range(data_number):
    for k in range(f_n):
        string_data=str(I[j][k][0])
        for feature in features:
            if str(feature) in string_data:
                if float(I[j][k][1])>0:
                    total_pos[np.where(features==feature)]+=abs(float(I[j][k][1]))
                else:
                    total_neg[np.where(features==feature)]+=abs(float(I[j][k][1]))
                #total[np.where(features==feature)]+=(float(I[j][k][1]))
                
            else:
                pass'''

for j in range(data_number):
    if int(y_train[j])==0:
        for k in range(f_n):
            string_data=str(I[j][k][0])
            for feature in features:
                if str(feature) in string_data:
                    if float(I[j][k][1])>0:
                        total_pos[np.where(features==feature)]+=abs(float(I[j][k][1]))
                        total[np.where(features==feature)]+=(float(I[j][k][1]))
                        total_abs[np.where(features==feature)]+=abs(float(I[j][k][1]))
                    else:
                        total_neg[np.where(features==feature)]+=abs(float(I[j][k][1]))
                        total[np.where(features==feature)]+=(float(I[j][k][1]))
                        total_abs[np.where(features==feature)]+=abs(float(I[j][k][1]))
                    #total[np.where(features==feature)]+=(float(I[j][k][1]))
                    
                else:
                    pass


print(total_pos)
print(total_neg)
print(total)
print(total_abs)

sorted_total_pos=np.argsort(total_pos)[::-1]
sorted_total_neg=np.argsort(total_neg)[::-1]
sorted_total=np.argsort(total)[::-1]
sorted_total_abs=np.argsort(total_abs)[::-1]
print(sorted_total_pos)
print(sorted_total_neg)
print(sorted_total)
print(sorted_total_abs)
sorted_features_pos=np.empty(f_n,dtype='str')
sorted_features_neg=np.empty(f_n,dtype='str')
sorted_features=np.empty(f_n,dtype='str')
sorted_features_abs=np.empty(f_n,dtype='str')
i=0
for arg in sorted_total_pos:
    print(features[sorted_total_pos[i]])
    #sorted_features[i]=str(features[sorted_total[i]])
    i+=1
print('----------------------')
i=0
for arg in sorted_total_neg:
    print(features[sorted_total_neg[i]])
    #sorted_features[i]=str(features[sorted_total[i]])
    i+=1
print('----------------------')
i=0
for arg in sorted_total:
    print(features[sorted_total[i]])
    #sorted_features[i]=str(features[sorted_total[i]])
    i+=1
print('----------------------')   
i=0
for arg in sorted_total_abs:
    print(features[sorted_total_abs[i]])
    #sorted_features[i]=str(features[sorted_total[i]])
    i+=1