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
from tensorflow.keras.layers import BatchNormalization, Activation, GaussianNoise
#from keras_layer_normalization import LayerNormalization
from numpy.random import seed 
import random
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperas import optim
from hyperas.distributions import choice, uniform
import random
from keras import regularizers
from sklearn.metrics import roc_auc_score, auc, roc_curve
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import *

np.warnings.filterwarnings('ignore')
seed_value=0
seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed=1)
random.seed(1)



x_train=np.load('./dataset/0123_x_train.npy', allow_pickle=True)
y_train=np.load('./dataset/0123_y_train.npy', allow_pickle=True)
x_val=np.load('./dataset/0123_x_val.npy', allow_pickle=True)
y_val=np.load('./dataset/0123_y_val.npy', allow_pickle=True)
x_test=np.load('./dataset/0123_x_test.npy', allow_pickle=True)
y_test=np.load('./dataset/0123_y_test.npy', allow_pickle=True)


y_train=y_train.astype(int)
y_val=y_val.astype(int)
y_test=y_test.astype(int)

f_n=x_train.shape[1]
print(f_n)

x_eval=np.concatenate((x_test, x_val), axis=0)
y_eval=np.concatenate((y_test, y_val), axis=0)
y_eval=to_categorical(y_eval)

node=[32,64,128,256,512]
#node=[4]
act=['relu']
#act=['relu']
lr=[0.005,0.009,0.03,0.07]
#lr=[0.001]
best_score=-10
drp1=[0.2,0.4,0.6]
drp2=[0.2,0.4,0.6]
#drp1=[0.1]
#drp2=[0.1]

es=[EarlyStopping(monitor='val_loss', patience=5),
    ModelCheckpoint(filepath='./models/tmp3.h5', monitor='val_loss',
                        save_best_only=True)]

itr1=len(node)
itr2=len(act)
itr3=len(drp1)
itr4=len(drp2)
itr5=len(lr)

idx=0
for i in range(itr1):
    for j in range(itr1):
        for k in range(itr2):
            for l in range(itr3):
                for m in range(itr4):
                    for n in range(itr5):
                        idx+=1
                        print('---------{}/{}-----------'.format(idx, itr1*itr1*itr2*itr3*itr4*itr5))
                        model=Sequential()
                        model.add(Dense(node[i], input_shape=(f_n,)))
                        model.add(Activation(act[k]))
                        model.add(BatchNormalization())
                        model.add(Dropout(drp1[l]))

                        model.add(Dense(node[j]))
                        model.add(Activation(act[k]))
                        model.add(BatchNormalization())
                        model.add(Dropout(drp2[m]))

                        model.add(Dense(3))
                        model.add(Activation('softmax'))

                        model.compile(loss='sparse_categorical_crossentropy',
                                    optimizer=RMSprop(lr=lr[n]),
                                    metrics=['accuracy'])

                        model.fit(x_train, y_train, batch_size=64,
                                        epochs=40, 
                                        verbose=2, validation_data=(x_val, y_val),
                                        callbacks=es)
                        score, acc=model.evaluate(x_test, y_test, verbose=0)
                        print('Test accuracy:', acc)
                        print('the mse value is : ', model.evaluate(x_test, y_test))
                        print('train accuracy: ', model.evaluate(x_train, y_train))
                        y_pred=model.predict_proba(x_eval)
                        score=roc_auc_score(y_eval,y_pred, multi_class='ovo')
                        print('auc: ', score)
                        fpr=dict()
                        tpr=dict()
                        roc_auc=dict()
                        for i in range(3):
                            fpr[i], tpr[i], _=roc_curve(y_eval[:,i], np.array(y_pred)[:,i])
                            roc_auc[i]=auc(fpr[i], tpr[i])
                            print('roc_auc: ', roc_auc[i])

                        fpr['micro'], tpr['micro'], _= roc_curve(y_eval.ravel(), np.array(y_pred).ravel())
                        roc_auc['micro']=auc(fpr['micro'], tpr['micro'])

                        all_fpr=np.unique(np.concatenate([fpr[i] for i in range(3)]))
                        mean_tpr=np.zeros_like(all_fpr)
                        for i in range(3):
                            mean_tpr+=interp(all_fpr, fpr[i], tpr[i])

                        mean_tpr/=3
                        fpr['macro']=all_fpr
                        tpr['macro']=mean_tpr
                        roc_auc['macro']=auc(fpr['macro'], tpr['macro'])
                        score2=roc_auc['micro']

                        print('current auc: ', score2)

                        if score2>=best_score:
                            best_score=score2
                            model=tf.keras.models.load_model('./models/tmp3.h5')
                            model.save('./models/model53.h5')
                        print('current best auc: ', best_score)

#y_pred=y_pred.argmax(axis=-1)
#y_test=y_test.argmax(axis=-1)
#y_test=y_test.flatten()






