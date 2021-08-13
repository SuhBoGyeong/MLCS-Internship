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
from tensorflow.keras.layers import BatchNormalization, Activation, GaussianNoise, AveragePooling2D
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from numpy import *
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix


np.warnings.filterwarnings('ignore')
seed_value=0
seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed=1)
random.seed(1)


'''x_train=np.load('./dataset/0123_x_train.npy', allow_pickle=True)
y_train=np.load('./dataset/0123_y_train.npy', allow_pickle=True)
x_val=np.load('./dataset/0123_x_val.npy', allow_pickle=True)
y_val=np.load('./dataset/0123_y_val.npy', allow_pickle=True)
x_test=np.load('./dataset/0123_x_test.npy', allow_pickle=True)
y_test=np.load('./dataset/0123_y_test.npy', allow_pickle=True)'''


x_train=np.load('./dataset/age_0123_x_train.npy', allow_pickle=True)
y_train=np.load('./dataset/age_0123_y_train.npy', allow_pickle=True)
x_val=np.load('./dataset/age_0123_x_val.npy', allow_pickle=True)
y_val=np.load('./dataset/age_0123_y_val.npy', allow_pickle=True)
x_test=np.load('./dataset/age_0123_x_test.npy', allow_pickle=True)
y_test=np.load('./dataset/age_0123_y_test.npy', allow_pickle=True)


y_train=y_train.astype(int)
y_val=y_val.astype(int)
y_test=y_test.astype(int)

f_n=x_train.shape[1]
print(f_n)
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)


#class weight
'''
class_weights=class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights=dict(enumerate(class_weights))

class_weights[0]=1
class_weights[1]=1
class_weights[2]=1
print(class_weights)'''
#quit()


model=Sequential()

model.add(Dense(64, input_shape=(f_n,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

'''
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())'''



model.add(Dense(3))
model.add(Activation('softmax'))

es=[#EarlyStopping(monitor='val_acc', patience=10),
    ModelCheckpoint(filepath='./models/model54.h5', monitor='val_acc',
                        save_best_only=True)]
    #ReduceLROnPlateau(monitor='val_acc', factor=0.8, patience=10, mode='max')]

opt1=Adam(lr=0.008)
opt2=RMSprop(lr=0.001)
opt3=SGD(lr=0.008)

model.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt2,
                metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=64,
            epochs=30, 
            verbose=2, validation_data=(x_val, y_val),
            #class_weight=class_weights,
            callbacks=es)

score, acc=model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', acc)
model=tf.keras.models.load_model('./models/model54.h5')

model.summary()
x_eval=np.concatenate((x_test, x_val), axis=0)
y_eval=np.concatenate((y_test, y_val), axis=0)

print('the mse value is : ', model.evaluate(x_test, y_test))
print('train accuracy: ', model.evaluate(x_train, y_train))


#y_pred=y_pred.argmax(axis=-1)
#y_test=y_test.argmax(axis=-1)
#y_test=y_test.flatten()


y_pred=model.predict_proba(x_eval)
score=roc_auc_score(y_eval,y_pred, multi_class='ovo')
print('auc: ', score)
'''
y_test_dummies=pd.get_dummies(y_eval, drop_first=False).values
#y_test_roc=np.empty((len(y_test),3))

fpr=dict()
tpr=dict()
roc_auc=dict()
for i in range(3):
    fpr[i], tpr[i], _=roc_curve(y_test_dummies[:,i], y_pred[:,i])
    roc_auc[i]=auc(fpr[i],tpr[i])

#plot of a ROC curve for a specific class
for i in range(3):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area=%0.2f)' % roc_auc[i])
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.show()
'''


fpr=dict()
tpr=dict()
roc_auc=dict()
y_eval=to_categorical(y_eval)
y_pred=model.predict_proba(x_eval)
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

print('macro: ', roc_auc['macro'])
print('micro: ', roc_auc['micro'])

plt.figure(1)
plt.plot(fpr['micro'], tpr['micro'],
            label='micro-average ROC curve (area={0:0.2f})'.format(roc_auc['micro']))

plt.plot(fpr['macro'], tpr['macro'],
            label='macro-average ROC curve (area={0:0.2f})'.format(roc_auc['macro']))

plt.plot(fpr[0], tpr[0], label='Normal   (area={0:0.2f})'.format(roc_auc[0]))
plt.plot(fpr[1], tpr[1], label='Osteopenia   (area={0:0.2f})'.format(roc_auc[1]))
plt.plot(fpr[2], tpr[2], label='Osteoporosis    (area={0:0.2f})'.format(roc_auc[2]))

plt.legend(loc='lower right')

plt.show()




