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


#risk_factor_list = ['HE_ALP','age','sex','E_VS_TY','processed_edu','HE_hCHOL','HE_tb2','sm_presnt','DM1_pt','LQ_1EQL','DI1_pr','DF2_dg','HE_WBC','HE_obe','ho_incm5','DC1_lt','DE1_pt','HE_BMI','HE_HP','LQ4_00','DE1_lt','DX_Q_hsty','DI6_dg','T_Q_DZ1','DM2_ag','DF2_lt','DI3_pt','N_VA','dr_month','DM2_pr','BD2','HE_vitD','BE3_31','DM2_pt','HE_tb7','pa_walk','BE3_12','ho_incm','LQ4_05','DM1_dg','DM2_lt','BM7','BS6_2','T_Tymp_rt','LQ4_06','DK8_dg','BD1','E_Q_RM','DI2_dg','T_Q_SNST2','T_Prsn_rt','DJ4_pt','mh_stress','DI1_ag','DM2_dg','DK4_lt','HE_LHDL_st2','DI1_2','N_CAROT','DJ2_lt','DC4_lt','DM1_lt','HE_rPLS','mh_suicide','pa_mid','LQ4_03','DI1_pt','HE_Uglu','DI6_lt','BS2_1','HE_Upro','DI2_lt','DC11_pr','DI5_pr','HE_tb6','marri_2','DI4_pr','house','E_DL_2','DI4_pt','DI2_pt','LQ4_08','DJ2_pr','DI2_2','incm','HE_DM','DI3_dg','DC1_ag','DC1_dg','HE_PTH','HE_hepaB','N_ASH','DI6_pt','BO1_2','DC3_lt','N_NIAC','DI1_dg','HE_RBC','HE_tb1','DC11_pt','DC11_dg','marri_1','DI3_lt','HE_BUN','DE2_dg','DK4_dg','HE_ALC','DE1_dg','EC_stt_1','E_DR_2','E_Q_FAM','DI2_pr','DK8_pt','T_Tymp3_rt','BD2_32','BD2_31','DI4_lt','HE_DMdg','DC1_pr','N_VITC','incm5','DK8_lt','E_DL_1','E_VS_MYO','DM3_ag','HE_Ubil','DI3_pr','DM3_dg','BE3_22','DX_Q_MP','DE2_lt','HE_Bplt','HE_Uph','DK4_pt','DJ4_lt','BE3_13','BE5_1','HE_HDL_st2','E_VS_DS','DC3_ag','DC3_pr','DK8_ag','DC5_ag','EQ5D','DM1_pr','graduat','T_VCds','HE_HB', 'BD1_11', 'processed_incm', 'processed_smoking', 'processed_drinking','processed_diabetes'] 
f_n=153  ##same as len(risk_factor_list)

x_train=np.load('./dataset/DEF_KNN_x_train.npy', allow_pickle=True)
y_train=np.load('./dataset/DEF_KNN_y_train.npy', allow_pickle=True)
x_val=np.load('./dataset/DEF_KNN_x_val.npy', allow_pickle=True)
y_val=np.load('./dataset/DEF_KNN_y_val.npy', allow_pickle=True)
x_test=np.load('./dataset/DEF_KNN_x_test.npy', allow_pickle=True)
y_test=np.load('./dataset/DEF_KNN_y_test.npy', allow_pickle=True)

#print(x_train)
#quit()

y_train=y_train.astype(int)
y_val=y_val.astype(int)
y_test=y_test.astype(int)


#class weight
class_weights=class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights=dict(enumerate(class_weights))

class_weights[0]=1
class_weights[1]=1
class_weights[2]=1
print(class_weights)
#quit()


model=Sequential()

model.add(Dense(256, input_shape=(f_n,)))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(BatchNormalization())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
'''
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
'''


model.add(Dense(3))
model.add(Activation('softmax'))

es=[EarlyStopping(monitor='val_acc', patience=10),
    ModelCheckpoint(filepath='./models/model50.h5', monitor='val_acc',
                        save_best_only=True)]
    #ReduceLROnPlateau(monitor='val_acc', factor=0.8, patience=10, mode='max')]

opt1=Adam(lr=0.008)
opt2=RMSprop(lr=0.01)
opt3=SGD(lr=0.008)

model.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt2,
                metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=64,
            epochs=40, 
            verbose=2, validation_data=(x_val, y_val),
            class_weight=class_weights,
            callbacks=es)

score, acc=model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', acc)
model=tf.keras.models.load_model('./models/model49.h5')

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
            label='micro=average ROC curve (area={0:0.2f})'.format(roc_auc['micro']))

plt.plot(fpr['macro'], tpr['macro'],
            label='macro-average ROC curve (area={0:0.2f})'.format(roc_auc['macro']))

plt.plot(fpr[0], tpr[0], label='Normal   (area={0:0.2f})'.format(roc_auc[0]))
plt.plot(fpr[1], tpr[1], label='Osteopenia   (area={0:0.2f})'.format(roc_auc[1]))
plt.plot(fpr[2], tpr[2], label='Osteoporosis    (area={0:0.2f})'.format(roc_auc[2]))

plt.legend(loc='lower right')

plt.show()




