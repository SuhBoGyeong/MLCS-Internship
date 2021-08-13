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
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool1D, Dropout
from numpy.random import seed 
import random
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


risk_factor_list = ['HE_ALP','age','sex','E_VS_TY','processed_edu','HE_hCHOL','HE_tb2','sm_presnt','DM1_pt',
'LQ_1EQL','DI1_pr','DF2_dg','HE_WBC','HE_obe','ho_incm5','DC1_lt','DE1_pt','HE_BMI','HE_HP','LQ4_00','DE1_lt',
'DX_Q_hsty','DI6_dg','T_Q_DZ1','DM2_ag','DF2_lt','DI3_pt','N_VA','dr_month','DM2_pr','BD2','HE_vitD','BE3_31',
'DM2_pt','HE_tb7','pa_walk','BE3_12','ho_incm','LQ4_05','DM1_dg','DM2_lt','BM7','BS6_2','T_Tymp_rt','LQ4_06',
'DK8_dg','BD1','E_Q_RM','DI2_dg','T_Q_SNST2','T_Prsn_rt','DJ4_pt','mh_stress','DI1_ag','DM2_dg','DK4_lt',
'HE_LHDL_st2','DI1_2','N_CAROT','DJ2_lt','DC4_lt','DM1_lt','HE_rPLS','mh_suicide','pa_mid','LQ4_03','DI1_pt',
'HE_Uglu','DI6_lt','BS2_1','HE_Upro','DI2_lt','DC11_pr','DI5_pr','HE_tb6','marri_2','DI4_pr','house','E_DL_2',
'DI4_pt','DI2_pt','LQ4_08','DJ2_pr','DI2_2','incm','HE_DM','DI3_dg','DC1_ag','DC1_dg','HE_PTH','HE_hepaB','N_ASH',
'DI6_pt','BO1_2','DC3_lt','N_NIAC','DI1_dg','HE_RBC','HE_tb1','DC11_pt','DC11_dg','marri_1','DI3_lt','HE_BUN',
'DE2_dg','DK4_dg','HE_ALC','DE1_dg','EC_stt_1','E_DR_2','E_Q_FAM','DI2_pr','DK8_pt','T_Tymp3_rt','BD2_32','BD2_31',
'DI4_lt','HE_DMdg','DC1_pr','N_VITC','incm5','DK8_lt','E_DL_1','E_VS_MYO','DM3_ag','HE_Ubil','DI3_pr','DM3_dg',
'BE3_22','DX_Q_MP','DE2_lt','HE_Bplt','HE_Uph','DK4_pt','DJ4_lt','BE3_13','BE5_1','HE_HDL_st2','E_VS_DS','DC3_ag',
'DC3_pr','DK8_ag','DC5_ag','EQ5D','DM1_pr','graduat','T_VCds','HE_HB', 'BD1_11', 'processed_incm', 
'processed_smoking', 'processed_drinking','processed_diabetes'] 

print(len(risk_factor_list))


data = np.load('./DEF_data_wo_imputation.npy', allow_pickle=True)
# print(data.shape)   #8179,875
y = np.load('./DEF_data_y_wo_imputation.npy') -1
features = np.load('./DEF_data_columns_10.npy',allow_pickle=True)
y = np.expand_dims(y, axis=1)

print(data.shape)
print(features[0])

#quit()
ind = np.where(features == 'edu')[0][0] 
processed_edu = np.expand_dims(list(map(lambda x: x-1 if x in [3,4] else x, list(data[:,ind]))), axis=1)
data = np.concatenate((data, processed_edu), axis=1)
features = np.append(features,'processed_edu')

ind1 = np.where(features == 'ainc')[0][0] 
ind2 = np.where(features == 'cfam')[0][0]
processed_incm = np.empty((data[:,ind1].shape[0],1))
for jj in range(data[:,ind1].shape[0]):
    if data[jj,ind1] == np.nan or data[jj,ind1] == np.nan:
        processed_incm[jj,0] = np.nan
    else:
        processed_incm[jj,0] = data[jj,ind1]/np.sqrt(data[jj,ind2])
data = np.concatenate((data, processed_incm), axis=1)
features = np.append(features,'processed_incm')

ind1 = np.where(features == 'BS1_1')[0][0] 
ind2 = np.where(features == 'BS3_1')[0][0]
processed_smoking = np.empty((data[:,ind1].shape[0],1))
for jj in range(data[:,ind1].shape[0]):
    if data[jj,ind1] in [1,3]:
        processed_smoking[jj,0] = 1
    elif data[jj,ind1] == 2 and data[jj,ind2] in [1,2]:
        processed_smoking[jj,0] = 2
    elif data[jj,ind1] == 2 and data[jj,ind2] == 3:
        processed_smoking[jj,0] = 3
data = np.concatenate((data, processed_smoking), axis=1)
features = np.append(features,'processed_smoking')

ind1 = np.where(features == 'sex')[0][0] 
ind2 = np.where(features == 'BD2_1')[0][0]
ind3 = np.where(features == 'BD1_11')[0][0] 
processed_drinking = np.empty((data[:,ind1].shape[0],1))
for jj in range(data[:,ind1].shape[0]):
    if data[jj,ind1] == 1 and data[jj,ind2] in [4,5] and data[jj,ind3] in [5,6]:
        processed_drinking[jj,0] = 1
    elif data[jj,ind1] == 2 and data[jj,ind2] in [3,4,5] and data[jj,ind3] in [5,6]:
        processed_drinking[jj,0] = 1
    else:
        processed_drinking[jj,0] = 0
data = np.concatenate((data, processed_drinking), axis=1)
features = np.append(features,'processed_drinking')

ind1 = np.where(features == 'HE_glu')[0][0] 
ind2 = np.where(features == 'DE1_31')[0][0]
ind3 = np.where(features == 'DE1_32')[0][0] 
ind4 = np.where(features == 'DE1_dg')[0][0] 
processed_diabetes = np.empty((data[:,ind1].shape[0],1))
for jj in range(data[:,ind1].shape[0]):
    if data[jj,ind1] >= 126 or data[jj,ind2] == 1 or data[jj,ind3] == 1 or data[jj,ind4] == 1:
        processed_diabetes[jj,0] = 1
    else:
        processed_diabetes[jj,0] = 0
data = np.concatenate((data, processed_diabetes), axis=1)
features = np.append(features,'processed_diabetes')


for ii, risk_factor in enumerate(risk_factor_list):     
    ind = np.where(features == risk_factor)[0]-1
    
    if  ii == 0:
        data2 = data[:,ind]
    else:
        data2 = np.concatenate((data2, data[:,ind]),axis=1)

data = data2


df=pd.DataFrame(data)
df.to_csv('./DEF_processed_wo_imputation.csv')

features = np.array(risk_factor_list)
#print(features)


##########Imputation##############
imputer = KNNImputer(n_neighbors=int(data.shape[0]/10), weights = 'uniform')
data=imputer.fit_transform(data)


print(data.shape)
print(features.shape)


train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

data_merge = np.concatenate((data,y.reshape((y.shape[0],1))), axis=1)
np.random.shuffle(data_merge)

x_data = data_merge[:,:-1]
y = data_merge[:,-1]

train_val = int(x_data.shape[0] * train_ratio)
val_test = int(x_data.shape[0] * (train_ratio + val_ratio))

x_train = x_data[:train_val, ...]
y_train = y[:train_val, ...]
x_val = x_data[train_val:val_test, ...]
y_val = y[train_val:val_test, ...]
x_test = x_data[val_test:, ...] 
y_test = y[val_test:, ...]

#np.save('data_x.npy',x_data)
#np.save('data_y.npy',y)




Scaler = MinMaxScaler(feature_range=(0,1))
x_train = Scaler.fit_transform(x_train)
x_val = Scaler.transform(x_val)
x_test = Scaler.transform(x_test)

'''
#standard sclaer
Scaler = StandardScaler()
x_train = Scaler.fit_transform(x_train)
x_val = Scaler.transform(x_val)
x_test = Scaler.transform(x_test)
x_train = np.clip(x_train, -5, 5)
x_val = np.clip(x_val, -5, 5)
x_test = np.clip(x_test, -5, 5)'''

np.save('./dataset/DEF_KNN_x_train',x_train)
np.save('./dataset/DEF_KNN_y_train.npy',y_train)
np.save('./dataset/DEF_KNN_x_val.npy', x_val)
np.save('./dataset/DEF_KNN_y_val.npy', y_val)
np.save('./dataset/DEF_KNN_x_test.npy', x_test)
np.save('./dataset/DEF_KNN_y_test.npy', y_test)

'''
x_data=np.concatenate((x_train,x_val,x_test), axis=0)

df=pd.DataFrame(x_data)
df.to_csv('data.csv', index=False)'''