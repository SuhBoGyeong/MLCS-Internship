import numpy as np
import pandas as pd
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


data=np.load('./data_wo_imputation.npy')
y=np.load('./data_y.npy')
#print(data.shape)
#print(y.shape)

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

imputer = KNNImputer(n_neighbors=int(x_train.shape[0]/10), weights = 'uniform')
x_train=imputer.fit_transform(x_train)
x_val=imputer.transform(x_val)
x_test=imputer.transform(x_test)

Scaler = MinMaxScaler(feature_range=(0,1))
x_train = Scaler.fit_transform(x_train)
x_val = Scaler.transform(x_val)
x_test = Scaler.transform(x_test)

np.save('./dataset/test_x_train',x_train)
np.save('./dataset/test_y_train.npy',y_train)
np.save('./dataset/test_x_val.npy', x_val)
np.save('./dataset/test_y_val.npy', y_val)
np.save('./dataset/test_x_test.npy', x_test)
np.save('./dataset/test_y_test.npy', y_test)

x_data=np.concatenate((x_train,x_val,x_test), axis=0)

df=pd.DataFrame(x_data)
df.to_csv('test_data.csv', index=False)