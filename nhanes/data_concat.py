import numpy as np
import pandas as pd
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

years = ['2005-2006','2007-2008','2009-2010','2013-2014']
#years = ['2009-2010','2013-2014']

nan_percentile=0.1

df={}
df_y={}
for year in years:
    df[year]=pd.read_csv(os.path.join('./'+year+'_wo_imputation.csv'))
    df_y[year]=pd.DataFrame(np.load(os.path.join('./'+year+'_y.npy')))

#data merge
df_total = df['2005-2006']
#df_total = df['2007-2008']
#df_total = df['2009-2010']
df_total = df_total.append(df['2007-2008'], sort = False)
df_total = df_total.append(df['2009-2010'], sort=False)
df_total = df_total.append(df['2013-2014'], sort=False)

df_total_y = df_y['2005-2006']
#df_total_y = df_y['2007-2008']
#df_total_y = df_y['2009-2010']
df_total_y = df_total_y.append(df_y['2007-2008'], sort = False)
df_total_y = df_total_y.append(df_y['2009-2010'], sort=False)
df_total_y = df_total_y.append(df_y['2013-2014'], sort=False)

patient_number=len(df_total_y)

#print(df_total) 
print(patient_number)
#print(column_number)
#quit()

print('original data: ', df_total.shape)
y_total=df_total_y.to_numpy()
df_total['y'] = y_total



#print('droping age', df_total.shape)

#exclude age 50 below, not menopause




df_total = df_total.dropna(axis=1,thresh = int(patient_number * (1-nan_percentile)))
print('After excluding nan columns: ', df_total.shape)

#y_total=df_total_y.to_numpy()
#df_total['y'] = y_total
#print(df_total['y'])

df_total = df_total.dropna(axis=0,thresh = int(len(list(df_total.columns))* (1-nan_percentile)))
print('AFter excluding nan rows: ', df_total.shape)


#nan: 0.1
#0: 2005-2006, 1: 2007-2008, 2: 2009-2010, 3: 2013-2014
#[0 1 2 3]: 22683x5434/22683x222/20842x222
#[0 1 2]: 
#[0 1 3]: 
#[0 2 3]: 
#[1 2 3]: 16607x4857/16607x274/15166x274
#[2 3]: 10090x4149/10090x287/9239x287


##############use when x age
#y_total = list(df_total['y'])
#df_total.drop('y', axis=1, inplace=True)


df_total=df_total.iloc[:,1:] #exclude unnamed column

#print(df_total.columns)

#####age
df_total=df_total[df_total['RIDAGEYR']>=50]
print('after age: ', df_total.shape)
y_total = list(df_total['y'])
df_total.drop('y', axis=1, inplace=True)
print(df_total.shape)
######age

df_total.to_csv('./age_0123_wo_imputation.csv')

np.save('./age_0123_wo_imputation.npy', df_total.to_numpy())
np.save('./age_0123_y.npy', y_total)

data=df_total.to_numpy()
y=np.load('./age_0123_y.npy')

column=(df_total.columns)
np.save('./age_0123_column.npy', column)

#df_total=df_total[df_total['RIDAGEYR']>=50]
#print('after age: ', df_total.shape)

#df_total.to_csv('./age_0123_test.csv')


#scaling and impute

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

data_merge = np.concatenate((data,y.reshape((y.shape[0],1))), axis=1)
print(data_merge.shape)

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

imputer = KNNImputer(n_neighbors=int(x_train.shape[0]/10), weights = 'uniform')
x_train=imputer.fit_transform(x_train)
x_val = imputer.transform(x_val)
x_test = imputer.transform(x_test)

Scaler = MinMaxScaler(feature_range=(0,1))
x_train = Scaler.fit_transform(x_train)
x_val = Scaler.transform(x_val)
x_test = Scaler.transform(x_test)



np.save('./dataset/age_0123_x_train.npy',x_train)
np.save('./dataset/age_0123_y_train.npy',y_train)
np.save('./dataset/age_0123_x_val.npy', x_val)
np.save('./dataset/age_0123_y_val.npy', y_val)
np.save('./dataset/age_0123_x_test.npy', x_test)
np.save('./dataset/age_0123_y_test.npy', y_test)
