#This data defines osteoporosis only for 'femoral neck'
import numpy as np
import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler

#For error SettingWithCopyError
pd.set_option('mode.chained_assignment',None)

years = ['2008','2009','2010', '2011']
#years = ['2008','2009']

datasets = ['all','dxa','ent','eye','ijmt', 'OE']
#datasets = ['eye','ent']
# nan_percentiles = [0.1,0.2,0.3,0.4,0.5]
nan_percentiles = [0.1]

#E_Q_FAM1 in eye: multiple response exists but many NAN
Dontknow = [9,99,999,9999,99999,999999,9999999,99999999, 999999999, 9999999999, 999.99,999.9, 99999.9]
not_applicable = [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888, 8888888888, 888.8, 888.88, 88888.8]

def osteoporosis(df):
    df = df[['ID','DX_OST_FN']]
    df.replace(" ",float("NaN"),inplace=True)
    df.dropna(inplace=True)

    ID = list(df['ID'])
    y = list(df['DX_OST_FN'].apply(pd.to_numeric))

    return ID, y #only datas (ID) with osteoporosis info. 

for nan_percentile in nan_percentiles:
    column_names = {}
    ID = {}
    df_year = {}
    y_total = []
    for year in years:
        print(year)
        df = pd.read_csv('./raw/'+year+'/hn'+ year[-2:] + '_dxa.csv') #dxa.csv: osteoporosis info
        ID[year], y = osteoporosis(df)
        y_total += y
        column_names[year] = ['ID']
        j=0
        for dataset in datasets:
            print(dataset)
            df = pd.read_csv('./raw/'+year+'/hn'+ year[-2:] + '_'+dataset+'.csv')
            df = df.round(3)
            df.replace(" ",np.NaN,inplace=True)

            if dataset == 'dxa':
                df.drop('DX_OST_FN', axis=1, inplace=True) ####
            
            columns = list(df.iloc[:,df.columns.get_loc('ID')+1:].columns) #columns starting by next to id
 
            if list(set(column_names[year]).intersection(columns))!=[]: #To drop out repeating columns of each dataset in datasets
                #print(list(set(column_names[year]).intersection(columns)))
                df.drop(list(set(column_names[year]).intersection(columns)), axis=1, inplace=True)         

            columns = list(df.iloc[:,df.columns.get_loc('ID')+1:].columns)

            if list(set(column_names[year]).intersection(columns))!=[]:
                #print(dataset)
                #print(list(set(column_names[year]).intersection(columns)))
                quit()


            for column in columns:
                try:
                    df[column] = df[column].apply(pd.to_numeric)
                except:
                    # print(column)
                    df.drop(column, axis=1, inplace=True) #dropout x numerical datas

            columns = list(df.iloc[:,df.columns.get_loc('ID')+1:].columns)

            df = df[~df['ID'].isin(list(set(list(df['ID'])) - set(ID[year])))] #To collect only datas with ost info. 
            # print(df.shape)
            # quit()

            column_names[year] += columns
            # print(df)
            # quit()

            #remove 999,888
            for column in columns:
                # print(column)
                max_val = df[column].max()
                if max_val in Dontknow:
                    dontknow_val = max_val
                    df[column][(df[column] == dontknow_val)] = None
                    df[column][(df[column] == dontknow_val/9*8)] = 0  # 8,,,-->0
                elif max_val in not_applicable:
                    not_applicable_val = max_val
                    df[column][(df[column] == not_applicable_val)] = 0


            #empty_patients
            empty_patients = (list(set(ID[year]) - set(list(df['ID']))))

            if empty_patients != []:

                nan_array = np.empty((1,df.iloc[:,df.columns.get_loc('ID'):].shape[1]),dtype=object)
                nan_array[:] = np.NaN                
                data = np.array(df.iloc[:,df.columns.get_loc('ID'):])
                # print(len(empty_patients))
                for empty_patient in sorted(empty_patients):
                    nan_array[0,0] = empty_patient
             
                    data_temp = np.append(data[:,0], str(empty_patient))
                    
                    data_temp = np.sort(data_temp)
                    ind = np.where(data_temp == empty_patient)[0][0]
                    data = np.insert(data, ind,nan_array,0)
            else:
                data = np.array(df.iloc[:,df.columns.get_loc('ID'):])

            if j == 0:
                total_data = data[:,1:]
            else:
                total_data = np.concatenate((total_data,data[:,1:]), axis=1)
            j += 1

        ID[year] = list(map(lambda x: [x], ID[year])) #next year

        total_data = np.concatenate((np.array(ID[year]), total_data), axis=1)

        df_year[year] = pd.DataFrame(total_data,index=None,columns=column_names[year])


        
    df_total = df_year['2008'].append(df_year['2009'], sort = False)
    df_total = df_total.append(df_year['2010'], sort=False)
    df_total = df_total.append(df_year['2011'], sort=False)


    df_total.to_excel('./DEF_data_wo_imputation.xlsx',sheet_name='data')
    data = np.array(df_total)[:,1:]
    np.save('./DEF_data_wo_imputation.npy',data)
    np.save('./DEF_data_y_wo_imputation.npy', y_total)
    #quit()
    np.save('./DEF_data_columns_'+ str(int(nan_percentile*100))+'.npy', df_total.columns)
    #remove column
    if nan_percentile:
        df_total = df_total.dropna(axis=1,thresh = int((len(ID['2008'])+len(ID['2009'])+len(ID['2010'])+ len(ID['2011'])) * (1-nan_percentile)))

    df_total['y'] = y_total 

    df_total = df_total.dropna(axis=0,thresh = int(len(list(df_total.columns))* (1-nan_percentile)))

    y_total = list(df_total['y'])
    df_total.drop('y', axis=1, inplace=True)

    df_total.to_excel('./data_'+ str(int(nan_percentile*100)) +'.xlsx',sheet_name='data')

    

    data = np.array(df_total)[:,1:]

    y_data = y_total
    '''
    normalized_data = MinMaxScaler().fit_transform(data)

    imputer = KNNImputer(n_neighbors=int(normalized_data.shape[0]/10), weights = 'uniform')
    imputed_data = imputer.fit_transform(normalized_data)
    np.save('./data_imputed_' +str(int(nan_percentile*100)) + '.npy',imputed_data)
    np.save('./data_y_'+ str(int(nan_percentile*100))+'.npy', y_data)'''

    # imputer2 = IterativeImputer(sample_posterior=True)
    # imputed_data2 = imputer2.fit_transform(normalized_data)
    # np.save('./data_imputed_MICE' +str(int(nan_percentile*100)) + '.npy',imputed_data2)

