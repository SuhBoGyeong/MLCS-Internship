import numpy as np
import pandas as pd
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

years = ['2009-2010','2011-2012','2013-2014']
# years = ['2013-2014']
# number of patient 10537   9756   10173  
#     51624-62160    62161-71916   73557-83729   

datasets = ['demographics','dietary','examination','laboratory','questionnarire']
# datasets = ['examination']

# nan_percentiles = [0.1,0.2,0.3,0.4,0.5]
nan_percentile = 0.1  # None,0.1,0.2,0.3
#                           574  730  1926
teeth_removal_num = 1

except_list = ['DR1IFF_H', 'DR2IFF_H', 'DS1IDS_H', 'DS2IDS_H', 'DSQIDS_H', 'SSHPV_H', 'RXQ_RX_H','DR1IFF_G', 'DR2IFF_G', 'DS1IDS_G', 'DS2IDS_G', 'DSQIDS_G', 'SSHPV_G', 'RXQ_RX_G','DR1IFF_F', 'DR2IFF_F', 'DS1IDS_F', 'DS2IDS_F', 'DSQIDS_F', 'SSHPV_F', 'RXQ_RX_F','DTQ_F']

except_column_list = ['DXXLABEL','OHX02CSC', 'OHX03CSC','OHX04CSC','OHX05CSC','OHX06CSC','OHX07CSC','OHX08CSC','OHX09CSC','OHX10CSC','OHX11CSC','OHX12CSC','OHX13CSC','OHX14CSC','OHX15CSC','OHX16CSC','OHX17CSC','OHX18CSC','OHX19CSC','OHX20CSC','OHX21CSC','OHX22CSC','OHX23CSC','OHX24CSC','OHX25CSC','OHX26CSC','OHX27CSC','OHX28CSC','OHX29CSC','OHX30CSC','OHX31CSC', 'OHX02SE','OHX03SE','OHX04SE','OHX05SE','OHX07SE','OHX10SE','OHX12SE','OHX13SE','OHX14SE','OHX15SE','OHX18SE','OHX19SE','OHX20SE','OHX21SE','OHX28SE','OHX29SE','OHX30SE','OHX31SE','SMDUPCA','SMD100BR','SSTSUM']


Dontknow = [9,99,999,9999,99999,999999,9999999,99999999]

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def periodontitis(df, teeth_removal_num):

    df = df[df['OHDEXSTS'] == 1] 
    df = df[df['OHDEXCLU'] == 2] 

    del_col=['OHDEXSTS', 'OHDPDSTS', 'OHDEXCLU']
    df.drop(del_col, axis=1, inplace=True)
    df = df.iloc[:,1:]
    indices_99 = (df == 99)
    df[indices_99] = None

    # check num_teeth
    data = np.array(df.iloc[:,1:])
    row_remove = []
    for patient in range(data.shape[0]):

        patient_data = data[patient,:]
        
        status_matrix = np.empty([28,5])    #teeth_ind * (num of al3,al4,al6,pd4,pd5)

        num_teeth = 0

        for teeth_ind in range(28):
            teeth_data = patient_data[18*(teeth_ind):18*(teeth_ind+1)]          
            # teeth_data = np.array(map(lambda x: x if x != 99 else 0, teeth_data))
            red_box = teeth_data[:6]
            blue_box_pd = teeth_data[6:12]
            green_box_al = teeth_data[12:18]
            # if np.sum(np.isnan(blue_box_pd)) != 6 or np.sum(np.isnan(blue_box_pd)) !=6:
            #     num_teeth += 1

            if np.sum(np.isnan(blue_box_pd)) == 0 and np.sum(np.isnan(blue_box_pd)) == 0:
                num_teeth += 1

        if num_teeth == 0:
            row_remove.append(patient)
        if teeth_removal_num == 1:
            if num_teeth == 1:
                row_remove.append(patient)

    df.drop(df.index[row_remove], inplace = True)
    patient_numbers=list(df['SEQN'])


    y = [] # no:0, mi:1, mo:2, se:3 
    num_y_0 = 0 
    num_y_1 = 1 
    num_y_2 = 2 
    num_y_3 = 3 

    data = np.array(df.iloc[:,1:])

    for patient in range(data.shape[0]):
        patient_data = data[patient,:]
        
        status_matrix = np.empty([28,5])    #teeth_ind * (num of al3,al4,al6,pd4,pd5)

        for teeth_ind in range(28):
            teeth_data = patient_data[18*(teeth_ind):18*(teeth_ind+1)]
            # teeth_data = np.array(map(lambda x: x if x != 99 else 0, teeth_data))

            red_box = teeth_data[:6]
            blue_box_pd = teeth_data[6:12]
            green_box_al = teeth_data[12:18]
            # print(green_box_al)
            # print(sum(green_box_al>= 3))
            # quit()

            status_matrix[teeth_ind, 0] = sum(green_box_al>= 3)
            status_matrix[teeth_ind, 1] = sum(green_box_al>= 4)
            status_matrix[teeth_ind, 2] = sum(green_box_al>= 6)
            status_matrix[teeth_ind, 3] = sum(blue_box_pd>= 4)
            status_matrix[teeth_ind, 4] = sum(blue_box_pd>= 5)

        if sum(status_matrix[:,2] >= 1) >= 2 and sum(status_matrix[:,4] >= 1) >= 1:
            y.append(3)
            num_y_3 += 1
        elif sum(status_matrix[:,1] >= 1) >= 2 or sum(status_matrix[:,4] >= 1) >= 2:
            y.append(2)
            num_y_2 += 1
        elif ((sum(status_matrix[:,0] >= 2) >= 1 or sum(status_matrix[:,0] >= 1) >= 2) and sum(status_matrix[:,3] >= 1) >= 2) or sum(status_matrix[:,4] >= 1) >= 1:
            y.append(1)
            num_y_1 += 1
        else:
            y.append(0)
            num_y_0 += 1
    # print(num_y_0)
    # print(num_y_1)
    # print(num_y_2)
    # print(num_y_3)

    return patient_numbers, y

def connected_variable(df, var1, var1_val, var2, var2_val):
    a = df[var1] == var1_val
    df.loc[a, var2] = var2_val

    return df


column_names = {}
patient_numbers = {}
df_year = {}
y_total = []
for year in years:
    print(year)
    if year == '2009-2010':
        char = 'F'
    elif year == '2011-2012':
        char = 'G'
    elif year == '2013-2014':
        char = 'H'
        
    df = pd.read_csv('./'+year+'/examination/OHXPER_' + char +'.csv')


    patient_numbers[year], y = periodontitis(df,teeth_removal_num)
    y_total += y
    column_names[year] = [u'SEQN']
    j=0
    for dataset in datasets:
        print(dataset)
        xpts = sorted(os.listdir('./'+year+'/'+dataset))
        xpts = [f for f in xpts if 'XPT' in f]

        for xpt in xpts:
            print(dataset, xpt)
            df = pd.read_csv('./'+year+'/'+dataset+'/'+xpt.split('.')[0]+'.csv')
            df = df.round(3)

            if list(set(list(df.columns)).intersection(except_column_list)) != []:
                df.drop(list(set(list(df.columns)).intersection(except_column_list)), axis=1, inplace=True)

            if xpt.split('.')[0] == 'OHXDEN_'+char and year != '2009-2010': 

                df = df.replace({"b'D'":1, "b'E'": 2, "b'J'": 3, "b'K'":4, "b'M'":5, "b'P'":6, "b'Q'":7, "b'R'":8, "b'S'":9, "b'T'":10, "b'U'":11, "b'X'":12, "b'Y'":13, "b'Z'":14, "b''": np.NaN})

            elif xpt.split('.')[0] == 'CSX_'+char:
                df = df.replace({"b'A'": 1, "b'B'": 2,"b''": np.NaN})

            elif xpt.split('.')[0] == 'VID_'+char:
                new_column = df['LBXVIDMS'] + df['LBXVE3MS']
                df.drop(['LBXVIDMS','LBXVE3MS'],axis=1, inplace=True)
                df['LBXVIDMS_LBXVE3MS'] = new_column

            elif xpt.split('.')[0] == 'ALQ_'+char:
                conditions = [
                    (df['ALQ120Q'] == 0), 
                    (df['ALQ120Q'] != 0) & (df['ALQ120U'] == 1),
                    (df['ALQ120Q'] != 0) & (df['ALQ120U'] == 2),
                    (df['ALQ120Q'] != 0) & (df['ALQ120U'] == 3)
                ]
                choices = [0, df['ALQ120Q']*52*df['ALQ130'], df['ALQ120Q']*12*df['ALQ130'], df['ALQ120Q']*1*df['ALQ130']]

                df['ALQ120Q_ALQ120U_ALQ130'] = np.select(conditions, choices, default = np.nan)

                df.drop(['ALQ120Q','ALQ130', 'ALQ120U'],axis=1, inplace=True)

                df = connected_variable(df,'ALQ101',1,'ALQ110',1)

            elif xpt.split('.')[0] == 'RXQASA_'+char:
                conditions = [
                    (df['RXQ515'].isna()), 
                    (df['RXQ520'].isna())
                ]
                choices = [df['RXQ520'], df['RXQ515']]

                df['RXQ515_RXQ520'] = np.select(conditions, choices, default = np.nan)
                
                df.drop(['RXQ515','RXQ520'],axis=1, inplace=True)

            elif xpt.split('.')[0] == 'SPX_'+char and year != '2013-2014': 

                df = df.replace({"b'A'":1, "b'B'": 2, "b'C'": 3, "b'D'":4, "b'F'":5, "b''": np.NaN})
            
            elif xpt.split('.')[0] == 'CBC_'+char or xpt.split('.')[0] == 'COT_'+char:
        
                df.rename({'LBXHCT': 'LBXHCT_'+xpt.split('_')[0]}, axis=1, inplace=True)
            
            elif xpt.split('.')[0] == 'DR1TOT_'+char:
                df = connected_variable(df,'DBQ095Z',4,'DBD100',0)
                
            elif xpt.split('.')[0] == 'BPQ_'+char:
                df = connected_variable(df,'BPQ020', 2, ['BPQ030', 'BPD035', 'BPQ040A'], [2,100,2])
                df = connected_variable(df,'BPQ056', 2, 'BPD058', 0)

            elif xpt.split('.')[0] == 'DIQ_'+char:
                df = connected_variable(df,'DIQ010', 2, 'DIQ040', 100)
                df = connected_variable(df,'DIQ050', 2, 'DID060', 0)
                df = df.replace({666: 0})

                conditions = [
                    (df['DID060'] == 0), 
                    (df['DID060'] != 0) & (df['DIQ060U'] == 1),
                    (df['DID060'] != 0) & (df['DIQ060U'] == 2)
                ]
                choices = [0, df['DID060']*12, df['DID060']]

                df['DID060_DIQ060U'] = np.select(conditions, choices, default = np.nan)

                df.drop(['DID060','DIQ060U'],axis=1, inplace=True)

            elif xpt.split('.')[0] == 'DBQ_'+char:
                df = connected_variable(df,'DBQ197', 0, ['DBQ223A', 'DBQ223B', 'DBQ223C', 'DBQ223D', 'DBQ223E', 'DBQ223U'], 0)
                df = connected_variable(df,'DBQ229', 2, ['DBQ235A', 'DBQ235B', 'DBQ235C'], 0)
                
            elif xpt.split('.')[0] == 'DUQ_'+char:
                df = connected_variable(df,'DUQ200', 2, ['DUQ210', 'DUQ211', 'DUQ213', 'DUQ217', 'DUQ219', 'DUQ230'], [100,2,100, 0, 0, 0])
                df = connected_variable(df,'DUQ240', 2, ['DUQ250', 'DUQ300', 'DUQ320', 'DUQ330', 'DUQ340', 'DUQ352', 'DUQ360'], [2,100, 0,2,100, 0, 0])
                df = connected_variable(df,'DUQ250', 2, ['DUQ260', 'DUQ272', 'DUQ280'], [100, 0, 0])
                df = connected_variable(df,'DUQ370', 2, ['DUQ380A', 'DUQ380B', 'DUQ380C', 'DUQ380D', 'DUQ380E', 'DUQ390','DUQ410','DUQ420'], [0, 0, 0,0,0,100,0,0])

            elif xpt.split('.')[0] == 'ECQ_'+char:
                df = connected_variable(df,'MCQ080E', 2, 'ECQ150', 2)

            elif xpt.split('.')[0] == 'FSQ_'+char:
                df = connected_variable(df,'FSD041', 2, 'FSD052', 4)
                df = connected_variable(df,'FSQ165', 2, 'FSQ171', 2)
            
            elif xpt.split('.')[0] == 'HIQ_'+char:
                df = connected_variable(df,'HIQ011', 2, ['HIQ031A','HIQ031B','HIQ031C','HIQ031D','HIQ031E','HIQ031F','HIQ031H','HIQ031I','HIQ031J'], [0,0,0,0,0,0,0,0,0])

            elif xpt.split('.')[0] == 'HUQ_'+char:
                df = connected_variable(df,'HUQ071', 2, 'HUD080', 0)

            elif xpt.split('.')[0] == 'IMQ_'+char:
                df = connected_variable(df,'IMQ040', 2, 'IMQ045', 0)
            
            elif xpt.split('.')[0] == 'INQ_'+char:
                df = connected_variable(df,'INQ244', 1, 'INQ247', 7)

            elif xpt.split('.')[0] == 'KIQ_'+char:
                df = connected_variable(df,'KIQ022', 2, 'KIQ025', 2)
                df = connected_variable(df,'KIQ026', 2, 'KIQ028', 0)
                df = connected_variable(df,'KIQ005', 1, 'KIQ010', 0)
                df = connected_variable(df,'KIQ042', 2, 'KIQ430', 0)
                df = connected_variable(df,'KIQ044', 2, 'KIQ450', 0)
                df = connected_variable(df,'KIQ046', 2, 'KIQ470', 0)

            elif xpt.split('.')[0] == 'MCQ_'+char:
                df = connected_variable(df,'MCQ010', 2, ['MCQ025', 'MCQ035','MCQ040','MCQ050'], [100,2,2,2])
                df = connected_variable(df,'MCQ160M', 2, 'MCQ170M', 2)
                df = connected_variable(df,'MCQ160K', 2, 'MCQ170K', 2)
                df = connected_variable(df,'MCQ160L', 2, 'MCQ170L', 2)

            elif xpt.split('.')[0] == 'OCQ_'+char and year != '2013-2014':
                df = connected_variable(df,'OCQ510',2 , 'OCQ520',0 )  
                df = connected_variable(df,'OCQ530',2 , 'OCQ540',0 )  
                df = connected_variable(df,'OCQ550',2 , 'OCQ560',0 )  
                df = connected_variable(df,'OCQ570',2 , 'OCQ580',0 ) 

            elif xpt.split('.')[0] == 'PAQ_'+char:
                df = connected_variable(df,'PAQ605', 2, ['PAQ610','PAD615'], [0, 0])
                df = connected_variable(df,'PAQ620', 2, ['PAQ625','PAD630'], [0, 0])
                df = connected_variable(df,'PAQ635', 2, ['PAQ640','PAD645'], [0, 0])
                df = connected_variable(df,'PAQ650', 2, ['PAQ655','PAD660'], [0, 0])
                df = connected_variable(df,'PAQ665', 2, ['PAQ670','PAD675'], [0, 0])

            elif xpt.split('.')[0] == 'SMQ_'+char:
                df = connected_variable(df,'SMQ020', 2, ['SMD030','SMQ040'], [0, 3] ) 

                conditions = [
                    (df['SMQ050Q'] == 0), 
                    (df['SMQ050Q'] != 0) & (df['SMQ050U'] == 1),
                    (df['SMQ050Q'] != 0) & (df['SMQ050U'] == 2),
                    (df['SMQ050Q'] != 0) & (df['SMQ050U'] == 3),
                    (df['SMQ050Q'] != 0) & (df['SMQ050U'] == 4)
                ]
                choices = [0, df['SMQ050Q'], df['SMQ050Q']*7, df['SMQ050Q']*30, df['SMQ050Q']*365]

                df['SMQ050Q_SMQ050U'] = np.select(conditions, choices, default = np.nan)

                df.drop(['SMQ050Q','SMQ050U'],axis=1, inplace=True)

            elif xpt.split('.')[0] == 'SMQRTU_'+char and year != '2013-2014':
                df = connected_variable(df,'SMQ680', 2, ['SMQ690A','SMQ690B','SMQ690C','SMQ690D','SMQ690E','SMQ690F', 'SMQ710','SMQ720','SMQ725', 'SMQ740', 'SMQ750', 'SMQ755', 'SMQ770', 'SMQ780', 'SMQ785', 'SMQ800','SMQ815', 'SMQ817', 'SMQ819'], [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 5,0, 5, 0, 5])
     
            if xpt.split('.')[0] in except_list or not u'SEQN' in df.columns:
                continue
    
            elif df.astype('object').isin(["b''"]).sum().sum() != 0:
                print(xpt)
                quit()

            else:
                if 'OHXPER' in xpt:
                    continue
                else:
                    columns = list(df.iloc[:,df.columns.get_loc(u'SEQN')+1:].columns)

                    if list(set(column_names[year]).intersection(columns))!=[]:
                        if dataset == 'dietary':
                            df.drop(list(set(column_names[year]).intersection(columns)), axis=1, inplace=True)
                            columns = list(df.iloc[:,df.columns.get_loc(u'SEQN')+1:].columns)

                        for jj, col in enumerate(list(set(column_names[year]).intersection(columns))):
                            dup_cols = [s for s in column_names[year] if col in s]
                            dup_cols_nums = sorted([x.split('_')[-1] for x in dup_cols if RepresentsInt(x.split('_')[-1])],key=int)
                            # print(dup_cols)
                            # print(dup_cols_nums)
                            if dup_cols_nums !=[]:
                                df.rename({col: col+'_'+str(int(dup_cols_nums[-1])+1)}, axis=1, inplace=True)
                        
                            else:
                                df.rename({col: col+'_2'}, axis=1, inplace=True)

                        columns = list(df.iloc[:,df.columns.get_loc(u'SEQN')+1:].columns)
                    if list(set(column_names[year]).intersection(columns))!=[]:
                        print(xpt)
                        print(list(set(column_names[year]).intersection(columns)))

                    # print(list(map(int, list(set(list(df[u'SEQN'])) - set(patient_numbers)))))
                    # quit()


                    df = df[~df[u'SEQN'].isin(list(map(int, list(set(list(df[u'SEQN'])) - set(patient_numbers[year])))))]
                    # print(df.shape)
                    # quit()


                    column_names[year] += columns
                    # print(df)
                    # quit()
                    #remove 999,777
                    for column in columns:
                        max_val = df[column].max()
                        if max_val in Dontknow:
                            dontknow_val = max_val
                            refused_val = max_val/9*7
                            df[column][(df[column] == dontknow_val)] = None
                            df[column][(df[column] == refused_val)] = None
                    
                    #empty_patients
                    empty_patients = (list(set(patient_numbers[year]) - set(list(df[u'SEQN']))))
                    
                    if empty_patients != []:

                        nan_array = np.empty((1,df.iloc[:,df.columns.get_loc(u'SEQN'):].shape[1]))
                        nan_array[:] = np.nan
                        
                        data = np.array(df.iloc[:,df.columns.get_loc(u'SEQN'):])
                        # inds = []
                        for empty_patient in sorted(empty_patients):
                            # nan_array[0,0] = empty_patient
                            # ind = data[data[:,0]< empty_patient].shape[0]
                            # data = np.insert(data, ind,nan_array,0)


                            nan_array[0,0] = empty_patient
                    
                            data_temp = np.append(data[:,0], empty_patient)
                            
                            data_temp = np.sort(data_temp)
                            ind = np.where(data_temp == empty_patient)[0][0]
                            data = np.insert(data, ind,nan_array,0)                                    

                    else:
                        data = np.array(df.iloc[:,df.columns.get_loc(u'SEQN'):])

            if j == 0:
                total_data = data[:,1:]
            else:
                total_data = np.concatenate((total_data,data[:,1:]), axis=1)
            j += 1
            

    # print(total_data.shape)
    # print(len(column_names))

    patient_numbers[year] = list(map(lambda x: [x], patient_numbers[year]))
    total_data = np.concatenate((np.array(patient_numbers[year]), total_data), axis=1)

    df_year[year] = pd.DataFrame(total_data,index=None,columns=column_names[year])
    

    # print(len(list(df_year['2009-2010'].columns))==len(list(set(list(df_year['2009-2010'].columns)))))

# print(df_year)
# print(len(patient_numbers['2009-2010']))
# print(len(patient_numbers['2011-2012']))
# print(len(patient_numbers['2013-2014']))

df_total = df_year['2009-2010'].append(df_year['2011-2012'], sort = False)
df_total = df_total.append(df_year['2013-2014'], sort=False)

print(df_total)

# print(column_names)
# print(len(column_names['2009-2010']))    #3168
# print(len(column_names['2011-2012']))    #3140
# print(len(column_names['2013-2014']))    #5722
# print(len(list(set(column_names['2009-2010']).intersection(column_names['2011-2012']))))    #2395
# print(len(list(set(column_names['2009-2010']).intersection(column_names['2013-2014']))))    #1602
# print(len(list(set(column_names['2011-2012']).intersection(column_names['2013-2014']))))    #1841
# print(len(list(set(column_names['2011-2012']).intersection(column_names['2013-2014']).intersection(column_names['2009-2010']))))    #1467


#remove column
if nan_percentile:
    df_total = df_total.dropna(axis=1,thresh = int((len(patient_numbers['2009-2010'])+len(patient_numbers['2011-2012'])+len(patient_numbers['2013-2014'])) * (1-nan_percentile)))
# print(df)
df_total['y'] = y_total

df_total = df_total.dropna(axis=0,thresh = int(len(list(df_total.columns))* (1-nan_percentile)))

y_total = list(df_total['y'])
df_total.drop('y', axis=1, inplace=True)


df_total.to_excel('data_paper_'+ str(int(nan_percentile*100)) + '_' + str(teeth_removal_num) +'.xlsx',sheet_name='data')

np.save('data_paper_'+ str(int(nan_percentile*100)) +'_wo_imputation.npy', np.array(df_total)[:,1:])
np.save('data_paper_'+ str(int(nan_percentile*100)) +'_y_wo_imputation.npy', y_total)

np.save('data_paper_columns_'+ str(int(nan_percentile*100))+'.npy', df_total.columns)

data = np.array(df_total)[:,1:]
y_data = y_total

normalized_data = MinMaxScaler().fit_transform(data)

imputer = KNNImputer(n_neighbors=int(normalized_data.shape[0]/10), weights = 'uniform')
imputed_data = imputer.fit_transform(normalized_data)

np.save('./data_imputed_paper_' +str(int(nan_percentile*100)) + '_' + str(teeth_removal_num) + '.npy',imputed_data)
np.save('data_paper_y_'+ str(int(nan_percentile*100))  + '_' + str(teeth_removal_num)+'.npy', y_data)
