import numpy as np
import pandas as pd
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.set_option('mode.chained_assignment', None)

#NO DATA FOR 2011-2012!!!!!!!
#years = ['2005-2006','2007-2008','2009-2010','2013-2014']
#years = ['2009-2010','2013-2014']
years = ['2013-2014']
# number of patient 10537   9756   10173  
#     51624-62160    62161-71916   73557-83729   

datasets = ['demographics','dietary','examination','laboratory','questionnaire']
#datasets = ['demographics', 'examination', 'laboratory', 'questionnaire']

# nan_percentiles = [0.1,0.2,0.3,0.4,0.5]
nan_percentile = 0.1  # None,0.1,0.2,0.3
#                           574  730  1926

#Exclude dataset / columns that directly relate to osteoporosis
except_list=['DXXSPN_D', 'DXXFEM_D', 'OSQ_D', 'DXXFEM_E','DXXSPN_E', 'OSQ_E', 'DXXFEM_F', 'DXXSPN_F', 'OSQ_F', 'DXXFEM_H', 'DXXL1_H','DXXL2_H', 'DXXL3_H', 'DXXL4_H', 'DXXSPN_H', 'OSQ_H',
            'DXXAG_D','dxx','dxx_b','dxx_c','dxx_d','DXXAAC_H','DXXFRX_H','DXXT4_H', 'DXXT5_H','DXXT6_H', 'DXXT7_H','DXXT8_H','DXXT9_H','DXXT10_H','DXXT11_H','DXXT12_H','DXXVFA_H',
            'OHXDEN_H', 'RXQ_RX_D', 'RXQ_RX_E','RXQ_RX_F','RXQ_RX_H', 'PAQIAF_D', 'SSHPV_F',
            'DR1IFF_D', 'DR1IFF_E','DR1IFF_F','DR1IFF_H', 'DR2IFF_D', 'DR2IFF_E','DR2IFF_F','DR2IFF_H','DSQ2_D','DSQ2_E','DSQ2_F','DSQ2_H',
            'FFQDC_D','FFQDC_E','FFQDC_F','FFQDC_H', 'DS1IDS_E','DS1IDS_F','DS1IDS_H','DS2IDS_E','DS2IDS_F','DS2IDS_H',
            'DSQIDS_E','DSQIDS_F','DSQIDS_H']
#except_list=['DXXFEM_H', 'DXXSPN_H', 'OSQ_H', 'DR1IFF_H', 'DR1TOT_H', 'DR2IFF_H', 'DR2TOT_H', 'DRXFCD_H','DS1IDS_H','DS1TOT_H','DS2IDS_H','DS2TOT_H','DSBI','DSII','DSPI','DSQIDS_H']
except_column_list=['ARD118AE','ARD118AO','ARD118BE', 'ARD118BO', 'DTQ020A', 'DTD020AF', 'DTD020B','DTD020BF','DXXLABEL', 'SMD100BR',
                    'SMDUPCA', 'SPXNQFV1', 'SPXNQEFF', 'SPXNQFVC', 'SPXBQFVC','SPXBQFV1','SPXBQEFF','DTQ020A','DTQ020B']

#91: other, 90: other
Dontknow = [9,99,999,9999,99999,999999,9999999,99999999]

'''
#pseudo code
some patients empty-->process
1. Education concatenate ('DMDEDUC3', 'DMDEDUC2')

2. Income concatenate ('INDHHIN2', 'INDFMIN2', 'INDFMPIR', )

3. Alcohol concatenate ('ALQ_F', 'ALQY_F', )'''

#NHANES III femoral neck BMD non-hispanic white women 20-29 years old
fem_neck_mean=0.858
fem_neck_SD=0.120

#NHANES III total femur BMD non-hispanic white women 20-29 years old
fem_tot_mean=0.942
fem_tot_SD=0.122

#Hologic reference database of 30-year-old white women l1-l4 mean
lumb_mean=1.047
lumb_SD=0.11


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def osteoporosis(df):
    #Before labeling, exclude 'nan' value patients. (drop 'nan' containing row)
    df.dropna(subset=['DXXNKBMD'], inplace=True)
    #df.dropna(subset=['DXXOFBMD'], inplace=True)
    #df.dropna(subset=['DXXL1BMD'], inplace=True)
    #df.dropna(subset=['DXXL2BMD'], inplace=True)
    #df.dropna(subset=['DXXL3BMD'], inplace=True)
    #df.dropna(subset=['DXXL4BMD'], inplace=True)

    df_n = df['DXXNKBMD']
    #df_t = df['DXXOFBMD']
    #df_l1 = df['DXXL1BMD']
    #df_l2 = df['DXXL2BMD']
    #df_l3 = df['DXXL3BMD']
    #df_l4 = df['DXXL4BMD']

    #1. Option: femoral neck
    df_result = (df_n - fem_neck_mean)/fem_neck_SD

    #2. Option: Total femur
    #df_result = (df_t-fem_tot_mean)/fem_tot_SD

    #3. Option: Lumbar Spine l1~l4
    #df_result = ((df_l1+df_l2+df_l3+df_l4)/4-lumb_mean)/lumb_SD

    y=[]
    label1=0 #normal
    label2=1 #osteopenia
    label3=2 #osteoporosis
    data=np.array(df_result)

    for patient in range(df.shape[0]):
        value = data[patient]
        
        if value<=-2.5:
            y.append(label3)
        elif value<=-1.0:
            y.append(label2)
        else:
            y.append(label1)
        #print(value, y[patient])
    
    #Eliminate label columns in dataframe
    df.drop('DXXNKBMD', inplace=True, axis=1)
    df.drop('DXXOFBMD', inplace=True, axis=1)
    #df.drop('DXXL1BMD', inplace=True, axis=1)
    #df.drop('DXXL2BMD', inplace=True, axis=1)
    #df.drop('DXXL3BMD', inplace=True, axis=1)
    #df.drop('DXXL4BMD', inplace=True, axis=1)

    patient_numbers=list(df['SEQN'])
    
    return patient_numbers, y

def connected_variable(df, var1, var1_val, var2, var2_val):
    a = df[var1] == var1_val
    df.loc[a, var2] = var2_val

    return df


column_names={}
patient_numbers={}
df_year={}
y_total=[]
df_total={}
df_total_year={}
column_numbers=0
patient_len=0
more_except=['name']
df_y={}


for year in years:
    l=0
    print(year)
    if year == '2005-2006':
        char = ['D']
    if year == '2007-2008':
        char = ['E']
    elif year == '2009-2010':
        char = ['F']
    elif year == '2013-2014':
        char = ['H']

    j=0
    for c in char:
        df = pd.read_csv('./'+year+'/examination/DXXFEM_' + c +'.csv')
        #For option 3, choose df as below
        #df = pd.read_csv('./'+year+'/examination/DXXSPN_'+c+'.csv')
        
        patient_numbers[year], y = osteoporosis(df)
        y_total +=y
        df_y[year]=y

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

                #if xpt.split('.')[0] in except_list or not u'SEQN' in df.columns:
                    #pass when except_list dataset or no SEQN in data
                #    pass
                #else:

                #    if list(set(list(df.columns)).intersection(except_column_list)) != []:
                #        df.drop(list(set(list(df.columns)).intersection(except_column_list)), axis=1, inplace=True)
                    
                xpt_name=xpt.split('.')[0]

                    

                if xpt_name=='OHXDEN_'+c and year != '2009-2010':
                    df = df.replace({"b'D'":1, "b'E'": 2, "b'J'": 3, "b'K'":4, "b'M'":5, "b'P'":6, "b'Q'":7, "b'R'":8, "b'S'":9, "b'T'":10, "b'U'":11, "b'X'":12, "b'Y'":13, "b'Z'":14, "b''": np.NaN})
                    
                elif xpt_name=='CSX_'+c:
                    df = df.replace({"b'A'": 1, "b'B'": 2,"b''": np.NaN})
                elif xpt_name=='VID_'+c and year != '2005-2006':
                    new_column=df['LBXVIDMS']+df['LBXVE3MS']
                    df.drop(['LBXVIDMS','LBXVE3MS'], axis=1, inplace=True)
                    df['LBXVIDMS_LBXVE3MS'] = new_column
                elif xpt_name == 'ALQ_'+c:
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
                    
                elif xpt_name == 'RXQASA_'+c:
                    conditions = [
                        (df['RXQ515'].isna()),
                        (df['RXQ520'].isna())
                    ]
                    choices = [df['RXQ520'], df['RXQ515']]

                    df['RXQ515_RXQ520'] = np.select(conditions, choices, default = np.nan)
                    df.drop(['RXQ515','RXQ520'],axis=1, inplace=True)

                elif xpt_name == 'SPX_'+c and year != '2013-2014' and year != '2007-2008' and year != '2009-2010':
                    print('-----------')
                    df = df.replace({"b'A'":1, "b'B'": 2, "b'C'": 3, "b'D'":4, "b'F'":5, "b''": np.NaN})

                elif xpt_name == 'CBC_'+c or xpt_name == 'COT_'+c:
                    df.rename({'LBXHCT': 'LBXHCT_'+xpt.split('_')[0]}, axis=1, inplace=True)

                elif xpt_name == 'DR1TOT_'+c:
                    df = connected_variable(df,'DBQ095Z',4,'DBD100',0)
                elif xpt_name == 'BPQ_'+c:
                    if year == '2005-2006':
                        df = connected_variable(df, 'BPQ020', 2, ['BPQ030', 'BPQ040A'], [2,2])
                    elif year == '2013-2014' or year == '2007-2008':
                        df = connected_variable(df,'BPQ020', 2, ['BPQ030', 'BPD035', 'BPQ040A'], [2,100,2])
                    else:

                        df = connected_variable(df,'BPQ020', 2, ['BPQ030', 'BPD035', 'BPQ040A'], [2,100,2])
                        df = connected_variable(df,'BPQ056', 2, 'BPD058', 0)
                    
                elif xpt_name == 'DIQ_'+c:
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

                elif xpt_name == 'DBQ_'+c and year != '2005-2006' and year != '2007-2008':
                    df = connected_variable(df,'DBQ197', 0, ['DBQ223A', 'DBQ223B', 'DBQ223C', 'DBQ223D', 'DBQ223E', 'DBQ223U'], 0)
                    df = connected_variable(df,'DBQ229', 2, ['DBQ235A', 'DBQ235B', 'DBQ235C'], 0)
                        
                elif xpt_name  == 'DUQ_'+c:
                    if year == '2005-2006' or year =='2007-2008':
                        df = connected_variable(df,'DUQ200', 2, ['DUQ210', 'DUQ230'], [100, 0])
                        df = connected_variable(df,'DUQ240', 2, ['DUQ250', 'DUQ300', 'DUQ320', 'DUQ330', 'DUQ340', 'DUQ352', 'DUQ360'], [2,100, 0,2,100, 0, 0])
                        df = connected_variable(df,'DUQ250', 2, ['DUQ260', 'DUQ272', 'DUQ280'], [100, 0, 0])
                        df = connected_variable(df,'DUQ370', 2, ['DUQ380A', 'DUQ380B', 'DUQ380C', 'DUQ380D', 'DUQ380E', 'DUQ390','DUQ410','DUQ420'], [0, 0, 0,0,0,100,0,0])
                    else:
                        df = connected_variable(df,'DUQ200', 2, ['DUQ210', 'DUQ211', 'DUQ213', 'DUQ217', 'DUQ219', 'DUQ230'], [100,2,100, 0, 0, 0])
                        df = connected_variable(df,'DUQ240', 2, ['DUQ250', 'DUQ300', 'DUQ320', 'DUQ330', 'DUQ340', 'DUQ352', 'DUQ360'], [2,100, 0,2,100, 0, 0])
                        df = connected_variable(df,'DUQ250', 2, ['DUQ260', 'DUQ272', 'DUQ280'], [100, 0, 0])
                        df = connected_variable(df,'DUQ370', 2, ['DUQ380A', 'DUQ380B', 'DUQ380C', 'DUQ380D', 'DUQ380E', 'DUQ390','DUQ410','DUQ420'], [0, 0, 0,0,0,100,0,0])

                elif xpt_name  == 'ECQ_'+c:
                    df = connected_variable(df,'MCQ080E', 2, 'ECQ150', 2)

                elif xpt_name  == 'FSQ_'+c and year != '2005-2006':
                    df = connected_variable(df,'FSD041', 2, 'FSD052', 4)
                    df = connected_variable(df,'FSQ165', 2, 'FSQ171', 2)
                    
                elif xpt_name == 'HIQ_'+c:
                    df = connected_variable(df,'HIQ011', 2, ['HIQ031A','HIQ031B','HIQ031C','HIQ031D','HIQ031E','HIQ031F','HIQ031H','HIQ031I','HIQ031J'], [0,0,0,0,0,0,0,0,0])

                elif xpt_name  == 'HUQ_'+c:
                    df = connected_variable(df,'HUQ071', 2, 'HUD080', 0)

                elif xpt_name == 'IMQ_'+c and year != '2005-2006':
                    df = connected_variable(df,'IMQ040', 2, 'IMQ045', 0)
                    
                elif xpt_name == 'INQ_'+c:
                    df = connected_variable(df,'INQ244', 1, 'INQ247', 7)

                elif xpt_name == 'KIQ_'+c:
                    df = connected_variable(df,'KIQ022', 2, 'KIQ025', 2)
                    df = connected_variable(df,'KIQ026', 2, 'KIQ028', 0)
                    df = connected_variable(df,'KIQ005', 1, 'KIQ010', 0)
                    df = connected_variable(df,'KIQ042', 2, 'KIQ430', 0)
                    df = connected_variable(df,'KIQ044', 2, 'KIQ450', 0)
                    df = connected_variable(df,'KIQ046', 2, 'KIQ470', 0)

                elif xpt_name == 'MCQ_'+c:
                    df = connected_variable(df,'MCQ010', 2, ['MCQ025', 'MCQ035','MCQ040','MCQ050'], [100,2,2,2])
                    df = connected_variable(df,'MCQ160M', 2, 'MCQ170M', 2)
                    df = connected_variable(df,'MCQ160K', 2, 'MCQ170K', 2)
                    df = connected_variable(df,'MCQ160L', 2, 'MCQ170L', 2)

                elif xpt_name == 'OCQ_'+c and year != '2013-2014' and year != '2005-2006':
                    df = connected_variable(df,'OCQ510',2 , 'OCQ520',0 )  
                    df = connected_variable(df,'OCQ530',2 , 'OCQ540',0 )  
                    df = connected_variable(df,'OCQ550',2 , 'OCQ560',0 )  
                    df = connected_variable(df,'OCQ570',2 , 'OCQ580',0 ) 

                elif xpt_name == 'PAQ_'+c and year != '2005-2006':
                    df = connected_variable(df,'PAQ605', 2, ['PAQ610','PAD615'], [0, 0])
                    df = connected_variable(df,'PAQ620', 2, ['PAQ625','PAD630'], [0, 0])
                    df = connected_variable(df,'PAQ635', 2, ['PAQ640','PAD645'], [0, 0])
                    df = connected_variable(df,'PAQ650', 2, ['PAQ655','PAD660'], [0, 0])
                    df = connected_variable(df,'PAQ665', 2, ['PAQ670','PAD675'], [0, 0])

                elif xpt_name == 'SMQ_'+c:
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

                elif xpt_name == 'SMQRTU_'+c and year != '2013-2014':
                    df = connected_variable(df,'SMQ680', 2, ['SMQ690A','SMQ690B','SMQ690C','SMQ690D','SMQ690E','SMQ690F', 'SMQ710','SMQ720','SMQ725', 'SMQ740', 'SMQ750', 'SMQ755', 'SMQ770', 'SMQ780', 'SMQ785', 'SMQ800','SMQ815', 'SMQ817', 'SMQ819'], [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 5,0, 5, 0, 5])

                if xpt.split('.')[0] in except_list or not u'SEQN' in df.columns:
                    continue  

                elif df.astype('object').isin(["b''"]).sum().sum() != 0:
                       
                    print('Found non numeric column in :', xpt)
                    more_except.append(xpt)
                        #quit()
                    
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
                            
                        df = df[~df[u'SEQN'].isin(list(map(int, list(set(list(df[u'SEQN'])) - set(patient_numbers[year])))))]
                    
                        column_names[year] += columns
                

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
                    print(total_data.shape)
                else:
                    print(total_data.shape)
                    print(data.shape)
                    total_data = np.concatenate((total_data,data[:,1:]), axis=1)
                    
                j += 1
                                
                    #print(columns)
                        
                    #if list(set(column_names[year]).intersection(columns))!=[]:
                    #data = np.array(df.iloc[:,df.columns.get_loc(u'SEQN'):])

                        
                    #print(patient_numbers[year])

                    #only append patient datas that have DXX info--patient_numbers
                    #k=0
                    #for patient in patient_numbers[year]:
                    #    if k==0:
                    #        df_tmp=df.loc[df[u'SEQN']==patient]
                    #        k+=1
                    #    else:
                    #        df_tmp=df_tmp.append(df.loc[df[u'SEQN']==patient], sort=False)
                    #print(df_tmp)


        patient_numbers[year] = list(map(lambda x: [x], patient_numbers[year]))
        total_data = np.concatenate((np.array(patient_numbers[year]), total_data), axis=1)

        df_year[year] = pd.DataFrame(total_data,index=None,columns=column_names[year])
                        

                    #for column in columns:
                        #print('current column: ', column, xpt)
                        #max_val=df[column].max()
                        #if max_val in Dontknow:
                            
                            #dontknow_val = max_val
                            #if max_val != 9:
                                #print(column)
                                #refused_val = max_val/9*7 
                            #else:
                                #refused_val=12345
                            
                            #df[column][(df[column] == dontknow_val)] = None
                            #df[column][(df[column] == refused_val)] = None

                #if df_tmp.shape[0]==len(patient_numbers[year]):
                
                #if list(set(column_names[year]).intersection(columns))!=[]:
                    #if j==0:
                        #df_total_year=df_tmp
                        #print('start: ', df_tmp.shape)
                        #j+=1
                    #else:

                        #print('now: ', df_tmp.shape)
                        #remove duplicated columns   
                        #for column in columns:
                            #if column in column_names[year] and column!=u'SEQN':
                                #print('-----------', column)
                                #df_tmp=df_tmp.drop([column], axis=1)
                                #print(df_tmp.columns)
                        #columns=list(df_tmp.iloc[:,df_tmp.columns.get_loc(u'SEQN')+1:].columns)

                        #df_total_year=pd.merge(df_total_year,df_tmp, on=u'SEQN')
                        
                        
                #else:
                    #if j==0:
                        #df_total_year=df_tmp
                        #print('start: ', df_tmp.shape)
                        #j+=1
                    #else:
                        #print('------------', df_total_year)
                       # df_total_year=pd.merge(df_total_year,df_tmp, on=u'SEQN')
                
                #print('df_total_year: ', df_total_year.shape)
               # column_names[year] +=columns
        
        #df_year[year]=df_total_year
        #print(df_year[year].shape)
        #df_year[year] = df_year[year].loc[:, ~df.columns.str.contains('^Unnamed')]
        #print(df_year[year].shape)
        
        #data = np.array(df_year[year].iloc[:,df_year[year].columns.get_loc(u'SEQN'):])


        #column_names[year]=list(df_year[year].iloc[:,df_year[year].columns.get_loc(u'SEQN')+1:].columns)
        #print(len(column_names[year]))

       #patient_numbers[year]=list(map(lambda x: [x], patient_numbers[year]))
        #patient_len+=len(patient_numbers[year])

        #if year=='2005-2006':
            #df_year['2005-2006'].to_csv('./2005-2006_wo_imputation.csv')
            #np.save('./2005-2006_y.npy',np.array(df_y['2005-2006']))
            #quit()
            
        #elif year=='2007-2008':
            #df_year['2007-2008'].to_csv('./2007-2008_wo_imputation.csv')
            #np.save('./2007-2008_y.npy',np.array(df_y['2007-2008']))
        #elif year=='2009-2010':
            #df_year['2009-2010'].to_csv('./2009-2010_wo_imputation.csv')
            #np.save('./2009-2010_y.npy',np.array(df_y['2009-2010']))
        #else:
            #df_year['2013-2014'].to_csv('./2013-2014_wo_imputation.csv')
            #np.save('./2013-2014_y.npy',np.array(df_y['2013-2014']))

df_total = df_year['2013-2014']

print(df_total)
df_year['2013-2014'].to_csv('./2013-2014_wo_imputation.csv')

np.save('./2013-2014_y.npy',y_total)



    #df_year[year] = pd.DataFrame(data, index=None, columns=column_names[year])
    #df_year[year] = df_year[year].loc[:, ~df.columns.str.contains('^Unnamed')]



#print(df_total.shape)
#print(len(y_total))
print(more_except)





#df_y['2005-2006'].to_csv('./2005-2006_y.csv')
#df_y['2007-2008'].to_csv('./2007-2008_y.csv')
#df_y['2009-2010'].to_csv('./2009-2010_y.csv')
#df_y['2013-2014'].to_csv('./2013-2014_y.csv')



'''
df_total=df_total.iloc[:,df.columns.get_loc(u'SEQN')+1:]
df_total.to_csv('./total_data_wo_imputation.csv')'''
#np.save('./3_data_wo_imputation.npy', np.array(df_total))
#np.save('./3_data_y.npy', y_total)
