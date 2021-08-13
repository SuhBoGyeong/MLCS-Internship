import xport, csv 
import os 
import pandas as pd 

data_dir='./2007-2008'

def xpt_to_csv(filename, filepath, save_dir):
    path=os.path.join(filepath+'/'+filename)
    with open(path, 'rb') as f:
        df=xport.to_dataframe(f)

    savepath=os.path.join(filepath+'/'+filename.split('.')[0]+'.csv')

    df.to_csv(savepath)


for folder in os.listdir(data_dir):
    print(folder)
    path_1=os.path.join(data_dir, folder)
    for xpt in os.listdir(path_1):
        if xpt.split('.')[1]=='XPT' or xpt.split('.')[1]=='xpt':
            xpt_loc=os.path.join(path_1)
            sav_loc=path_1
            xpt_to_csv(xpt, xpt_loc, sav_loc)
