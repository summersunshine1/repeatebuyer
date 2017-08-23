import pandas as pd
from getPath import *
pardir = getparentdir()

def analyze_train():
    data = pd.read_csv(pardir+'/data/user_log_format1.csv')
    # users = len(data.groupby('user_id').size())
    # merchants = len(data.groupby('merchant_id').size())
    # positives = data['label'][data['label']==1]
    data.rename(index=str, columns={'seller_id':'merchant_id'}, inplace=True)
    data.to_csv(pardir+'/data/user_log_format1.csv',encoding='utf-8',mode = 'w', index = False)
    del data
    # print(data)
    # print(users)
    # print(merchants)
    # print(len(positives))
    # print(len(data))
    # print(len(positives)/len(data))
    
analyze_train()
    



