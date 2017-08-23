import pandas as pd
from getPath import *
pardir = getparentdir()

def analyze_train():
    data = pd.read_csv(pardir+'/data/user_log_format1.csv')
    # users = len(data.groupby('user_id').size())
    # merchants = len(data.groupby('merchant_id').size())
    # positives = data['label'][data['label']==1]
    print(data.columns.values)
    print(data[1:5])
    # print(users)
    # print(merchants)
    # print(len(positives))
    # print(len(data))
    # print(len(positives)/len(data))
    
analyze_train()
    



