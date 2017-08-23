import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()

user_log_path = pardir+'/data/user_log_format1.csv'

def merchantFeature(data):
    merchant = pd.DataFrame()
    merchant['item_set']=data.groupby(["seller_id")['item_id'].apply(set)
    merchant['item_num']=(merchant['item_set'].map(len)).as_type(np.int16)
    merchant.drop('item_set',1,inplace=True)
    merchant['cate_set']=data.groupby(["seller_id")['item_id'].apply(set)
    merchant['cate_num']=(merchant['item_set'].map(len)).as_type(np.int16)
    
    
    print(merchant)
    
if __name__=="__main__":
    data = pd.read_csv(user_log_path,encoding='utf-8')
    merchantFeature(data)
    


