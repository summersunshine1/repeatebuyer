import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()

user_log_path = pardir+'/data/user_log_format1.csv'

def merchantFeature(data):
    merchant = pd.DataFrame()
    merchant['item_set']=data.groupby("seller_id")['item_id'].apply(set)
    merchant['item_num']=(merchant['item_set'].map(len)).astype(np.int16)
    merchant.drop('item_set',1,inplace=True)
    merchant['cate_set']=data.groupby("seller_id")['cat_id'].apply(set)
    merchant['cate_num']=(merchant['cate_set'].map(len)).astype(np.int16)
    merchant.drop('cate_set',1,inplace=True)
    merchant['brand_set']=data.groupby("seller_id")['brand_id'].apply(set)
    merchant['brand_num']=(merchant['brand_set'].map(len)).astype(np.int16)
    merchant.drop('brand_set',1,inplace=True)
    group = data.groupby("seller_id")
    merchant['click']=group.apply(lambda g:len(g[g['action_type']==0]))
    merchant['add_to_carts'] =group.apply(lambda g:len(g[g['action_type']==1]))
    merchant['purchase']=group.apply(lambda g:len(g[g['action_type']==2]))
    merchant['add_to_favourite'] =group.apply(lambda g:len(g[g['action_type']==3]))
    del group
    print(merchant)
    
if __name__=="__main__":
    data = pd.read_csv(user_log_path,encoding='utf-8')
    merchantFeature(data)
    


