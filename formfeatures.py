import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()

user_log_path = pardir+'/data/user_log_format1.csv'
train_path = pardir+'/data/train_format1.csv'
test_path = pardir+'/data/test_format1.csv'

train_log_path = pardir+'/data/train_log_format1.csv'
test_log_path = pardir+'/data/test_log_format1.csv'

merchant_path = pardir +'/middledata/merchant.csv'
item_path = pardir +'/middledata/item.csv'


def merchantFeature(data):
    merchant = pd.DataFrame()
    merchant['item_set']=data.groupby("merchant_id")['item_id'].apply(set)
    merchant['item_num']=(merchant['item_set'].map(len)).astype(np.int16)
    merchant.drop('item_set',1,inplace=True)
    merchant['cate_set']=data.groupby("merchant_id")['cat_id'].apply(set)
    merchant['cate_num']=(merchant['cate_set'].map(len)).astype(np.int16)
    merchant.drop('cate_set',1,inplace=True)
    merchant['brand_set']=data.groupby("merchant_id")['brand_id'].apply(set)
    merchant['brand_num']=(merchant['brand_set'].map(len)).astype(np.int16)
    merchant.drop('brand_set',1,inplace=True)
    group = data.groupby("merchant_id")
    merchant['click']=group.apply(lambda g:len(g[g['action_type']==0]))
    merchant['add_to_carts'] =group.apply(lambda g:len(g[g['action_type']==1]))
    merchant['purchase']=group.apply(lambda g:len(g[g['action_type']==2]))
    merchant['add_to_favourite'] =group.apply(lambda g:len(g[g['action_type']==3]))
    del group
    merchant.reset_index(level=['merchant_id'],inplace = True)
    merchant.to_csv(merchant_path,encoding='utf-8',mode = 'w', index = False)
    del merchant
    
def itemFeature(data):
    item = pd.DataFrame()
    group = data.groupby("item_id")
    item['click']=group.apply(lambda g:len(g[g['action_type']==0]))
    item['add_to_carts'] =group.apply(lambda g:len(g[g['action_type']==1]))
    item['purchase']=group.apply(lambda g:len(g[g['action_type']==2]))
    item['add_to_favourite'] =group.apply(lambda g:len(g[g['action_type']==3]))
    del group
    item.reset_index(level=['item_id'],inplace = True)
    item.to_csv(item_path,encoding='utf-8',mode = 'w', index = False)
    del item

def brandFeature(data):
    item = pd.DataFrame()
    group = data.groupby("brand_id")
    item['click']=group.apply(lambda g:len(g[g['action_type']==0]))
    item['add_to_carts'] =group.apply(lambda g:len(g[g['action_type']==1]))
    item['purchase']=group.apply(lambda g:len(g[g['action_type']==2]))
    item['add_to_favourite'] =group.apply(lambda g:len(g[g['action_type']==3]))
    del group
    item.reset_index(level=['item_id'],inplace = True)
    item.to_csv(item_path,encoding='utf-8',mode = 'w', index = False)
    del item    


def identify_duplicate():
    train= pd.read_csv(train_path,encoding='utf-8')
    test = pd.read_csv(test_path,encoding='utf-8')
    train_ = train.groupby(['user_id','merchant_id']).count()
    train_.reset_index(level=['user_id','merchant_id'],inplace = True)
    test_ = test.groupby(['user_id','merchant_id']).count()
    test_.reset_index(level=['user_id','merchant_id'],inplace = True)
    s1 = pd.merge(train_, test_, how='inner', on=['user_id','merchant_id'])
    print(s1)
    
def split_train_test(data):
    train= pd.read_csv(train_path,encoding='utf-8')
    s1 = pd.merge(train[['user_id','merchant_id']],data,how='inner', on=['user_id','merchant_id'])
    s1.to_csv(train_log_path,encoding='utf-8',mode = 'w', index = False)
    del s1
    test= pd.read_csv(test_path,encoding='utf-8')
    s2 = pd.merge(test[['user_id','merchant_id']],data,how='inner', on=['user_id','merchant_id'])
    del test
    s2.to_csv(test_log_path,encoding='utf-8',mode = 'w', index = False)
    del s2


if __name__=="__main__":
    data = pd.read_csv(user_log_path,encoding='utf-8')
    newdata = data[0:10]
    print(newdata)
    del data
   # Q
    split_train_test(newdata)
    # test()
    


