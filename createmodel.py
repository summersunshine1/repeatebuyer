from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score,KFold, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVR,LinearSVR
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

from getPath import *
pardir = getparentdir()

train_split_path = pardir+'/middledata/train_split.csv'
test_split_path = pardir+'/middledata/test_split.csv'
exceptcolumns = ['user_id','merchant_id','item_id','brand_id','cat_id','label']

def get_train_data():
    data = pd.read_csv(train_split_path,encoding='utf-8')
    columns = list(data.columns.values)
    features = list(set(columns)-set(exceptcolumns))
    x = data[features]
    y = data['label']
    
    