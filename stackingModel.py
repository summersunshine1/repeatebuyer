from sklearn.model_selection import KFold
from createmodel import *
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

stacking_res_path = pardir+'/middledata/stackingres.csv'

def get_k_fold(data):
    kf = KFold(n_splits=5,random_state=1,shuffle=True)
    train_indexs = []
    test_indexs = []
    for train_index, test_index in kf.split(data):
        train_indexs.append(train_index)
        test_indexs.append(test_index)
    return train_indexs,test_indexs
    
def createmodel():
    x,y = getTrainData()
    test_data = getPredictData()
    train_indexs,test_indexs = get_k_fold(x)
    #stacking linearsvr
    firstlayer = []
    test = []
    x_arr = np.array(x)
    y_arr = np.array(y)
    for i in range(len(train_indexs)):
        print("first train"+str(i))
        train = x_arr[train_indexs[i]]
        label = y_arr[train_indexs[i]]
        regr = LinearSVR(random_state=0)
        regr.fit(train,label)
        joblib.dump(regr, pardir+'/model/lr'+str(i)+".pkl")
        # regr = joblib.load(pardir+'/model/lr'+str(i)+".pkl")
        res = regr.predict(x_arr[test_indexs[i]])
        firstlayer+=list(res)
        res = regr.predict(test_data)
        test.append(res)
    test = np.mean(test,axis = 0)
    test = [[t] for t in test]
    # firstlayer = x_arr
    secondlayer =[]
    finalres = []
    firstlayer = np.array([[f] for f in firstlayer])
    for i in range(len(train_indexs)):
        print("second train"+str(i))
        train = firstlayer[train_indexs[i]]
        label = y_arr[train_indexs[i]]
        # clf = joblib.load(pardir+'/model/rf'+str(i)+".pkl")
        clf = RandomForestClassifier(random_state=0)
        clf.fit(train,label)
        joblib.dump(clf, pardir+'/model/rf'+str(i)+".pkl")
        # res = clf.predict(test_data[test_indexs[i]])
        res = clf.predict_proba(test)
        test.append(res[:,1])
    res = np.mean(test,axis = 0)
    test_data['prob'] = res
    res = pd.DataFrame({'prob':test_data.groupby(['user_id','merchant_id'])['prob'].max()}).reset_index()
    res.to_csv(stacking_res_path,encoding='utf-8',mode = 'w', index = False)
 
if __name__=="__main__":
    createmodel()
        
        
        
    
