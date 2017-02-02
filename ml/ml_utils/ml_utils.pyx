import numpy as np
cimport numpy as np
import pandas as pd
from os import listdir
cimport cython

from libc.math cimport isnan


def train_test_split(data,only_interesting=True):
    from sklearn import preprocessing
    data = data.dropna(how='any')
    y = data['score']
    del data['score']
    
    if only_interesting:
        half_std = y.std() / 2
        data = data.ix[y.abs()>half_std]
        y = y[y.abs()>half_std]
    
    split = len(data) * .7
    
    train_X = data.values[:split,:]
    test_X = data.values[split:,:]
    train_y = y.values[:split]
    test_y = y.values[split:]
    
    std_scale = preprocessing.StandardScaler().fit(train_X)
    X_train_std = std_scale.transform(train_X)
    X_test_std = std_scale.transform(test_X)
    
    return X_train_std,X_test_std,train_y,test_y

def train_test_split_robust(X,y, test_size=0.3,scale=True):  
    ntrn = int(round((X.shape[0]) * (1 - test_size)))
    X_train, y_train = X[:ntrn],y[:ntrn]
    X_test, y_test = X[ntrn:],y[ntrn:]
    
    from sklearn.preprocessing import RobustScaler
    RS = RobustScaler()
    
    if not scale:
        return X_train, X_test, y_train, y_test
    X_train = RS.fit_transform(X_train)
    return X_train, RS.transform(X_test), y_train, y_test

def list_h5s(mypath,enriched=False, dates_only=True):
    s = pd.Series([f.strip() for f in listdir(mypath)])
    dates = s.ix[s.str[-2:]=='h5']
    if not enriched:
        dates = dates.ix[np.logical_not(dates.str.contains("enriched"))]
    else:
        dates = dates.ix[(dates.str.contains("enriched"))]
    if not dates_only:
        return dates
    return dates.ix[dates.str[0]=='2']

def optimize_gbt(trainX,trainY,testX,testY,fspace=None,max_calls=50):
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    if fspace is None:
        fspace = {
            'n_estimators': hp.choice('n_estimators', range(20,200,20)),
            'max_depth': hp.choice('max_depth', range(2,6)),
            'subsample': hp.choice('subsample', [ x / 10. for x in range(5,10,1)]),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(20,100,20)),
            'min_samples_split': hp.choice('min_samples_split', range(20,100,20)),
            'learning_rate': hp.choice('learning_rate', [.001,.005,.01,.05,.1,.5])
        }

    best = 0

    def hyperopt_train_test(params):
        clf = GradientBoostingRegressor(**params)
        clf.fit(trainX, trainY)
        test_preds = clf.predict(testX)
        return r2_score(testY,test_preds)

    def f(params):
        global best
        
        r2 = hyperopt_train_test(params)
        if r2 > best:
            best = r2
        print 'new best:', best, params
        return {'loss': -r2, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=max_calls, trials=trials)
    print 'best:', best
    return trials

@cython.cdivision(True)
cdef single_path_score(np.ndarray[double,ndim=1] wmids, int side, long dex, double start, double win, double loss):
    cdef:
        long i = dex, w_len = wmids.shape[0]

    if side == 1:
        for i from dex <= i < w_len:
            if isnan(wmids[i]):
                continue
            if wmids[i] - win > start:
                return 1
            if wmids[i] < start - loss:
                return 0
        return np.NaN
    else:
        for i from dex <= i < w_len:
            if isnan(wmids[i]):
                continue
            if wmids[i] < start - win:
                return 1
            if wmids[i] - loss > start:
                return 0
        return np.NaN

@cython.cdivision(True)
def path_score(np.ndarray[double,ndim=1] wmids, int side, double win, double loss, double tw):
    cdef:
        long i =0
        long w_len = wmids.shape[0]
        np.ndarray[double,ndim=1] res = np.zeros(w_len,dtype=np.double) * np.NaN
        double winner = win * tw
        double loser = loss * tw
    for i from 0 <= i < w_len:
        res[i] = single_path_score(wmids,side,i,wmids[i],winner,loser)
    return res