##SIMPLEST POSSIBLE CUBIC REGRESSION SPLINES
import numpy as np
import pandas as pd

def basis(x, degree, i,knots):
    if degree == 0:
        B = np.zeros(len(x))
        B[np.logical_and(x>=knots[i],x<knots[i+1])] = 1
    else:
        if knots[degree+i] - knots[i] == 0:
            alpha1 = 0
        else:
            alpha1 = (x - knots[i]) / (knots[degree+i] - knots[i])
        if knots[i+degree+1] - knots[i+1] == 0:
            alpha2 = 0
        else:
            alpha2 = (knots[i+degree+1] - x) / (knots[i+degree+1] - knots[i+1])
        B = alpha1*basis(x,degree-1, i, knots) + alpha2*basis(x,degree-1,i+1,knots)
    return B

#KNOTS MUST BE IN ORDER -- no checks because this is going to be cython
def bspline(x,degree,knots,intercept=False):
    nknots = np.concatenate((np.repeat(knots[0],(degree+1)),np.array(knots[1:-1]),np.repeat(knots[-1],(degree+1))))
    K = len(knots[1:-1]) + degree + 1
    basis_matrix = np.zeros((len(x),K))
    for j in range(0,K):
        basis_matrix[:,j] = basis(x,degree,j,nknots)
    if not intercept:
        return basis_matrix[:,1:]
    return np.matrix(basis_matrix)

#rounds to nearest 1000th moneyness
def altmoneys(underlying_array, strikes, tte):
    us = (np.log(np.outer(strikes,1/underlying_array))).transpose()
    us = us / np.sqrt(tte)
    us = (us*1000).astype(int) * 1.0 / 1000
    return us
    
def placeKnots(some_moneys,some_vols):
    min_vol = np.argmin(some_vols)
    put_knots = np.append([-2.5,-1],np.linspace(-.4,some_moneys[min_vol]-.0025,4))
    call_knots = np.append(np.linspace(some_moneys[min_vol]+.0025,.4,3),[1,2.5])
    return np.concatenate([put_knots,call_knots])

def deltas_to_weights(sdeltas):
    return np.maximum((sdeltas - sdeltas** 2)/.25,.1)

def crs_vols(raw_vols,all_moneys,deltas=[],knots = [-5,-2.5,-1,-.5,-.3,-.1,0,.1,.2,.3,.5,1,2.5,5,10],degree=3,autoPlace=True):
    preds = []
    delta_weight = deltas.shape == raw_vols.shape
    i= 0
    assert all_moneys.shape == raw_vols.shape, 'Vols and Moneys Frame sizes do NOT match!'
    for vols,moneys in zip(raw_vols.values,all_moneys.values):
        actual_vol_values =  np.logical_not(np.isnan(vols))
        not_balls_far_out = np.logical_and(moneys > -2.5, moneys < 2.5)
        useful_vols = np.logical_and(actual_vol_values,not_balls_far_out)
        res = np.zeros(moneys.shape[0]) * np.NaN
        if np.any(actual_vol_values):
            if delta_weight:
                wm = np.diag(deltas_to_weights(deltas[i,useful_vols]))
                knots = placeKnots(moneys[actual_vol_values],vols[actual_vol_values])
                bm = np.dot(wm,bspline(moneys[useful_vols],degree,knots,True))
                v = np.dot(wm,vols[useful_vols])
            else:
                wm = np.diag(np.repeat(1,np.sum(useful_vols)))
                #knots = placeKnots(moneys[actual_vol_values],vols[actual_vol_values])
                bm = np.dot(wm,bspline(moneys[useful_vols],degree,knots,True))
                v = np.dot(wm,vols[useful_vols])
            c, resid,rank,sigma = np.linalg.lstsq(bm,v)
            res[useful_vols] = np.array(np.dot(bm,c)).ravel() * 1/np.diag(wm) #careful bro
        i+=1
        preds.append(res)
    return pd.DataFrame(preds,columns=raw_vols.columns,index=raw_vols.index)
