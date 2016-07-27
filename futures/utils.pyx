import numpy as np
cimport numpy as np
import pandas as pd
cimport cython
from libc.math cimport exp, sqrt, pow, log, erf, abs, M_PI

ctypedef np.double_t DTYPE_t
from libc.math cimport isnan

cdef inline double abz(double a) : return a if a >= 0. else -1 * a

cpdef inline double wmid(double bidp, double bids, double askp, double asks,double tick_width):
    if bids < 1 or asks < 1:
        return np.NaN
    if (askp - bidp) > 10 * tick_width:
        return np.NaN    
    if (askp - bidp) > tick_width:
        return (bidp+askp)/2
    else:
        return (bidp*asks+askp*bids)/(bids+asks)

def rolling_vwap_cython(np.ndarray[long,ndim=1] times, np.ndarray[double,ndim=1] prices, np.ndarray[double,ndim=1] volumes,long window_size,long min_size_not_nan):
    
    cdef long t_len = times.shape[0], s_len = prices.shape[0], i =0, win_size = window_size, t_diff, j, window_start
    cdef double tots_volume
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.zeros(t_len, dtype=np.double) * np.NaN
    assert(t_len==s_len)
    
    for i in range(0,t_len):
        window_start = times[i] - win_size
        j = i
        tots_volume = 0
        while times[j]>= window_start and j>=0:
            if prices[j] > 0:
                if tots_volume==0:
                    res[i] = 0
                tots_volume+=volumes[j]
                res[i]+= prices[j]*volumes[j]
            j-=1
        res[i]/=tots_volume
        if tots_volume <= min_size_not_nan:
            res[i] = np.NaN
    return res

def joins(np.ndarray[long,ndim=1] times,np.ndarray[double,ndim=2] md, long window_size):
    
    cdef long t_len = times.shape[0], md_len = md.shape[0], win_size = window_size, window_start, i,j,k
    cdef double start_bid, start_ask, tick_width
    cdef long start_bid_size, start_ask_size
    cdef double eps = .00001
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.zeros((t_len,2), dtype=np.double)
    assert(t_len == md_len)
    
    for i in range(1,t_len):
        window_start = times[i] - win_size
        j = i
        
        while times[j] >= window_start and j>=0:
            start_bid = md[j,0]
            start_ask = md[j,2]
            start_bid_size = long(md[j,1])
            start_ask_size = long(md[j,3])
            j-=1
        
        for k in [0,6,10,14,18]:
            if md[i,k] >= start_bid - eps:
                res[i,0] += md[i,1]
                if md[i,k] - start_bid < eps:
                    res[i,0] -= start_bid_size
                    
        for k in [2,8,12,16,20]:
            if md[i,k] <= start_ask + eps:
                res[i,1] += md[i,3]
                if md[i,k] - start_ask < eps:
                    res[i,1] -= start_ask_size           
    return res    

def add_vwaps(md):
    cdef:
        long sz = len(md),i=0
        dict last_info = {}
        np.ndarray[object, ndim=1] syms = md.symbol.values
        np.ndarray[DTYPE_t, ndim=1] t = md.turnover.values.astype(np.double)
        np.ndarray[DTYPE_t, ndim=1] v = md.volume.values.astype(np.double)
        np.ndarray[DTYPE_t, ndim=1] ts = md.tick_size.values.astype(np.double)
        np.ndarray[DTYPE_t, ndim=1] res = np.zeros(sz) * np.NaN
        object sym
    
    for i in range(0,sz):
        sym = syms[i]
        if v[i]>0:
            if last_info.has_key(sym) and v[i]>last_info[sym][1]:
                res[i] = (t[i] - last_info[sym][0]) / (v[i] - last_info[sym][1]) / ts[i]
            last_info[sym] = (t[i],v[i])
    return res 