import numpy as np
cimport numpy as np
import pandas as pd
from os import listdir
from kit_kospi import liquid_underlying,wmid
from kit_kospi import net_effect_cython
from tsutil import streaker,streaker_with_refs
from kit_kospi import net_effect_cython
from tsutil import streaker,streaker_with_refs
from kit_kospi import rolling_vwap_cython,joins
cimport cython

include "../ml_utils/ml_utils.pyx"
ctypedef np.double_t DTYPE_t
cdef extern from "math.h":
    bint isnan(double x)

cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b

cdef inline double abz(double a) : return a if a >= 0. else -1 * a
def opposing_deltas_cython(np.ndarray[double,ndim=2] implieds_info,np.ndarray[double,ndim=2] streaker_prices):
    cdef:
        long i_len = implieds_info.shape[0], s_len = streaker_prices.shape[0]
        long i_wid = implieds_info.shape[1]
        double buy_ref,sell_ref
        double bid_deltas_through,ask_deltas_through
        long i,j
        np.ndarray[DTYPE_t,ndim=2] res = np.zeros([i_len,2], dtype=np.double) * np.NaN
    assert(i_len == s_len)
    
    for i in range(0,i_len):
        buy_ref = streaker_prices[i][0]
        sell_ref = streaker_prices[i][1]
        bid_deltas_through = 0
        ask_deltas_through = 0
        for j in range(0,i_wid/2,2):
            if (implieds_info[i][j]>sell_ref):#there's a bid THROUGH our streak sell price
                ask_deltas_through+=implieds_info[i][j+1]
        for j in range(10,i_wid,2):
            if (implieds_info[i][j]<buy_ref):#there's a ask THROUGH our streak buy price
                bid_deltas_through+=implieds_info[i][j+1]
        res[i][0] = bid_deltas_through
        res[i][1] = ask_deltas_through
    return res

#bid1,bidsize1,bid2,bidsize2 etc
def deltas_in_face(np.ndarray[double,ndim=1] prices, np.ndarray[double,ndim=2] price_sizes, int buy):
    cdef:
        long p_len = prices.shape[0]
        long ps_len = price_sizes.shape[0]
        np.ndarray[DTYPE_t, ndim=1] res = np.zeros(p_len, dtype=np.double)
    assert(p_len==ps_len)
    if buy>0:
        for i in range(0,p_len):
            if price_sizes[i][0]<prices[i]:
                res[i]+=price_sizes[i][5]
            if price_sizes[i][1]<prices[i]:
                res[i]+=price_sizes[i][6]
            if price_sizes[i][2]<prices[i]:
                res[i]+=price_sizes[i][7]
            if price_sizes[i][3]<prices[i]:
                res[i]+=price_sizes[i][8]
            if price_sizes[i][4]<prices[i]:
                res[i]+=price_sizes[i][9]
    else:
        for i in range(0,p_len):
            if price_sizes[i][0]>prices[i]:
                res[i]+=price_sizes[i][5]
            if price_sizes[i][1]>prices[i]:
                res[i]+=price_sizes[i][6]
            if price_sizes[i][2]>prices[i]:
                res[i]+=price_sizes[i][7]
            if price_sizes[i][3]>prices[i]:
                res[i]+=price_sizes[i][8]
            if price_sizes[i][4]>prices[i]:
                res[i]+=price_sizes[i][9]
    return res

def implieds_streaker(np.ndarray[long,ndim=1] times, np.ndarray[double,ndim=1] prices, 
    np.ndarray[double,ndim=1] volumes,long window_size, double min_streak_size = 25):
    cdef:
        long t_len = times.shape[0]
        long s_len = prices.shape[0]
        long i =0, win_size = window_size, t_diff, j, window_start
        double tots_buys,tots_sells,agg_buys,agg_sells
        np.ndarray[DTYPE_t, ndim=2] buys = np.zeros([t_len,2], dtype=np.double) * np.NaN
        np.ndarray[DTYPE_t, ndim=2] sells = np.zeros([t_len,2], dtype=np.double) * np.NaN
    assert(t_len==s_len)

    for i in range(1,t_len):
        window_start = times[i] - win_size
        j = i
        tots_buys = 0
        tots_sells = 0
        agg_buys = 0
        agg_sells = 0
        #climb back to find the start of the window
        while times[j]>= window_start and j>=0:
            j-=1
        j+=1
        #now step FORWARD through to calculate the streaks
        while j<=i:
            if prices[j] > 0: #it's a trade
                if volumes[j]>0: #it's a buy 
                    tots_buys+=volumes[j]
                    agg_buys += prices[j]*volumes[j]
                else:
                    tots_sells+=abz(volumes[j])
                    agg_sells+= prices[j]*abz(volumes[j])
            j+=1
        if tots_buys>=min_streak_size:
            agg_buys /= tots_buys
            buys[i][0] = agg_buys
            buys[i][1] = tots_buys
        if tots_sells>=min_streak_size:
            agg_sells /= tots_sells
            sells[i][0] = agg_sells
            sells[i][1] = tots_sells  
    return pd.DataFrame(np.hstack((buys,sells)),columns=['BuyPrc','BuyQty','SellPrc','SellQty'],index=times)

def kospi_implieds_enriched_features(d,front,implieds,AM_exclude_seconds=180,PM_exclude_seconds=120, score_seconds=30):
    implieds = implieds.fillna(method='ffill')
    liquid_future = liquid_underlying(front.symbol)
    futs = front.ix[front.symbol == liquid_future]

    t1 = pd.Timestamp(d+'T09:00:00',tz='Asia/Seoul') + np.timedelta64(AM_exclude_seconds,'s')
    t2 = pd.Timestamp(d+'T15:05:00',tz='Asia/Seoul')  -  np.timedelta64(AM_exclude_seconds,'s')

    f = futs.ix[t1:t2,].copy()
    f = f.ix[(f.bz0 > 0) & (f.az0 >0)]

    original_columns = f.columns

    wmids = pd.Series(map(lambda bp,bs,ap,az: wmid(bp,bs,ap,az,.05), f.bp0,f.bz0,f.ap0,f.az0),index=f.index)
    f['effect'] = net_effect_cython(f.ix[:,['bp0','bz0','ap0','az0','last','lastsize']].values,f['id'].values)
    f.effect.fillna(0,inplace=True)

    buckets = [1,5,10,30,60,300]

    for i,b in enumerate(buckets):
        f['lag'+str(b)] =  wmids.asof(wmids.index - np.timedelta64(i,'s'))
        f['vwap'+str(b)] =rolling_vwap_cython(f.index.astype(long),f['last'].values,f.lastsize.values.astype(np.double),i*1e9,1)
        
    streak_bucks = [1,5,10,30]

    for i,b in enumerate(streak_bucks):
        streak_buys_bid,streak_sells_bid = [x.ravel() for x in \
            np.hsplit(streaker_with_refs(f.index.astype(long),f['last'].values,\
            f.effect.values.astype(long),f.bp0.values,b*1e9),2)]
        streak_buys_ask,streak_sells_ask = [x.ravel() for x in \
            np.hsplit(streaker_with_refs(f.index.astype(long),f['last'].values,\
            f.effect.values.astype(long),f.ap0.values,b*1e9),2)]
        f['streak_buys_bid'+str(b)] = streak_buys_bid
        f['streak_sells_bid'+str(b)] = streak_sells_bid
        f['streak_buys_ask'+str(b)] = streak_buys_ask
        f['streak_sells_ask'+str(b)] = streak_sells_ask

    f['time_of_day'] = (f.groupby(pd.TimeGrouper('5Min'),as_index=False).apply(lambda x: \
                        x['effect'])).index.get_level_values(0).values
    f['total_bids'] = (f.bz4+f.bz3+f.bz2+f.bz1+f.bz0).values
    f['total_asks'] = (f.az4+f.az3+f.az2+f.az1+f.az0).values
    f['width'] = (f.ap0 - f.bp0)    

    def hit_lift_ratio(x):
            return (np.sum(x>0) - np.sum(x<0)) * 1.0 / np.sum(x!=0)
    f['hit_lifts_100'] = f.effect.rolling(100).apply(hit_lift_ratio)
    f['hit_lifts_1000'] = f.effect.rolling(1000).apply(hit_lift_ratio)


    implieds_at_f = implieds.ix[f.index]
    f['face'] = deltas_in_face(wmids.values,implieds_at_f.values[:,10:],1)
    f['supporting'] = deltas_in_face(wmids.values,implieds_at_f.values,-1)
    adjusted_trade_size = (front.delta < 0).astype(int)*-1*front.imp_fut_tz
    adjusted_trade_size[front.delta >= 0] = front.imp_fut_tz

    implied_streak_bucks = [1,5,10]
    for b in implied_streak_bucks:
        streak_info = implieds_streaker(front.index.astype(np.int64), front.imp_fut_tp.values, adjusted_trade_size.values,b*1e9,10)
        streak_info = streak_info.ix[f.index]
        f['streak_implieds_bp'] = streak_info.BuyPrc
        f['streak_implieds_bz'] = streak_info.BuyQty
        f['streak_implieds_ap'] = streak_info.SellPrc
        f['streak_implieds_az'] = streak_info.SellQty

    for i in range(0,5): 
        f['implied_bp'+str(i)] = wmids.values - implieds_at_f.iloc[:,i].values
        f['implied_bz'+str(i)] = implieds_at_f.iloc[:,i+5].values
        f['implied_ap'+str(i)] = wmids.values - implieds_at_f.iloc[:,i+10].values
        f['implied_az'+str(i)] = implieds_at_f.iloc[:,i+15].values

    for c in original_columns:
        del f[c]

    f['score'] = wmids.asof(wmids.index + np.timedelta64(30,'s')).values - wmids.values

    return f

def kospi_futures_longer_term(d,futs,AM_exclude_seconds=180,PM_exclude_seconds=120,score_minutes=2):
    from kit_kospi import net_effect_cython
    from tsutil import streaker,streaker_with_refs
    from kit_kospi import rolling_vwap_cython,joins
    
    t1 = pd.Timestamp(d+'T09:00:00',tz='Asia/Seoul') + np.timedelta64(AM_exclude_seconds,'s')
    t2 = pd.Timestamp(d+'T15:05:00',tz='Asia/Seoul')  -  np.timedelta64(AM_exclude_seconds,'s')
    
    f = futs.ix[t1:t2,].copy()
    f = f.ix[(f.bz0 > 0) & (f.az0 >0)]
    wmids = pd.Series(map(lambda bp,bs,ap,az: wmid(bp,bs,ap,az,.05), f.bp0,f.bz0,f.ap0,f.az0),index=f.index)

    enum_helper = pd.DataFrame({'symbol': f.symbol.unique(), 'id':range(len(f.symbol.unique()))})
    ids = f.merge(enum_helper, on='symbol',how='left')['id']

    f['effect'] = net_effect_cython(f.ix[:,['bp0','bz0','ap0','az0','last','lastsize']].values,ids.values)
    f.effect.fillna(0,inplace=True)

    buckets = [5,10,30,60,120,300,600,1200]
    lags_wmids = []
    vwaps = []
    for i in buckets:
        lags_wmids.append(wmids.asof(wmids.index - np.timedelta64(i,'s')))
        vwaps.append(rolling_vwap_cython(f.index.astype(long),f['last'].values,f.lastsize.values.astype(np.double),i*1e9,1))

    streak_buys_bid5, streak_sells_bid5 = [x.ravel() for x in \
        np.hsplit(streaker_with_refs(f.index.astype(long),f['last'].values,\
        f.effect.values.astype(long),f.bp0.values,5e9),2)]

    streak_buys_ask5, streak_sells_ask5 = [x.ravel() for x in \
        np.hsplit(streaker_with_refs(f.index.astype(long),f['last'].values,\
        f.effect.values.astype(long),f.ap0.values,5e9),2)]

    streak_buys_bid30, streak_sells_bid30 = [x.ravel() for x in \
        np.hsplit(streaker_with_refs(f.index.astype(long),f['last'].values,\
        f.effect.values.astype(long),f.bp0.values,30e9),2)]

    streak_buys_ask30, streak_sells_ask30 = [x.ravel() for x in \
        np.hsplit(streaker_with_refs(f.index.astype(long),f['last'].values,\
        f.effect.values.astype(long),f.ap0.values,30e9),2)]

    f['time_of_day'] = (f.groupby(pd.TimeGrouper('5Min'),as_index=False).apply(lambda x: \
                                                x['effect'])).index.get_level_values(0).values
    f['total_bids'] = (f.bz4+f.bz3+f.bz2+f.bz1+f.bz0).values
    f['total_asks'] = (f.az4+f.az3+f.az2+f.az1+f.az0).values
    f['width'] = (f.ap0 - f.bp0)
                                               
    def hit_lift_ratio(x):
        return (np.sum(x>0) - np.sum(x<0)) * 1.0 / np.sum(x!=0)

    f = f.ix[:,['effect','time_of_day','total_bids','total_asks','width']]
    
    f['hit_lifts_100'] = f.effect.rolling(100).apply(hit_lift_ratio)
    f['hit_lifts_1000'] = f.effect.rolling(1000).apply(hit_lift_ratio)

    for i,b in enumerate(buckets):
        f['lag'+str(b)] = wmids.values - lags_wmids[i].values
        f['vwap'+str(b)] = wmids.values - vwaps[i]

    f['streak_buys_bid5'] = streak_buys_bid5
    f['streak_sells_bid5'] = streak_sells_bid5
    f['streak_buys_ask5'] = streak_buys_ask5
    f['streak_sells_ask5'] = streak_sells_ask5
    f['streak_buys_bid30'] = streak_buys_bid30
    f['streak_sells_bid30'] = streak_sells_bid30
    f['streak_buys_ask30'] = streak_buys_ask30
    f['streak_sells_ask30'] = streak_sells_ask30
    
    f['score'] = wmids.asof(wmids.index + np.timedelta64(score_minutes,'m')).values - wmids.values
    return f


def kospi_futures_features(d,futs,AM_exclude_seconds=180,PM_exclude_seconds=120,score_seconds=5):
    from kit_kospi import net_effect_cython
    from tsutil import streaker,streaker_with_refs
    from kit_kospi import rolling_vwap_cython,joins
    
    t1 = pd.Timestamp(d+'T09:00:00',tz='Asia/Seoul') + np.timedelta64(AM_exclude_seconds,'s')
    t2 = pd.Timestamp(d+'T15:05:00',tz='Asia/Seoul')  -  np.timedelta64(AM_exclude_seconds,'s')
    
    f = futs.ix[t1:t2,].copy()
    f = f.ix[(f.bz0 > 0) & (f.az0 >0)]
    wmids = pd.Series(map(lambda bp,bs,ap,az: wmid(bp,bs,ap,az,.05), f.bp0,f.bz0,f.ap0,f.az0),index=f.index)

    lagged_wmids_1s = wmids.asof(wmids.index - np.timedelta64(1,'s'))
    lagged_wmids_5s = wmids.asof(wmids.index - np.timedelta64(5,'s'))
    lagged_wmids_10s = wmids.asof(wmids.index - np.timedelta64(10,'s'))
    lagged_wmids_30s = wmids.asof(wmids.index - np.timedelta64(30,'s'))

    vwaps_1s = rolling_vwap_cython(f.index.astype(long),f['last'].values,f.lastsize.values.astype(np.double),1e9,1)
    vwaps_5s = rolling_vwap_cython(f.index.astype(long),f['last'].values,f.lastsize.values.astype(np.double),5e9,1)
    vwaps_10s = rolling_vwap_cython(f.index.astype(long),f['last'].values,f.lastsize.values.astype(np.double),1e10,1)
    vwaps_30s = rolling_vwap_cython(f.index.astype(long),f['last'].values,f.lastsize.values.astype(np.double),3e10,1)
    
    enum_helper = pd.DataFrame({'symbol': f.symbol.unique(), 'id':range(len(f.symbol.unique()))})
    ids = f.merge(enum_helper, on='symbol',how='left')['id']

    f['effect'] = net_effect_cython(f.ix[:,['bp0','bz0','ap0','az0','last','lastsize']].values,ids.values)
    f.effect.fillna(0,inplace=True)

    streak_buys_bid, streak_sells_bid = [x.ravel() for x in \
        np.hsplit(streaker_with_refs(f.index.astype(long),f['last'].values,\
        f.effect.values.astype(long),f.bp0.values,1e9),2)]

    streak_buys_ask, streak_sells_ask = [x.ravel() for x in \
        np.hsplit(streaker_with_refs(f.index.astype(long),f['last'].values,\
        f.effect.values.astype(long),f.ap0.values,1e9),2)]
    
    j1_bid,j1_ask = [x.ravel() for x in \
                     np.hsplit(joins(f.index.astype(long), \
                        f.ix[:,['bp0','bz0','ap0','az0','last','lastsize',
                       'bp1','bz1','ap1','az1',
                       'bp2','bz2','ap2','az2',
                       'bp3','bz3','ap3','az3',
                       'bp4','bz4','ap4','az4']].values,1e9),2)]
    
    j5_bid,j5_ask = [x.ravel() for x in \
                     np.hsplit(joins(f.index.astype(long), \
                        f.ix[:,['bp0','bz0','ap0','az0','last','lastsize',
                       'bp1','bz1','ap1','az1',
                       'bp2','bz2','ap2','az2',
                       'bp3','bz3','ap3','az3',
                       'bp4','bz4','ap4','az4']].values,5e9),2)]
    
    f['time_of_day'] = (f.groupby(pd.TimeGrouper('5Min'),as_index=False).apply(lambda x: \
                                                x['effect'])).index.get_level_values(0).values

    def hit_lift_ratio(x):
        return (np.sum(x>0) - np.sum(x<0)) * 1.0 / np.sum(x!=0)
    hit_lifts_100 = f.effect.rolling(100).apply(hit_lift_ratio)
    
    f['total_bids'] = (f.bz4+f.bz3+f.bz2+f.bz1+f.bz0).values
    f['total_asks'] = (f.az4+f.az3+f.az2+f.az1+f.az0).values
    f['width'] = (f.ap0 - f.bp0)
    
    f = f.ix[:,['bz0','az0','effect','time_of_day','total_bids','total_asks','width']]

    f['lagged_wmid1'] = lagged_wmids_1s.values - wmids.values
    f['lagged_wmid5'] = lagged_wmids_5s.values - wmids.values
    f['lagged_wmid10'] = lagged_wmids_10s.values - wmids.values
    f['lagged_wmid30'] = lagged_wmids_30s.values - wmids.values
    
    f['vwap1'] = vwaps_1s - wmids.values
    f['vwap5'] = vwaps_5s - wmids.values
    f['vwap10'] = vwaps_10s - wmids.values
    f['vwap30'] = vwaps_30s - wmids.values
    
    f['streak_buys_bid'] = streak_buys_bid
    f['streak_sells_bid'] = streak_sells_bid
    f['streak_buys_ask'] = streak_buys_ask
    f['streak_sells_ask'] = streak_sells_ask
    
    f['joins_bid_1s'] = j1_bid
    f['joins_ask_1s'] = j1_ask
    f['joins_bid_5s'] = j5_bid
    f['joins_ask_5s'] = j5_ask
    
    f['hits_lifts'] = hit_lifts_100.values
    
    f['score'] = wmids.asof(wmids.index + np.timedelta64(score_seconds,'s')).values - wmids.values
    return f

def read_data(md_root = "/home/jgreenwald/md/", fn = "kospi_futures_longer_term", max_days = 0):
    all_data = []
    h5s = list_h5s(md_root)
    if max_days == 0:
        max_days = len(h5s)
    for f in h5s[:max_days]:
        d = f[:-3]
        try:
            print d
            store = pd.HDFStore(md_root+d+'.h5','r')
            hi = store['md']
            liquid_future = liquid_underlying(hi.symbol)
            futs = hi.ix[hi.symbol == liquid_future]
            store.close()
            all_data.append( eval(fn)(d,futs))
        except:
            store.close()
            raise
    dat = pd.concat(all_data)
    return dat