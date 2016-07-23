import numpy as np
cimport numpy as np
import pandas as pd
from os import listdir
from kit_kospi import liquid_underlying,wmid
cimport cython

include "../ml_utils/ml_utils.pyx"

def kospi_futures_features(d,futs,AM_exclude_seconds=180,PM_exclude_seconds=120,score_seconds=5):
    from kit_kospi import net_effect_cython
    from tsutil import streaker,streaker_with_refs
    from kit_kospi import rolling_vwap_cython,joins
    
    t1 = pd.Timestamp(d+'T09:00:00',tz='Asia/Seoul') + np.timedelta64(AM_exclude_seconds,'s')
    t2 = pd.Timestamp(d+'T15:05:00',tz='Asia/Seoul')  -  np.timedelta64(AM_exclude_seconds,'s')
    
    f = futs.ix[t1:t2,].copy()
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

def read_data(md_root = "/home/jgreenwald/md/"):
    all_data = []
    
    for f in list_h5s(md_root):
        d = f[:-3]
        try:
            store = pd.HDFStore(md_root+d+'.h5','r')
            hi = store['md']
            liquid_future = liquid_underlying(hi.symbol)
            futs = hi.ix[hi.symbol == liquid_future]
            store.close()
            all_data.append( kospi_futures_features(d,futs))
        except:
            store.close()
            raise
    dat = pd.concat(all_data)
    dat = dat.ix[(dat.bz0 > 0) & (dat.az0 >0)]
    return dat