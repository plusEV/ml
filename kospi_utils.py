from kf_linear import *
import pandas as pd
from kit_kospi import liquid_underlying, twwmid_cython,option_symbols
from kit_kospi import start_time as st
from cubic_regression_spline import crs_vols,altmoneys
from kit_kospi import imp_vols_cython
from kit_kospi import tick_theos_cython_front
import numpy as np

kospi_expiry_dict = {
    'L4' : '20160414',
    'L5' : '20160512',
    'L6' : '20160609',
    'L7' : '20160714',
    'L8' : '20160811',
    'L9' : '20160908'
}

def fetch_expiry(code):
    return kospi_expiry_dict[code]

def dte(code, from_date):
    expiry = fetch_expiry(code)
    return len(pd.bdate_range(pd.Timestamp(from_date),pd.Timestamp(expiry)))
    

def interest_rate(from_date):
    return 0.015

def kospi_kalman_synthetic(smids,strikes,front_sym,return_both=False):
    
    syn_spread = []
    
    for i in range(smids.shape[0]):
        if i<10:
            syn_spread.append(np.NaN)
            continue
        dex = smids.index[i]
        spawt =  smids[front_sym].values[i]
        valid_ics = (smids.ix[dex,np.logical_and(abs(strikes-spawt)<7.5, smids.columns.str[4:6]=='01')])
        valid_ics.sort_index(inplace=True)
        valid_strikes = pd.Series(valid_ics.index).str[8:11].astype(float)
        valid_strikes[valid_strikes%5 != 0 ] += .5
        stacked = (valid_ics.values).reshape((len(valid_ics)/2,2),order='F')
        half_strikes = valid_strikes[0:len(valid_strikes)/2]
        impliedz =(half_strikes + stacked[:,0] - stacked[:,1]).dropna()
        if len(impliedz) < 3:
            syn_spread.append(np.NaN)
            continue
        implied_synthetic = impliedz.mean()
        syn_spread.append(implied_synthetic-spawt)
    #print syn_spread
    skip_early = pd.Series(np.asarray(syn_spread)[10:])
    first_valid = skip_early.first_valid_index()

    kalmaned_synthetic = univariate_filter(pd.Series(syn_spread),skip_early.values[first_valid],.0001,.10)
    if return_both:
        return pd.Series(syn_spread),kalmaned_synthetic
    return kalmaned_synthetic

def resmooth_synthetic(k,und,p):
    und_rolling_changes = und.rolling(window=60,center=False).apply(lambda x: x.max() - x.min())
    k2 = pd.Series(np.zeros(k.shape),index=k.index)
    for i in range(len(k)):
        change = und_rolling_changes.iloc[i]
        if np.isnan(change) or i < 1:
            k2.iloc[i] = k.iloc[i]
            continue

        ratio = min(1.,change / p)
        k2.iloc[i] = (ratio) * k.iloc[i] + (1-ratio) * k2.iloc[i-1]
    return k2

def fit_vols(hi,d,dst=0):
    expiry_counts = hi.symbol.str[6:8].value_counts()
    front_opt_code = expiry_counts.index[0]

    opt_code = front_opt_code
    biz_days = 260.
 
    liquid_future = liquid_underlying(hi.symbol)
    front_opts_index = option_symbols(opt_code,hi.symbol).index
    liquid_futs_index = hi.ix[hi.symbol == liquid_future].index
    joined_dex = liquid_futs_index.append(front_opts_index).sort_values()
    front_md = hi.ix[joined_dex]
    tte = dte(opt_code,d) / biz_days
    ir = interest_rate(d)
    enum_helper = pd.DataFrame({'symbol': sorted(front_md.symbol.unique()), 'id':range(len(front_md.symbol.unique()))})
    ids = front_md.merge(enum_helper, on='symbol',how='left')['id']
    front_md['id'] = ids.values

    start_time = pd.Series(pd.to_datetime([d])).astype('datetime64[ns, Asia/Seoul]') - np.timedelta64(9,'h')#.values[0]
    times = (front_md.index - start_time[0]).astype(np.int64)

    widths = np.asarray(map(lambda x: .05 if x == '41' else .01, front_md.symbol.str[2:4]))
    mids = pd.DataFrame(twwmid_cython(front_md.ix[:,['bp0','bz0','ap0','az0','last','lastsize']].values,\
                                      times / 1000,front_md['id'].values,widths,0)).replace(0,np.NaN)

    mids.index = np.arange(0,len(mids))*1e6 + st(dst)

    column_names = map(lambda x: enum_helper.ix[enum_helper.id == x].symbol.values[0], mids.columns)
    mids.columns = column_names
    mids.sort_index(inplace=True,axis=1)

    strikes = map(lambda x: x + 0.5 if x % 5 !=0 else x, mids.columns.str[8:11].astype(float))
    types = map(lambda x:1 if x == '42' else -1 if x == '43' else 0, mids.columns.str[2:4])
    
    front_sym = mids.columns[np.where(pd.Series(types) == 0)[0][0]]
    raw, k = kospi_kalman_synthetic(mids, strikes, front_sym,return_both=True)
    und = pd.Series(k.values+mids.ix[:,front_sym].values,index=mids.index)
    k2 = resmooth_synthetic(k,und,8) #optimized to be 8!
    
    vols = pd.DataFrame(imp_vols_cython(mids.values,k2.values+mids.ix[:,front_sym].values,\
                                    np.asarray(strikes),np.asarray(types),tte,ir),columns=mids.columns,index=mids.index).replace(-1,np.NaN)


    moneys = pd.DataFrame(altmoneys(k2.values+mids[front_sym].values,np.asarray(strikes),tte),columns=vols.columns)
    splined_vols = crs_vols(vols,moneys,np.asarray([]),degree=3)
    
    
    tick_theos,tick_deltas, tick_vegas, tick_vols, tick_futs = tick_theos_cython_front(\
        front_md.ix[:,['bp0','bz0','ap0','az0','last','lastsize']].values,times / 1000,\
        front_md['id'].values,vols.values,vols.index.astype(long).values,k2.values,
        np.asarray(strikes),np.asarray(types),tte,ir)
    
    und = pd.Series(k2.values+mids.ix[:,front_sym].values,index=vols.index)
    return front_md, vols, und, raw, k ,moneys, splined_vols, [tick_theos,tick_deltas,tick_vegas,tick_vols,tick_futs]