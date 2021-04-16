import sys
sys.path.append('..')

import core.config as conf
from utils.util import save_memory, split_join

import numpy as np
import cudf, cupy, time
import pandas as pd, numpy as np, gc
from sklearn.model_selection import KFold

from tqdm import tqdm

def read_data(path, type='csv'):
    if type == 'csv':
        df = pd.read_csv(f'{path}', sep='\x01', header=None, names=conf.raw_features+conf.labels)
        return df
    else:
        print('cannot read data')



def feature_extraction(raw_df, features, train=False):
    labels = conf.labels
    if train:
        df = raw_df[features + labels].copy()
        for label in (labels):
            label_name = label.split('_')[0]
            df.loc[df[label]<=0, label_name ] = 0
            df.loc[df[label]>0, label_name ] = 1
            df = df.drop([label], axis=1)
    else:
        df = raw_df[features].copy()
        for label in (labels):
            label_name = label.split('_')[0]
            # df[label_name] = 0

    del raw_df
    # # for labels
    # for label in (labels):
    #         label_name = label.split('_')[0]
    #         df.loc[df[label]<=0, label_name ] = 0
    #         df.loc[df[label]>0, label_name ] = 1
    #         df = df.drop([label], axis=1)
    #     else:
    #         label_name = label.split('_')[0]
    #         df[label_name] = 0

    # for timestamp
    df['dt_day']  = pd.to_datetime( df['tweet_timestamp'] , unit='s' ).dt.day.values.astype( np.int8 )
    df['dt_dow']  = pd.to_datetime( df['tweet_timestamp'] , unit='s' ).dt.dayofweek.values.astype( np.int8 )
    df['dt_hour'] = pd.to_datetime( df['tweet_timestamp'] , unit='s' ).dt.hour.values.astype( np.int8 )

    save_memory(df)

    if 'media' in (df.columns):
        df['media'] = df['media'].fillna( '' ).copy()
        gc.collect()
        df['media'] = df['media'].apply( lambda x:  '_'.join(x.split('\t')[:2]) )
        gc.collect()

    for col in (['language','tweet_type','media']):
        if col in df.columns:
            df[col] = pd.factorize( df[col], sort=True )[0]
            gc.collect()
            df[col] = df[col].astype( np.uint8 )
            gc.collect()

    # most_frequent_token
    for col in (['domains','links','hashtags']):
        if col in df.columns:
            df = most_frequent_token(df, col=col, sep='\t')
    
    return df

def most_frequent_token(df, col, sep='\t'):
    df['len_'+col] = df[col].apply(lambda x: str(x).count('\t')+1 if not(pd.isnull(x)) else 0)
    var = df[col].fillna('').values.copy()
    gc.collect()

    PD = {}
    null_value = ''
    PD[null_value] = [0,0]
    count = 1
    for vs in var:
        if vs != null_value:
            for v in vs.split('\t'):
                if v not in PD:
                    PD[v] = [count,1]
                    count +=1
                else:
                    x = PD[v]
                    x[1] += 1
                    PD[v] = x
        else:
            x = PD[null_value]
            x[1] += 1
            PD[null_value] = x

    vari = []
    for vs in var:
        if vs != null_value:
            li=[]
            lf=[]
            for v in vs.split('\t'):
                if v!='':
                    li.append(PD[v][0])
                    lf.append(-PD[v][1])
            vari.append( list(np.array(li)[np.argsort(lf)].astype(np.int32) ) )
        else:
            vari.append( [0] )
    del PD

    gc.collect()
    df[col] = np.array( [v[0] for v in vari ] ).astype( np.int32 )
    gc.collect()
    del vari
    del var
    gc.collect()

    return df

    

def tartget_encoding( tra, col, tar, L=1, smooth_method=0):
    np.random.seed(L)

    cols = col+[tar]
    gf = cudf.from_pandas(tra[cols])
    mn = gf[tar].mean().astype('float32')
    
    predtrain = np.zeros( tra.shape[0] )
    
    for fold in [7,8,9,10,11,12]:
        px = np.where( tra.dt_day <fold )[0]
        py = np.where( tra.dt_day==fold )[0]
        mn = gf[tar].iloc[px].mean().astype('float32')
        if smooth_method==0:
            te = gf.iloc[px].groupby(col)[tar].agg(['mean','count'])
            te['smooth']  = (te['mean']*te['count'])
            te['smooth'] += (mn*L)
            te['smooth'] /= (te['count']+L)
            te = te.drop( ['mean','count'], axis=1 )
        elif smooth_method==1:
            te = gf.iloc[px].groupby(col)[tar].agg(['sum','count'])
            te['smooth'] = (te['sum']+L) / (te['count']+1)
            te = te.drop( ['sum','count'], axis=1 )
        gf2 = gf.iloc[py].copy()
        gf2 = gf2.set_index( col )
        gf2['id'] = cupy.arange( gf2.shape[0] )
        gf2 = gf2.join( te, how='left' )
        gf2 = gf2.sort_values( 'id' )
        del te
        del gf2['id']
        predtrain[py] = gf2.smooth.fillna(-999).to_array()
        del gf2

    px = np.where( tra.dt_day <13 )[0]
    py = np.where( tra.dt_day>=13 )[0]
    mn = gf[tar].iloc[px].mean().astype('float32')
    if smooth_method==0:
        te = gf.iloc[px].groupby(col)[tar].agg(['mean','count'])
        te['smooth']  = (te['mean']*te['count'])
        te['smooth'] += (mn*L)
        te['smooth'] /= (te['count']+L)
        te = te.drop( ['mean','count'], axis=1 )
    elif smooth_method==1:
        te = gf.iloc[px].groupby(col)[tar].agg(['sum','count'])
        te['smooth'] = (te['sum']+L) / (te['count']+1)
        te = te.drop( ['sum','count'], axis=1 )
    gf2 = gf.iloc[py].copy()
    gf2 = gf2.set_index( col )
    gf2['id'] = cupy.arange( gf2.shape[0] )
    gf2 = gf2.join( te, how='left' )
    gf2 = gf2.sort_values( 'id' )
    del te
    del gf2['id']
    predtrain[py] = gf2.smooth.fillna(-999).to_array()            
    del gf2

    px = np.where( (tra.dt_day>=7)&(tra.dt_day<=11) )[0]
    py = np.where( tra.dt_day==6 )[0]
    mn = gf[tar].iloc[px].mean().astype('float32')
    if smooth_method==0:
        te = gf.iloc[px].groupby(col)[tar].agg(['mean','count'])
        te['smooth']  = (te['mean']*te['count'])
        te['smooth'] += (mn*L)
        te['smooth'] /= (te['count']+L)
        te = te.drop( ['mean','count'], axis=1 )
    elif smooth_method==1:
        te = gf.iloc[px].groupby(col)[tar].agg(['sum','count'])
        te['smooth'] = (te['sum']+L) / (te['count']+1)
        te = te.drop( ['sum','count'], axis=1 )
    gf2 = gf.iloc[py].copy()
    gf2 = gf2.set_index( col )
    gf2['id'] = cupy.arange( gf2.shape[0] )
    gf2 = gf2.join( te, how='left' )
    gf2 = gf2.sort_values( 'id' )
    del te
    del gf2['id']
    predtrain[py] = gf2.smooth.fillna(-999).to_array()            
    del gf2
    
    _ = gc.collect()
    predtrain[predtrain <= -999 ] = np.nan
    return predtrain.astype(np.float32)


def target_encode_cudf_v3(train, valid, col, tar, n_folds=5, min_ct=0, smooth=20, 
                          seed=42, shuffle=False, t2=None, v2=None, x=-1):
    #
    # col = column to target encode (or if list of columns then multiple groupby)
    # tar = tar column encode against
    # if min_ct>0 then all classes with <= min_ct are consider in new class "other"
    # smooth = Bayesian smooth parameter
    # seed = for 5 Fold if shuffle==True
    # if x==-1 result appended to train and valid
    # if x>=0 then result returned in column x of t2 and v2
    #    
    
    # SINGLE OR MULTIPLE COLUMN
    if not isinstance(col, list): col = [col]
    if (min_ct>0)&(len(col)>1): 
        print('WARNING: Setting min_ct=0 with multiple columns. Not implemented')
        min_ct = 0
    name = "_".join(col)
        
    # FIT ALL TRAIN
    gf = cudf.from_pandas(train[col+[tar]]).reset_index(drop=True)
    gf['idx'] = gf.index #needed because cuDF merge returns out of order
    if min_ct>0: # USE MIN_CT?
        other = gf.groupby(col[0]).size(); other = other[other<=min_ct].index
        save = gf[col[0]].values.copy()
        gf.loc[gf[col[0]].isin(other),col[0]] = -1
    te = gf.groupby(col)[[tar]].agg(['mean','count']).reset_index(); te.columns = col + ['m','c']
    mn = gf[tar].mean().astype('float32')
    te['smooth'] = ((te['m']*te['c'])+(mn*smooth)) / (te['c']+smooth)
    if min_ct>0: gf[col[0]] = save.copy()
    
    # PREDICT VALID
    gf2 = cudf.from_pandas(valid[col]).reset_index(drop=True); gf2['idx'] = gf2.index
    if min_ct>0: gf2.loc[gf2[col[0]].isin(other),col[0]] = -1
    gf2 = gf2.merge(te[col+['smooth']], on=col, how='left', sort=False).sort_values('idx')
    if x==-1: valid[f'TE_{name}_{tar}'] = gf2['smooth'].fillna(mn).astype('float32').to_array()
    elif x>=0: v2[:,x] = gf2['smooth'].fillna(mn).astype('float32').to_array()
    
    # KFOLD ON TRAIN
    tmp = cupy.zeros((train.shape[0]),dtype='float32'); gf['fold'] = 0
    if shuffle: # shuffling is 2x slower
        kf = KFold(n_folds, random_state=seed, shuffle=shuffle)
        for k,(idxT,idxV) in enumerate(kf.split(train)): gf.loc[idxV,'fold'] = k
    else:
        fsize = train.shape[0]//n_folds
        gf['fold'] = cupy.clip(gf.idx.values//fsize,0,n_folds-1)
    for k in range(n_folds):
        if min_ct>0: # USE MIN CT?
            if k<n_folds-1: save = gf[col[0]].values.copy()
            other = gf.loc[gf.fold!=k].groupby(col[0]).size(); other = other[other<=min_ct].index
            gf.loc[gf[col[0]].isin(other),col[0]] = -1
        te = gf.loc[gf.fold!=k].groupby(col)[[tar]].agg(['mean','count']).reset_index(); 
        te.columns = col + ['m','c']
        mn = gf.loc[gf.fold!=k,tar].mean().astype('float32')
        te['smooth'] = ((te['m']*te['c'])+(mn*smooth)) / (te['c']+smooth)
        gf = gf.merge(te[col+['smooth']], on=col, how='left', sort=False).sort_values('idx')
        tmp[(gf.fold.values==k)] = gf.loc[gf.fold==k,'smooth'].fillna(mn).astype('float32').values
        gf.drop_column('smooth')
        if (min_ct>0)&(k<n_folds-1): gf[col[0]] = save.copy()
    if x==-1: train[f'TE_{name}_{tar}'] = cupy.asnumpy(tmp.astype('float32'))
    elif x>=0: t2[:,x] = cupy.asnumpy(tmp.astype('float32'))