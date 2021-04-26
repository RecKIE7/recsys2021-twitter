import sys
sys.path.append('..')

import core.config as conf
from utils.util import save_memory, split_join

import numpy as np
import cudf, cupy, time
import pandas as pd, numpy as np, gc
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


from tqdm import tqdm

def read_data(path, type='csv'):
    if type == 'csv':
        df = pd.read_csv(f'{path}', sep='\x01', header=None, names=conf.raw_features+conf.labels)
        return df
    else:
        print('cannot read data')


def scaling(df, dense_features):
    mms = MinMaxScaler(feature_range = (0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])
    return df

def label_encoder(df, sparse_features):
    for feat in sparse_features :
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    return df

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

