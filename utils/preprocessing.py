import sys
sys.path.append('..')

import dask as dask, dask_cudf
from dask.distributed import Client, wait, progress
from dask_cuda import LocalCUDACluster
from utils.cuda_cluster import client
import tensorflow as tf
import core.config as conf

from utils.util import save_memory, split_join

import numpy as np
import cudf, cupy, time
import pandas as pd, numpy as np, gc

from tqdm import tqdm

def read_data(path, type='parquet', index=False, n_partitions=0):
    if type == 'parquet':
        df = dask_cudf.read_parquet(f'{path}/*.parquet', index=False)
        if n_partitions > 0:
            df = df.repartition(npartitions=n_partitions)
        df, = dask.persist(df)
        df = df.set_index('id', drop=True)
        _ = wait(df)
        #print('number of rows:',len(df)) 
        return df
    else:
        print('cannot read data')

def factorize_small_cardinality(df, col, tmp=None, is_several=False):
    df['id'] = df.index
    tmp_col = f'{col}_encode'

    if not is_several:
        tmp = df[col].unique().compute()
    
    idx_to_col = dict(zip(tmp.index.to_array(), tmp.to_array()))
    # idx_to_col = dict(zip(tmp.index.values , tmp.values))
    tmp = tmp.to_frame().reset_index()

    if is_several:
        tmp[col] = tmp[0]
        tmp = tmp.drop(0, axis=1)
    
    df = df.merge(tmp,on=col,how='left')
    df, = dask.persist(df)
    wait(df)
    del tmp

    df[tmp_col] = df['index']
    df = df.drop('index',axis=1)
    df = df.set_index('id', drop=True)
    df, = dask.persist(df)
    wait(df)
    return df, idx_to_col


def factorize_small_cardinality_with_index(df, col, tmp_col):
    tmp = df[col].unique().compute()
    tmp = tmp.to_frame().reset_index()
    df = df.merge(tmp,on=col,how='left')
    df, = dask.persist(df)
    wait(df)
    head=df.head()
    df[tmp_col] = df['index']
    df = df.drop('index', axis=1)
    df, = dask.persist(df)
    wait(df)
    tmp_col_list = ["media_type", "engagement_type", "language_encode", "user_encode", "tweet_id_encode"]
    if tmp_col in tmp_col_list :
        index_list = tmp
        del tmp
        return df, index_list, head
    del tmp
    return df,head


def get_media_index(media_index) :
    media_index = media_index.to_pandas()
    media_index["media"] = media_index['present_media'].apply(lambda x:  split_join(x,'\t'))
    media_index["number_of_Video"] = media_index['media'].apply(lambda x : x.count("Video"))
    media_index["number_of_Photo"] = media_index['media'].apply(lambda x : x.count("Photo"))
    media_index["number_of_GIF"] = media_index['media'].apply(lambda x : x.count("GIF"))
    media_index["number_of_media"] = media_index['media'].apply(lambda x : len(x))
    media_index = media_index.drop('present_media', axis = 1)
    return media_index

def df_to_tfdataset(df, col='label', shuffle=True, batch_size=32):
    df = df.copy()
    labels = df.pop(col)
    ds = tf.data.Dataset.from_tensor_slices((df.to_pandas().to_dict('series'), labels.to_array()))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


def most_frequent_token(df, col, idcol, outcol, sep='\t'):
    df['id'] = df.index        
    ds = df[[idcol,col]].compute()
    tokens = ds[col].str.tokenize(sep)
    
    token_counts = ds[col].str.token_count('\t').to_frame()
    token_counts[idcol] = ds[idcol]
    del ds
    token_rows = reconstruct_token_rows(token_counts,col,idcol)
    del token_counts
    
    tokens = tokens.to_frame()
    tokens.columns = ['token']
    tokens['token'],_ = tokens['token'].factorize()
    tokens['token'] = tokens['token']+1
    tokens[idcol] = token_rows
    
    global_token_counts = tokens['token'].value_counts().reset_index()
    global_token_counts.columns = ['token','count']
    
    tokens = tokens.merge(global_token_counts,on='token',how='left')
    del global_token_counts
    
    token_counts = tokens.groupby(idcol).agg({'count':'max'}).reset_index()
    tokens = tokens.merge(token_counts,on=idcol,how='left')
    del token_counts
    mask = tokens.count_x == tokens.count_y
    tokens = tokens[mask]
    tokens = tokens.drop(['count_x','count_y'],axis=1)
    tokens = tokens.drop_duplicates(subset=['token',idcol])
    tokens = tokens.rename(columns = {'token':outcol})

    
    if outcol == col:
        df = df.drop(col,axis=1)
    df = df.merge(tokens,on=idcol,how='left')
    df[outcol] = df[outcol].fillna(0)
    del tokens
    return df


def reconstruct_token_rows(token_counts, col, idcol):
    ids = cupy.asnumpy(token_counts[idcol].values)
    rep = cupy.asnumpy(token_counts[col].values)
    token_rows = np.repeat(ids,rep)
    token_rows = cupy.asarray(token_rows)
    return token_rows


def feature_extraction(df, features, labels):
    df = df[features + labels]

    # for labels
    for label in (labels):
        label_name = label.split('_')[0]
        df[label_name] = df[label].compute().applymap(lambda x: 1 if x > 0 else 0).astype(np.int32)
        df = df.drop(label, axis=1)

    # for timestamp
    df['dt_day'] = df['tweet_timestamp'].astype('datetime64[s]').dt.day
    df['dt_dow'] = df['tweet_timestamp'].astype('datetime64[s]').dt.dayofweek
    df['dt_hour'] = df['tweet_timestamp'].astype('datetime64[s]').dt.hour    

    save_memory(df)

    if 'media' in (df.columns):
        df['media'] = df['media'].fillna('')
        df['media'] = df['media'].map_partitions( lambda x:  split_join(x,'\t'), meta=('O'))

    for col in (['language','tweet_type','media']):
        if col in df.columns:
            df,_ = factorize_small_cardinality(df,col)
            df = df.drop(col, axis=1)
            df = df.rename(columns = {f'{col}_encode':f'{col}'})
            df, = dask.persist(df)
            _ = wait(df)

    for col in (['domains','links','hashtags']):
        if col in df.columns:
            df = most_frequent_token(df, col=col, idcol='id', outcol=col, sep='\t')
            df, = dask.persist(df)
            _ = wait(df)
    
    return df


def tartget_encoding( tra, col, tar, L=1, smooth_method=0  ):
    
    np.random.seed(L)

    cols = col+[tar]
    gf = cudf.from_pandas(tra[cols])
    gf = tra[cols]
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
        gf2['id'] = cupy.arange( gf2.shape[0] ).get()
        gf2 = gf2.join( te, how='left' )
        gf2 = gf2.sort_values( 'id' )
        del te
        del gf2['id']
        predtrain[py] = gf2.smooth.fillna(-999)
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
    gf2['id'] = cupy.arange( gf2.shape[0] ).get()
    gf2 = gf2.join( te, how='left' )
    gf2 = gf2.sort_values( 'id' )
    del te
    del gf2['id']
    predtrain[py] = gf2.smooth.fillna(-999)
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
    gf2['id'] = cupy.arange( gf2.shape[0] ).get()
    gf2 = gf2.join( te, how='left' )
    gf2 = gf2.sort_values( 'id' )
    del te
    del gf2['id']
    predtrain[py] = gf2.smooth.fillna(-999)
    del gf2
    
    _ = gc.collect()
    predtrain[predtrain <= -999 ] = np.nan
    return predtrain.astype(np.float32)

def save_parquet(df, file_path):
    df.to_parquet( f'{file_path}' )
    gc.collect()
