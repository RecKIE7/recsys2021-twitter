import sys
sys.path.append('../..')

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import subprocess

import dask.multiprocessing
dask.config.set(schedular='process')

import core.config as conf


class SplitNew(object):
    def __init__(self, file):
        data_path = f'{conf.preproc_path}/step1_output/{file}'
        df = dask_cudf.read_parquet(f'{data_path}/*.parquet')
        df, = dask.persist(df)
        _ = wait(df)

        def count_token(ds,token):
            not_null = ds.isnull()==0
            return ((ds.str.count(token)+1)*not_null).fillna(0)

        df['len_hashtags'] = df['hashtags'].map_partitions(lambda ds: count_token(ds,'\t'))
        df['len_domains']  = df['domains'].map_partitions(lambda ds: count_token(ds,'\t'))
        df['len_links']    = df['links'].map_partitions(lambda ds: count_token(ds,'\t'))

        df, = dask.persist(df)
        _ = wait(df)

        def most_frequent_token(df, col, idcol, outcol, sep='\t'):
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

        def reconstruct_token_rows(token_counts,col,idcol):
            ids = cupy.asnumpy(token_counts[idcol].values)
            rep = cupy.asnumpy(token_counts[col].values)
            token_rows = np.repeat(ids,rep)
            token_rows = cupy.asarray(token_rows)
            return token_rows


        for col in ['domains','links','hashtags']: # 
            df = most_frequent_token(df, col=col, idcol='id', outcol=col, sep='\t')
            df, = dask.persist(df)
            _ = wait(df)

        TRAIN_SIZE = len(df)

        train = df[df['id']<=TRAIN_SIZE] 


        path = f'{conf.preproc_path}/step2_output/{file}'
        train.to_parquet(f'{path}/train.parquet', write_index=False)


        del train
        del df
        client.close()
        cluster.close()


if __name__ == '__main__':
    fire.Fire(SplitNew)
