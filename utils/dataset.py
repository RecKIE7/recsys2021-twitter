import dask as dask, dask_cudf
from dask.distributed import Client, wait, progress
from dask_cuda import LocalCUDACluster
from utils.cuda_cluster import client

def read_data(path, type='parquet', index=False):
    if type == 'parquet':
        df = dask_cudf.read_parquet(f'{path}/*.parquet', index=False)
        df, = dask.persist(df)
        df = df.set_index('id', drop=True)
        _ = wait(df)
        print('number of rows:',len(df)) 
        return df
    else:
        print('cannot read data')



def factorize_small_cardinality(df, col, tmp=None, is_several=False):
    df['id'] = df.index
    tmp_col = f'{col}_encode'

    if not is_several:
        tmp = df[col].unique().compute()
    

    idx_to_col = dict(zip(tmp.index.to_array(), tmp.to_array()))
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

def map_features_to_dict(df, col, sereis):
    df['id'] = df.index
    tmp_col = f'{col}_encode'

    tmp = sereis
    idx_to_col = dict(zip(tmp.index.to_array(), tmp.to_array()))
    tmp = tmp.to_frame().reset_index()
    
    
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
    