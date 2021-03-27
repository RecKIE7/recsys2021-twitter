import dask as dask, dask_cudf
from dask.distributed import Client, wait, progress
from dask_cuda import LocalCUDACluster
from utils.cuda_cluster import client

def read_data(path, type='parquet'):


    if type == 'parquet':
        df = dask_cudf.read_parquet(f'{path}/*.parquet')
        df, = dask.persist(df)
        _ = wait(df)
        print('number of rows:',len(df)) 
        return df
    else:
        print('cannot read data')

