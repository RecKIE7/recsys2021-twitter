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


def arr_to_series(arr):
    return dask_cudf.Seriese(arr)
