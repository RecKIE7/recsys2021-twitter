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
    tmp_col_list = ["media_type", "engagement_type", "language_encode", "user_encode"]
    if tmp_col in tmp_col_list :
        index_list = tmp
        del tmp
        return df, index_list, head
    del tmp
    return df,head



def split_join(ds,sep):
    df = ds.split(sep)
    return df



def get_media_index(media_index) :
    media_index = media_index.to_pandas()
    media_index["media"] = media_index['present_media'].apply(lambda x:  split_join(x,'\t'))
    media_index["number_of_Video"] = media_index['media'].apply(lambda x : x.count("Video"))
    media_index["number_of_Photo"] = media_index['media'].apply(lambda x : x.count("Photo"))
    media_index["number_of_GIF"] = media_index['media'].apply(lambda x : x.count("GIF"))
    media_index["number_of_media"] = media_index['media'].apply(lambda x : len(x))
    media_index = media_index.drop('present_media', axis = 1)
    return media_index
