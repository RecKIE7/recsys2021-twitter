import sys
sys.path.append('../..')

import fire

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

class LoadAll(object):
    def __init__(self, file):
        # data_path = conf.raw_data_path + 'part-00175'
        data_path = conf.raw_data_path + file
        df = read_data(data_path, n_partitions=conf.n_partitions)
        features = [
            'text_tokens',    ###############
            'hashtags',       #Tweet Features
            'tweet_id',       #
            'media',          #
            'links',          #
            'domains',        #
            'tweet_type',     #
            'language',       #
            'timestamp',      ###############
            'creator_user_id',              ###########################
            'creator_follower_count',       #Engaged With User Features
            'creator_following_count',      #
            'creator_is_verified',          #
            'creator_account_creation',     ###########################
            'engager_user_id',              #######################
            'engager_follower_count',       #Engaging User Features
            'engager_following_count',      #
            'engager_is_verified',          #
            'engager_account_creation',     #######################
            'engager_follows_creator',    #################### Engagement Features
            'reply',          #Target Reply
            'retweet',        #Target Retweet    
            'retweet_comment',#Target Retweet with comment
            'like',           #Target Like
                            ####################
        ]

        df.columns = features

        df = df.drop('text_tokens', axis=1)
        df, = dask.persist(df)
        _ = wait(df)


        df['id']   = 1
        df['id']   = df['id'].cumsum()
        df['id'] = df['id'].astype('int32')

        df['reply']   = df['reply'].fillna(0)
        df['retweet'] = df['retweet'].fillna(0)
        df['retweet_comment'] = df['retweet_comment'].fillna(0)
        df['like']    = df['like'].fillna(0)

        df['reply']   = df['reply'].astype('int32')
        df['retweet'] = df['retweet'].astype('int32')
        df['retweet_comment'] = df['retweet_comment'].astype('int32')
        df['like']    = df['like'].astype('int32')
        df, = dask.persist(df)
        _ = wait(df)




        df['timestamp']         = df['timestamp'].astype( np.int32 )
        df['creator_follower_count']  = df['creator_follower_count'].astype( np.int32 )
        df['creator_following_count'] = df['creator_following_count'].astype( np.int32 )
        df['creator_account_creation']= df['creator_account_creation'].astype( np.int32 )
        df['engager_follower_count']  = df['engager_follower_count'].astype( np.int32 )
        df['engager_following_count'] = df['engager_following_count'].astype( np.int32 )
        df['engager_account_creation']= df['engager_account_creation'].astype( np.int32 )

        df, = dask.persist(df)
        _ = wait(df)

        train_size = len(df)

        df['media'] = df['media'].fillna('')

        def split_join(ds, sep):
            return ds.str.replace('\t', '_')

        df['media'] = df['media'].map_partitions( lambda x:  split_join(x,'\t'), meta=('O'))

        for col in ['language','tweet_type','media']:
            df,_ = factorize_small_cardinality(df,col)


        df = df.drop('language', axis=1)
        df = df.drop('tweet_type', axis=1)
        df = df.drop('media', axis=1)
        df = df.rename(columns = {'language_encode':'language'})
        df = df.rename(columns = {'tweet_type_encode':'tweet_type'})
        df = df.rename(columns = {'media_encode':'media'})

        tweet = df[['tweet_id']]
        tweet = tweet.drop_duplicates(split_out=16)
        tweet['tweet_encode'] = 1
        tweet['tweet_encode'] = tweet['tweet_encode'].cumsum()
        tweet, = dask.persist(tweet)
        _ = wait(tweet)

        df = df.merge(tweet,on='tweet_id',how='left')
        df = df.drop('tweet_id',axis=1)
        df.columns = [i if i!='tweet_encode' else 'tweet_id' for i in df.columns]
        df, = dask.persist(df)
        wait(df)
        del tweet


        user_a = df[['creator_user_id']].drop_duplicates(split_out=16)
        user_a, = dask.persist(user_a)
        _ = wait(user_a)

        user_b = df[['engager_user_id']].drop_duplicates(split_out=16)
        user_b, = dask.persist(user_b)
        wait(user_b)

        print(len(user_a),len(user_b),len(df))

        user_a.columns = ['user_id']
        user_b.columns = ['user_id']
        user_b['dummy'] = 1
        user_a = user_a.merge(user_b,on='user_id',how='outer')
        user_a = user_a.drop('dummy',axis=1)
        user_a, = dask.persist(user_a)
        wait(user_a)

        print(len(user_a),len(user_b),len(df))
        del user_b

        user_a['user_encode'] = 1
        user_a['user_encode'] = user_a['user_encode'].cumsum()
        user_a, = dask.persist(user_a)
        _ = wait(user_a)

        df = df.merge(user_a,left_on='creator_user_id',right_on='user_id',how='left')
        df = df.drop(['creator_user_id','user_id'],axis=1)
        df.columns = [i if i!='user_encode' else 'creator_user_id' for i in df.columns]
        df, = dask.persist(df)
        _ = wait(df)

        df = df.merge(user_a,left_on='engager_user_id',right_on='user_id',how='left')
        df = df.drop(['engager_user_id','user_id'],axis=1)
        df.columns = [i if i!='user_encode' else 'engager_user_id' for i in df.columns]
        df, = dask.persist(df)
        wait(df)
        del user_a

        df['id']   = 1
        df['id']   = df['id'].cumsum()
        df['id'] = df['id'].astype('int32')

        df = df.repartition(npartitions=conf.n_partitions)
        df, = dask.persist(df)
        _ = wait(df)


        df.to_parquet(f'{conf.preproc_path}/step1_output/{file}',write_index=False)
        print('saved step1_output')

        
        del df
        client.close()
        cluster.close()


if __name__ == '__main__':
    from utils.cuda_cluster import *
    from utils.dataset import read_data, factorize_small_cardinality

    # freeze_support()
    fire.Fire(LoadAll)
    # main()

