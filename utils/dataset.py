'''
모든 parquet에 대해서 preprocessing 적용시키기
'''
import sys, os
sys.path.append('..')

from tqdm import tqdm

from utils.preprocessing import *
import core.config as conf

class Dataset():
    def __init__(self, training_flag=True):
        self.all_features_to_idx = dict(zip(conf.raw_features, range(len(conf.raw_features))))

        # save trian datas
        # self.load_data_all(path=conf.raw_lzo_path, save=True, save_dir='/hdd/preprocessing/train/')
        
    def load_data_all(self, path=conf.raw_lzo_path, save=False, save_dir='.'):
        file_list = os.listdir(path)
        file_list = sorted(file_list)

        for file_name in tqdm(file_list):
            if not os.path.exists(save_dir+file_name+'.parquet'):
                client.restart()
                df = dask_cudf.read_csv(f'{path}/{file_name}', sep='\x01', header=None, names=conf.raw_features+conf.labels)
                df = df.repartition(npartitions=conf.n_partitions)
                df = self.lzo_to_dataframe(df)
                df = df.set_index('id', drop=True)
                df, = dask.persist(df)
                _ = wait(df)

                df = self.preprocess(df)

                if save:
                    save_parquet(df, save_dir+file_name+'.parquet')
                
                del df

    def preprocess(self, df):
        df.columns = conf.raw_features + conf.labels
        df = df.drop('text_tokens', axis=1)

        df, = dask.persist(df)
        _ = wait(df)

        features = ['creator_id', 'engager_id', 'tweet_id', 'tweet_type', 'language', 'creator_follower_count', 'creator_following_count', 'domains', 'media', 'tweet_timestamp']
        df = feature_extraction(df, features=features, labels=conf.labels)

        target = 'like' ########### engagement 
        df = df.compute().to_pandas() # to pandas
        for c in ([
            ['engager_id'],
            ['engager_id','tweet_type','language'],
            ['creator_id'],
            ['domains','media','tweet_type','language']
            ]):
            fname = 'TE_'+'_'.join(c)+'_'+target
            print( fname )
            df[fname] = tartget_encoding( df, c, target, 20, 0 )
        df = cudf.from_pandas(df)
        df = dask_cudf.from_cudf(df,  npartitions=conf.n_partitions).reset_index().drop('index', axis=1)

        return df

    def lzo_to_dataframe(self, df):
        df['id']   = 1
        df['id']   = df['id'].cumsum()
        df['id'] = df['id'].astype('int32')

        df['reply_timestamp']   = df['reply_timestamp'].fillna(0)
        df['retweet_timestamp'] = df['retweet_timestamp'].fillna(0)
        df['retweet_with_comment_timestamp'] = df['retweet_with_comment_timestamp'].fillna(0)
        df['like_timestamp']    = df['like_timestamp'].fillna(0)

        df['reply_timestamp']   = df['reply_timestamp'].astype('int32')
        df['retweet_timestamp'] = df['retweet_timestamp'].astype('int32')
        df['retweet_with_comment_timestamp'] = df['retweet_with_comment_timestamp'].astype('int32')
        df['like_timestamp']    = df['like_timestamp'].astype('int32')

        df['tweet_timestamp']         = df['tweet_timestamp'].astype( np.int32 )
        df['creator_follower_count']  = df['creator_follower_count'].astype( np.int32 )
        df['creator_following_count'] = df['creator_following_count'].astype( np.int32 )
        df['creator_account_creation']= df['creator_account_creation'].astype( np.int32 )
        df['engager_follower_count']  = df['engager_follower_count'].astype( np.int32 )
        df['engager_following_count'] = df['engager_following_count'].astype( np.int32 )
        df['engager_account_creation']= df['engager_account_creation'].astype( np.int32 )

        df, = dask.persist(df)
        _ = wait(df)

        return df

    def parse_input_line(self, line):
        features = line.split("\x01")
        tweet_id = features[all_features_to_idx['tweet_id']]
        user_id = features[all_features_to_idx['engaging_user_id']]
        input_feats = features[all_features_to_idx['text_tokens']]
        return tweet_id, user_id, input_feats
