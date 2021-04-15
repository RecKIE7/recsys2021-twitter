import sys, os
sys.path.append('..')

from tqdm import tqdm

from utils.preprocessing import *
import core.config as conf

class Dataset:
    def __init__(self, training_flag=True):
        self.all_features_to_idx = dict(zip(conf.raw_features, range(len(conf.raw_features))))


    def preprocess(self, df, TARGET_id=conf.LIKE):
        df = self.set_dataframe_types(df)
        # df = df.set_index('id')
        # df.columns = conf.raw_features + conf.labels
        df = df.drop('text_tokens', axis=1)
        df = feature_extraction(df, features=conf.used_features, labels=conf.labels) # extract 'used_features'

        target = conf.target[TARGET_id]
        for c in ([
            ['engager_id'],
            ['engager_id','tweet_type','language'],
            ['creator_id'],
            ['domains','media','tweet_type','language']
            ]):
            fname = 'TE_'+'_'.join(c)+'_'+target
            print( fname )
            df[fname] = tartget_encoding( df, c, target, 20, 0 )

        return df
    

    def set_dataframe_types(self, df):
        df['id']   = np.arange( df.shape[0] )
        df['id']   = df['id'].astype(np.uint32)

        df['reply_timestamp']   = df['reply_timestamp'].fillna(0)
        df['retweet_timestamp'] = df['retweet_timestamp'].fillna(0)
        df['retweet_with_comment_timestamp'] = df['retweet_with_comment_timestamp'].fillna(0)
        df['like_timestamp']    = df['like_timestamp'].fillna(0)

        df['reply_timestamp']   = df['reply_timestamp'].astype(np.uint32)
        df['retweet_timestamp'] = df['retweet_timestamp'].astype(np.uint32)
        df['retweet_with_comment_timestamp'] = df['retweet_with_comment_timestamp'].astype(np.uint32)
        df['like_timestamp']    = df['like_timestamp'].astype(np.uint32)

        df['tweet_timestamp']         = df['tweet_timestamp'].astype( np.uint32 )
        df['creator_follower_count']  = df['creator_follower_count'].astype( np.uint32 )
        df['creator_following_count'] = df['creator_following_count'].astype( np.uint32 )
        df['creator_account_creation']= df['creator_account_creation'].astype( np.uint32 )
        df['engager_follower_count']  = df['engager_follower_count'].astype( np.uint32 )
        df['engager_following_count'] = df['engager_following_count'].astype( np.uint32 )
        df['engager_account_creation']= df['engager_account_creation'].astype( np.uint32 )

        return df
