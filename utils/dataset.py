import sys, os
sys.path.append('..')

from tqdm import tqdm
import pickle

from utils.target_encode import MTE_one_shot
from utils.preprocessing import *
import core.config as conf

class Dataset:
    def __init__(self, train=False, target_encoding=conf.target_encoding):
        self.all_features_to_idx = dict(zip(conf.raw_features, range(len(conf.raw_features))))
        self.train = train
        self.target_encoding = target_encoding

    def preprocess(self, df, TARGET_id=conf.LIKE):
        df = self.set_dataframe_types(df)
        df = df.drop('text_tokens', axis=1)
        
        df = feature_extraction(df, features=conf.used_features, train=self.train) 
        target = conf.target[TARGET_id]

        if conf.target_encoding == 1:    
            for c in ([
                ['engager_id'],
                ['engager_id','tweet_type','language'],
                ['creator_id'],
                ['domains','media','tweet_type','language']
                ]):
                fname = 'TE_'+'_'.join(c)+'_'+target
                print( fname )
                df[fname] = tartget_encoding( df, c, target, 20, 0 )

        elif conf.target_encoding == 2:
            for c in tqdm([
                ['engager_id'],
                ['engager_id','tweet_type','language'],
                ['creator_id'],
                ['domains','media','tweet_type','language']
                ]):
                
                out_col = 'TE_'+'_'.join(c)+'_'+target
                encoder_path = f'{out_col}.pkl'
                
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'rb') as f:
                        encoder = pickle.load(f)
                else:
                    encoder = MTE_one_shot(folds=5,smooth=20)

                if self.train:
                    
                    df = encoder.fit_transform(df, c, target, out_col=out_col, out_dtype='float32')
                    with open(encoder_path, 'wb') as f:
                        pickle.dump(encoder, f)
                else:
                    df = encoder.transform(df, c, out_col=out_col, out_dtype='float32')
        
            del encoder
            
        elif conf.target_encoding == 3:
            df['creator_account_creation_group'] = df['creator_account_creation'].apply(grouping, args=(1136041200, 31536000))
            df['creator_follower_count_group'] = df['creator_follower_count'].apply(grouping, args=(0, 0, [10, 100, 1000, 100000]))

            for c in tqdm([
                ['creator_account_creation_group'],
                ['creator_follower_count_group'],
                ['creator_account_creation_group', 'creator_follower_count_group'],
                ]):
                
                out_col = 'TE_'+'_'.join(c)+'_'+target
                encoder_path = f'{out_col}.pkl'
                
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'rb') as f:
                        encoder = pickle.load(f)
                else:
                    encoder = MTE_one_shot(folds=5,smooth=20)

                if self.train:
                    df = encoder.fit_transform(df, c, target, out_col=out_col, out_dtype='float32')
                    with open(encoder_path, 'wb') as f:
                        pickle.dump(encoder, f)
                else:
                    df = encoder.transform(df, c, out_col=out_col, out_dtype='float32')
        
            del encoder

        return df
    
    def raw_preprocess(self, df, TARGET_id=conf.LIKE) : 
        df = self.set_dataframe_types(df)
        labels = ['reply_timestamp', 'retweet_timestamp', 'comment_timestamp', 'like_timestamp']
        df['reply'] = df['reply_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32)
        df['retweet'] = df['retweet_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32)
        df['comment'] = df['comment_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32)
        df['like'] = df['like_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32) 
        df = df.drop(labels, axis=1)
       
        return df
    
    def set_dataframe_types(self, df):
        df['id']   = np.arange( df.shape[0] )
        df['id']   = df['id'].astype(np.uint32)

        if self.train:
            df['reply_timestamp']   = df['reply_timestamp'].fillna(0)
            df['retweet_timestamp'] = df['retweet_timestamp'].fillna(0)
            df['comment_timestamp'] = df['comment_timestamp'].fillna(0)
            df['like_timestamp']    = df['like_timestamp'].fillna(0)

            df['reply_timestamp']   = df['reply_timestamp'].astype(np.uint32)
            df['retweet_timestamp'] = df['retweet_timestamp'].astype(np.uint32)
            df['comment_timestamp'] = df['comment_timestamp'].astype(np.uint32)
            df['like_timestamp']    = df['like_timestamp'].astype(np.uint32)

        df['tweet_timestamp']         = df['tweet_timestamp'].astype( np.uint32 )
        df['creator_follower_count']  = df['creator_follower_count'].astype( np.uint32 )
        df['creator_following_count'] = df['creator_following_count'].astype( np.uint32 )
        df['creator_account_creation']= df['creator_account_creation'].astype( np.uint32 )
        df['engager_follower_count']  = df['engager_follower_count'].astype( np.uint32 )
        df['engager_following_count'] = df['engager_following_count'].astype( np.uint32 )
        df['engager_account_creation']= df['engager_account_creation'].astype( np.uint32 )

        return df
