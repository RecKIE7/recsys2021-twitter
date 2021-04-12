import sys
sys.path.append('../..')

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import subprocess
import gc
from transformers import *
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

import core.config as conf


class SplitText(object):
    def __init__(self, file):
        data_path = conf.raw_data_path + file
        df = pd.read_parquet(data_path, engine='pyarrow')
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
            'id'                     ####################
        ]
        df.columns = features
        gc.collect()

        df['tweet'] = [ tokenizer.decode( [ int(n) for n in t.split('\t') ] ) for t in tqdm(df.text_tokens.values) ]
        df['tweet'] = df['tweet'].apply( lambda x: x.replace('https : / / t. co / ', 'https://t.co/') )
        df['tweet'] = df['tweet'].apply( lambda x: x.replace('@ ', '@') )


        path = f'{conf.preproc_path}/step3_output/{file}'
        df.to_parquet(f'{path}/train-tweet-1.parquet' )

        del df




if __name__ == '__main__':
    from utils.cuda_cluster import *
    from utils.dataset import read_data, factorize_small_cardinality

    fire.Fire(SplitText)
