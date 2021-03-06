import sys
sys.path.append('..')

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, gc
from tqdm import tqdm
import joblib

from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import save_model,load_model
import tensorflow as tf

pd.set_option('display.max_columns', 500)

from utils.util import *
from utils.evaluate import calculate_ctr, compute_rce, average_precision_score
from utils.dataset import *

import core.config as conf

class FFNN_ALL_DEFAULT:
    def __init__(self, df, TARGET_id='reply'):
        super().__init__()
        self.df = df
        self.TARGET_id = TARGET_id
#         if TARGET_id != 4 :
#             self.TARGETS = ['reply', 'retweet', 'comment', 'like'][self.TARGET_id]
#             self.LR = [0.05,0.03,0.07,0.01][self.TARGET_id]
#         else :
        self.TARGETS = ['reply', 'retweet', 'comment', 'like']
        self.LR = [0.05,0.03,0.07,0.01]
        self.default_values = {'engager_feature_number_of_previous_like_engagement': 16.68406226808318,
                             'engager_feature_number_of_previous_reply_engagement': 3.9166628750988446,
                             'engager_feature_number_of_previous_retweet_engagement': 7.943690435417255,
                             'engager_feature_number_of_previous_comment_engagement': 2.397117827194066,
                             'creator_feature_number_of_previous_like_engagement': 18.650278982078916,
                             'creator_feature_number_of_previous_reply_engagement': 4.005221886495085,
                             'creator_feature_number_of_previous_retweet_engagement': 8.378531979240039,
                             'creator_feature_number_of_previous_comment_engagement': 2.465194979899623,
                             'creator_number_of_engagements_positive': 8.374806956928415,
                             'number_of_engagements_positive': 7.735383351448337,
                             'number_of_engagements_ratio_like': 2.1568500887495556,
                             'number_of_engagements_ratio_retweet': 1.0269291222560957,
                             'number_of_engagements_ratio_reply': 0.5063308044539909,
                             'number_of_engagements_ratio_comment': 0.30988998454035777,
                             'creator_number_of_engagements_ratio_like': 2.226950313959139,
                             'creator_number_of_engagements_ratio_retweet': 1.0004447890358288,
                             'creator_number_of_engagements_ratio_reply': 0.4782464726761964,
                             'creator_number_of_engagements_ratio_comment': 0.29435842432883613,
                             'creator_main_language': 0,
                             'engager_main_language': 0,
                             'is_tweet_in_creator_main_language': 0.5,
                             'is_tweet_in_engager_main_language': 0.5,
                             'creator_and_engager_have_same_main_language': 0.5,
                             'number_of_tweet_like': 1.2568421511772854,
                             'number_of_tweet_reply': 1.0636431744005945,
                             'number_of_tweet_retweet': 1.0998849163606805,
                             'number_of_tweet_comment': 1.0363500723064045,
                             'number_of_tweet_engagements': 1.3275094196173265
                              }
                       

    def feature_extract(self, train=True):
        label_names = ['reply', 'retweet', 'comment', 'like']
        DONT_USE = ['tweet_timestamp','creator_account_creation','engager_account_creation','engage_time',
                    'creator_account_creation', 'engager_account_creation',
                    'fold','tweet_id', 
                    'tr','dt_day','','',
                    'engager_id','creator_id','engager_is_verified',
                    'elapsed_time',
                    'links','domains','hashtags0','hashtags1',
                    'hashtags','tweet_hash','dt_second','id',
                    'tw_hash0',
                    'tw_hash1',
                    'tw_rt_uhash',
                    'same_language', 'nan_language','language',
                    'tw_hash', 'tw_freq_hash','tw_first_word', 'tw_second_word', 'tw_last_word', 'tw_llast_word',
                    'ypred','creator_count_combined','creator_user_fer_count_delta_time','creator_user_fing_count_delta_time','creator_user_fering_count_delta_time','creator_user_fing_count_mode','creator_user_fer_count_mode','creator_user_fering_count_mode'
                ]
        DONT_USE += label_names
        DONT_USE += conf.labels
        return [c for c in DONT_USE if c in train.columns]
    
    def scaling(self, df, TRAIN):
        print(df.columns)
        scaling_columns = ['creator_following_count', 'creator_follower_count', 'engager_follower_count', 
                           'engager_following_count', 'dt_dow', 'dt_hour', 'len_domains', 'creator_main_language', 
                           'engager_main_language',
                           'engager_feature_number_of_previous_like_engagement',
                           'engager_feature_number_of_previous_reply_engagement',
                           'engager_feature_number_of_previous_retweet_engagement',
                           'engager_feature_number_of_previous_comment_engagement',
                           'number_of_engagements_positive',
                           'creator_feature_number_of_previous_like_engagement',
                           'creator_feature_number_of_previous_reply_engagement',
                           'creator_feature_number_of_previous_retweet_engagement',
                           'creator_feature_number_of_previous_comment_engagement',
                           'creator_number_of_engagements_positive',
                           'len_text_tokens',
                           'len_text_tokens_unique',
                           'cnt_mention',
                           'number_of_tweet_engagements',
                           'number_of_tweet_like',
                           'number_of_tweet_reply',
                           'number_of_tweet_retweet',
                           'number_of_tweet_comment',
                           ]
        df = df.reset_index(drop=True)

        if TRAIN:
            standard_scaler = preprocessing.StandardScaler()
            
            standard_scaler.fit(df[scaling_columns])
            pickle.dump(standard_scaler, open(conf.scaler_path + 'scaler.pkl','wb'))
        else :
            standard_scaler = pickle.load(open(conf.scaler_path + 'scaler.pkl', 'rb'))
            
        sc = standard_scaler.transform(df[scaling_columns])
        df[scaling_columns] = pd.DataFrame(sc, columns = scaling_columns)
        df = df.fillna(df.mean())
        return df
    
    def train(self):
        input_dim = 25 #17

        models = [Sequential([
            Dense(16, activation = 'relu', input_dim = input_dim),
            Dense(8, activation = 'relu'),
            Dense(4, activation = 'relu'),
            Dense(1, activation = 'sigmoid')
        ])] * 4
        
        for model in models :
            model.compile(optimizer = 'adam',
                          loss = 'binary_crossentropy', # softmax : sparse_categorical_crossentropy, sigmoid : binary_crossentropy
                          metrics=['binary_crossentropy']) # sigmoid :binary_crossentropy

        for i, train in tqdm(enumerate(self.df)):
            RMV = self.feature_extract(train)
            yt_train = train[self.TARGETS]
            Xt_train = train.drop(RMV, axis=1)
            del train
            
            Xt_train = self.scaling(Xt_train, True)
            
            
            gc.collect()
            
            for target in self.TARGETS :
                print(target, self.TARGETS)
                idx = conf.target_to_idx[target]
                X_train = Xt_train.drop(conf.drop_features[idx], axis = 1)
                y_train = yt_train[target]
                model = models[idx]
                model.fit(x = X_train,
                          y = y_train,
                          validation_split=0.2,
                          epochs = 1,
                          batch_size=32) 

                #save model
                model_path = f'{conf.model_path}/ffnn--{target}-{i}'
                model.save(model_path)
                
                del X_train
                del y_train
            del Xt_train
            del yt_train

            gc.collect()  

    def predict(self, model_path):
        TARGET = self.TARGETS[self.TARGET_id]
        valid = self.df
        
        RMV = self.feature_extract(valid)
        X_valid = valid.drop(RMV, axis=1)
        
        del valid
        
        X_valid = self.scaling(X_valid, False)
        X_valid = X_valid.drop(conf.drop_features[self.TARGET_id], axis = 1)
        
        gc.collect()
                             
        model = tf.keras.models.load_model(f'{conf.model_path}ffnn--{TARGET}-0')
        print(X_valid.shape)

        pred = model.predict(X_valid)
        
        return pred
        