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

class Ensemble_FFNN_ALL:
    def __init__(self, df, TARGET_id):
        super().__init__()
        self.df = df
        self.TARGET_id = TARGET_id
        self.TARGETS = ['reply', 'retweet', 'comment', 'like']
        self.LR = [0.05,0.03,0.07,0.01]
        self.default_values = conf.default_values
        
                       
    def fill_with_default_value(self, df, ensemble_num=0):
                
        default_values = self.default_values
        
        tmp = df.sample(frac=0.1, random_state=conf.random_states[ensemble_num])
        
        for key in default_values.keys():
            tmp[key] = default_values[key]
            
        df.loc[tmp.index] = tmp.loc[tmp.index]
        
        return df


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
    
        
    
    def train(self, ensemble_num=0):
        input_dim = 25 #17
        
        self.models = [Sequential([
            Dense(16, activation = 'relu', input_dim = input_dim),
            Dense(8, activation = 'relu'),
            Dense(4, activation = 'relu'),
            Dense(1, activation = 'sigmoid')
        ])] * 4
        
        for model in self.models :
            model.compile(optimizer = 'adam',
                          loss = 'binary_crossentropy', 
                          metrics=['binary_crossentropy']) 
            
        for i, train in tqdm(enumerate(self.df)):
            for j in range(5):
                print(f'------ train ensemble model {j} ------')
                self._train(train, i, j)


    def _train(self, train, i, ensemble_num):


        train = self.fill_with_default_value(train, ensemble_num)
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
            model = self.models[idx]
            model.fit(x = X_train,
                      y = y_train,
                      validation_split=0.2,
                      epochs = 1,
                      batch_size=32) 

            #save model
            model_path = f'{conf.model_path}/ensemble-{ensemble_num}/ffnn--{target}-{i}'
            model.save(model_path)

            del X_train
            del y_train
        del Xt_train
        del yt_train

        gc.collect()  

            

    def predict(self, model_path, model_num=0):
        TARGET = self.TARGETS[self.TARGET_id]
        valid = self.df
        
        RMV = self.feature_extract(valid)
        X_valid = valid.drop(RMV, axis=1)
        
        del valid
        
        X_valid = self.scaling(X_valid, False)
        X_valid = X_valid.drop(conf.drop_features[self.TARGET_id], axis = 1)
        
        gc.collect()
        
        result = []
        for i in range(5):
            print(f'------ predict ensemble model {i} ------')
                             
            model = tf.keras.models.load_model(f'{conf.model_path}/ensemble-{i}/ffnn--{TARGET}-{model_num}')
            pred = model.predict(X_valid)
            result.append(pred)
            
        result = np.mean(result, axis=0)
        
        return result
        