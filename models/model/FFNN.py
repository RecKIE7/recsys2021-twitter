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

class FFNN:
    def __init__(self, df, TARGET_id):
        super().__init__()
        self.df = df
        self.TARGET_id = TARGET_id
        self.TARGETS = ['reply', 'retweet', 'comment', 'like']
        self.LR = [0.05,0.03,0.07,0.01]
                 
    def feature_extract(self, df, train=True):
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
        if train:
            DONT_USE += label_names
            DONT_USE += conf.labels
        return [c for c in DONT_USE if c in df.columns]
    
    def scaling(self, df, target, TRAIN):
        scaling_columns = ['creator_following_count', 'creator_follower_count', 'engager_follower_count', 
                           'engager_following_count', 'dt_dow', 'dt_hour', 'len_domains', 'creator_main_language', 'engager_main_language',
                           f'engager_feature_number_of_previous_{target}_engagement', 'number_of_engagements_positive']
        
        df = df.reset_index(drop=True)

        if TRAIN or True :
            standard_scaler = preprocessing.StandardScaler()
            standard_scaler.fit(df[scaling_columns])
            pickle.dump(standard_scaler, open(conf.scaler_path + 'scaler_{target}.pkl','wb'))
        else :
            standard_scaler = pickle.load(open(conf.scaler_path + 'scaler_{target}.pkl', 'rb'))
            
        sc = standard_scaler.transform(df[scaling_columns])
        df[scaling_columns] = pd.DataFrame(sc, columns = scaling_columns)
        df = df.fillna(df.mean())
        return df
    
    def train(self):
        model_prev = None

        TARGET = self.TARGETS[self.TARGET_id]
        lr = self.LR[self.TARGET_id]
        input_dim = 17

        model = Sequential([
            Dense(16, activation = 'relu', input_dim = input_dim),
            Dense(8, activation = 'relu'),
            Dense(4, activation = 'relu'),
            Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer = 'adam',
                      loss = 'binary_crossentropy', # softmax : sparse_categorical_crossentropy, sigmoid : binary_crossentropy
                      metrics=['binary_crossentropy']) # sigmoid :binary_crossentropy
        
        for i, train in tqdm(enumerate(self.df)):
            
            RMV = self.feature_extract(train)
            y_train = train[TARGET]
            X_train = train.drop(RMV, axis=1)
            del train
            
            X_train = self.scaling(X_train, TARGET, True)
            
            gc.collect()
            
            model.fit(x = X_train,
                      y = y_train,
                      epochs = 5,
                      batch_size=32) 

            #save model
            model_path = f'/hdd/models/ffnn_pkl/ffnn--{TARGET}-{i}'
            model.save(model_path)
                
            del X_train
            del y_train

            gc.collect()  

    def predict(self, model_path):
        TARGET = self.TARGETS[self.TARGET_id]
        valid = self.df
        RMV = self.feature_extract(valid, train=False)
        X_valid = valid.drop(RMV, axis=1)
        del valid
        
        X_valid = self.scaling(X_valid, TARGET, TRAIN=True)
        print(X_valid.columns)
        
        gc.collect()
                             
        model = tf.keras.models.load_model(f'{model_path}/ffnn_{TARGET}')

        pred = model.predict(X_valid)
        _=gc.collect()
        
        return pred
        