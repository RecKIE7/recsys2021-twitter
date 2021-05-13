import sys
sys.path.append('..')

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, gc
from tqdm import tqdm
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import save_model,load_model

pd.set_option('display.max_columns', 500)

from utils.util import *
from utils.evaluate import calculate_ctr, compute_rce, average_precision_score

import core.config as conf

class DNN:
    def __init__(self, df, TARGET_id):
        super().__init__()
        self.df = df
        self.TARGETS = ['reply', 'retweet', 'retweet_comment', 'like']
        self.LR = [0.05,0.03,0.07,0.01]
                 
                 
    def feature_extract(self, train):
        label_names = ['reply', 'retweet', 'retweet_comment', 'like']
        DONT_USE = ['text_tokens', 'hashtags', 'tweet_id', 'links', 'domains', 'language', 'tweet_timestamp', 
                    'creator_id', 'engager_id', 'enaging_user_account_creation', 'id']
        DONT_USE += label_names
        return [c for c in DONT_USE if c in train.columns]
    
    def sparse_encoding(self, train) : # Label Encoding
        sparse_features = ['media', 'tweet_type']
        for feat in sparse_features :
            lbe = LabelEncoder()
            train[feat] = lbe.fit_transform(train[feat])
        return train
    
    def scaling(self, train) :
        col = train.columns
        mms = MinMaxScaler(feature_range = (0, 1))
        train = mms.fit_transform(train)
        train = pd.DataFrame(train, columns = col)
        return train
    
    def train(self, TARGET_id=3):
        model_prev = None
        TARGET = self.TARGETS[TARGET_id]
        lr = self.LR[TARGET_id]
        
        model = Sequential([Dense(64, activation = 'relu', input_dim = 12),
                            Dense(32, activation = 'relu'),
                            Dense(16, activation = 'relu'),
                            Dense(1, activation = 'sigmoid')])
        model.compile(optimizer = optimizers.Adam(learning_rate = lr),
                      loss = 'binary_crossentropy', # softmax : sparse_categorical_crossentropy, sigmoid : binary_crossentropy
                      metrics=['binary_crossentropy'])
                
        for i, train in tqdm(enumerate(self.df)):
            RMV = self.feature_extract(train)
            y_train = train[TARGET]
            X_train = train.drop(RMV, axis=1)
            X_train = self.sparse_encoding(X_train)
            X_train = self.scaling(X_train)
            X_train = X_train.fillna(-1)
            del train
            
            gc.collect()
            
            model.fit(x = X_train,
                          y = y_train,
                          epochs = 1,
                          batch_size=64) 
            #save model
            model_path = f'/hdd/cpu_models_DNN/model-{TARGET}-{i}'
            model.save(model_path)
                
            del X_train
            del y_train

            gc.collect()  

    def predict(self, TARGET_id=3):
        TARGET = self.TARGETS[TARGET_id]
        valid = self.df
        RMV = self.feature_extract(valid)
        model = joblib.load( f'/hdd/cpu_models/model-{TARGET}-288' )
        y_valid = valid[TARGET]
        X_valid = valid.drop(RMV, axis=1)
        X_valid = X_valid.drop(TARGET, axis=1)

        pred = model.predict(X_valid)
        del dvalid
        _=gc.collect()
        
        return pred
        