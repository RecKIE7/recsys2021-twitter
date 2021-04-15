import sys
sys.path.append('..')

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, gc
from tqdm import tqdm
import joblib
import cudf, cupy, time
import xgboost as xgb

pd.set_option('display.max_columns', 500)

from utils.util import *
from utils.evaluate import calculate_ctr, compute_rce, average_precision_score
from utils.dataiter import Dataiter

import core.config as conf

class XGBoost:
    def __init__(self, df):
        self.df = df
        self.xgb_parms = { 
                'max_depth':8, 
                'learning_rate':0.025, 
                'subsample':0.85,
                'colsample_bytree':0.35, 
                'eval_metric':'logloss',
                'objective':'binary:logistic',
                'tree_method':'gpu_hist',
                #'predictor': 'gpu_predictor',
                'seed': 1,
            }
        self.TARGETS = ['reply', 'retweet', 'retweet_comment', 'like']
        self.LR = [0.05,0.03,0.07,0.01]
    
    def feature_extract(self, train):
        label_names = ['reply', 'retweet', 'retweet_comment', 'like']
        DONT_USE = ['timestamp','creator_account_creation','engager_account_creation','engage_time',
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
        return [c for c in DONT_USE if c in train.columns]
    
    def incremental_train(self, TARGET_id=3):
        model_prev = None
        TARGET = self.TARGETS[TARGET_id]
        self.xgb_parms['learning_rate'] = self.LR[TARGET_id]

        for i, train in tqdm(enumerate(self.df)):
            
            RMV = self.feature_extract(train)

            dtrain = xgb.DMatrix(data=train.drop(RMV, axis=1).compute().to_pandas(),
                                label=train[TARGET].compute().values)

            gc.collect()

            if model_prev:
                model = xgb.train(self.xgb_parms, 
                                dtrain=dtrain,
                                num_boost_round=500,
                                xgb_model=model_prev,
                                ) 

            else:
                model = xgb.train(self.xgb_parms, 
                                dtrain=dtrain,
                                num_boost_round=500,
                                ) 
            del dtrain
            gc.collect()  

            #save model
            model_path = f'/hdd/models/model-{TARGET}-{i}.xgb'
            joblib.dump(model, model_path) 
            model_prev = model
            del model
            gc.collect()  

            # print(f'saved models - {path}')




            
        