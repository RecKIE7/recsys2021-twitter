import sys, os
sys.path.append('../..')

import tqdm
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import optimizers
from deepctr.models import DeepFM as DeepCTR
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.layers import custom_objects

from sklearn import preprocessing
from tqdm import tqdm

from utils.preprocessing import *
import core.config as conf

class DeepFM:
    def __init__(self, df, TARGET_id):
        self.model_name = conf.net_structure
        self.df = df
        self.TARGETS = conf.target
        self.TARGET_id = TARGET_id

        self.sparse_features = conf.sparse_features
        self.dense_features = conf.dense_features

    def feature_extract(self, df):
        DONT_USE = ['text_tokens', 'hashtags', 'tweet_id', 'creator_id', 'engager_id', 'language','links', 'domains', 'id']
        DONT_USE += self.TARGETS
        DONT_USE += conf.labels
        return [c for c in DONT_USE if c in df.columns]
    
    def preprocess(self, df):
        RMV = self.feature_extract(df)
        df = df.drop(RMV, axis=1)
        df = label_encoder(df, self.sparse_features)
        df = scaling(df, self.dense_features)

        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size = df[feat].max() +1, embedding_dim = 4) for feat in self.sparse_features]  + [DenseFeat(feat, 1,) for feat in self.dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        train_model_input = {name:df[name].values for name in feature_names}

        return train_model_input,  fixlen_feature_columns


    def train(self):
        model_prev = None
        TARGET = self.TARGETS[self.TARGET_id]

        for i, train in tqdm(enumerate(self.df)):
            
            train_model_input, fixlen_feature_columns = self.preprocess(train)

            if model_prev:
                model = model_prev
            else:
                model = DeepCTR(fixlen_feature_columns, fixlen_feature_columns, task='binary')
                model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])
            
            history = model.fit(train_model_input, train[TARGET].values,
                batch_size = 256,
                epochs = 1,
                verbose = 1,
                validation_split = 0.2,)

            model_prev = model
            self.save_model(model, i)
            del model

            gc.collect()  


    def save_model(self, model, idx):
        TARGET = self.TARGETS[self.TARGET_id]
        path = f'/hdd/models/{self.model_name}/{self.model_name}_{TARGET}'
        os.makedirs(path, exist_ok=True)
        model.save(f'{path}/{self.model_name}_{TARGET}_{idx}')
    
    
    def predict(self, model='0'):
        TARGET = self.TARGETS[self.TARGET_id]
        path = f'/hdd/models/{self.model_name}/{self.model_name}_{TARGET}'

        test_model_input, fixlen_feature_columns = self.preprocess(self.df)
        model = tf.keras.models.load_model(f'{path}/{self.model_name}_{TARGET}_{model}', custom_objects)
        pred = model.predict(test_model_input, batch_size = 256)

        del model
        _ = gc.collect()

        return pred
