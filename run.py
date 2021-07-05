import os

from utils.dataiter import Dataset
from utils.preprocessing import *
from models.baseline.random_model import random_prediction_model
from models.model.Ensemble_FFNN_ALL import Ensemble_FFNN_ALL
# from models.model.XGBoost import XGBoost
from core.config import raw_features
import core.config as conf
from tqdm import tqdm

all_features_to_idx = dict(zip(raw_features, range(len(raw_features))))
float_formatter = "{:.15f}".format

def parse_input_line(line):
    features = line.split("\x01")
    tweet_id = features[all_features_to_idx['tweet_id']]
    user_id = features[all_features_to_idx['engager_id']]
    input_feats = features[all_features_to_idx['text_tokens']]
    return tweet_id, user_id, input_feats


def evaluate_test_set():
#     path = './test' # ./test
#     path = '/dataset/final_data/small_dataset/valid_1000'
#     path = conf.valid_dataset_path
    path = '/dataset/final_data/dataset/train_tmp'
    path = '/dataset/final_data/dataset/valid/'
    model_path = conf.model_path
#     model_path = '/hdd/models/test.ensemble_ffnn_pkl/'
    

    part_files = sorted([os.path.join(path, f) for f in os.listdir(path) if 'part' in f])
    ds = Dataset()
    file = path
    with open('results.csv', 'w') as output:
        for file in tqdm(part_files):
            df = read_data(file)
            df = ds.preprocess(df, TARGET_id=conf.REPLY)

            df = ds.tweet_engagements(df) # tweet engagement
            df = ds.user_engagements(df, train=False) # user engagement
            df = ds.tweet_features(df) # tweet features
            df = ds.set_engager_follows_creator(df) 

            pred_reply = Ensemble_FFNN_ALL(df, conf.REPLY).predict(model_path, model_num=1)
            pred_retweet = Ensemble_FFNN_ALL(df, conf.RETWEET).predict(model_path, model_num=1) 
            pred_comment = Ensemble_FFNN_ALL(df, conf.COMMNET).predict(model_path, model_num=1) 
            pred_like = Ensemble_FFNN_ALL(df, conf.LIKE).predict(model_path, model_num=1) 

            with open(file, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    tweet_id, user_id, features = parse_input_line(line)
                    '''
                    ## XGBoost
                    reply_pred = pred_reply[i]
                    retweet_pred = pred_retweet[i]
                    quote_pred = pred_comment[i]
                    fav_pred = pred_like[i]
                    '''
                    reply_pred = pred_reply[i][0]
                    retweet_pred = pred_retweet[i][0]
                    quote_pred = pred_comment[i][0]
                    fav_pred = pred_like[i][0]

                    output.write(f'{tweet_id},{user_id},{reply_pred},{retweet_pred},{quote_pred},{fav_pred}\n')

        del df

if __name__ == "__main__":
    evaluate_test_set()
