import os

from utils.dataiter import Dataset
from utils.preprocessing import *
from models.baseline.random_model import random_prediction_model
from models.model.FFNN_ALL import FFNN_ALL
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
    path = '/hdd/twitter/test_data/' # ./test
    part_files = sorted([os.path.join(path, f) for f in os.listdir(path) if 'part' in f])[:1]
    model_path = '/hdd/models/ffnn_pkl/'
    ds = Dataset()
    with open('results.csv', 'w') as output:
        for file in tqdm(part_files):
            df = read_data(file)
            df = ds.pickle_matching(ds.preprocess(df, TARGET_id=conf.REPLY))

            pred_reply = FFNN_ALL(df, conf.REPLY).predict(model_path) 
            pred_retweet = FFNN_ALL(df, conf.RETWEET).predict(model_path) 
            pred_comment = FFNN_ALL(df, conf.COMMNET).predict(model_path) 
            pred_like = FFNN_ALL(df, conf.LIKE).predict(model_path) 

            with open(file, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    tweet_id, user_id, features = parse_input_line(line)
                    reply_pred = pred_reply[i][0]
                    retweet_pred = pred_retweet[i][0]
                    quote_pred = pred_comment[i][0]
                    fav_pred = pred_like[i][0]
                    output.write(f'{tweet_id},{user_id},{reply_pred},{retweet_pred},{quote_pred},{fav_pred}\n')

            del df

if __name__ == "__main__":
    evaluate_test_set()
