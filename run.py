import os

from utils.dataiter import Dataset
from utils.preprocessing import *
from models.baseline.random_model import random_prediction_model
from models.model.DeepFM import DeepFM
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
    path = '/hdd/twitter/raw_lzo/' # ./test
    part_files = sorted([os.path.join(path, f) for f in os.listdir(path) if 'part' in f])[:3]
    ds = Dataset()
    with open('submission/results.csv', 'w') as output:
        for file in tqdm(part_files):
            test_df = ds.preprocess(read_data(file))
            pred_reply = DeepFM(test_df, conf.REPLY).predict(model='0') 
            pred_retweet = DeepFM(test_df, conf.RETWEET).predict(model='0') 
            pred_comment = DeepFM(test_df, conf.COMMNET).predict(model='0') 
            pred_like = DeepFM(test_df, conf.LIKE).predict(model='0') 

            with open(file, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    tweet_id, user_id, features = parse_input_line(line)
                    reply_pred = pred_reply[i][0]
                    retweet_pred = pred_retweet[i][0]
                    quote_pred = pred_comment[i][0]
                    fav_pred = pred_like[i][0]
                    # fav_pred = float_formatter(pred_like[i])
                    output.write(f'{tweet_id},{user_id},{reply_pred},{retweet_pred},{quote_pred},{fav_pred}\n')

            del test_df

if __name__ == "__main__":
    evaluate_test_set()
