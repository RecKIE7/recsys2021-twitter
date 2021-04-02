#!/env/bin/python

import os

from models.baseline.random_model import random_prediction_model
from core.config import raw_features
import core.config as conf
from tqdm import tqdm

all_features_to_idx = dict(zip(raw_features, range(len(raw_features))))

def parse_input_line(line):
    features = line.split("\x01")
    tweet_id = features[all_features_to_idx['tweet_id']]
    user_id = features[all_features_to_idx['engaging_user_id']]
    input_feats = features[all_features_to_idx['text_tokens']]
    return tweet_id, user_id, input_feats


def evaluate_test_set():
    model = random_prediction_model
    path = '/hdd/twitter/raw_lzo/'
    part_files = [os.path.join(path, f) for f in os.listdir(path) if 'part' in f]
    with open('submission/results.csv', 'w') as output:
        for file in tqdm(part_files):
            with open(file, 'r') as f:
                for line in f.readlines():
                    tweet_id, user_id, features = parse_input_line(line)
                    reply_pred = model(features)
                    retweet_pred = model(features)
                    quote_pred = model(features)
                    fav_pred = model(features)
                    output.write(f'{tweet_id},{user_id},{reply_pred},{retweet_pred},{quote_pred},{fav_pred}\n')

if __name__ == "__main__":
    evaluate_test_set()
