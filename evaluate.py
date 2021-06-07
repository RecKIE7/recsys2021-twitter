

from utils.dataiter import Dataset
from utils.preprocessing import *
from models.baseline.random_model import random_prediction_model
from models.model.FFNN_ALL import FFNN_ALL
from core.config import raw_features
from utils.evaluate import calculate_ctr, compute_rce, average_precision_score

import core.config as conf
from tqdm import tqdm


path = '/hdd/twitter/test_data/part-210210-.7'
path = '/hdd/twitter/test_data2/part-test'
df = read_data(path)

df['reply_timestamp']   = df['reply_timestamp'].fillna(0)
df['retweet_timestamp'] = df['retweet_timestamp'].fillna(0)
df['comment_timestamp'] = df['comment_timestamp'].fillna(0)
df['like_timestamp']    = df['like_timestamp'].fillna(0)

df['reply_timestamp']   = df['reply_timestamp'].astype(np.uint32)
df['retweet_timestamp'] = df['retweet_timestamp'].astype(np.uint32)
df['comment_timestamp'] = df['comment_timestamp'].astype(np.uint32)
df['like_timestamp']    = df['like_timestamp'].astype(np.uint32)

df['reply'] = df['reply_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32)
df['retweet'] = df['retweet_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32)
df['comment'] = df['comment_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32)
df['like'] = df['like_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32) 

pred = pd.read_csv('results.csv', names=['tweet_id','user_id','reply_pred','retweet_pred','comment_pred','like_pred'])


reply_rce = compute_rce(pred['reply_pred'], df['reply'])
like_rce = compute_rce(pred['like_pred'], df['like'])
comment_rce = compute_rce(pred['comment_pred'], df['comment'])
retweet_rce = compute_rce(pred['retweet_pred'], df['retweet'])
print()
print('***************rce***************')
print('reply:', reply_rce)
print('like:', like_rce)
print('comment:', comment_rce)
print('retweet:', retweet_rce)


reply_ap = average_precision_score(df['reply'], pred['reply_pred'])
like_ap = average_precision_score(df['like'], pred['like_pred'])
comment_ap = average_precision_score(df['comment'], pred['comment_pred'])
retweet_ap = average_precision_score(df['retweet'], pred['retweet_pred'])

print()
print('***************ap***************')
print('reply:', reply_ap)
print('like:', like_ap)
print('comment:', comment_ap)
print('retweet:', retweet_ap)


