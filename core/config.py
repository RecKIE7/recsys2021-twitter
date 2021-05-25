# use gpu
gpu = False

# network structure
# structures = ['xgboost', 'deepfm', 'dnn', 'ffnn']
net_structure = 'ffnn'

# path
data_root = '/dataset/'
project_root = '~/kie/recsys2021-twitter/'
raw_data_path = data_root + 'raw/'
submission_path = project_root + 'submission/'
preproc_path = '/hdd/preprocessing/'
raw_lzo_path = '/hdd/twitter/raw_lzo/'
dataset_path = '/hdd/twitter/dataset/'
dataset_mini_path = '/hdd/twitter/dataset_mini/'
dict_path = '/dataset/pickle/'
pickle_data = '/dataset/pickle/'
scaler_path = '/dataset/preprocessing'

# features
raw_features = ["text_tokens", "hashtags", "tweet_id", "media", "links", "domains", "tweet_type","language", "tweet_timestamp", "creator_id", "creator_follower_count", "creator_following_count", "creator_is_verified", "creator_account_creation","engager_id", "engager_follower_count", "engager_following_count", "engager_is_verified", "engager_account_creation", "creator_follows_engager"]
labels = ["reply_timestamp", "retweet_timestamp", "comment_timestamp", "like_timestamp"]

# used raw features

used_features = ['creator_id', 'engager_id', 'tweet_id', 'tweet_type', 'language', "creator_account_creation", 'creator_follower_count', 'creator_following_count', 'engager_follower_count', 'engager_following_count', 'domains', 'media', 'tweet_timestamp']

# for Deep FM
if net_structure == 'deepfm':
    sparse_features = ['media', 'tweet_type', 'creator_is_verified', 'engager_is_verified', 'creator_follows_engager']
    dense_features = ['tweet_timestamp', 'creator_follower_count', 'creator_following_count', 'creator_account_creation', 'engager_follower_count', 'engager_following_count', 'engager_account_creation']
    used_features = used_features + sparse_features + dense_features
    used_features = list(set(used_features))

# parameters
n_workers = 1
n_partitions = 16


# target name
target = ['reply', 'retweet', 'comment', 'like']
target_to_idx = {'reply':0, 'retweet':1, 'comment':2, 'like':3}
REPLY = 0
RETWEET = 1
COMMNET = 2
LIKE = 3
target_encoding = 0 # run target encoding? (0:False, 1:simple_encoder, 2:MTE_encoder, 3:Grouping_encoder)
