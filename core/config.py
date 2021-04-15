# use gpu
gpu = False

# path
data_root = '/dataset/'
project_root = '~/kie/recsys2021-twitter/'
raw_data_path = data_root + 'raw/'
submission_path = project_root + 'submission/'
preproc_path = '/hdd/preprocessing/'
raw_lzo_path = '/hdd/twitter/raw_lzo/'

# features
raw_features = ["text_tokens", "hashtags", "tweet_id", "media", "links", "domains", "tweet_type","language", "tweet_timestamp", "creator_id", "creator_follower_count", "creator_following_count", "creator_is_verified", "creator_account_creation","engager_id", "engager_follower_count", "engager_following_count", "engager_is_verified", "engager_account_creation", "creator_follows_engager"]
labels = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]

# used raw features
used_features = ['creator_id', 'engager_id', 'tweet_id', 'tweet_type', 'language', 'creator_follower_count', 'creator_following_count', 'domains', 'media', 'tweet_timestamp']

# parameters
n_workers = 1
n_partitions = 16

# network structure
net_structure = 'xgboost'

# target name
target = ['reply', 'retweet', 'retweet_comment', 'like']
REPLY = 0
RETWEET = 1
COMMNET = 2
LIKE = 3