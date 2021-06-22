# use gpu
gpu = False

# network structure
# structures = ['xgboost', 'deepfm', 'dnn', 'ffnn']
net_structure = 'ffnn_all_default' # ffnn_all
#net_structure = 'ensemble_ffnn_all' # ffnn_all


random_states = [1111, 2222, 3333, 4444, 5555]

# path
data_root = '/dataset/'
project_root = '~/kie/recsys2021-twitter/'
raw_data_path = data_root + 'raw/'
submission_path = project_root + 'submission/'
preproc_path = '/hdd/preprocessing/'
raw_lzo_path = '/hdd/twitter/raw_lzo/'
dataset_mini_path = '/hdd/twitter/dataset_mini/'
pickle_data = '/dataset/pickle/'
scaler_path = '/dataset/preprocessing/'

dataset_path = '/dataset/final_data/dataset/train_split/'
dataset_path = '/dataset/final_data/small_dataset/'

valid_dataset_path = '/dataset/final_data/dataset/valid/'

cross_valid_data = '/dataset/final_data/dataset/train_split/'
pred_pickle_path = '/dataset/pred_pickle/'
model_path = f'/hdd/models/ffnn_default/'


# features
raw_features = ["text_tokens", "hashtags", "tweet_id", "media", "links", "domains", "tweet_type","language", "tweet_timestamp", "creator_id", "creator_follower_count", "creator_following_count", "creator_is_verified", "creator_account_creation","engager_id", "engager_follower_count", "engager_following_count", "engager_is_verified", "engager_account_creation", "creator_follows_engager"]
labels = ["reply_timestamp", "retweet_timestamp", "comment_timestamp", "like_timestamp"]

# used raw features
used_features = ['text_tokens', 'creator_id', 'engager_id', 'tweet_id', 'tweet_type', 'language', "hashtags", "creator_account_creation", 'creator_follower_count', 'creator_following_count', 'engager_follower_count', 'engager_following_count', 'domains', 'media', 'tweet_timestamp']

# 'reply', 'retweet', 'comment', 'like'
drop_features = [['engager_feature_number_of_previous_like_engagement', 
                  'engager_feature_number_of_previous_retweet_engagement',
                  'engager_feature_number_of_previous_comment_engagement', 'number_of_engagements_ratio_like', 
                  'number_of_engagements_ratio_retweet', 'number_of_engagements_ratio_comment',
                  'number_of_tweet_like', 'number_of_tweet_retweet', 'number_of_tweet_comment',
                  'creator_number_of_engagements_ratio_like', 'creator_number_of_engagements_ratio_retweet',
                  'creator_number_of_engagements_ratio_comment',
                  'creator_feature_number_of_previous_like_engagement', 
                  'creator_feature_number_of_previous_retweet_engagement',
                  'creator_feature_number_of_previous_comment_engagement'
                 ],
                 
                 ['engager_feature_number_of_previous_like_engagement', 
                  'engager_feature_number_of_previous_reply_engagement', 
                  'engager_feature_number_of_previous_comment_engagement', 'number_of_engagements_ratio_like', 
                  'number_of_engagements_ratio_reply', 'number_of_engagements_ratio_comment',
                  'number_of_tweet_like', 'number_of_tweet_reply', 'number_of_tweet_comment',
                  'creator_number_of_engagements_ratio_like', 'creator_number_of_engagements_ratio_reply',
                  'creator_number_of_engagements_ratio_comment',
                  'creator_feature_number_of_previous_like_engagement', 
                  'creator_feature_number_of_previous_reply_engagement',
                  'creator_feature_number_of_previous_comment_engagement'
                 ],
                 
                 ['engager_feature_number_of_previous_like_engagement',
                  'engager_feature_number_of_previous_retweet_engagement', 
                  'engager_feature_number_of_previous_reply_engagement', 'number_of_engagements_ratio_like', 
                  'number_of_engagements_ratio_retweet', 'number_of_engagements_ratio_reply',
                  'number_of_tweet_like', 'number_of_tweet_retweet', 'number_of_tweet_reply',
                  'creator_number_of_engagements_ratio_like', 'creator_number_of_engagements_ratio_retweet',
                  'creator_number_of_engagements_ratio_reply',
                  'creator_feature_number_of_previous_like_engagement', 
                  'creator_feature_number_of_previous_retweet_engagement',
                  'creator_feature_number_of_previous_reply_engagement'
                 ],
                 
                 ['engager_feature_number_of_previous_retweet_engagement',
                  'engager_feature_number_of_previous_reply_engagement', 
                  'engager_feature_number_of_previous_comment_engagement', 'number_of_engagements_ratio_retweet', 
                  'number_of_engagements_ratio_reply', 'number_of_engagements_ratio_comment',
                  'number_of_tweet_reply', 'number_of_tweet_retweet', 'number_of_tweet_comment',
                  'creator_number_of_engagements_ratio_reply', 'creator_number_of_engagements_ratio_retweet',
                  'creator_number_of_engagements_ratio_comment',
                  'creator_feature_number_of_previous_reply_engagement', 
                  'creator_feature_number_of_previous_retweet_engagement',
                  'creator_feature_number_of_previous_comment_engagement'
                 ]]

default_values = {'engager_feature_number_of_previous_like_engagement': 16.68406226808318,
                             'engager_feature_number_of_previous_reply_engagement': 3.9166628750988446,
                             'engager_feature_number_of_previous_retweet_engagement': 7.943690435417255,
                             'engager_feature_number_of_previous_comment_engagement': 2.397117827194066,
                             'creator_feature_number_of_previous_like_engagement': 18.650278982078916,
                             'creator_feature_number_of_previous_reply_engagement': 4.005221886495085,
                             'creator_feature_number_of_previous_retweet_engagement': 8.378531979240039,
                             'creator_feature_number_of_previous_comment_engagement': 2.465194979899623,
                             'creator_number_of_engagements_positive': 8.374806956928415,
                             'number_of_engagements_positive': 7.735383351448337,
                             'number_of_engagements_ratio_like': 2.1568500887495556,
                             'number_of_engagements_ratio_retweet': 1.0269291222560957,
                             'number_of_engagements_ratio_reply': 0.5063308044539909,
                             'number_of_engagements_ratio_comment': 0.30988998454035777,
                             'creator_number_of_engagements_ratio_like': 2.226950313959139,
                             'creator_number_of_engagements_ratio_retweet': 1.0004447890358288,
                             'creator_number_of_engagements_ratio_reply': 0.4782464726761964,
                             'creator_number_of_engagements_ratio_comment': 0.29435842432883613,
                             'creator_main_language': 0,
                             'engager_main_language': 0,
                             'is_tweet_in_creator_main_language': 0.5,
                             'is_tweet_in_engager_main_language': 0.5,
                             'creator_and_engager_have_same_main_language': 0.5,
                             'number_of_tweet_like': 1.2568421511772854,
                             'number_of_tweet_reply': 1.0636431744005945,
                             'number_of_tweet_retweet': 1.0998849163606805,
                             'number_of_tweet_comment': 1.0363500723064045,
                             'number_of_tweet_engagements': 1.3275094196173265}


# default_values = {'engager_feature_number_of_previous_like_engagement': 0,
#                              'engager_feature_number_of_previous_reply_engagement': 0,
#                              'engager_feature_number_of_previous_retweet_engagement': 0,
#                              'engager_feature_number_of_previous_comment_engagement':0,
#                              'creator_feature_number_of_previous_like_engagement': 0,
#                              'creator_feature_number_of_previous_reply_engagement': 0,
#                              'creator_feature_number_of_previous_retweet_engagement': 0,
#                              'creator_feature_number_of_previous_comment_engagement': 0,
#                              'creator_number_of_engagements_positive': 0,
#                              'number_of_engagements_positive': 0,
#                              'number_of_engagements_ratio_like': 0,
#                              'number_of_engagements_ratio_retweet': 0,
#                              'number_of_engagements_ratio_reply': 0,
#                              'number_of_engagements_ratio_comment': 0,
#                              'creator_number_of_engagements_ratio_like': 0,
#                              'creator_number_of_engagements_ratio_retweet': 0,
#                              'creator_number_of_engagements_ratio_reply': 0,
#                              'creator_number_of_engagements_ratio_comment': 0,
#                              'creator_main_language': 0,
#                              'engager_main_language': 0,
#                              'is_tweet_in_creator_main_language': 0.5,
#                              'is_tweet_in_engager_main_language': 0.5,
#                              'creator_and_engager_have_same_main_language': 0.5,
#                              'number_of_tweet_like': -1,
#                              'number_of_tweet_reply': -1,
#                              'number_of_tweet_retweet': -1,
#                              'number_of_tweet_comment': -1,
#                              'number_of_tweet_engagements': -1}

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
target_to_idx = {'reply':0, 'retweet':1, 'comment':2, 'like':3, 'all' : 4}
REPLY = 0
RETWEET = 1
COMMNET = 2
LIKE = 3
ALL = 4
target_encoding = 0 # run target encoding? (0:False, 1:simple_encoder, 2:MTE_encoder, 3:Grouping_encoder)
