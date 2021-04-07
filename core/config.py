# path
data_root = '/dataset/'
project_root = '~/kie/recsys2021-twitter/'
raw_data_path = data_root + 'raw/'
<<<<<<< HEAD
submission_path = project_root + 'submission/'

# features
raw_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "engaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
               "enaging_user_account_creation", "engagee_follows_engager"]
labels = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]

n_workers = 1
=======
>>>>>>> 8e2cc1bb3a0e7a4c33d308448c2297e256cfeac8
n_partitions = 16