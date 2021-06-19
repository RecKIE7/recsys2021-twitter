import sys, os
sys.path.append('..')

from utils.preprocessing import *
from utils.dataset import Dataset
import core.config as conf

class Dataiter(Dataset):
    def __init__(self, path, TARGET_id=3, train=False):
        self.dir = path
        self.file_list = sorted(os.listdir(path))[:1]
        self.current = 0
        self.stop = len(self.file_list)
        self.TARGET_id = TARGET_id 
        self.random_state = conf.random_states[TARGET_id]
        self.train = train
        self.default_values = {'engager_feature_number_of_previous_like_engagement': 16.68406226808318,
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
                             'creator_and_engager_have_same_main_language': 0.5}

    def __iter__(self):
        return self         
 
    def __next__(self):
        if self.current < self.stop:    
            r = self.current            
            self.current += 1           
            current_file = self.file_list[r]
            df = read_data(self.dir + current_file) # read data (to dataframe)
                
            if conf.net_structure == 'dnn':
                df = self.raw_preprocess(df, self.TARGET_id) # DNN    
            else:
                df = self.preprocess(df, self.TARGET_id) # preprocessing using dataset.py

            df = self.tweet_engagements(df) # tweet engagement
            df = self.user_engagements(df, self.train) # user engagement
            df = self.tweet_features(df) # tweet features
            
#             df = self.fill_with_default_value(df) # for ensemble
            
            self.current_file = current_file

            gc.collect()
            save_memory(df)
            return df
        else:                           
            raise StopIteration 

    def __len__(self):
        return self.stop
        