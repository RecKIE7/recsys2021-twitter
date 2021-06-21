import sys, os
sys.path.append('..')

from utils.preprocessing import *
from utils.dataset import Dataset
import core.config as conf

class Dataiter(Dataset):
    def __init__(self, path, TARGET_id=3, train=False):
        self.dir = path
        self.file_list = sorted(os.listdir(path))
        self.current = 0
        self.stop = len(self.file_list)
        self.TARGET_id = TARGET_id 
        self.random_state = conf.random_states[TARGET_id]
        self.train = train
        self.default_values = conf.default_values


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
        