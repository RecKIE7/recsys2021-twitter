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
        # self.dataset = Dataset()

    def __iter__(self):
        return self         
 
    def __next__(self):
        if self.current < self.stop:    
            r = self.current            
            self.current += 1           
            current_file = self.file_list[r]
            df = read_data(self.dir + current_file) # read data (to dataframe)
            df = self.preprocess(df, self.TARGET_id) # preprocessing using dataset.py

            gc.collect()
            save_memory(df)
            return df
        else:                           
            raise StopIteration 
