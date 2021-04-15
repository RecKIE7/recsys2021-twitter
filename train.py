import os
import fire
from tqdm import tqdm

import core.config as conf


from utils.dataiter import Dataiter
from models.model.XGBoost import XGBoost
from models.network import Network

class Train(object):
    def __init__(self, target='like'):
        TARGET_id = conf.target_to_idx[target]
        self.df = Dataiter(conf.raw_lzo_path, TARGET_id) 

        if conf.net_structure == 'xgboost':
            model = XGBoost(self.df, TARGET_id)
        else:
            print('Unidentified Network... exit')
            exit()

        self.model = Network(model)
        
    def train(self):
        self.model.train() 
    


if __name__ == "__main__":
    fire.Fire(Train)
