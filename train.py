import os
import fire
from tqdm import tqdm

import core.config as conf


from utils.dataiter import Dataiter
from models.model.XGBoost import XGBoost
from models.network import Network

class Train(object):
    def __init__(self):
        self.df = Dataiter(conf.raw_lzo_path) 

        if conf.net_structure == 'xgboost':
            model = XGBoost(self.df)
        else:
            print('Unidentified Network... exit')
            exit()

        self.model = Network(model)
        
    def train(self, target=conf.LIKE):
        self.model.train(target) 
    


if __name__ == "__main__":
    fire.Fire(Train)
