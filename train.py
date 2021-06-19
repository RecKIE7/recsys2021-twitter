import os
import fire
from tqdm import tqdm

import core.config as conf


from utils.dataiter import Dataiter
from models.model.XGBoost import XGBoost
from models.model.DNN import DNN
# from models.model.DeepFM import DeepFM
from models.model.FFNN import FFNN
from models.model.FFNN_ALL import FFNN_ALL
from models.network import Network

class Train(object):
    def __init__(self, target='like'):
        TARGET_id = conf.target_to_idx[target]
        self.df = Dataiter(conf.dataset_path, TARGET_id, train=True) 
        
        if conf.net_structure == 'xgboost':
            model = XGBoost(self.df, TARGET_id)

        elif conf.net_structure == 'deepfm':                
            model = DeepFM(self.df, TARGET_id)
            
        elif conf.net_structure == 'dnn' :
            model = DNN(self.df, TARGET_id)
        elif conf.net_structure == 'ffnn' :
            model = FFNN(self.df, TARGET_id)
        elif conf.net_structure == 'ffnn_all':
            model = FFNN_ALL(self.df, TARGET_id)
        else:
            print('Unidentified Network... exit')
            exit()

        self.model = Network(model)
        
    def train(self):
        self.model.train() 
    


if __name__ == "__main__":
    fire.Fire(Train)
