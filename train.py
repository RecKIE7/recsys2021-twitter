import os
import fire
import core.config as conf
from tqdm import tqdm


class Train(object):
    def __init__(self):
        # self.dataset = Dataset(training_flag=True)
        self.df = Dataiter(conf.raw_lzo_path) # test => ./test
        self.model = XGBoost(self.df)

        self.train() 
        # for i, d in enumerate(self.df):
        #     print(len(d))
        #     if i == 10:
        #         break

    def train(self):
        # train
        self.model.incremental_train(3) # Like only
    


if __name__ == "__main__":
    
    # from utils.dataset import *
    from utils.dataiter import Dataiter
    from models.XGBoost import XGBoost

    fire.Fire(Train)
