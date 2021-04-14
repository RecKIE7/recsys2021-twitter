import os
import fire


class Train(object):
    def __init__(self):
        # self.dataset = Dataset(training_flag=True)
        self.df = Dataiter(conf.preproc_path + 'train/')
        self.model = XGBoost(self.df)
        self.model.incremental_train(3) # Like
    


if __name__ == "__main__":
    from utils.dataset import *
    from utils.dataiter import Dataiter
    from models.XGBoost import XGBoost

    fire.Fire(Train)
