import os
import fire


class Train(object):
    def __init__(self):
        self.dataset = Dataset(training_flag=True)
        



if __name__ == "__main__":
    from utils.dataset import *

    fire.Fire(Train)
