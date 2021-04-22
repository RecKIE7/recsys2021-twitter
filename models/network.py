

class Network(object):
    def __init__(self, model):
        self.model = model
    
    def train(self, target):
        self.model.train(target)

