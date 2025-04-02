from classify.dataset import ActiveLearningDataset
from classify.models import ActiveLearningNet

class BaseStrategy:
    def __init__(self, dataset: ActiveLearningDataset, net: ActiveLearningNet):
        self.dataset = dataset
        self.net = net

    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data)

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs
    
    def predict_prob_dropout(self, data):
        probs = self.net.predict_prob_dropout(data)
        return probs
    
    def predict_prob_bald(self, data):
        probs = self.net.predict_prob_bald(data)
        return probs
