from .base_strategy import BaseStrategy


class DropoutSampling(BaseStrategy):
    def __init__(self, dataset, net):
        super(DropoutSampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout(unlabeled_data) # torch.Size([N_UNLABELED_DATA, N_CLASS])
        uncertainties = probs.max(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
