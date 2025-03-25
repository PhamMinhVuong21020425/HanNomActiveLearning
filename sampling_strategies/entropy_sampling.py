import torch
from .base_strategy import BaseStrategy


class EntropySampling(BaseStrategy):
    def __init__(self, dataset, net):
        super(EntropySampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs + 1e-9)
        uncertainties = -torch.sum(probs*log_probs, dim=1)
        _, indices = uncertainties.sort(descending=True)
        selected_indices = indices[:n]
        return unlabeled_idxs[selected_indices.cpu()]
