import torch
from .base_strategy import BaseStrategy


class DropoutSampling(BaseStrategy):
    def __init__(self, dataset, net):
        super(DropoutSampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout(unlabeled_data) # torch.Size([N_UNLABELED_DATA, N_CLASS])

        # Calculate uncertainty with entropy-based method
        log_probs = torch.log(probs + 1e-9)
        uncertainties = -torch.sum(probs*log_probs, dim=1) # torch.size([N_UNLABELED_DATA])
        _, indices = uncertainties.sort(descending=True)
        selected_indices = indices[:n]
        return unlabeled_idxs[selected_indices.cpu()]
