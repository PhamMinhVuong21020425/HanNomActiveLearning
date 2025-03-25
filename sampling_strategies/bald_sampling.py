import torch
from .base_strategy import BaseStrategy


class BALDSampling(BaseStrategy):
    def __init__(self, dataset, net):
        super(BALDSampling, self).__init__(dataset, net)
    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_bald(unlabeled_data) # torch.Size([N_INFER, N_UNLABELED_DATA, N_CLASS])
        
        # Calculate BALD score: entropy of expected predictions - expected entropy of predictions
        mean_prob = probs.mean(0) # torch.Size([N_UNLABELED_DATA, N_CLASS])
        entropy_mean = (-mean_prob * torch.log(mean_prob + 1e-9)).sum(1) # Entropy of the mean probability
        average_entropy = (-probs * torch.log(probs + 1e-9)).sum(2).mean(0) # Mean of the entropy

        # Higher BALD score means higher uncertainty
        uncertainties = entropy_mean - average_entropy

        # Sort and select indices
        _, indices = uncertainties.sort(descending=True)
        selected_indices = indices[:n]
        
        return unlabeled_idxs[selected_indices.cpu()]
