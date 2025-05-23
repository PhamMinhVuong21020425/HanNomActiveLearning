from .base_strategy import BaseStrategy


class MarginSampling(BaseStrategy):
    def __init__(self, dataset, net):
        super(MarginSampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data) # torch.Size([N_UNLABELED_DATA, N_CLASS])
        probs_sorted, _ = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
        _, indices = uncertainties.sort()
        selected_indices = indices[:n]
        return unlabeled_idxs[selected_indices.cpu()]
