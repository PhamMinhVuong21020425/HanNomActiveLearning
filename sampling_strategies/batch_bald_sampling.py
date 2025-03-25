import torch
import numpy as np
from torch.utils.data import Subset
from itertools import combinations_with_replacement
from .base_strategy import BaseStrategy


class BatchBALDSampling(BaseStrategy):
    def __init__(self, dataset, net, num_sub_pool=100, m=1e4):
        """
        BatchBALD acquisition strategy implementation
        
        Args:
            dataset: Dataset object with labeled/unlabeled data
            net: Neural network model
            num_sub_pool: Number of datapoints in the subpool from which we acquire
            m: Number of MC samples for label combinations
        """
        super(BatchBALDSampling, self).__init__(dataset, net)
        self.num_sub_pool = int(num_sub_pool)  # number of datapoints in the subpool
        self.m = int(m)  # number of MC samples for label combinations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_entropy(self, x):
        """
        Compute entropy for given probability distributions
        
        Args:
            x: Tensor of probabilities
            
        Returns:
            Entropy values
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-9
        return -x * torch.log(x + eps)
    
    def class_combinations(self, c, n, m=np.inf):
        """ Generates an array of n-element combinations where each element is one of
        the c classes (an integer). If m is provided and m < n^c, then instead of all
        n^c combinations, m combinations are randomly sampled.
        Arguments:
            c {int} -- the number of classes
            n {int} -- the number of elements in each combination
        Keyword Arguments:
            m {int} -- the number of desired combinations (default: {np.inf})
        Returns:
            tensor -- An [m, n] or [(n + c -1)C(n), n] array of integers in [0, c)
        """
        if m < c**n:
            # randomly sample combinations
            return torch.randint(c, size=(int(m), n), device=self.device) # [m, n]
        else:
            # all combinations: số tổ hợp có lặp của n phần tử từ c lớp
            p_c = combinations_with_replacement(range(c), n)
            comb_list = list(p_c)
            return torch.tensor(comb_list, dtype=torch.long, device=self.device) # [(n + c -1)C(n), n]

    def query(self, n):
        """
        Select a batch of n samples using BatchBALD acquisition function
        
        Args:
            n: Number of samples to select
            
        Returns:
            Indices of selected samples
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        
        # If we have fewer unlabeled samples than requested, return all unlabeled samples
        if len(unlabeled_idxs) <= n:
            return unlabeled_idxs
        
        # Get a subset of the pool if it's too large
        if len(unlabeled_idxs) > self.num_sub_pool:
            sub_unlabeled_idxs = np.random.choice(len(unlabeled_idxs), self.num_sub_pool, replace=False)
            sub_unlabeled_data = Subset(unlabeled_data, sub_unlabeled_idxs)
        else:
            sub_unlabeled_idxs = unlabeled_idxs
            sub_unlabeled_data = unlabeled_data
        
        # Get MC dropout samples for the subpool
        print("Running MC Dropout predictions...")
        probs = self.predict_prob_bald(sub_unlabeled_data) # torch.size([N_INFER, N_SAMPLES, N_CLASS])
        
        # Transpose to shape: [N_SAMPLES, N_CLASS, N_INFER]
        probs = probs.permute(1, 2, 0)
        
        # Get number of classes
        n_samples, n_classes, k = probs.shape
        
        # Compute H2 term (only needs to be calculated once)
        # Expected entropy of each prediction across MC dropout samples
        H2 = (self.compute_entropy(probs).sum(axis=(1,2))/k).to(self.device)
        
        # Generate all possible class combinations or sample a subset
        c_1_to_n = self.class_combinations(n_classes, n, self.m) # [m, n]
        
        # Initialize tensor for probability of class combinations
        p_y_1_to_n_minus_1 = None # [m, k]
        
        # Keep track of selected indices
        best_indices = []
        
        # Create a mask to keep track of which indices we've chosen
        remaining_indices = torch.ones(len(sub_unlabeled_idxs), dtype=torch.bool, device=self.device)

        print("Starting greedy batch selection...")
        for i in range(n):
            # Get class probabilities for the current position in the batch
            # Shape: [N_SAMPLES, m, N_INFER]
            p_y_n = probs[:, c_1_to_n[:, i], :].to(self.device)
            
            if p_y_1_to_n_minus_1 is not None:
                # Compute joint probability with previously selected datapoints
                # Pre-allocate result tensor for improved performance
                p_y_1_to_n = torch.zeros_like(p_y_n)
                # Use einsum for efficient batch matrix multiplication
                p_y_1_to_n = torch.einsum('mk,pmk->pmk', p_y_1_to_n_minus_1, p_y_n)
            else:
                p_y_1_to_n = p_y_n
            
            # Compute H1 term (entropy of the marginal)
            # Average over MC samples first, then compute entropy
            mean_p_y_1_to_n = p_y_1_to_n.mean(dim=2)  # [N_SAMPLES, m]
            H1 = self.compute_entropy(mean_p_y_1_to_n).sum(dim=1)  # [N_SAMPLES]
            
            # scores is a vector of scores for each element in the pool.
            # mask by the remaining indices and find the highest scoring element
            scores = H1 - H2
        
            best_idx = torch.argmax(scores - np.inf*(~remaining_indices)).item()
            
            # Add the best index to our selection
            best_indices.append(best_idx)
            
            # Save the computation for the next batch
            p_y_1_to_n_minus_1 = p_y_1_to_n[best_idx]  # [m, N_INFER]
            
            # Mark this index as selected
            remaining_indices[best_idx] = False

            print(f"Selected sample {i+1}/{n}, score: {scores[best_idx].item():.4f}")

        # Convert subpool indices back to original unlabeled indices
        if len(unlabeled_idxs) > self.num_sub_pool:
            selected_idxs = sub_unlabeled_idxs[best_indices]
            return unlabeled_idxs[selected_idxs]
        else:
            return unlabeled_idxs[best_indices]
