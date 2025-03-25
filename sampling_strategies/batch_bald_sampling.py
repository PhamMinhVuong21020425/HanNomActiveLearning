import torch
import numpy as np
from torch.utils.data import Subset
from .base_strategy import BaseStrategy


class BatchBALDSampling(BaseStrategy):
    def __init__(self, dataset, net, num_sub_pool=100):
        """
        BatchBALD acquisition strategy implementation
        
        Args:
            dataset: Dataset object with labeled/unlabeled data
            net: Neural network model
            num_sub_pool: Number of datapoints in the subpool from which we acquire
        """
        super(BatchBALDSampling, self).__init__(dataset, net)
        self.num_sub_pool = int(num_sub_pool)  # number of datapoints in the subpool
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        average_entropy = (-probs * torch.log(probs + 1e-9)).sum(2).mean(0) # torch.Size([N_SAMPLES])
        
        # Initialize batch selection tracking
        best_indices = []
        remaining_indices = list(range(len(sub_unlabeled_idxs)))
        
        # Greedy batch selection
        print("Starting greedy batch selection...")
        for i in range(n):
            max_mutual_info = -float('inf')
            best_index = None
            
            # Evaluate each remaining datapoint
            for j in remaining_indices:
                # Temporarily add this point to the batch
                current_batch = best_indices + [j]
                
                # Compute conditional probabilities
                batch_probs = probs[:, current_batch, :]
                
                # Compute conditional entropy
                mean_prob = batch_probs.mean(0) # torch.Size([N_BATCH, N_CLASS])
                entropy_mean = (-mean_prob * torch.log(mean_prob + 1e-9)).sum(1) # torch.Size([N_BATCH])

                # Compute mutual information
                mutual_info = (entropy_mean - average_entropy[current_batch]).sum()
                
                # Update best index
                if mutual_info > max_mutual_info:
                    max_mutual_info = mutual_info
                    best_index = j
            
            # Add best index to selection
            if best_index is not None:
                best_indices.append(best_index)
                remaining_indices.remove(best_index)

            print ("Best index: ", best_index)
            print(f"Selected sample {i+1}/{n}, score: {max_mutual_info.item():.4f}")

        # Convert subpool indices back to original unlabeled indices
        if len(unlabeled_idxs) > self.num_sub_pool:
            selected_idxs = sub_unlabeled_idxs[best_indices]
            return unlabeled_idxs[selected_idxs]
        else:
            return unlabeled_idxs[best_indices]
