import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset

from classify.hyperparams import *


class SinoNomDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image, self.phase)

        label = int(img_path.split('/')[-2])
        
        return image, label

class ActiveLearningDataset:
    def __init__(self, dataset, initial_labeled=2000):
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.labeled_idxs = np.zeros(self.n_samples, dtype=bool)
        
        # Initially label random samples
        initial_indices = np.random.choice(np.arange(self.n_samples), initial_labeled, replace=False)
        self.labeled_idxs[initial_indices] = True
        
    def get_labeled_data(self):
        labeled_idxs = np.where(self.labeled_idxs)[0]
        return labeled_idxs, Subset(self.dataset, labeled_idxs)
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.where(self.labeled_idxs == 0)[0]
        if len(unlabeled_idxs) > N_SAMPLES_PREDICT:
            unlabeled_idxs = np.random.choice(unlabeled_idxs, N_SAMPLES_PREDICT, replace=False)
        return unlabeled_idxs, Subset(self.dataset, unlabeled_idxs)
    
    def label_samples(self, indices):
        self.labeled_idxs[indices] = True
