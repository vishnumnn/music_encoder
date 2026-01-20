import numpy as np
from src.models.archive import Archive
from torch.utils.data import Dataset


class Trainer(Dataset):
    def __init__(
        self, dataset: np.ndarray,
    ):
        self.dataset=dataset

    @classmethod
    def from_archive(cls, archive: Archive):
        return cls(dataset=archive.core_dataset())
    
    def __getitem__(self, idx):
        return self.dataset[idx, :]
    
    def __len__(self):
        return self.dataset.shape[0]