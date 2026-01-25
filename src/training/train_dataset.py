import numpy as np
from src.models.archive import Archive
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(
        self, core_dataset: np.ndarray,
    ):
        self.core_dataset=core_dataset

    @classmethod
    def from_archive(cls, archive: Archive):
        return cls(core_dataset=archive.core_dataset())
    
    def __getitem__(self, idx):
        return self.core_dataset[idx, :]
    
    def __len__(self):
        return self.core_dataset.shape[0]