import numpy as np

from kgcnn.data.base import MemoryGraphDataset, MemoryGraphList
from kgcnn.data.qm import QMDataset


def idx_generator(mol_count, val_ratio, test_ratio):
    """
    Generates random lists in the range of mol_count to split of validation set with val_ratio and test_set with test_ratio
    """
    if val_ratio + test_ratio > 1 or val_ratio + test_ratio < 0:
        raise  ValueError("idx_generator: the val_ratio and test_ratio must be in between 0 and 1")
    
    shuffled_indices = np.random.permutation(mol_count)
    
    val_set_size = int(mol_count * val_ratio)
    test_set_size= int(mol_count * test_ratio)

    val_idx   = shuffled_indices[:val_set_size]
    test_idx  = shuffled_indices[val_set_size:val_set_size+test_set_size]
    train_idx = shuffled_indices[val_set_size + test_set_size:]

    # ensures same data order after boolean indexing and vector indexing 
    val_idx.sort()
    test_idx.sort()
    train_idx.sort()

    # Check whether it is totally splitted
    if train_idx.shape[0] + test_idx.shape[0] + val_idx.shape[0] != mol_count:
        raise ValueError("Splitting Test does not equal to the entire set!")
        
    return train_idx, val_idx, test_idx

def subsample_dataset(dataset: QMDataset, max_dataset_size: int) -> MemoryGraphList:
    """Subsample a dataset to a maximum size.

    Args:
        dataset (QMDataset): The original dataset.
        max_dataset_size (int): The maximum size of the subsampled dataset.

    Returns:
        MemoryGraphList: The subsampled dataset.
    """
    dataset_name = dataset.dataset_name
    np.random.seed(42)
    subsample_indices = np.random.choice(len(dataset), size=min(max_dataset_size, len(dataset)), replace=False)
    dataset: MemoryGraphList = dataset[subsample_indices]
    dataset.dataset_name = dataset_name  # hack to keep the name after subsampling
    return dataset