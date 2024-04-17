import os
import pickle
import numpy as np

def save_history(history_objects: list, filename: str = "histories.pkl"):
    """_summary_

    Args:
        history_objects (list): _description_
        filename (str, optional): _description_. Defaults to "histories.pkl".
    """
    hist_dicts = [hist.history for hist in history_objects]
    with open(os.path.join("", filename), "wb") as f:
        pickle.dump(hist_dicts, f)

def load_history(filename: str = "histories.pkl"):
    with open(os.path.join("", filename), "rb") as f:
        hist_dicts = pickle.load(f)
    return hist_dicts


def save_training_indices(train_indices: list, test_indices: list, filename: str ="training_indices.pkl"):
    """_summary_

    Args:
        train_indices (list): _description_
        test_indices (list): _description_
        filename (str, optional): _description_. Defaults to "training_indices.pkl".
    """
    index_dict = {
        "train": train_indices,
        "test": test_indices
    }
    with open(os.path.join("", filename), "wb") as f:
        pickle.dump(index_dict, f)


def load_training_indices(filename: str ="training_indices.pkl", expected_length: int= None):
    with open(filename, 'rb') as f:
        index_dict = pickle.load(f)
    train_indices: list[np.ndarray] = index_dict["train"]
    test_indices: list[np.ndarray] = index_dict["test"]
    if expected_length is not None:
        assert len(train_indices) == expected_length, "Expected_length and amount of train indices differ"
        assert len(test_indices) == expected_length, "Expected_length and amount of test indices differ"

    return train_indices, test_indices

    
    