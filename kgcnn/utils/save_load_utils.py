import os
import pickle
import numpy as np
from ase import Atoms
from ase.io import read, write
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf

def save_history(history_objects: List[tf.keras.callbacks.History], filename: str = "histories.pkl"):
    """Save the training history of Keras models to a file.

    Args:
        history_objects (List[tf.keras.src.callbacks.History]): List of history objects to save.
        filename (str, optional): _description_. Defaults to "histories.pkl".
    """
    hist_dicts = [hist.history for hist in history_objects]
    with open(os.path.join("", filename), "wb") as f:
        pickle.dump(hist_dicts, f)

def load_history(filename: str = "histories.pkl"):
    with open(os.path.join("", filename), "rb") as f:
        hist_dicts = pickle.load(f)
    return hist_dicts


def save_training_indices(
        train_indices: List[np.ndarray], 
        val_indices: List[np.ndarray],
        test_index: np.ndarray, 
        filename: str ="training_indices.pkl"
    ) -> None:
    """Save training, validation, and test indices to a file.

    Args:
        train_indices (List[np.ndarray]): Indices for training set.
        val_indices (List[np.ndarray]): Indices for validation set.
        test_indices (np.ndarray): Indices for test set.
        filename (str, optional): Filename to save the indices. Defaults to "training_indices.pkl
    """
    index_dict = {
        "train": train_indices,
        "val": val_indices,
        "test": test_index
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

def save_extxyz(atomic_numbers_list: List[np.ndarray],
                positions_list: List[np.ndarray],
                ref_infos: Dict[str, np.ndarray],
                ref_arrays: Dict[str, np.ndarray],
                pred_infos: Dict[str, np.ndarray],
                pred_arrays: Dict[str, np.ndarray],
                filename: str = "geoms.extxyz") -> None:
    """Creates ase.Atoms object from the given data and saves it to an extxyz file.
    Args:
        atomic_numbers_list (List[np.ndarray]): List of atomic numbers.
        positions_list (List[np.ndarray]): List of atomic positions.
        ref_infos (Dict[str, np.ndarray]): Reference information dictionary.
        ref_arrays (Dict[str, np.ndarray]): Reference arrays dictionary.
        pred_infos (Dict[str, np.ndarray]): Predicted information dictionary.
        pred_arrays (Dict[str, np.ndarray]): Predicted arrays dictionary.
        filename (str, optional): Filename for the output extxyz file. Defaults to "geoms.extxyz".
    """
    atoms_list = []
    
    for i in range(len(atomic_numbers_list)):
        atomic_numbers = atomic_numbers_list[i]
        positions = positions_list[i]
        
        # Create ASE Atoms object
        atoms = Atoms(numbers=atomic_numbers, positions=positions)
        
        # Add reference information to Atoms object
        for key, val in ref_infos.items():
            if i < len(val):
                atoms.info[f"ref_{key}"] = val[i]
        
        # Add predicted information to Atoms object
        for key, val in pred_infos.items():
            if i < len(val):
                atoms.info[f"pred_{key}"] = val[i]
                
        # Add reference arrays to Atoms object
        for key, val in ref_arrays.items():
            if i < len(val):
                atoms.arrays[f"ref_{key}"] = val[i]
                
        # Add predicted arrays to Atoms object
        for key, val in pred_arrays.items():
            if i < len(val):
                atoms.arrays[f"pred_{key}"] = val[i]
                
        atoms_list.append(atoms)
    
    # Write Atoms objects to extxyz file
    write(filename, atoms_list, format='extxyz')
    