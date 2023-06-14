import argparse
from datetime import timedelta
import os
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
ks=tf.keras
print(tf.config.list_physical_devices('GPU'))

from kgcnn.data.base import MemoryGraphList, MemoryGraphDataset

data_directory="data/B3LYP_aug-cc-pVTZ_water/"
dataset_name="ThiolDisulfidExchange"

file_name=f"{dataset_name}.csv"
print("Dataset:", data_directory+file_name)

data_directory = os.path.join(os.path.dirname(__file__), data_directory)
dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=dataset_name)
dataset.load()
print(dataset[0].keys())


inputs = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
            {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
            {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True}]

X = dataset[[0,1]].tensor(inputs)
[print(x.shape) for x in X]

model_energy_force = tf.keras.models.load_model("model_energy_force", compile=False)

results = model_energy_force.predict(X)

[print(y.shape for y in results.values())]
