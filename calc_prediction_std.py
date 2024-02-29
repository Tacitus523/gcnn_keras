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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(tf.config.list_physical_devices('GPU'))

from kgcnn.data.base import MemoryGraphList, MemoryGraphDataset
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.model.force import EnergyForceModel
from kgcnn.utils.plots import plot_predict_true

data_directory="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_combined/"
#data_directory="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/"
dataset_name="ThiolDisulfidExchange"

file_name=f"{dataset_name}.csv"
print("Dataset:", data_directory+file_name)

dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=dataset_name)
dataset.load()
#dataset=dataset[:10]

inputs = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
          {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
          {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
          {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
          {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
          {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
          {"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}]

outputs = [
    {"name": "charge", "shape": (None, 1), "ragged": True},
    {"name": "graph_labels", "ragged": False},
    {"name": "force", "shape": (None, 3), "ragged": True}
]
charge_output = {"name": "charge", "shape": (None, 1), "ragged": True}

model_paths = [
    "/data/lpetersen/Behler_training/thiol_disulfide/07_esp_derivative/B3LYP_aug-cc-pVTZ_combined/multi_kgcnn_vacuum_minimum/model_energy_force0",
    "/data/lpetersen/Behler_training/thiol_disulfide/07_esp_derivative/B3LYP_aug-cc-pVTZ_combined/multi_kgcnn_vacuum_minimum/model_energy_force1",
    "/data/lpetersen/Behler_training/thiol_disulfide/07_esp_derivative/B3LYP_aug-cc-pVTZ_combined/multi_kgcnn_vacuum_minimum/model_energy_force2"
]

models = [tf.keras.models.load_model(model_path, compile=False) for model_path in model_paths]

predicted_charges, predicted_energies, predicted_forces = [], [], []
for model in models:
    predicted_charge, predicted_energy, predicted_force = model.predict(dataset.tensor(inputs), verbose=0)

    predicted_charges.append(predicted_charge)
    predicted_energies.append(predicted_energy)
    predicted_forces.append(predicted_force)

del models
 
predicted_charges = tf.stack(predicted_charges, axis=0)
predicted_energies = tf.stack(predicted_energies, axis=0)
predicted_forces = tf.stack(predicted_forces, axis=0)

charge_mean = tf.reduce_mean(predicted_charges, axis=0)
energy_mean = tf.reduce_mean(predicted_energies, axis=0)
force_mean = tf.reduce_mean(predicted_forces, axis=0)

charge_std = tf.math.reduce_std(tf.squeeze(predicted_charges, axis=-1), axis=0)
energy_std = tf.math.reduce_std(predicted_energies, axis=0)
force_std = tf.math.reduce_std(predicted_forces, axis=0)


max_charge_std = tf.reduce_max(charge_std)
max_energy_std = tf.reduce_max(energy_std)
max_force_std = tf.reduce_max(force_std)

print("Max Charge Std:", max_charge_std)
print("Max Energy Std:", max_energy_std)
print("Max Force Std:", max_force_std)

true_charge = np.array(dataset.get("charge")).reshape(-1,1)
predicted_charge = np.array(predicted_charge).reshape(-1,1)
plot_predict_true(predicted_charge, true_charge,
    filepath="", data_unit="e",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Charge",
    error="RMSE", file_name=f"predict_charge_full_dataset.png", show_fig=False)