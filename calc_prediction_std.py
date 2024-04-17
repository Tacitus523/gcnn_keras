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
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.model.force import EnergyForceModel
from kgcnn.utils.plots import plot_predict_true
from kgcnn.utils import constants
from kgcnn.utils.devices import set_devices_gpu

model_paths = [
    "model_energy_force0",
    "model_energy_force1",
    "model_energy_force2"
]

data_directory="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water"
dataset_name="ThiolDisulfidExchange"
# data_directory="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water"
# dataset_name="Alanindipeptide"

file_name=f"{dataset_name}.csv"
print("Dataset:", os.path.join(data_directory, file_name))

# Ability to restrict the model to only use a certain GPU, which is passed with python -g gpu_id
ap = argparse.ArgumentParser(description="Handle gpu_ids")
ap.add_argument("-g", "--gpuid", type=int)
args = ap.parse_args()
if args.gpuid is not None:
    set_devices_gpu([args.gpuid])

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

models = [tf.keras.models.load_model(model_path, compile=False) for model_path in model_paths]

predicted_charges, predicted_energies, predicted_forces = [], [], []

for model in models:
    predicted_charge, predicted_energy, predicted_force = model.predict(dataset.tensor(inputs), verbose=2)

    predicted_charges.append(predicted_charge)
    predicted_energies.append(predicted_energy)
    predicted_forces.append(predicted_force)

del models
 
predicted_charges = tf.stack(predicted_charges, axis=0) # shape(n_models,n_molecules,n_atoms,1)
predicted_energies = tf.stack(predicted_energies, axis=0) # shape(n_models,n_molecules)
predicted_forces = tf.stack(predicted_forces, axis=0) # shape(n_models,n_molecules,n_atoms,3)

charge_mean = tf.reduce_mean(predicted_charges, axis=0) # shape(n_molecules,n_atoms,1)
energy_mean = tf.reduce_mean(predicted_energies, axis=0) # shape(n_molecules)
force_mean = tf.reduce_mean(predicted_forces, axis=0) # shape(n_molecules,n_atoms,3)

charge_std = tf.math.reduce_std(tf.squeeze(predicted_charges, axis=-1), axis=0) # shape(n_molecules,n_atoms)
energy_std = tf.math.reduce_std(predicted_energies, axis=0) # shape(n_molecules)
force_std = tf.math.reduce_std(predicted_forces, axis=0) # shape(n_molecules,n_atoms,3)

mean_charge_mean = tf.reduce_mean(predicted_charges) # shape(1)
mean_energy_mean = tf.reduce_mean(predicted_energies) # shape(1)
mean_force_mean = tf.reduce_mean(predicted_forces) # shape(1)

mean_charge_std = tf.reduce_mean(charge_std) # shape(1)
mean_energy_std = tf.reduce_mean(energy_std) # shape(1)
mean_force_std = tf.reduce_mean(force_std) # shape(1)

std_charge_std = tf.math.reduce_std(charge_std) # shape(1)
std_energy_std = tf.math.reduce_std(energy_std) # shape(1)
std_force_std = tf.math.reduce_std(force_std) # shape(1)

max_charge_std = tf.reduce_max(charge_std) # shape(1)
max_energy_std = tf.reduce_max(energy_std) # shape(1)
max_force_std = tf.reduce_max(force_std) # shape(1)

force_threshold = mean_force_std + 4*std_force_std

print("Mean Charge Mean:", mean_charge_mean)
print("Mean Energy Mean:", mean_energy_mean)
print("Mean Force Mean:", mean_force_mean)

print("Mean Charge Std:", mean_charge_std)
print("Mean Energy Std:", mean_energy_std)
print("Mean Force Std:", mean_force_std)

print("Std Charge Std:", std_charge_std)
print("Std Energy Std:", std_energy_std)
print("Std Force Std:", std_force_std)

print("Max Charge Std:", max_charge_std)
print("Max Energy Std:", max_energy_std)
print("Max Force Std:", max_force_std)

print("Force Threshold:", force_threshold)

# poi = 12395
# print(dataset.tensor(inputs)[1][poi]/constants.angstrom_to_bohr/10)
# print(force_std[poi])

# true_charge = np.array(dataset.get("charge")).reshape(-1,1)
# true_energy = np.array(dataset.get("graph_labels")).reshape(-1,1)*constants.hartree_to_kcalmol
# true_force = np.array(dataset.get("force")).reshape(-1,1)

# predicted_charge = np.array(predicted_charge).reshape(-1,1)
# predicted_energy = np.array(predicted_energy).reshape(-1,1)*constants.hartree_to_kcalmol
# predicted_force = np.array(predicted_force).reshape(-1,1)

# plot_predict_true(predicted_charge, true_charge,
#     filepath="", data_unit="e",
#     model_name="HDNNP", dataset_name=dataset_name, target_names="Charge",
#     error="RMSE", file_name=f"predict_charge_std_calc.png", show_fig=False)

# plot_predict_true(predicted_energy, true_energy,
#     filepath="", data_unit=r"$\frac{kcal}{mol}$",
#     model_name="HDNNP", dataset_name=dataset_name, target_names="Energy",
#     error="RMSE", file_name=f"predict_energy_std_calc.png", show_fig=False)

# plot_predict_true(predicted_force, true_force,
#     filepath="", data_unit="Eh/B",
#     model_name="HDNNP", dataset_name=dataset_name, target_names="Force",
#     error="RMSE", file_name=f"predict_force_std_calc.png", show_fig=False)