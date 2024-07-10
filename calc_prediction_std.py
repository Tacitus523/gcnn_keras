import argparse
from datetime import timedelta
import os
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

MODEL_PATHS = [
    "model_energy_force_painn0",
    "model_energy_force_painn1",
    "model_energy_force_painn2"
]

# DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water"
# DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
# DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/07_vacuum_sampling_retry/adaptive_sampling/current_training_data"
DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
# DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water"
# DATASET_NAME="ThiolDisulfidExchange"
DATASET_NAME="Alanindipeptide"

USE_SCALER = True
SCALER_PATH = "scaler.json"

file_name=f"{DATASET_NAME}.csv"
print("Dataset:", os.path.join(DATA_DIRECTORY, file_name))

# Ability to restrict the model to only use a certain GPU, which is passed with python -g gpu_id
ap = argparse.ArgumentParser(description="Handle gpu_ids")
ap.add_argument("-g", "--gpuid", type=int)
args = ap.parse_args()
if args.gpuid is not None:
    set_devices_gpu([args.gpuid])

dataset = MemoryGraphDataset(data_directory=DATA_DIRECTORY, dataset_name=DATASET_NAME)
dataset.load()
#dataset=dataset[:10]

inputs = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
          {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
          {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
          #{"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
          #{"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
          #{"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
          #{"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}
          ]

outputs = [
    #{"name": "charge", "shape": (None, 1), "ragged": True},
    {"name": "graph_labels", "ragged": False},
    {"name": "force", "shape": (None, 3), "ragged": True}
]
charge_output = {"name": "charge", "shape": (None, 1), "ragged": True}

models = [tf.keras.models.load_model(model_path, compile=False) for model_path in MODEL_PATHS]

#predicted_charges = []
predicted_energies, predicted_forces = [], []

for model in models:
    #predicted_charge, predicted_energy, predicted_force = model.predict(dataset.tensor(inputs), batch_size=128, verbose=2)
    predicted_energy, predicted_force = model.predict(dataset.tensor(inputs), batch_size=128, verbose=2)

    #predicted_charges.append(predicted_charge)
    predicted_energies.append(predicted_energy)
    predicted_forces.append(predicted_force)

del models

if USE_SCALER:
    scaler = EnergyForceExtensiveLabelScaler()
    scaler.load(SCALER_PATH)
    for index,(predicted_energy, predicted_force) in enumerate(zip(predicted_energies, predicted_forces)):
        predicted_energy, predicted_force = scaler.inverse_transform(
            y=(predicted_energy.flatten(), predicted_force), X=dataset.get("node_number"))
        predicted_energies[index] = predicted_energy
        predicted_forces[index] = predicted_force
 
#predicted_charges = tf.stack(predicted_charges, axis=0) # shape(n_models,n_molecules,n_atoms,1)
predicted_energies = tf.stack(predicted_energies, axis=0) # shape(n_models,n_molecules)
predicted_forces = tf.stack(predicted_forces, axis=0) # shape(n_models,n_molecules,n_atoms,3)

#charge_mean = tf.reduce_mean(predicted_charges, axis=0) # shape(n_molecules,n_atoms,1)
energy_mean = tf.reduce_mean(predicted_energies, axis=0) # shape(n_molecules)
force_mean = tf.reduce_mean(predicted_forces, axis=0) # shape(n_molecules,n_atoms,3)

#charge_std = tf.math.reduce_std(tf.squeeze(predicted_charges, axis=-1), axis=0) # shape(n_molecules,n_atoms)
energy_std = tf.math.reduce_std(predicted_energies, axis=0) # shape(n_molecules)
force_std = tf.math.reduce_std(predicted_forces, axis=0) # shape(n_molecules,n_atoms,3)

#mean_charge_mean = tf.reduce_mean(predicted_charges) # shape(1)
mean_energy_mean = tf.reduce_mean(predicted_energies) # shape(1)
mean_force_mean = tf.reduce_mean(predicted_forces) # shape(1)

#mean_charge_std = tf.reduce_mean(charge_std) # shape(1)
mean_energy_std = tf.reduce_mean(energy_std) # shape(1)
mean_force_std = tf.reduce_mean(force_std) # shape(1)

#std_charge_std = tf.math.reduce_std(charge_std) # shape(1)
std_energy_std = tf.math.reduce_std(energy_std) # shape(1)
std_force_std = tf.math.reduce_std(force_std) # shape(1)

#max_charge_std = tf.reduce_max(charge_std) # shape(1)
max_energy_std = tf.reduce_max(energy_std) # shape(1)
max_force_std = tf.reduce_max(force_std) # shape(1)

energy_threshold = mean_energy_std + 5*std_energy_std
force_threshold = mean_force_std + 10*std_force_std

#print("Mean Charge Mean:", mean_charge_mean)
print("Mean Energy Mean:", mean_energy_mean)
print("Mean Force Mean:", mean_force_mean)

#print("Mean Charge Std:", mean_charge_std)
print("Mean Energy Std:", mean_energy_std)
print("Mean Force Std:", mean_force_std)

#print("Std Charge Std:", std_charge_std)
print("Std Energy Std:", std_energy_std)
print("Std Force Std:", std_force_std)

#print("Max Charge Std:", max_charge_std)
print("Max Energy Std:", max_energy_std)
print("Max Force Std:", max_force_std)

print("Suggested Energy Treshold:", energy_threshold)
print("Suggested Force Threshold:", force_threshold)


energy_dict = {
    "Energy Std": tf.reshape(energy_std,(-1,))
}
energy_df = pd.DataFrame(energy_dict)

# plt.hist(tf.reshape(force_std,(-1,)), bins=50, color='skyblue', edgecolor='black')
sns.histplot(data=energy_df, x="Energy Std")
plt.xlabel('Energy Standard Deviation')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_std_histogram.png")
plt.close()

force_dict = {
    "Force Std": tf.reshape(force_std,(-1,)),
    "Atom Types": np.array(dataset.get("node_number")).repeat(3).flatten()
}
force_df = pd.DataFrame(force_dict)
force_df["Atom Types"] = force_df["Atom Types"].replace(constants.atomic_number_to_element)

sns.histplot(data=force_df, x="Force Std", hue="Atom Types", stat='probability', common_norm=False)
plt.xlabel('Force Standard Deviation')
plt.ylabel('Probability')
plt.grid(True)
plt.tight_layout()
plt.savefig("force_std_histogram.png")
plt.close()