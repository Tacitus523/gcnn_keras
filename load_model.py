import argparse
from datetime import timedelta
import os
import time
import warnings
from scipy.spatial import distance_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
ks=tf.keras
print(tf.config.list_physical_devices('GPU'))

from kgcnn.graph.base import GraphDict
from kgcnn.data.base import MemoryGraphList, MemoryGraphDataset
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.model.force import EnergyForceModel
from kgcnn.utils.plots import plot_predict_true
from kgcnn.utils import constants

# MODEL_PATHS = ["model_energy_force0"]
# MODEL_PATHS = ["model_energy_force_painn0"]
MODEL_PATHS = ["../../../model_energy_force0", "../../../model_energy_force1", "../../../model_energy_force2"]

#DATA_DIRECTORY = "/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_combined/"
#DATA_DIRECTORY = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water"
#DATA_DIRECTORY = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
#DATA_DIRECTORY = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water"
DATA_DIRECTORY = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"

#DATASET_NAME = "ThiolDisulfidExchange"
DATASET_NAME = "Alanindipeptide"

USE_SCALER = False
SCALER_PATH = "scaler.json"

file_name=f"{DATASET_NAME}.csv"
print("Dataset:", os.path.join(DATA_DIRECTORY, file_name))

dataset = MemoryGraphDataset(data_directory=DATA_DIRECTORY, dataset_name=DATASET_NAME)
dataset.load()
#dataset=dataset[:10]
np.set_printoptions(precision=5)

input_configs = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
          {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
          {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
          #{"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
          #{"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
          #{"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
          #{"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}
          ]

output_configs = [
    #{"name": "charge", "shape": (None, 1), "ragged": True},
    {"name": "graph_labels", "ragged": False},
    {"name": "force", "shape": (None, 3), "ragged": True}
]
charge_output = {"name": "charge", "shape": (None, 1), "ragged": True}

models = [tf.keras.models.load_model(model_path, compile=False) for model_path in MODEL_PATHS]

models[0].summary()

kf = KFold(n_splits=3, random_state=42, shuffle=True)

for train_index, test_index in kf.split(X=np.expand_dims(np.array(dataset.get("graph_labels")), axis=-1)): 
    test_index, train_index = train_index, test_index # Switched train and test indices to keep training data separate, Could also be read from the json-file now
    #predicted_charge, predicted_energy, predicted_force= models[0].predict(dataset[test_index].tensor(input_configs), batch_size=128, verbose=2)
    predicted_energy, predicted_force= models[0].predict(dataset[test_index].tensor(input_configs), batch_size=128, verbose=2)
    break

if USE_SCALER:
    scaler = EnergyForceExtensiveLabelScaler()
    scaler.load(SCALER_PATH)
    predicted_energy, predicted_force = scaler.inverse_transform(
        y=(predicted_energy.flatten(), predicted_force), X=dataset[test_index].get("node_number"))

#true_charge = np.array(dataset[test_index].get("charge")).reshape(-1,1)
true_energy = np.array(dataset[test_index].get("graph_labels")).reshape(-1,1)*constants.hartree_to_kcalmol
true_force = np.array(dataset[test_index].get("force")).reshape(-1,1)

#predicted_charge = np.array(predicted_charge).reshape(-1,1)
predicted_energy = np.array(predicted_energy).reshape(-1,1)*constants.hartree_to_kcalmol
predicted_force = np.array(predicted_force).reshape(-1,1)

# plot_predict_true(predicted_charge, true_charge,
#     filepath="", data_unit="e",
#     model_name="HDNNP", dataset_name=dataset_name, target_names="Charge",
#     error="RMSE", file_name=f"predict_charge_full_dataset.png", show_fig=False)

plot_predict_true(predicted_energy, true_energy,
    filepath="", data_unit=r"$\frac{kcal}{mol}$",
    model_name="HDNNP", dataset_name=DATASET_NAME, target_names="Energy",
    error="RMSE", file_name=f"predict_energy_full_dataset.png", show_fig=False)

plot_predict_true(predicted_force, true_force,
    filepath="", data_unit="Eh/B",
    model_name="HDNNP", dataset_name=DATASET_NAME, target_names="Force",
    error="RMSE", file_name=f"predict_force_full_dataset.png", show_fig=False)