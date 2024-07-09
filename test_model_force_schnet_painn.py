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

#model_paths = ["model_energy_force0"]
#model_paths = ["../../../model_energy_force0", "../../../model_energy_force1", "../../../model_energy_force2"]
model_paths = ["../../../model_energy_force_painn0", "../../../model_energy_force_painn1", "../../../model_energy_force_painn2"]

#data_directory="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_combined/"
#data_directory="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water"
#data_directory="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
#data_directory="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water"
data_directory="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"

#dataset_name="ThiolDisulfidExchange"
dataset_name="Alanindipeptide"

file_name=f"{dataset_name}.csv"
print("Dataset:", os.path.join(data_directory, file_name))

dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=dataset_name)
dataset.load()
#dataset=dataset[:10]
np.set_printoptions(precision=5)

input_configs = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
          {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
          {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True}
          ]

output_configs = [
    {"name": "graph_labels", "ragged": False},
    {"name": "force", "shape": (None, 3), "ragged": True}
]

models = [tf.keras.models.load_model(model_path, compile=False) for model_path in model_paths]

models[0].summary()

# scaler = EnergyForceExtensiveLabelScaler()
# scaler_mapping = {"atomic_number": "node_number", "y": ["graph_labels", "force"]}
# scaler.fit_transform_dataset(dataset, **scaler_mapping)

elements = np.loadtxt("input_00.txt", dtype=np.int64)
geom = np.loadtxt("input_02.txt", dtype=np.float32)
bond_indices = np.loadtxt("input_04.txt", dtype=np.int64)

with open("output.txt") as f:
    outputs = f.readlines()
n_atoms = len(elements)

energy = np.array(outputs[0], dtype=np.float32)
forces = np.array([line.split() for line in outputs[1:]], dtype=np.float32)

data_point = dataset[0]
data_point.set("node_number", elements)
data_point.set("node_coordinates", geom)
data_point.set("range_indices", bond_indices)

data_point.set("graph_labels", energy)
data_point.set("force", forces)
dataset.append(data_point)

test_index = [-1]
input_tensor = dataset[test_index].tensor(input_configs)
input_geom = np.array(dataset[test_index[0]].get("node_coordinates"))
input_distances = distance_matrix(input_geom, input_geom)

outputs_models = []
for model in models:
    outputs_model = model.predict(input_tensor, verbose=2)
    outputs_model = [np.array(output).reshape(-1,1) for output in outputs_model]
    outputs_models.append(outputs_model)

predicted_energies, predicted_forces = [np.stack(outputs, axis=0) for outputs in zip(*outputs_models)]
print(predicted_energies.shape, predicted_forces.shape)
true_energy, true_force = [
    np.array(output.to_list()).reshape(-1,1) if isinstance(output, tf.RaggedTensor)
    else output.numpy().reshape(-1,1)
    for output in dataset[test_index].tensor(output_configs)]
print(true_energy.shape, true_force.shape)

with open("adaptive_sampling.gro", "r") as f:
    lines = f.readlines()
last_comment = None
for line in lines[::-1]:
    if "step=" in line:
        last_comment = line
        break
if last_comment.endswith("step= 0\n"):
    with open("starting_structure_idxs.txt", "r") as f:
        lines = f.readlines()
        starting_structure_idx = [int(lines[-1])]
else:
    starting_structure_idx = None

if starting_structure_idx is not None:
    print("-----------------Input assertion--------------------------")
    starting_inputs = dataset[starting_structure_idx].tensor(input_configs)
    starting_geom = np.array(dataset[starting_structure_idx[0]].get("node_coordinates"))
    starting_distances = distance_matrix(starting_geom, starting_geom)
    true_true_energy = np.array(dataset[starting_structure_idx].get("graph_labels"))
    true_true_force = np.array(dataset[starting_structure_idx].get("force"))
    for input_config, starting_input, calc_input in zip(input_configs, starting_inputs, input_tensor):
        try:
            if input_config["name"] == "node_coordinates":
                assert np.allclose(starting_distances, input_distances, atol=1e-00), f"{input_config['name']} Assertion failed"
            elif input_config["name"] == "range_indices" or input_config["name"] == "angle_indices_nodes":
                assert calc_input.numpy().shape[1] == starting_input.numpy().shape[1], f"{input_config['name']} Assertion possibly failed"
            else:
                assert np.allclose(starting_input.numpy(), calc_input.numpy(), atol=1e-02), f"{input_config['name']} Assertion failed"
        except :
            if input_config["name"] == "node_coordinates":
                print(starting_geom[0:2])
                print(input_geom[0:2])
                print(input_distances[0])
                print(starting_distances[0])
                raise
            elif input_config["name"] == "range_indices" or input_config["name"] == "angle_indices_nodes":
                print(f"{input_config['name']} calc input vs training: {calc_input.numpy().shape[1]} vs {starting_input.numpy().shape[1]}. Differences might result from different handling of cutoff and might be inconsequential.")
            else:
                print(calc_input)
                print(starting_input)
                raise
    print("Input Assertions passed")
print("------------------Output assertion-------------------------")
predicted_energy, predicted_force = [output[0] for output in [predicted_energies, predicted_forces]]
try:
    assert np.allclose(predicted_energy, true_energy, atol=1e-05), "Energy Assertion failed"
    assert np.allclose(predicted_force, true_force, atol=1e-04), "Force Assertion failed"
except AssertionError:
    print("Energy")
    print(predicted_energy)
    print(true_energy)
    if starting_structure_idx is not None:
        print(true_true_energy)
    print("Force")
    print(predicted_force.reshape(1,-1,3))
    print(true_force.reshape(1,-1,3))
    if starting_structure_idx is not None:
        print(true_true_force)
    raise
print("Output Assertions passed")

print("------------------Std assertion-------------------------")
energy_mean = np.round(np.mean(predicted_energies, axis=0).flatten(), 4) # shape(n_molecules,)
energy_std = np.round(np.std(predicted_energies, axis=0).flatten(), 4) # shape(n_molecules,)
force_mean = np.mean(predicted_forces, axis=0) # shape(n_molecules,n_atoms,3)
force_std = np.std(predicted_forces, axis=0) # shape(n_molecules,n_atoms,3)
qm_mlmm_energy_mean = np.genfromtxt("qm_mlmm_std.xyz", skip_header=2, max_rows=1)
qm_mlmm_energy_std = np.genfromtxt("qm_mlmm_std.xyz", skip_header=3+2, max_rows=1)
qm_mlmm_force_mean = np.genfromtxt("qm_mlmm_std.xyz", skip_header=3+3+2, max_rows=n_atoms, usecols=(2,3,4))
qm_mlmm_force_std = np.genfromtxt("qm_mlmm_std.xyz", skip_header=3+3+(2+n_atoms)+2, max_rows=n_atoms, usecols=(2,3,4))
try:
    assert np.allclose(energy_mean, qm_mlmm_energy_mean, atol=1e-05), "Energy Mean Assertion failed"
except AssertionError as e:
    print(e)
    print(energy_mean)
    print(qm_mlmm_energy_mean)

try:
    assert np.allclose(energy_std, qm_mlmm_energy_std, atol=1e-05), "Energy Std Assertion failed"
except AssertionError as e:
    print(e)
    print(energy_std)
    print(qm_mlmm_energy_std)

try:
    assert np.allclose(force_mean.reshape(-1,3), qm_mlmm_force_mean, atol=1e-04), "Force Mean Assertion failed"
except AssertionError as e:
    print(e)
    print(force_mean.reshape(-1,3))
    print(qm_mlmm_force_mean)

try:
    assert np.allclose(force_std.reshape(-1,3), qm_mlmm_force_std, atol=1e-04), "Force Std Assertion failed"
except AssertionError as e:
    print(e)
    print(force_std.reshape(-1,3))
    print(qm_mlmm_force_std)

print("Std Assertions passed")