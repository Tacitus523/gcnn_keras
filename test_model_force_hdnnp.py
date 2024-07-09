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

def save_poi_inputs(poi, input_configs):
    # Saves inputs as .txt
    for x, input_config in zip(poi,input_configs):
        if isinstance(x, tf.RaggedTensor):
            # Convert the RaggedTensor to a regular tensor
            regular_tensor = x.to_tensor()
            print(regular_tensor.shape)
            # Convert the regular tensor to a NumPy array
            numpy_array = regular_tensor.numpy()

            if numpy_array.ndim > 2:
                numpy_array = numpy_array.reshape(-1, numpy_array.shape[-1])

            if input_config["name"] in ["edge_indices", "range_indices", "angle_indices_nodes"]:
                numpy_array = np.array(numpy_array, dtype=np.int32)
                np.savetxt(input_config["name"]+".txt", numpy_array, fmt='%d')

            # Write the NumPy array to a file using numpy.savetxt()
            else:
                np.savetxt(input_config["name"]+".txt", numpy_array, "%3.5f")
        else:
            print(x)
            numpy_array = x.numpy()
            if numpy_array.ndim > 2:
                numpy_array = numpy_array.reshape(-1, numpy_array.shape[-1])
            np.savetxt(input_config["name"]+".txt", numpy_array, "%3.5f")


model_paths = ["model_energy_force0"]
# model_paths = ["../../../model_energy_force0", "../../../model_energy_force1", "../../../model_energy_force2"]

#data_directory="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_combined/"
#data_directory="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water"
data_directory="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
#data_directory="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water"

dataset_name="ThiolDisulfidExchange"
#dataset_name="Alanindipeptide"

file_name=f"{dataset_name}.csv"
print("Dataset:", os.path.join(data_directory, file_name))

dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=dataset_name)
dataset.load()
#dataset=dataset[:10]
np.set_printoptions(precision=5)

input_configs = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
          {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
          {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
          {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
          {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
          {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
          {"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}]

output_configs = [
    {"name": "charge", "shape": (None, 1), "ragged": True},
    {"name": "graph_labels", "ragged": False},
    {"name": "force", "shape": (None, 3), "ragged": True}
]
charge_output = {"name": "charge", "shape": (None, 1), "ragged": True}

# print(dataset[0].keys())
# poi = dataset[765]
# print(poi["node_coordinates"])
# print(len(poi["range_indices"]))
# print(len(poi["angle_indices_nodes"]))
# print(poi["esp"])
# print(poi["esp_grad"])
# X = dataset[[1300]].tensor(input_configs)
# [print(x.shape) for x in X]
# save_poi_inputs(X,input_configs)
# exit()

models = [tf.keras.models.load_model(model_path, compile=False) for model_path in model_paths]

models[0].summary()

# scaler = EnergyForceExtensiveLabelScaler()
# scaler_mapping = {"atomic_number": "node_number", "y": ["graph_labels", "force"]}
# scaler.fit_transform_dataset(dataset, **scaler_mapping)

elements = np.loadtxt("input_00.txt", dtype=np.int64)
geom = np.loadtxt("input_02.txt", dtype=np.float32)
bond_indices = np.loadtxt("input_04.txt", dtype=np.int64)
angle_indices = np.loadtxt("input_06.txt", dtype=np.int64)
total_charge = np.loadtxt("input_08.txt", dtype=np.float32)
esp = np.loadtxt("input_09.txt", dtype=np.float32)
esp_grad = np.loadtxt("input_11.txt", dtype=np.float32)
with open("output.txt") as f:
    outputs = f.readlines()
n_atoms = len(elements)
charges = np.array(outputs[:n_atoms], dtype=np.float32)
energy = np.array(outputs[n_atoms], dtype=np.float32)
forces = np.array([line.split() for line in outputs[n_atoms+1:]], dtype=np.float32)

data_point = dataset[0]
data_point.set("node_number", elements)
data_point.set("node_coordinates", geom)
data_point.set("range_indices", bond_indices)
data_point.set("angle_indices_nodes", angle_indices)
data_point.set("total_charge", total_charge)
data_point.set("esp", esp)
data_point.set("esp_grad", esp_grad)
data_point.set("charge", charges)
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

predicted_charges, predicted_energies, predicted_forces = [np.stack(outputs, axis=0) for outputs in zip(*outputs_models)]
print(predicted_charges.shape, predicted_energies.shape, predicted_forces.shape)

true_charge, true_energy, true_force = [
    np.array(output.to_list()).reshape(-1,1) if isinstance(output, tf.RaggedTensor)
    else output.numpy().reshape(-1,1)
    for output in dataset[test_index].tensor(output_configs)]
print(true_charge.shape, true_energy.shape, true_force.shape)


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
    true_true_charge = np.array(dataset[starting_structure_idx].get("charge"))
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
predicted_charge, predicted_energy, predicted_force = [output[0] for output in [predicted_charges, predicted_energies, predicted_forces]]
try:
    assert np.allclose(predicted_charge, true_charge, atol=1e-05), "Charge Assertion failed"
    assert np.allclose(predicted_energy, true_energy, atol=1e-05), "Energy Assertion failed"
    assert np.allclose(predicted_force, true_force, atol=1e-05), "Force Assertion failed"
except AssertionError:
    print("Charge")
    print(predicted_charge.reshape(1,-1))
    print(true_charge.reshape(1,-1))
    if starting_structure_idx is not None:
        print(true_true_charge.reshape(1,-1))
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
force_std = np.std(predicted_forces, axis=0) # shape(n_molecules,n_atoms,3)
qm_mlmm_std = np.loadtxt("qm_mlmm_std.xyz", skiprows=2, usecols=(2,3,4))
try:
    assert np.allclose(force_std.reshape(-1,3), qm_mlmm_std, atol=1e-04), "Std Assertion failed"
except AssertionError:
    print("Std")
    print(force_std.reshape(-1,3))
    print(qm_mlmm_std)
