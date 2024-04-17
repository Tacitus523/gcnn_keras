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

def save_poi_inputs(poi, inputs):
    # Saves inputs as .txt
    for x, input_dict in zip(poi,inputs):
        if isinstance(x, tf.RaggedTensor):
            # Convert the RaggedTensor to a regular tensor
            regular_tensor = x.to_tensor()
            print(regular_tensor.shape)
            # Convert the regular tensor to a NumPy array
            numpy_array = regular_tensor.numpy()

            if numpy_array.ndim > 2:
                numpy_array = numpy_array.reshape(-1, numpy_array.shape[-1])

            if input_dict["name"] in ["edge_indices", "range_indices", "angle_indices_nodes"]:
                numpy_array = np.array(numpy_array, dtype=np.int32)
                np.savetxt(input_dict["name"]+".txt", numpy_array, fmt='%d')

            # Write the NumPy array to a file using numpy.savetxt()
            else:
                np.savetxt(input_dict["name"]+".txt", numpy_array, "%3.5f")
        else:
            print(x)
            numpy_array = x.numpy()
            if numpy_array.ndim > 2:
                numpy_array = numpy_array.reshape(-1, numpy_array.shape[-1])
            np.savetxt(input_dict["name"]+".txt", numpy_array, "%3.5f")


model_path = "../../../model_energy_force0"

#data_directory="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_combined/"
data_directory="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water"
#data_directory="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water"

dataset_name="ThiolDisulfidExchange"
#dataset_name="Alanindipeptide"

file_name=f"{dataset_name}.csv"
print("Dataset:", os.path.join(data_directory, file_name))

dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=dataset_name)
dataset.load()
#dataset=dataset[:10]
np.set_printoptions(precision=5)

# print(dataset[0].keys())
# poi = dataset[765]
# print(poi["node_coordinates"])
# print(len(poi["range_indices"]))
# print(len(poi["angle_indices_nodes"]))
# print(poi["esp"])
# print(poi["esp_grad"])
# exit()

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

# X = dataset[[1300]].tensor(inputs)
# [print(x.shape) for x in X]
# save_poi_inputs(X,inputs)

@ks.utils.register_keras_serializable(package="kgcnn", name="zero_loss_function")
def zero_loss_function(y_true, y_pred):
    return 0

model = tf.keras.models.load_model(model_path, compile=False)

model.summary()

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
input_tensor = dataset[test_index].tensor(inputs)
input_geom = np.array(dataset[test_index[0]].get("node_coordinates"))
input_distances = distance_matrix(input_geom, input_geom)
predicted_charge, predicted_energy, predicted_force= model.predict(input_tensor, verbose=2)
predicted_charge = np.array(predicted_charge).reshape(-1,1)
predicted_energy = np.array(predicted_energy).reshape(-1,1)
predicted_force = np.array(predicted_force).reshape(-1,1)
true_charge = np.array(dataset[test_index].get("charge")).reshape(-1,1)
true_energy = np.array(dataset[test_index].get("graph_labels")).reshape(-1,1)
true_force = np.array(dataset[test_index].get("force")).reshape(-1,1)

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
    starting_inputs = dataset[starting_structure_idx].tensor(inputs)
    starting_geom = np.array(dataset[starting_structure_idx[0]].get("node_coordinates"))
    starting_distances = distance_matrix(starting_geom, starting_geom)
    true_true_charge = np.array(dataset[starting_structure_idx].get("charge"))
    true_true_energy = np.array(dataset[starting_structure_idx].get("graph_labels"))
    true_true_force = np.array(dataset[starting_structure_idx].get("force"))
    for network_input_dict, starting_input, calc_input in zip(inputs, starting_inputs, input_tensor):
        try:
            if network_input_dict["name"] == "node_coordinates":
                assert np.allclose(starting_distances, input_distances, atol=1e-00), f"{network_input_dict['name']} Assertion failed"
            elif network_input_dict["name"] == "range_indices" or network_input_dict["name"] == "angle_indices_nodes":
                assert calc_input.numpy().shape[1] == starting_input.numpy().shape[1], f"{network_input_dict['name']} Assertion possibly failed"
            else:
                assert np.allclose(starting_input.numpy(), calc_input.numpy(), atol=1e-02), f"{network_input_dict['name']} Assertion failed"
        except :
            if network_input_dict["name"] == "node_coordinates":
                print(starting_geom[0:2])
                print(input_geom[0:2])
                print(input_distances[0])
                print(starting_distances[0])
                raise
            elif network_input_dict["name"] == "range_indices" or network_input_dict["name"] == "angle_indices_nodes":
                print(f"{network_input_dict['name']} calc input vs training: {calc_input.numpy().shape[1]} vs {starting_input.numpy().shape[1]}. Differences might result from different handling of cutoff and might be inconsequential.")
            else:
                print(calc_input)
                print(starting_input)
                raise
    print("Input Assertions passed")
print("------------------Output assertion-------------------------")
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
exit()

kf = KFold(n_splits=3, random_state=42, shuffle=True)

for test_index, train_index in kf.split(X=np.expand_dims(np.array(dataset.get("graph_labels")), axis=-1)): # Switched train and test indices to keep training data separate, Could also be read from the json-file now
    predicted_charge, predicted_energy, predicted_force= model.predict(dataset[test_index].tensor(inputs), verbose=2)
    break

#scaler.inverse_transform_dataset(dataset, **scaler_mapping)
true_charge = np.array(dataset[test_index].get("charge")).reshape(-1,1)
true_energy = np.array(dataset[test_index].get("graph_labels")).reshape(-1,1)*constants.hartree_to_kcalmol
true_force = np.array(dataset[test_index].get("force")).reshape(-1,1)

predicted_charge = np.array(predicted_charge).reshape(-1,1)
predicted_energy = np.array(predicted_energy).reshape(-1,1)*constants.hartree_to_kcalmol
predicted_force = np.array(predicted_force).reshape(-1,1)

plot_predict_true(predicted_charge, true_charge,
    filepath="", data_unit="e",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Charge",
    error="RMSE", file_name=f"predict_charge_full_dataset.png", show_fig=False)

plot_predict_true(predicted_energy, true_energy,
    filepath="", data_unit=r"$\frac{kcal}{mol}$",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Energy",
    error="RMSE", file_name=f"predict_energy_full_dataset.png", show_fig=False)

plot_predict_true(predicted_force, true_force,
    filepath="", data_unit="Eh/B",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Force",
    error="RMSE", file_name=f"predict_force_full_dataset.png", show_fig=False)