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
from kgcnn.model.force import EnergyForceModel

data_directory="data/B3LYP_aug-cc-pVTZ_water/"
dataset_name="ThiolDisulfidExchange"

file_name=f"{dataset_name}.csv"
print("Dataset:", data_directory+file_name)

data_directory = os.path.join(os.path.dirname(__file__), data_directory)
dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=dataset_name)
dataset.load()
print(dataset[0].keys())
poi = dataset[765]
print(poi["node_coordinates"])
print(len(poi["range_indices"]))
print(len(poi["angle_indices_nodes"]))
print(poi["esp"])
print(poi["esp_grad"])
exit()

# to remove esp data
# print("Setting ESP to 0")
# for i in range(len(dataset)):
#    dataset[i].set("esp", np.zeros_like(dataset[i]["node_number"],dtype=np.float64))
# esp_0 = [-0.422, 0.064, -0.762, -0.262, 0.573, 0.781, 1.541, 1.140, 0.510, 0.473, 0.491, 0.236, -0.625, 0.217, 0.983]
# geom_0 = np.array([
# [22.903, 17.839, 21.241],
# [21.448, 18.519, 18.274],
# [24.718, 18.689, 21.203],
# [21.751, 18.538, 22.809],
# [15.250, 21.316, 15.893],
# [23.187, 24.415, 24.321],
# [25.492, 27.061, 23.962],
# [21.467, 25.209, 23.565],
# [23.546, 22.941, 22.941],
# [17.102, 18.387, 15.571],
# [16.006, 19.672, 16.989],
# [18.122, 21.108, 19.426],
# [23.225, 15.798, 21.354],
# [23.149, 23.886, 26.324],
# [14.551, 18.425, 17.877]])
# dataset[0].set("esp", esp_0)
# dataset[0].set("node_coordinates", geom_0)

esp_1 = [-0.008, -0.226, -0.376, -0.096, -0.251, 0.203, 0.441, 0.244, 0.464, -0.400, -0.456, 0.274, 0.377, 0.046, -0.717]
geom_1 = np.array([
[210.402, 205.186, 208.437],
[208.947, 205.867, 205.470],
[212.216, 206.037, 208.399],
[209.249, 205.886, 210.005],
[202.749, 208.664, 203.089],
[210.686, 211.763, 211.517],
[212.991, 214.408, 211.158],
[208.966, 212.556, 210.761],
[211.045, 210.289, 210.138],
[204.601, 205.734, 202.768],
[203.505, 207.020, 204.185],
[205.621, 208.456, 206.623],
[210.723, 203.146, 208.550],
[210.648, 211.234, 213.520],
[202.050, 205.772, 205.073]])
esp_grad_1 = np.array([
[-0.014, -0.005, -0.006],
[ 0.006,  0.002,  0.003],
[ 0.008,  0.002,  0.003],
[-0.002, -0.004, -0.026],
[ 0.002,  0.003,  0.015],
[ 0.001,  0.002,  0.011],
[-0.013, -0.008, -0.009],
[ 0.007,  0.004,  0.004],
[ 0.007,  0.004,  0.005],
[-0.011, -0.012,  0.003],
[ 0.006,  0.006, -0.002],
[ 0.006,  0.007, -0.001],
[-0.017, -0.009,  0.003],
[ 0.010,  0.005, -0.002],
[ 0.008,  0.004, -0.001]])
dataset[0].set("esp", esp_1)
dataset[0].set("node_coordinates", geom_1)
dataset[0].set("esp_grad", esp_grad_1)

inputs = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
          {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
          {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
          {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
          {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
          {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
          {"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}]

X = dataset[[1300]].tensor(inputs)

# Saves inputs as .txt
for x, input_dict in zip(X,inputs):
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


[print(x.shape) for x in X]

model = tf.keras.models.load_model("model_mlmm", compile=False)

# Get the number of outputs from the last layer
# results = model_energy_force.predict(X)
# print(results)
# [print(type(y)) for y in results.values()]
# [print(y.shape) for y in results.values()]
# [print(y.dtype) for y in results.values()]
# print(type(X[0]))

predicted_y = model.predict(X)
true_charge = np.array(dataset[1300].get("charge"))
true_energy = np.array(dataset[1300].get("graph_labels"))
true_force = np.array(dataset[1300].get("force"))
print(predicted_y)
print(true_charge)
print(true_energy)
print(true_force)

model.summary()