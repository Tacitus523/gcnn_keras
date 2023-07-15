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
from kgcnn.model.force import EnergyForceModel, NonRaggedEnergyForceModel, make_model_non_ragged_energy_force

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

# # Saves inputs are .txt
# for x, input_dict in zip(X,inputs):
#     if isinstance(x, tf.RaggedTensor):
#         # Convert the RaggedTensor to a regular tensor
#         regular_tensor = x.to_tensor()
#         print(regular_tensor.shape)
#         # Convert the regular tensor to a NumPy array
#         numpy_array = regular_tensor.numpy()

#         if numpy_array.ndim > 2:
#             numpy_array = numpy_array.reshape(-1, numpy_array.shape[-1])

#         if input_dict["name"] == "angle_indices_nodes":
#             numpy_array = np.array(numpy_array, dtype=np.int32)
#             np.savetxt(input_dict["name"]+".txt", numpy_array, fmt='%d')

#         # Write the NumPy array to a file using numpy.savetxt()
#         else:
#             np.savetxt(input_dict["name"]+".txt", numpy_array)
#     else:
#         print(x)
#         numpy_array = x.numpy()
#         if numpy_array.ndim > 2:
#             numpy_array = numpy_array.reshape(-1, numpy_array.shape[-1])
#         np.savetxt(input_dict["name"]+".txt", numpy_array)


[print(x.shape) for x in X]

model_energy_force = tf.keras.models.load_model("model_energy_force", compile=False)

# results = model_energy_force.predict(X)
# print(results)
# [print(type(y)) for y in results.values()]
# [print(y.shape) for y in results.values()]
# [print(y.dtype) for y in results.values()]
# print(type(X[0]))

#model_energy_force = NonRaggedEnergyForceModel(model_energy_force)
#model_energy_force = make_model_non_ragged_energy_force(inputs, model_energy_force)

predicted_y = model_energy_force.predict(X)
print(predicted_y)
model_energy_force.summary()
#model_energy_force.save("model_non_ragged_energy_force")
