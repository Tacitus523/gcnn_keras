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

from kgcnn.graph.base import GraphDict
from kgcnn.data.base import MemoryGraphList, MemoryGraphDataset
from kgcnn.data.qm import QMDataset
from kgcnn.training.scheduler import LinearLearningRateScheduler
from kgcnn.literature.HDNNP4th import make_model_behler as make_model
from kgcnn.literature.HDNNP4th import make_model_rep, make_model_learn
from kgcnn.data.transform.scaler.mol import ExtensiveMolecularScaler
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss
from kgcnn.utils.devices import set_devices_gpu

#data_directory="data/B3LYP_def2-TZVPP_water"
data_directory="data/B3LYP_aug-cc-pVTZ_vacuum"
dataset_name="ThiolDisulfidExchange"

file_name=f"{dataset_name}.csv"
print("Dataset:", data_directory+file_name)

# Ability to restrict the model to only use a certain GPU, which is passed with python -g gpu_id
ap = argparse.ArgumentParser(description="Handle gpu_ids")
ap.add_argument("-g", "--gpuid", type=int)
args = ap.parse_args()
if args.gpuid is not None:
    set_devices_gpu([args.gpuid])

data_directory = os.path.join(os.path.dirname(__file__), data_directory)
dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=dataset_name)
dataset.load()
print(dataset[0].keys())

# to remove esp data
#print("Setting ESP to 0")
#for i in range(len(dataset)):
#    dataset[i].set("esp", np.zeros_like(dataset[i]["node_number"],dtype=np.float64))

# model_config = {
#     "name": "HDNNP4th",
#     "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
#                {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
#                {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
#                {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
#                {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
#                {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True}],
#     "g2_kwargs": {"eta": [0.0, 0.3], "rs": [0.0, 3.0], "rc": 10.0, "elements": [1, 6, 16]},
#     "g4_kwargs": {"eta": [0.0, 0.3], "lamda": [-1.0, 1.0], "rc": 6.0,
#                   "zeta": [1.0, 8.0], "elements": [1, 6, 16], "multiplicity": 2.0},
#     "normalize_kwargs": {},
#     "mlp_charge_kwargs": {"units": [64, 64, 1],
#                           "num_relations": 96,
#                           "activation": ["swish", "swish", "linear"]},
#     "mlp_local_kwargs": {"units": [64, 64, 1],
#                          "num_relations": 96,
#                          "activation": ["swish", "swish", "linear"]},
#     "cent_kwargs": {},
#     "electrostatic_kwargs": {"name": "electrostatic_layer"},
#     "node_pooling_args": {"pooling_method": "sum"},
#     "verbose": 10,
#     "output_embedding": "graph", "output_to_tensor": True,
#     "use_output_mlp": False,
#     "output_mlp": {"use_bias": [True, True], "units": [64, 1],
#                    "activation": ["swish", "linear"]}
# }

# model = make_model(**model_config)

elemental_mapping = [1,6,16]

# Radial parameters
cutoff_rad = 20
Rs_array   = [0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
eta_array  = [0.0, 0.03, 0.08, 0.16, 0.3, 0.5]

# Angular parameters
cutoff_ang    = 12
lambd_array   = [-1, 1]
zeta_array    = [1, 2, 4, 8, 16]
eta_ang_array = eta_array

rep_model_configs = {
    "name": "HDNNP2nd",
    "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True}],
    "g2_kwargs": {"eta": eta_array, "rs": Rs_array, "rc": cutoff_rad, "elements": elemental_mapping},
    "g4_kwargs": {"eta": eta_ang_array, "zeta": zeta_array, "lamda": lambd_array, "rc": cutoff_ang, "elements": elemental_mapping, "multiplicity": 2.0}
}
rep_model = make_model_rep(**rep_model_configs)
rep = rep_model.predict(dataset.tensor(rep_model_configs["inputs"]), verbose=2)
for i in range(len(dataset)):
   dataset[i].set("rep", rep[i])

model_learn_config = {
    "name": "HDNNP4th",
    "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
               {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
               {"shape": rep.shape[1:], "name": "rep", "dtype": "float32", "ragged": True},
               {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True}],
    "normalize_kwargs": {},
    "mlp_charge_kwargs": {"units": [15, 1],
                          "num_relations": 96,
                          "activation": ["tanh", "linear"]},
    "mlp_local_kwargs": {"units": [35, 35, 1],
                         "num_relations": 96,
                         "activation": ["tanh", "tanh", "linear"]},
    "cent_kwargs": {},
    "electrostatic_kwargs": {"name": "electrostatic_layer"},
    "qmmm_kwargs": {"name": "qmmm_layer"},
    "node_pooling_args": {"pooling_method": "sum"},
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "use_output_mlp": False,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["swish", "linear"]}
}

inputs = dataset.tensor(model_learn_config["inputs"])
print("Amount of inputs:", len(inputs))
for i in range(len(inputs)):
    print(f"Shape {model_learn_config['inputs'][i]['name']}:", inputs[i].shape)

scaler = ExtensiveMolecularScaler(standardize_scale=False)
labels = np.expand_dims(np.array(dataset.get("graph_labels")), axis=-1)
labels = scaler.fit_transform(labels, atomic_number=dataset.get("node_number"))
kf = KFold(n_splits=5, random_state=42, shuffle=True)

hists = []
for train_index, test_index in kf.split(X=labels):
    model = make_model_learn(**model_learn_config)
    model.compile(loss="mean_squared_error", metrics=["mean_absolute_error"], 
                  optimizer=ks.optimizers.Adam(learning_rate=5e-5))
    
    x_train, y_train = dataset[train_index].tensor(model_learn_config["inputs"]), labels[train_index]
    x_test, y_test = dataset[test_index].tensor(model_learn_config["inputs"]), labels[test_index]
    
    scheduler = LinearLearningRateScheduler(
        learning_rate_start=1e-3, learning_rate_stop=1e-8, epo_min=0, epo=500)
    
    start = time.process_time()
    hist = model.fit(
        x_train, y_train,
        callbacks=[scheduler
        ],
        validation_data=(x_test, y_test),
        epochs=500,
        verbose=2
    )
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))
    hists.append(hist)

model.summary()
#model.save("model_energy")

true_y = scaler.inverse_transform(labels[test_index], atomic_number=dataset[test_index].get("node_number"))
predicted_y = scaler.inverse_transform(model.predict(x_test, verbose=0), atomic_number=dataset[test_index].get("node_number"))

plot_predict_true(predicted_y, true_y,
    filepath="", data_unit="Eh",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Energy",
    error="RMSE", file_name=f"predict.png", show_fig=False)

plot_train_test_loss(hists,
    filepath="", data_unit="Eh",
    model_name="HDNNP", dataset_name=dataset_name, file_name="loss.png", show_fig=False)