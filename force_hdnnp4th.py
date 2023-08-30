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
from kgcnn.literature.HDNNP4th import make_model_behler_charge_separat as make_model
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.model.force import EnergyForceModel
from kgcnn.metrics.loss import RaggedMeanAbsoluteError

data_directory="data/B3LYP_aug-cc-pVTZ_water/"
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
#dataset=dataset[:10]
print(dataset[0].keys())

# to inverse force data
for i in range(len(dataset)):
    dataset[i].set("force", -1*dataset[i]["force"])

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

model_config = {
    "name": "HDNNP4th",
    "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
               {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
               {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True}],
    "g2_kwargs": {"eta": eta_array, "rs": Rs_array, "rc": cutoff_rad, "elements": elemental_mapping},
    "g4_kwargs": {"eta": eta_ang_array, "zeta": zeta_array, "lamda": lambd_array, "rc": cutoff_ang
                  , "elements": elemental_mapping, "multiplicity": 2.0},
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
    "output_embedding": "charge+total_energy", "output_to_tensor": True,
    "use_output_mlp": False,
    "output_mlp": {"use_bias": [True, True], "units": [64, 1],
                   "activation": ["swish", "linear"]}
}

charge_output = {"name": "charge", "shape": (None, 1), "ragged": True}

outputs = [
    {"name": "charge", "shape": (None, 1), "ragged": True},
    {"name": "graph_labels", "ragged": False},
    {"name": "force", "shape": (None, 3), "ragged": True}
]
inputs = dataset.tensor(model_config["inputs"])
print("Amount of inputs:", len(inputs))
for i in range(len(inputs)):
    print(f"Shape {model_config['inputs'][i]['name']}:", inputs[i].shape)


# Scaling energy and forces.
# scaler = EnergyForceExtensiveLabelScaler()
# scaler_mapping = {"atomic_number": "node_number", "y": ["graph_labels", "force"]}
# scaler.fit_transform_dataset(dataset, **scaler_mapping)

def zero_loss_function(y_true, y_pred):
    return 0

kf = KFold(n_splits=2, random_state=42, shuffle=True)
charge_hists = []
hists = []
for train_index, test_index in kf.split(X=np.expand_dims(np.array(dataset.get("graph_labels")), axis=-1)):
    model_charge, model_energy = make_model(**model_config)

    model_charge.compile(
        loss="mean_squared_error",
        optimizer=ks.optimizers.Adam(),
        metrics=None,
    )

    x_train, y_train = dataset[train_index].tensor(model_config["inputs"]), dataset[train_index].tensor(charge_output)
    x_test, y_test = dataset[test_index].tensor(model_config["inputs"]), dataset[test_index].tensor(charge_output)

    scheduler = LinearLearningRateScheduler(
        learning_rate_start=1e-3, learning_rate_stop=1e-8, epo_min=0, epo=1000)

    start = time.process_time()
    charge_hist = model_charge.fit(
        x_train, y_train,
        callbacks=[scheduler
        ],
        validation_data=(x_test, y_test),
        epochs=1000,
        batch_size=128,
        verbose=2
    )
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))
    charge_hists.append(charge_hist)

    charge_mlp_layer = model_energy.layers[10]
    assert "relational_mlp" in charge_mlp_layer.name, "This is not a relational MLP, double check your model"
    charge_mlp_layer.trainable = False
    
    model_energy_force = EnergyForceModel(
        model_energy = model_energy,
        energy_output = 1,
        output_to_tensor = True,
        output_as_dict = False,
        output_squeeze_states = True
    )

    model_energy_force.compile(
        loss=[zero_loss_function, "mean_squared_error", "mean_squared_error"],
        optimizer=ks.optimizers.Adam(),
        metrics=None,
        loss_weights=[0, 1, 199]
    )
    
    x_train, y_train = dataset[train_index].tensor(model_config["inputs"]), dataset[train_index].tensor(outputs)
    x_test, y_test = dataset[test_index].tensor(model_config["inputs"]), dataset[test_index].tensor(outputs)
    
    scheduler = LinearLearningRateScheduler(
        learning_rate_start=1e-3, learning_rate_stop=1e-8, epo_min=0, epo=1000)
    
    start = time.process_time()
    hist = model_energy_force.fit(
        x_train, y_train,
        callbacks=[scheduler
        ],
        validation_data=(x_test, y_test),
        epochs=1000,
        batch_size=128,
        verbose=2
    )
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))
    hists.append(hist)

model_energy.summary()
model_energy_force.save("model_energy_force")

#scaler.inverse_transform_dataset(dataset, **scaler_mapping)
true_charge, true_energy, true_force = np.array(dataset[test_index].get("charge")), np.array(dataset[test_index].get("graph_labels")), np.array(dataset[test_index].get("force"))
predicted_charge, predicted_energy, predicted_force = model_energy_force.predict(x_test, verbose=0)

# predicted_energy, predicted_force = scaler.inverse_transform(
#     y=(predicted_energy, predicted_force), X=dataset[test_index].get("node_number"))

plot_predict_true(np.array(predicted_charge).reshape(-1,1), np.array(true_charge).reshape(-1,1),
    filepath="", data_unit="e",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Charge",
    error="RMSE", file_name=f"predict_charge.png", show_fig=False)

plot_predict_true(np.array(predicted_energy), true_energy,
    filepath="", data_unit="Eh",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Energy",
    error="RMSE", file_name=f"predict_energy.png", show_fig=False)


plot_predict_true(np.array(predicted_force).reshape(-1,1), np.array(true_force).reshape(-1,1),
    filepath="", data_unit="Eh/B",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Force",
    error="RMSE", file_name=f"predict_force.png", show_fig=False)

plot_train_test_loss(charge_hists,
    filepath="", data_unit="e",
    model_name="HDNNP", dataset_name=dataset_name, file_name="charge_loss.png", show_fig=False)

plot_train_test_loss(hists,
    filepath="", data_unit="Eh",
    model_name="HDNNP", dataset_name=dataset_name, file_name="loss.png", show_fig=False)