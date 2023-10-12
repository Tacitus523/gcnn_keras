import argparse
from datetime import timedelta
import json
import os
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss, plot_test_set_prediction
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.utils import constants
from kgcnn.model.force import EnergyForceModel
from kgcnn.metrics.loss import RaggedMeanAbsoluteError

data_directory="data/B3LYP_aug-cc-pVTZ_water/"
#data_directory="data/B3LYP_def2-TZVPP_water/"
dataset_name="ThiolDisulfidExchange"

file_name=f"{dataset_name}.csv"
print("Dataset:", data_directory+file_name)

# Ability to restrict the model to only use a certain GPU, which is passed with python -g gpu_id
ap = argparse.ArgumentParser(description="Handle gpu_ids")
ap.add_argument("-g", "--gpuid", type=int)
args = ap.parse_args()
if args.gpuid is not None:
    set_devices_gpu([args.gpuid])

data_directory = os.path.join(os.path.dirname(__file__), os.path.normpath(data_directory))
dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=dataset_name)
dataset.load()
#dataset=dataset[:10]
print(dataset[0].keys())

# # to inverse force data
# for i in range(len(dataset)):
#     dataset[i].set("force", -1*dataset[i]["force"])

# to inverse esp grad
for i in range(len(dataset)):
    dataset[i].set("esp_grad", -1*dataset[i]["esp_grad"])

elemental_mapping = [1, 6, 16]

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
               {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
               {"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}],
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
    "output_embedding": "charge+qm_energy", "output_to_tensor": True,
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


# # Scaling energy and forces.
# means_and_stds = [(0,1),
#                   (np.mean(np.array(dataset.get("graph_labels"))), np.std(np.array(dataset.get("graph_labels")))),
#                   (np.mean(np.array(dataset.get("force"))), np.std(np.array(dataset.get("force"))))]
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
        esp_input = 5,
        esp_grad_input = 6,
        output_to_tensor = True,
        output_as_dict = False,
        output_squeeze_states = True,
        is_physical_force = False
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
true_charge = np.array(dataset[test_index].get("charge")).reshape(-1,1)
true_energy = np.array(dataset[test_index].get("graph_labels")).reshape(-1,1)*constants.hartree_to_kcalmol
true_force = np.array(dataset[test_index].get("force")).reshape(-1,1)
predicted_charge, predicted_energy, predicted_force = model_energy_force.predict(x_test, verbose=0)
del model_energy_force
#predicted_energy, predicted_force = scaler.inverse_transform(
#    y=(predicted_energy.flatten(), predicted_force), X=dataset[test_index].get("node_number"))
predicted_charge = np.array(predicted_charge).reshape(-1,1)
predicted_energy = np.array(predicted_energy).reshape(-1,1)*constants.hartree_to_kcalmol
predicted_force = np.array(predicted_force).reshape(-1,1)

plot_predict_true(predicted_charge, true_charge,
    filepath="", data_unit="e",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Charge",
    error="RMSE", file_name=f"predict_charge.png", show_fig=False)

plot_predict_true(predicted_energy, true_energy,
    filepath="", data_unit=r"$\frac{kcal}{mol}$",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Energy",
    error="RMSE", file_name=f"predict_energy.png", show_fig=False)

plot_predict_true(predicted_force, true_force,
    filepath="", data_unit="Eh/B",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Force",
    error="RMSE", file_name=f"predict_force.png", show_fig=False)

plot_train_test_loss(charge_hists,
    filepath="", data_unit="e",
    model_name="HDNNP", dataset_name=dataset_name, file_name="charge_loss.png", show_fig=False)

plot_train_test_loss(hists,
    filepath="", data_unit="Eh",
    model_name="HDNNP", dataset_name=dataset_name, file_name="loss.png", show_fig=False)

rmse_charge = mean_squared_error(true_charge, predicted_charge, squared=False)
mae_charge  = mean_absolute_error(true_charge, predicted_charge)
r2_charge   = r2_score(true_charge, predicted_charge)

rmse_energy = mean_squared_error(true_energy, predicted_energy, squared=False)
mae_energy  = mean_absolute_error(true_energy, predicted_energy)
r2_energy   = r2_score(true_energy, predicted_energy)

rmse_force = mean_squared_error(true_force, predicted_force, squared=False)
mae_force  = mean_absolute_error(true_force, predicted_force)
r2_force   = r2_score(true_force, predicted_force)

error_dict = {
    "RMSE Charge": f"{rmse_charge:.3f}",
    "MAE Charge": f"{mae_charge:.3f}",
    "R2 Charge": f"{r2_charge:.2f}",
    "RMSE Energy": f"{rmse_energy:.1f}",
    "MAE Energy": f"{mae_energy:.1f}",
    "R2 Energy": f"{r2_energy:.2f}",
    "RMSE Force": f"{rmse_force:.3f}",
    "MAE Force": f"{mae_force:.3f}",
    "R2 Force": f"{r2_force:.2f}"
}

for key, value in error_dict.items():
    print(f"{key}: {value}")

with open(os.path.join("", "errors.json"), "w") as f:
    json.dump(error_dict, f, indent=2, sort_keys=True)

charge_df = pd.DataFrame({"charge_reference": true_charge.flatten(), "charge_prediction": predicted_charge.flatten()})
energy_df = pd.DataFrame({"energy_reference": true_energy.flatten(), "energy_prediction": predicted_energy.flatten()})
force_df = pd.DataFrame({"force_reference": true_force.flatten(), "force_prediction": predicted_force.flatten()})

atomic_numbers = np.array(dataset[test_index].get("node_number")).flatten()
at_types_column = pd.Series(atomic_numbers, name="at_types").replace(constants.atomic_number_to_element)
charge_df["at_types"] = at_types_column
force_df["at_types"] =  at_types_column

plot_test_set_prediction(charge_df, "charge_reference", "charge_prediction",
    "Charge", "e", rmse_charge, r2_charge, "")
plot_test_set_prediction(energy_df, "energy_reference", "energy_prediction",
    "Energy", r"$\frac{kcal}{mol}$", rmse_energy, r2_energy, "")
plot_test_set_prediction(force_df, "force_reference", "force_prediction",
    "Force", r"$\frac{E_h}{B}$", rmse_force, r2_force, "")