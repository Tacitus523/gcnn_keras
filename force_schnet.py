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
from kgcnn.training.history import save_history_score
from kgcnn.literature.Schnet import make_model
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss, plot_test_set_prediction
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.utils import constants
from kgcnn.model.force import EnergyForceModel

# DEFAULT VALUES
# DATA READ AND SAVE
DATA_DIRECTORY = "/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_vacuum" # Folder containing DATASET_NAME.kgcnn.pickle
DATASET_NAME = "ThiolDisulfidExchange" # Used in naming plots and looking for data
MODEL_PREFIX = "model_energy_force_schnet" # Will be used to save the models

# ENERGY MODEL HYPER PARAMETERS
ENERGY_EPOCHS                = 150 # Epochs during training
ENERGY_INITIAL_LEARNING_RATE = 1e-3 # Initial learning rate during training
ENERGY_FINAL_LEARNING_RATE   = 1e-8 # Initial learning rate during training
ENERGY_HIDDEN_LAYERS         = [35, 35] # List of number of nodes per hidden layer 
ENERGY_HIDDEN_ACTIVATION     = ["tanh", "tanh"] # List of activation functions of hidden layers
ENERGY_BATCH_SIZE            = 64 # Batch size during training
ENERGY_EARLY_STOPPING        = 0 # Patience of Early Stopping. If 0, no Early Stopping
FORCE_LOSS_FACTOR            = 200

# Ability to restrict the model to only use a certain GPU, which is passed with python -g gpu_id
ap = argparse.ArgumentParser(description="Handle gpu_ids and training parameters")
ap.add_argument("-g", "--gpuid", type=int)
ap.add_argument("-c", "--conf", default=None, type=str, dest="config_path", action="store", required=False, help="Path to config file, default: None", metavar="config")
args = ap.parse_args()
if args.gpuid is not None:
    set_devices_gpu([args.gpuid])
if args.config_path is not None:
    try:
        with open(args.config_path, 'r') as config_file:
            config_data = json.load(config_file)
    except FileNotFoundError:
        print(f"Config file {args.config_path} not found.")
        exit(1)

    for key, value in config_data.items():
        print(f"{key}: {value}")

    #TODO: Input validation function instead, or try except with raise in except block?
    DATA_DIRECTORY = config_data["DATA_DIRECTORY"]
    DATASET_NAME = config_data["DATASET_NAME"]
    MODEL_PREFIX = config_data.get("MODEL_PREFIX", MODEL_PREFIX)

    ENERGY_EPOCHS                = config_data.get("ENERGY_EPOCHS", ENERGY_EPOCHS)
    ENERGY_INITIAL_LEARNING_RATE = config_data.get("ENERGY_INITIAL_LEARNING_RATE", ENERGY_INITIAL_LEARNING_RATE)
    ENERGY_FINAL_LEARNING_RATE   = config_data.get("ENERGY_FINAL_LEARNING_RATE", ENERGY_FINAL_LEARNING_RATE)
    ENERGY_HIDDEN_LAYERS         = config_data.get("ENERGY_HIDDEN_LAYERS", ENERGY_HIDDEN_LAYERS)
    ENERGY_HIDDEN_ACTIVATION     = config_data.get("ENERGY_HIDDEN_ACTIVATION", ENERGY_HIDDEN_ACTIVATION)
    ENERGY_BATCH_SIZE            = config_data.get("ENERGY_BATCH_SIZE", ENERGY_BATCH_SIZE)
    ENERGY_EARLY_STOPPING        = config_data.get("ENERGY_EARLY_STOPPING", ENERGY_EARLY_STOPPING)
    FORCE_LOSS_FACTOR            = config_data.get("ENERGY_EARLY_STOPPING", FORCE_LOSS_FACTOR)  

file_name = f"{DATASET_NAME}.csv"
print("Dataset:", DATA_DIRECTORY+file_name)
data_directory = os.path.normpath(DATA_DIRECTORY)
dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=DATASET_NAME)
dataset.load()
#dataset=dataset[:10]
print(dataset[0].keys())

model_config = {
    "name": "Schnet",
    "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True}],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
    "make_distance": True, "expand_distance": True,
    "interaction_args": {"units": 250, "use_bias": True,
                         "activation": "swish", "cfconv_pool": "sum"},
    "node_pooling_args": {"pooling_method": "sum"},
    "depth": 4,
    "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
    "verbose": 10,
    "last_mlp": {"use_bias": True, "units": [250, 150, 100],
                 "activation": "swish"},
    "output_embedding": "graph", "output_to_tensor": True,
    "use_output_mlp": True,
    "output_mlp": {"use_bias": True, "units": [100, 50, 25, 1],
                   "activation": ["swish"]*3 + ["linear"]}
}

outputs = [
    {"name": "graph_labels", "ragged": False},
    {"name": "force", "shape": (None, 3), "ragged": True}
]

inputs = dataset.tensor(model_config["inputs"])
print("Amount of inputs:", len(inputs))
for i in range(len(inputs)):
    print(f"Shape {model_config['inputs'][i]['name']}:", inputs[i].shape)


# # Scaling energy and forces.
# scaler = EnergyForceExtensiveLabelScaler()
# scaler_mapping = {"atomic_number": "node_number", "y": ["graph_labels", "force"]}
# scaler.fit_transform_dataset(dataset, **scaler_mapping)

kf = KFold(n_splits=3, random_state=42, shuffle=True)
hists = []
model_index = 0
for test_index, train_index in kf.split(X=np.expand_dims(np.array(dataset.get("graph_labels")), axis=-1)): # Switched train and test indices to keep training data separate
    x_train = dataset[train_index].tensor(model_config["inputs"])
    x_test = dataset[test_index].tensor(model_config["inputs"])
    energy_force_train = dataset[train_index].tensor(outputs)
    energy_force_test = dataset[test_index].tensor(outputs)

    model_energy = make_model(**model_config)

    model_energy_force = EnergyForceModel(
        model_energy = model_energy,
        energy_output = 0,
        esp_input = None,
        esp_grad_input = None,
        output_to_tensor = True,
        output_as_dict = False,
        output_squeeze_states = True,
        is_physical_force = False
    )

    model_energy_force.compile(
        loss=["mean_squared_error", "mean_squared_error"],
        optimizer=ks.optimizers.Adam(),
        metrics=None,
        loss_weights=[1/FORCE_LOSS_FACTOR, 1-1/FORCE_LOSS_FACTOR]
    )
    
    lr_schedule = ks.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=1e3, t_mul=1.5, m_mul=0.7, alpha=1e-3)
    model_energy_force.compile(
        loss=["mean_squared_error", "mean_squared_error"],
        optimizer=ks.optimizers.Adam(lr_schedule),
        metrics=None,
        loss_weights=[1/FORCE_LOSS_FACTOR, 1-1/FORCE_LOSS_FACTOR]
    )

    callbacks = []
    # scheduler = LinearLearningRateScheduler(
    #     learning_rate_start=ENERGY_INITIAL_LEARNING_RATE,
    #     learning_rate_stop=ENERGY_FINAL_LEARNING_RATE,
    #     epo_min=0,
    #     epo=ENERGY_EPOCHS)
    # callbacks.append(scheduler)

    if ENERGY_EARLY_STOPPING > 0:
        earlystop = ks.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=ENERGY_EARLY_STOPPING,
            verbose=0
        )
        callbacks.append(earlystop)
    
    start = time.process_time()
    hist = model_energy_force.fit(
        x_train, energy_force_train,
        callbacks=callbacks,
        validation_data=(x_test, energy_force_test),
        epochs=ENERGY_EPOCHS,
        batch_size=ENERGY_BATCH_SIZE,
        verbose=2
    )
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))
    hists.append(hist)
    model_energy_force.save(MODEL_PREFIX+str(model_index))
    model_index += 1

hist_dicts = [hist.history for hist in hists]
with open(os.path.join("", "histories.json"), "w") as f:
    json.dump(hist_dicts, f, indent=2)

model_energy.summary()

#scaler.inverse_transform_dataset(dataset, **scaler_mapping)
true_energy = np.array(dataset[test_index].get("graph_labels")).reshape(-1,1)*constants.hartree_to_kcalmol
true_force = np.array(dataset[test_index].get("force")).reshape(-1,1)
predicted_energy, predicted_force = model_energy_force.predict(x_test, verbose=0)
del model_energy_force
#predicted_energy, predicted_force = scaler.inverse_transform(
#    y=(predicted_energy.flatten(), predicted_force), X=dataset[test_index].get("node_number"))
predicted_energy = np.array(predicted_energy).reshape(-1,1)*constants.hartree_to_kcalmol
predicted_force = np.array(predicted_force).reshape(-1,1)

plot_predict_true(predicted_energy, true_energy,
    filepath="", data_unit=r"$\frac{kcal}{mol}$",
    model_name="Schnet", dataset_name=DATASET_NAME, target_names="Energy",
    error="RMSE", file_name=f"predict_energy.png", show_fig=False)

plot_predict_true(predicted_force, true_force,
    filepath="", data_unit="Eh/B",
    model_name="Schnet", dataset_name=DATASET_NAME, target_names="Force",
    error="RMSE", file_name=f"predict_force.png", show_fig=False)

plot_train_test_loss(hists,
    filepath="", data_unit="Eh",
    model_name="Schnet", dataset_name=DATASET_NAME, file_name="loss.png", show_fig=False)

rmse_energy = mean_squared_error(true_energy, predicted_energy, squared=False)
mae_energy  = mean_absolute_error(true_energy, predicted_energy)
r2_energy   = r2_score(true_energy, predicted_energy)

rmse_force = mean_squared_error(true_force, predicted_force, squared=False)
mae_force  = mean_absolute_error(true_force, predicted_force)
r2_force   = r2_score(true_force, predicted_force)

error_dict = {
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

energy_df = pd.DataFrame({"energy_reference": true_energy.flatten(), "energy_prediction": predicted_energy.flatten()})
force_df = pd.DataFrame({"force_reference": true_force.flatten(), "force_prediction": predicted_force.flatten()})

atomic_numbers = np.array(dataset[test_index].get("node_number")).flatten()
at_types_column = pd.Series(atomic_numbers, name="at_types").replace(constants.atomic_number_to_element)
force_df["at_types"] = at_types_column.repeat(3).reset_index(drop=True)

plot_test_set_prediction(energy_df, "energy_reference", "energy_prediction",
    "Energy", r"$\frac{kcal}{mol}$", rmse_energy, r2_energy, "")
plot_test_set_prediction(force_df, "force_reference", "force_prediction",
    "Force", r"$\frac{E_h}{B}$", rmse_force, r2_force, "")