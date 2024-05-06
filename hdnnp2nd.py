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
from kgcnn.literature.HDNNP2nd import make_model_behler as make_model
from kgcnn.data.transform.scaler.mol import ExtensiveMolecularScaler
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss, plot_test_set_prediction
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.utils import constants, save_load_utils
DATA_DIRECTORY = "data/B3LYP_aug-cc-pVTZ_vacuum" # Folder containing DATASET_NAME.kgcnn.pickle
DATASET_NAME = "ThiolDisulfidExchange" # Used in naming plots and looking for data
MODEL_PREFIX = "model_energy" # Will be used to save the models

# SYMMETRY FUNCTION HYPER PARAMETERS
# Radial parameters
CUTOFF_RAD = 20 # Radial cutoff distance in Bohr
RS_ARRAY   = [0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] # Shift parameter Rs in Bohr
ETA_ARRAY  = [0.0, 0.03, 0.08, 0.16, 0.3, 0.5] # Width parameter eta in 1/Bohr^2

# Angular parameters
CUTOFF_ANG    = 12 # Angular cutoff distance in Bohr
LAMBD_ARRAY   = [-1, 1] # Lambda parameter
ZETA_ARRAY    = [1, 2, 4, 8, 16] # Zeta parameter
ETA_ANG_ARRAY = ETA_ARRAY # Width parameter eta in 1/Bohr^2

# Assignment of parameters to elements
MAX_ELEMENTS = 30 # Length of the array with the symmetry function parameters, the highest possible atomic number of the elements is determined by this
ELEMENTAL_MAPPING = [1, 6, 7, 8] # Parameters can be given individually per element. This list maps the parameters to its element

# ENERGY MODEL HYPER PARAMETERS
ENERGY_EPOCHS                = 500 # Epochs during training
ENERGY_INITIAL_LEARNING_RATE = 1e-3 # Initial learning rate during training
ENERGY_FINAL_LEARNING_RATE   = 1e-8 # Initial learning rate during training
ENERGY_HIDDEN_LAYERS         = [35, 35] # List of number of nodes per hidden layer 
ENERGY_HIDDEN_ACTIVATION     = ["tanh", "tanh"] # List of activation functions of hidden layers
ENERGY_BATCH_SIZE            = 128 # Batch size during training
ENERGY_EARLY_STOPPING        = 0 # Patience of Early Stopping. If 0, no Early Stopping, Early Stopping breaks loss history plot
FORCE_LOSS_FACTOR            = 200 # Weight of the force loss relative to the energy loss, gets normalized

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

    CUTOFF_RAD = config_data.get("CUTOFF_RAD", CUTOFF_RAD)
    RS_ARRAY   = config_data.get("RS_ARRAY", RS_ARRAY)
    ETA_ARRAY  = config_data.get("ETA_ARRAY", ETA_ARRAY)

    CUTOFF_ANG    = config_data.get("CUTOFF_ANG", CUTOFF_ANG)
    LAMBD_ARRAY   = config_data.get("LAMBD_ARRAY", LAMBD_ARRAY)
    ZETA_ARRAY    = config_data.get("ZETA_ARRAY", ZETA_ARRAY)
    ETA_ANG_ARRAY = config_data.get("ETA_ANG_ARRAY", ETA_ANG_ARRAY)

    MAX_ELEMENTS = config_data.get("MAX_ELEMENTS", CUTOFF_ANG)
    ELEMENTAL_MAPPING = config_data["ELEMENTAL_MAPPING"]

    ENERGY_EPOCHS                = config_data.get("ENERGY_EPOCHS", ENERGY_EPOCHS)
    ENERGY_INITIAL_LEARNING_RATE = config_data.get("ENERGY_INITIAL_LEARNING_RATE", ENERGY_INITIAL_LEARNING_RATE)
    ENERGY_FINAL_LEARNING_RATE   = config_data.get("ENERGY_FINAL_LEARNING_RATE", ENERGY_FINAL_LEARNING_RATE)
    ENERGY_HIDDEN_LAYERS         = config_data.get("ENERGY_HIDDEN_LAYERS", ENERGY_HIDDEN_LAYERS)
    ENERGY_HIDDEN_ACTIVATION     = config_data.get("ENERGY_HIDDEN_ACTIVATION", ENERGY_HIDDEN_ACTIVATION)
    ENERGY_BATCH_SIZE            = config_data.get("ENERGY_BATCH_SIZE", ENERGY_BATCH_SIZE)
    ENERGY_EARLY_STOPPING        = config_data.get("ENERGY_EARLY_STOPPING", ENERGY_EARLY_STOPPING)
    FORCE_LOSS_FACTOR            = config_data.get("FORCE_LOSS_FACTOR", FORCE_LOSS_FACTOR)

file_name = f"{DATASET_NAME}.csv"
data_directory = os.path.normpath(DATA_DIRECTORY)
print("Dataset:", os.path.join(DATA_DIRECTORY, file_name))
dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=DATASET_NAME)
dataset.load()
#dataset=dataset[:10]

model_config = {
    "name": "HDNNP2nd",
    "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
               {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
               {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True}],
    "g2_kwargs": {"eta": ETA_ARRAY, "rs": RS_ARRAY, "rc": CUTOFF_RAD, "elements": ELEMENTAL_MAPPING},
    "g4_kwargs": {"eta": ETA_ANG_ARRAY, "zeta": ZETA_ARRAY, "lamda": LAMBD_ARRAY, "rc": CUTOFF_ANG
                  , "elements": ELEMENTAL_MAPPING, "multiplicity": 2.0},
    "mlp_kwargs": {"units": ENERGY_HIDDEN_LAYERS+[1],
                         "num_relations": MAX_ELEMENTS,
                         "activation": ENERGY_HIDDEN_ACTIVATION+["linear"]},
    "node_pooling_args": {"pooling_method": "sum"},
    "verbose": 10,
    "output_embedding": "graph", "output_to_tensor": True,
    "use_output_mlp": False
}

outputs = [
    {"name": "graph_labels", "ragged": False}
]
inputs = dataset.tensor(model_config["inputs"])
print(model_config)
print("Amount of inputs:", len(inputs))
for i in range(len(inputs)):
    print(f"Shape {model_config['inputs'][i]['name']}:", inputs[i].shape)

# # Scaling energy
# scaler = ExtensiveMolecularScaler(standardize_scale=False)
# labels = np.expand_dims(np.array(dataset.get("graph_labels")), axis=-1)
# labels = scaler.fit_transform(labels, atomic_number=dataset.get("node_number"))

N_SPLITS = 3 # Used to determine amount of splits in training
kf = KFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

hists = []
train_indices = []
test_indices = []
model_index = 0
for test_index, train_index in kf.split(X=np.expand_dims(np.array(dataset.get("graph_labels")), axis=-1)): # Switched train and test indices to keep training data separate
    x_train = dataset[train_index].tensor(model_config["inputs"])
    x_test = dataset[test_index].tensor(model_config["inputs"])
    energy_train = dataset[train_index].tensor(outputs)
    energy_test = dataset[test_index].tensor(outputs)
    model = make_model(**model_config)
    model.compile(
        loss="mean_squared_error",
        optimizer=ks.optimizers.Adam(),
        metrics=None
        )
    

    callbacks=[]
    scheduler = LinearLearningRateScheduler(
        learning_rate_start=ENERGY_INITIAL_LEARNING_RATE,
        learning_rate_stop=ENERGY_FINAL_LEARNING_RATE,
        epo_min=0,
        epo=ENERGY_EPOCHS)
    callbacks.append(scheduler)

    if ENERGY_EARLY_STOPPING > 0:
        earlystop = ks.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=ENERGY_EARLY_STOPPING,
            verbose=0
        )
        callbacks.append(earlystop)

    start = time.process_time()
    hist = model.fit(
        x_train, energy_train,
        callbacks=callbacks,
        validation_data=(x_test, energy_test),
        epochs=ENERGY_EPOCHS,
        batch_size=ENERGY_BATCH_SIZE,
        verbose=2
    )
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))
    hists.append(hist)
    train_indices.append(train_index)
    test_indices.append(test_index)
    model.save(MODEL_PREFIX+str(model_index))
    model_index += 1

save_load_utils.save_history(hists)
save_load_utils.save_training_indices(train_indices, test_indices)

model_energy.summary()

#scaler.inverse_transform_dataset(dataset, **scaler_mapping)
true_energy = np.array(dataset[test_index].get("graph_labels")).reshape(-1,1)*constants.hartree_to_kcalmol
predicted_energy = model.predict(x_test, batch_size=ENERGY_BATCH_SIZE, verbose=0)
#predicted_energy = scaler.inverse_transform(model.predict(x_test, verbose=0), atomic_number=dataset[test_index].get("node_number"))
predicted_energy = np.array(predicted_energy).reshape(-1,1)*constants.hartree_to_kcalmol

plot_predict_true(predicted_energy, true_energy,
    filepath="", data_unit=r"$\frac{kcal}{mol}$",
    model_name="HDNNP", dataset_name=DATASET_NAME, target_names="Energy",
    error="RMSE", file_name=f"predict_energy.png", show_fig=False)

plot_train_test_loss(hists,
    filepath="", data_unit="Eh",
    model_name="HDNNP", dataset_name=DATASET_NAME, file_name="loss.png", show_fig=False)

rmse_energy = mean_squared_error(true_energy, predicted_energy, squared=False)
mae_energy  = mean_absolute_error(true_energy, predicted_energy)
r2_energy   = r2_score(true_energy, predicted_energy)

error_dict = {
    "RMSE Energy": f"{rmse_energy:.1f}",
    "MAE Energy": f"{mae_energy:.1f}",
    "R2 Energy": f"{r2_energy:.2f}"
}

for key, value in error_dict.items():
    print(f"{key}: {value}")

with open(os.path.join("", "errors.json"), "w") as f:
    json.dump(error_dict, f, indent=2, sort_keys=True)

energy_df = pd.DataFrame({"energy_reference": true_energy.flatten(), "energy_prediction": predicted_energy.flatten()})

plot_test_set_prediction(energy_df, "energy_reference", "energy_prediction",
    "Energy", r"$\frac{kcal}{mol}$", rmse_energy, r2_energy, "")