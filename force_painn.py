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
from kgcnn.literature.PAiNN import make_model
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss, plot_test_set_prediction
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.utils import constants, callbacks, save_load_utils
from kgcnn.model.force import EnergyForceModel

# DEFAULT VALUES
# DATA READ AND SAVE
DATA_DIRECTORY = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum" # Folder containing DATASET_NAME.kgcnn.pickle
DATASET_NAME = "Alanindipeptide" # Used in naming plots and looking for data
MODEL_PREFIX = "model_energy_force_painn" # Will be used to save the models

# ENERGY MODEL HYPER PARAMETERS
ENERGY_EPOCHS                = 1000 # Epochs during training
ENERGY_INITIAL_LEARNING_RATE = 1e-4 # Initial learning rate during training
#ENERGY_FINAL_LEARNING_RATE   = 1e-8 # Final learning rate during training, Deprecated
#ENERGY_HIDDEN_LAYERS         = [35, 35] # List of number of nodes per hidden layer 
#ENERGY_HIDDEN_ACTIVATION     = ["tanh", "tanh"] # List of activation functions of hidden layers
ENERGY_BATCH_SIZE            = 64 # Batch size during training
ENERGY_EARLY_STOPPING        = 0 # Patience of Early Stopping. If 0, no Early Stopping
FORCE_LOSS_FACTOR            = 95

# SCALER PARAMETERS
USE_SCALER = True # If True, the scaler will be used
SCALER_PATH = "scaler.json" # None if no scaler is used
STANDARDIZE_SCALE = True # If True, the scaler will standardize the energy labels and force labels accordingly

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
    #ENERGY_FINAL_LEARNING_RATE   = config_data.get("ENERGY_FINAL_LEARNING_RATE", ENERGY_FINAL_LEARNING_RATE)
    #ENERGY_HIDDEN_LAYERS         = config_data.get("ENERGY_HIDDEN_LAYERS", ENERGY_HIDDEN_LAYERS)
    #ENERGY_HIDDEN_ACTIVATION     = config_data.get("ENERGY_HIDDEN_ACTIVATION", ENERGY_HIDDEN_ACTIVATION)
    ENERGY_BATCH_SIZE            = config_data.get("ENERGY_BATCH_SIZE", ENERGY_BATCH_SIZE)
    ENERGY_EARLY_STOPPING        = config_data.get("ENERGY_EARLY_STOPPING", ENERGY_EARLY_STOPPING)
    FORCE_LOSS_FACTOR            = config_data.get("FORCE_LOSS_FACTOR", FORCE_LOSS_FACTOR)  

    USE_SCALER = config_data.get("USE_SCALER", USE_SCALER)
    SCALER_PATH = config_data.get("SCALER_PATH", SCALER_PATH)
    STANDARDIZE_SCALE = config_data.get("STANDARDIZE_SCALE", STANDARDIZE_SCALE)


file_name = f"{DATASET_NAME}.csv"
print("Dataset:", DATA_DIRECTORY+file_name)
data_directory = os.path.normpath(DATA_DIRECTORY)
dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=DATASET_NAME)
dataset.load()
#dataset=dataset[:10]
print(dataset[0].keys())

model_config = {
        "name": "PAiNNEnergy",
        "inputs": [
            {"shape": [None], "name": "node_number", "dtype": "int64", "ragged": True},
            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
            #{"shape": (), "name": "total_nodes", "dtype": "int64"},
            #{"shape": (), "name": "total_ranges", "dtype": "int64"}
        ],
        #"cast_disjoint_kwargs": {"padded_disjoint": False},
        "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
        "equiv_initialize_kwargs": {"dim": 3, "method": "eps"},# "units": 128},
        #"input_node_embedding": {"input_dim": 95, "output_dim": 128},
        "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
        "pooling_args": {"pooling_method": "sum"},
        "conv_args": {"units": 128, "cutoff": None},
        "update_args": {"units": 128},
        "depth": 6, "verbose": 10,
        "output_embedding": "graph",
        "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
}

outputs = [
    {"name": "graph_labels", "ragged": False},
    {"name": "force", "shape": (None, 3), "ragged": True}
]

inputs = dataset.tensor(model_config["inputs"])
print("Amount of inputs:", len(inputs))
for i in range(len(inputs)):
    print(f"Shape {model_config['inputs'][i]['name']}:", inputs[i].shape)

# Scaling energy and forces.
if USE_SCALER:
    scaler = EnergyForceExtensiveLabelScaler(standardize_scale=STANDARDIZE_SCALE, standardize_coordinates = False,
        energy= "graph_labels", force = "force", atomic_number = "node_number", sample_weight = None)
    scaler.fit_transform_dataset(dataset)
    scaler.save(SCALER_PATH)

kf = KFold(n_splits=3, random_state=42, shuffle=True)
hists = []
train_indices = []
test_indices = []
model_index = 0
for train_index, test_index  in kf.split(X=np.expand_dims(np.array(dataset.get("graph_labels")), axis=-1)): 
    train_index, test_index = test_index, train_index # Switched train and test indices to keep training data separate
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
    
    lr_schedule = ks.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=ENERGY_INITIAL_LEARNING_RATE, first_decay_steps=1e3, t_mul=1.5, m_mul=0.7, alpha=1e-3)
    model_energy_force.compile(
        loss=["mean_squared_error", "mean_squared_error"],
        optimizer=ks.optimizers.Adam(lr_schedule),
        metrics=None,
        loss_weights=[1/FORCE_LOSS_FACTOR, 1-1/FORCE_LOSS_FACTOR]
    )

    lrlog = callbacks.LearningRateLoggingCallback()
    callback_list = [lrlog]
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
        callback_list.append(earlystop)
    
    start = time.process_time()
    hist = model_energy_force.fit(
        x_train, energy_force_train,
        callbacks=callback_list,
        validation_data=(x_test, energy_force_test),
        epochs=ENERGY_EPOCHS,
        batch_size=ENERGY_BATCH_SIZE,
        verbose=2
    )
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))
    hists.append(hist)
    train_indices.append(train_index)
    test_indices.append(test_index)
    model_energy_force.save(MODEL_PREFIX+str(model_index))
    model_index += 1

save_load_utils.save_history(hists, filename="histories.pkl")
save_load_utils.save_training_indices(train_indices, test_indices)

model_energy.summary()

predicted_energy, predicted_force = model_energy_force.predict(x_test, verbose=0)

if USE_SCALER:
    scaler.inverse_transform_dataset(dataset)
    predicted_energy, predicted_force = scaler.inverse_transform(
        y=(predicted_energy.flatten(), predicted_force), X=dataset[test_index].get("node_number"))

true_energy = np.array(dataset[test_index].get("graph_labels")).reshape(-1,1)*constants.hartree_to_kcalmol
true_force = np.array(dataset[test_index].get("force")).reshape(-1,1)

predicted_energy = np.array(predicted_energy).reshape(-1,1)*constants.hartree_to_kcalmol
predicted_force = np.array(predicted_force).reshape(-1,1)

plot_predict_true(predicted_energy, true_energy,
    filepath="", data_unit=r"$\frac{kcal}{mol}$",
    model_name="PaiNN", dataset_name=DATASET_NAME, target_names="Energy",
    error="RMSE", file_name=f"predict_energy.png", show_fig=False)

plot_predict_true(predicted_force, true_force,
    filepath="", data_unit="Eh/B",
    model_name="PaiNN", dataset_name=DATASET_NAME, target_names="Force",
    error="RMSE", file_name=f"predict_force.png", show_fig=False)

plot_train_test_loss(hists,
    filepath="", data_unit="Eh",
    model_name="PaiNN", dataset_name=DATASET_NAME, file_name="loss.png", show_fig=False)

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