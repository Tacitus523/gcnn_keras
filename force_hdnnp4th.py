import argparse
from datetime import timedelta
import json
import os
import pickle
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
from kgcnn.utils import constants, save_load_utils
from kgcnn.model.force import EnergyForceModel
from kgcnn.metrics.loss import RaggedMeanAbsoluteError, zero_loss_function

# DEFAULT VALUES
# DATA READ AND SAVE
DATA_DIRECTORY = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water" # Folder containing DATASET_NAME.kgcnn.pickle
DATASET_NAME = "Alanindipeptide" # Used in naming plots and looking for data
MODEL_PREFIX = "model_energy_force" # Will be used to save the models

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

# CHARGE MODEL HYPER PARAMETERS
CHARGE_EPOCHS                = 500 # Epochs during training
CHARGE_INITIAL_LEARNING_RATE = 1e-3 # Initial learning rate during training
CHARGE_FINAL_LEARNING_RATE   = 1e-8 # Initial learning rate during training
CHARGE_HIDDEN_LAYERS         = [15] # List of number of nodes per hidden layer 
CHARGE_HIDDEN_ACTIVATION     = ["tanh"] # List of activation functions of hidden layers
CHARGE_BATCH_SIZE            = 128 # Batch size during training
CHARGE_EARLY_STOPPING        = 0 # Patience of Early Stopping. If 0, no Early Stopping, Early Stopping breaks loss history plot

# ENERGY MODEL HYPER PARAMETERS
ENERGY_EPOCHS                = 500 # Epochs during training
ENERGY_INITIAL_LEARNING_RATE = 1e-3 # Initial learning rate during training
ENERGY_FINAL_LEARNING_RATE   = 1e-8 # Initial learning rate during training
ENERGY_HIDDEN_LAYERS         = [35, 35] # List of number of nodes per hidden layer 
ENERGY_HIDDEN_ACTIVATION     = ["tanh", "tanh"] # List of activation functions of hidden layers
ENERGY_BATCH_SIZE            = 128 # Batch size during training
ENERGY_EARLY_STOPPING        = 0 # Patience of Early Stopping. If 0, no Early Stopping, Early Stopping breaks loss history plot
FORCE_LOSS_FACTOR            = 200 # Weight of the force loss relative to the energy loss, gets normalized

def load_data(data_directory: str, dataset_name: str) -> MemoryGraphDataset:
    dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=dataset_name)
    dataset.load()
    # dataset = dataset[:10]
    
    file_name=f"{dataset_name}.csv"
    data_directory = os.path.normpath(data_directory)
    print("Dataset:", os.path.join(data_directory, file_name))
    print(dataset[0].keys())
    return dataset

def train_model(dataset: MemoryGraphDataset,
                model_config: dict,
                charge_output: dict,
                outputs: list[dict],
                train_config: dict
                ) -> tuple[
                    EnergyForceModel,
                    list[tf.keras.callbacks.History],
                    list[tf.keras.callbacks.History],
                    EnergyForceExtensiveLabelScaler|None
                ]:
    print(model_config)

    charge_initial_learning_rate = train_config["charge_initial_learning_rate"]
    charge_final_learning_rate = train_config["charge_final_learning_rate"]
    charge_epochs = train_config["charge_epochs"]
    charge_early_stopping = train_config["charge_early_stopping"]
    charge_batch_size = train_config["charge_batch_size"]

    energy_initial_learning_rate = train_config["energy_initial_learning_rate"]
    energy_final_learning_rate = train_config["energy_final_learning_rate"]
    energy_epochs = train_config["energy_epochs"]
    energy_early_stopping = train_config["energy_early_stopping"]
    energy_batch_size = train_config["energy_batch_size"]
    force_loss_factor = train_config["force_loss_factor"]

    model_prefix = train_config["model_prefix"]

    # # Scaling energy and forces.
    # scaler = EnergyForceExtensiveLabelScaler(standardize_coordinates = False,
    #  energy= "graph_labels", force = "force", atomic_number = "node_number",
    #  sample_weight = None)
    # scaler.fit_transform_dataset(dataset)
    scaler = None

    N_SPLITS = 3 # Used to determine amount of splits in training
    kf = KFold(n_splits=N_SPLITS, random_state=42, shuffle=True)
    charge_hists = []
    hists = []
    train_indices = []
    test_indices = []
    model_index = 0
    for test_index, train_index in kf.split(X=np.expand_dims(np.array(dataset.get("graph_labels")), axis=-1)): # Switched train and test indices to keep training data separate
        x_train = dataset[train_index].tensor(model_config["inputs"])
        x_test = dataset[test_index].tensor(model_config["inputs"])
        charge_train = dataset[train_index].tensor(charge_output)
        charge_test = dataset[test_index].tensor(charge_output)
        energy_force_train = dataset[train_index].tensor(outputs)
        energy_force_test = dataset[test_index].tensor(outputs)

        model_charge, model_energy = make_model(**model_config)

        model_charge.compile(
            loss="mean_squared_error",
            optimizer=ks.optimizers.Adam(),
            metrics=None
        )

        callbacks = []
        scheduler = LinearLearningRateScheduler(
            learning_rate_start=charge_initial_learning_rate,
            learning_rate_stop=charge_final_learning_rate,
            epo_min=0,
            epo=charge_epochs)
        callbacks.append(scheduler)

        if charge_early_stopping > 0:
            earlystop = ks.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=charge_early_stopping,
                verbose=0
            )
            callbacks.append(earlystop)

        start = time.process_time()
        charge_hist = model_charge.fit(
            x_train, charge_train,
            callbacks=callbacks,
            validation_data=(x_test, charge_test),
            epochs=charge_epochs,
            batch_size=charge_batch_size,
            verbose=2
        )
        stop = time.process_time()
        print("Print Time for training: ", str(timedelta(seconds=stop - start)))
        charge_hists.append(charge_hist)

        charge_mlp_layer = model_energy.layers[10]
        assert "relational_mlp" in charge_mlp_layer.name, "This is not a relational MLP, double check your model"
        charge_mlp_layer.trainable = False
        electrostatic_layer = model_energy.layers[13]
        assert "electrostatic_layer" in electrostatic_layer.name, "This is not an electrostatic_layer, double check your model"
        electrostatic_layer.trainable = False

        model_energy_force: EnergyForceModel = EnergyForceModel(
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
            loss=["mean_squared_error", "mean_squared_error", "mean_squared_error"],
            optimizer=ks.optimizers.Adam(),
            metrics=None,
            loss_weights=[0, 1/force_loss_factor, 1-1/force_loss_factor]
        )
        
        callbacks=[]
        scheduler = LinearLearningRateScheduler(
            learning_rate_start=energy_initial_learning_rate,
            learning_rate_stop=energy_final_learning_rate,
            epo_min=0,
            epo=energy_epochs)
        callbacks.append(scheduler)

        if energy_early_stopping > 0:
            earlystop = ks.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=energy_early_stopping,
                verbose=0
            )
            callbacks.append(earlystop)

        start = time.process_time()
        hist = model_energy_force.fit(
            x_train, energy_force_train,
            callbacks=callbacks,
            validation_data=(x_test, energy_force_test),
            epochs=energy_epochs,
            batch_size=energy_batch_size,
            verbose=2
        )
        stop = time.process_time()
        print("Print Time for training: ", str(timedelta(seconds=stop - start)))
        hists.append(hist)
        train_indices.append(train_index)
        test_indices.append(test_index)
        model_energy_force.save(model_prefix+str(model_index))
        model_index += 1
        break

    save_load_utils.save_history(charge_hists, filename="charge_histories.pkl")
    save_load_utils.save_history(hists, filename="histories.pkl")
    save_load_utils.save_training_indices(train_indices, test_indices)

    return model_energy_force, test_index, charge_hists, hists, scaler

def evaluate_model(dataset: MemoryGraphDataset,
                   model_energy_force: EnergyForceModel,
                   test_index: np.ndarray[int],
                   charge_hists: list[tf.keras.callbacks.History],
                   hists: list[tf.keras.callbacks.History],
                   scaler: EnergyForceExtensiveLabelScaler|None = None) -> None:
    
    x_test = dataset[test_index].tensor(model_config["inputs"])
    predicted_charge, predicted_energy, predicted_force = model_energy_force.predict(x_test, batch_size=ENERGY_BATCH_SIZE, verbose=0)
    
    if scaler is not None:
        scaler.inverse_transform_dataset(dataset)
        predicted_energy, predicted_force = scaler.inverse_transform(
        y=(predicted_energy.flatten(), predicted_force), X=dataset[test_index].get("node_number"))

    true_charge = np.array(dataset[test_index].get("charge")).reshape(-1,1)
    true_energy = np.array(dataset[test_index].get("graph_labels")).reshape(-1,1)*constants.hartree_to_kcalmol
    true_force = np.array(dataset[test_index].get("force")).reshape(-1,1)

    predicted_charge = np.array(predicted_charge).reshape(-1,1)
    predicted_energy = np.array(predicted_energy).reshape(-1,1)*constants.hartree_to_kcalmol
    predicted_force = np.array(predicted_force).reshape(-1,1)

    dataset_name: str = dataset.dataset_name
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
    force_df["at_types"] = at_types_column.repeat(3).reset_index(drop=True)

    plot_test_set_prediction(charge_df, "charge_reference", "charge_prediction",
        "Charge", "e", rmse_charge, r2_charge, "")
    plot_test_set_prediction(energy_df, "energy_reference", "energy_prediction",
        "Energy", r"$\frac{kcal}{mol}$", rmse_energy, r2_energy, "")
    plot_test_set_prediction(force_df, "force_reference", "force_prediction",
        "Force", r"$\frac{E_h}{B}$", rmse_force, r2_force, "")

if __name__ == "__main__":
    # Ability to restrict the model to only use a certain GPU, which is passed with python -g gpu_id, or to use a config file
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

        MAX_ELEMENTS = config_data.get("MAX_ELEMENTS", MAX_ELEMENTS)
        ELEMENTAL_MAPPING = config_data["ELEMENTAL_MAPPING"]

        CHARGE_EPOCHS                = config_data.get("CHARGE_EPOCHS", CHARGE_EPOCHS)
        CHARGE_INITIAL_LEARNING_RATE = config_data.get("CHARGE_INITIAL_LEARNING_RATE", CHARGE_INITIAL_LEARNING_RATE)
        CHARGE_FINAL_LEARNING_RATE   = config_data.get("CHARGE_FINAL_LEARNING_RATE", CHARGE_FINAL_LEARNING_RATE)
        CHARGE_HIDDEN_LAYERS         = config_data.get("CHARGE_HIDDEN_LAYERS", CHARGE_HIDDEN_LAYERS)
        CHARGE_HIDDEN_ACTIVATION     = config_data.get("CHARGE_HIDDEN_ACTIVATION", CHARGE_HIDDEN_ACTIVATION)
        CHARGE_BATCH_SIZE            = config_data.get("CHARGE_BATCH_SIZE", CHARGE_BATCH_SIZE)
        CHARGE_EARLY_STOPPING        = config_data.get("CHARGE_EARLY_STOPPING", CHARGE_EARLY_STOPPING)

        ENERGY_EPOCHS                = config_data.get("ENERGY_EPOCHS", ENERGY_EPOCHS)
        ENERGY_INITIAL_LEARNING_RATE = config_data.get("ENERGY_INITIAL_LEARNING_RATE", ENERGY_INITIAL_LEARNING_RATE)
        ENERGY_FINAL_LEARNING_RATE   = config_data.get("ENERGY_FINAL_LEARNING_RATE", ENERGY_FINAL_LEARNING_RATE)
        ENERGY_HIDDEN_LAYERS         = config_data.get("ENERGY_HIDDEN_LAYERS", ENERGY_HIDDEN_LAYERS)
        ENERGY_HIDDEN_ACTIVATION     = config_data.get("ENERGY_HIDDEN_ACTIVATION", ENERGY_HIDDEN_ACTIVATION)
        ENERGY_BATCH_SIZE            = config_data.get("ENERGY_BATCH_SIZE", ENERGY_BATCH_SIZE)
        ENERGY_EARLY_STOPPING        = config_data.get("ENERGY_EARLY_STOPPING", ENERGY_EARLY_STOPPING)
        FORCE_LOSS_FACTOR            = config_data.get("FORCE_LOSS_FACTOR", FORCE_LOSS_FACTOR)

    model_config = {
        "name": "HDNNP4th",
        "inputs": [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
                {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
                {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
                {"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}],
        "g2_kwargs": {"eta": ETA_ARRAY, "rs": RS_ARRAY, "rc": CUTOFF_RAD, "elements": ELEMENTAL_MAPPING},
        "g4_kwargs": {"eta": ETA_ANG_ARRAY, "zeta": ZETA_ARRAY, "lamda": LAMBD_ARRAY, "rc": CUTOFF_ANG
                    , "elements": ELEMENTAL_MAPPING, "multiplicity": 2.0},
        "normalize_kwargs": {},
        "mlp_charge_kwargs": {"units": CHARGE_HIDDEN_LAYERS+[1],
                            "num_relations": MAX_ELEMENTS,
                            "activation": CHARGE_HIDDEN_ACTIVATION+["linear"]},
        "mlp_local_kwargs": {"units": ENERGY_HIDDEN_LAYERS+[1],
                            "num_relations": MAX_ELEMENTS,
                            "activation": ENERGY_HIDDEN_ACTIVATION+["linear"]},
        "cent_kwargs": {},
        "electrostatic_kwargs": {"name": "electrostatic_layer",
                                "use_physical_params": True,
                                "param_trainable": False},
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

    train_config = {
        "charge_initial_learning_rate": CHARGE_INITIAL_LEARNING_RATE,
        "charge_final_learning_rate": CHARGE_FINAL_LEARNING_RATE,
        "charge_epochs": CHARGE_EPOCHS,
        "charge_early_stopping": CHARGE_EARLY_STOPPING,
        "charge_batch_size": CHARGE_BATCH_SIZE,
        "energy_initial_learning_rate": ENERGY_INITIAL_LEARNING_RATE,
        "energy_final_learning_rate": ENERGY_FINAL_LEARNING_RATE,
        "energy_epochs": ENERGY_EPOCHS,
        "energy_early_stopping": ENERGY_EARLY_STOPPING,
        "energy_batch_size": ENERGY_BATCH_SIZE,
        "force_loss_factor": FORCE_LOSS_FACTOR,
        "model_prefix": MODEL_PREFIX
    }

    dataset = load_data(DATA_DIRECTORY, DATASET_NAME)
    model_energy_force, test_index, charge_hists, hists, scaler = train_model(dataset, model_config, charge_output, outputs, train_config)
    model_energy_force.summary()
    evaluate_model(dataset, model_energy_force, test_index, charge_hists, hists, scaler)