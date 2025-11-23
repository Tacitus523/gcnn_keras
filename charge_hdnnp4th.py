import argparse
from datetime import timedelta
import json
import os
import time
import warnings
from typing import Dict, List, Tuple, Optional, Dict, Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
ks=tf.keras
print(tf.config.list_physical_devices('GPU'))

from kgcnn.graph.base import GraphDict
from kgcnn.data.base import MemoryGraphList, MemoryGraphDataset
from kgcnn.data.qm import QMDataset
from kgcnn.utils.callbacks import TrainingTimeCallback
from kgcnn.training.scheduler import LinearLearningRateScheduler
from kgcnn.literature.HDNNP4th import make_model_behler as make_model
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss, plot_test_set_prediction, print_error_dict
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.utils import constants, save_load_utils, activations, wandb_wizard
from kgcnn.utils.tools import get_git_commit_hash
from kgcnn.model.force import EnergyForceModel
from kgcnn.metrics.loss import RaggedMeanAbsoluteError, zero_loss_function

# DEFAULT VALUES
# DATA READ AND SAVE
DATA_DIRECTORY = os.getcwd() # Folder containing DATASET_NAME.kgcnn.pickle
DATASET_NAME = "Alanindipeptide" # Used in naming plots and looking for data
MODEL_PREFIX = "model_energy_force" # Will be used to save the models

# SYMMETRY FUNCTION HYPER PARAMETERS
# Radial parameters
CUTOFF_RAD = 20 # Radial cutoff distance in Bohr
RS_ARRAY   = [0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] # Shift parameter Rs in Bohr
ETA_ARRAY  = [0.03, 0.08, 0.16, 0.3, 0.5] # Width parameter eta in 1/Bohr^2

# Angular parameters
CUTOFF_ANG    = 12 # Angular cutoff distance in Bohr
LAMBD_ARRAY   = [-1, 1] # Lambda parameter
ZETA_ARRAY    = [1, 2, 4, 8, 16] # Zeta parameter
ETA_ANG_ARRAY = ETA_ARRAY # Width parameter eta in 1/Bohr^2

# Assignment of parameters to elements
MAX_ELEMENTS = 30 # Length of the array with the symmetry function parameters, the highest possible atomic number of the elements is determined by this
ELEMENTAL_MAPPING = [1, 6, 7, 8] # Parameters can be given individually per element. This list maps the parameters to its element

# CHARGE MODEL HYPER PARAMETERS
CHARGE_HIDDEN_LAYERS         = [15] # List of number of nodes per hidden layer 
CHARGE_HIDDEN_ACTIVATION     = ["tanh"] # List of activation functions of hidden layers

# ENERGY MODEL HYPER PARAMETERS
ENERGY_EPOCHS                = 500 # Epochs during training
ENERGY_INITIAL_LEARNING_RATE = 1e-3 # Initial learning rate during training
ENERGY_FINAL_LEARNING_RATE   = 1e-8 # Final learning rate during training
ENERGY_HIDDEN_LAYERS         = [35, 35] # List of number of nodes per hidden layer 
ENERGY_HIDDEN_ACTIVATION     = ["tanh", "tanh"] # List of activation functions of hidden layers
ENERGY_BATCH_SIZE            = 128 # Batch size during training
ENERGY_EARLY_STOPPING        = 0 # Patience of Early Stopping. If 0, no Early Stopping, Early Stopping breaks loss history plot

N_SPLITS = 3 # Number of splits for cross-validation, used in KFold

MAX_DATASET_SIZE = None  # we will subsample the dataset to this size for speed

# WANDB CONFIGURATION
USE_WANDB = False  # Whether to use wandb logging
WANDB_PROJECT = None  # Project name
WANDB_ENTITY = None # Team name or username
WANDB_NAME: Optional[str] = os.getenv("SLURM_JOB_NAME", os.path.basename(os.getcwd()))

# Set default configuration from global constants
CONFIG_DATA = {
    "data_directory": DATA_DIRECTORY,
    "dataset_name": DATASET_NAME,
    "model_prefix": MODEL_PREFIX,
    "cutoff_rad": CUTOFF_RAD,
    "rs_array": RS_ARRAY,
    "eta_array": ETA_ARRAY,
    "cutoff_ang": CUTOFF_ANG,
    "lambd_array": LAMBD_ARRAY,
    "zeta_array": ZETA_ARRAY,
    "eta_ang_array": ETA_ANG_ARRAY,
    "max_elements": MAX_ELEMENTS,
    "elemental_mapping": ELEMENTAL_MAPPING,
    "charge_hidden_layers": CHARGE_HIDDEN_LAYERS,
    "charge_hidden_activation": CHARGE_HIDDEN_ACTIVATION,
    "energy_epochs": ENERGY_EPOCHS,
    "energy_initial_learning_rate": ENERGY_INITIAL_LEARNING_RATE,
    "energy_final_learning_rate": ENERGY_FINAL_LEARNING_RATE,
    "energy_hidden_layers": ENERGY_HIDDEN_LAYERS,
    "energy_hidden_activation": ENERGY_HIDDEN_ACTIVATION,
    "energy_batch_size": ENERGY_BATCH_SIZE,
    "energy_early_stopping": ENERGY_EARLY_STOPPING,
    "n_splits": N_SPLITS,
    "max_dataset_size": MAX_DATASET_SIZE,
    "use_wandb": USE_WANDB,
    "wandb_project": WANDB_PROJECT,
    "wandb_entity": WANDB_ENTITY,
    "wandb_name": WANDB_NAME
}

BASE_MODEL_CONFIG = {
    "name": "HDNNP4th",
    "inputs": [
        {"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
        {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
        {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
        {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
        {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
        {"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}
    ],
}

def load_data(config: Dict) -> MemoryGraphDataset:
    data_directory = config["data_directory"]
    dataset_name = config["dataset_name"]
    dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=dataset_name)
    dataset.load()
    
    file_name=f"{dataset_name}.csv"
    data_directory = os.path.normpath(data_directory)
    print("Dataset:", os.path.join(data_directory, file_name))
    print(dataset[0].keys())
    
    return dataset

def create_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create model configuration for HDNNP4th."""
    model_config = {
        "g2_kwargs": {
            "eta": config["eta_array"], 
            "rs": config["rs_array"], 
            "rc": config["cutoff_rad"], 
            "elements": config["elemental_mapping"]
        },
        "g4_kwargs": {
            "eta": config["eta_ang_array"], 
            "zeta": config["zeta_array"], 
            "lamda": config["lambd_array"], 
            "rc": config["cutoff_ang"], 
            "elements": config["elemental_mapping"],
            "multiplicity": 2.0
        },
        "normalize_kwargs": {},
        "mlp_charge_kwargs": {
            "units": config["charge_hidden_layers"]+[1],
            "num_relations": config["max_elements"],
            "activation": [lambda x: activations.custom_activation(x, charge_activation) 
                          for charge_activation in config["charge_hidden_activation"]]+["linear"]
        },
        "mlp_local_kwargs": {
            "units": config["energy_hidden_layers"] + [1],
            "num_relations": config["max_elements"],
            "activation": [lambda x: activations.custom_activation(x, energy_activation) 
                          for energy_activation in config["energy_hidden_activation"]] + ["linear"]
        },
        "cent_kwargs": {},
        "electrostatic_kwargs": {
            "name": "electrostatic_layer",
            "use_physical_params": True,
            "param_trainable": False
        },
        "qmmm_kwargs": {"name": "qmmm_layer"},
        "node_pooling_args": {"pooling_method": "sum"},
        "verbose": 10,
        "output_embedding": "charge", 
        "output_to_tensor": True,
        "use_output_mlp": False,
        "output_mlp": None
    }
    model_config.update(BASE_MODEL_CONFIG)  # Merge with base config to ensure inputs are included
    print("Model config:")
    print(model_config) 
    return model_config

def create_model(train_config: Dict, model_config: Dict) -> EnergyForceModel:
    """
    Create and return charge model, energy model, and energy-force model.
    
    Args:
        model_config: Configuration dictionary for model creation
        
    Returns:
        model_energy_force: Compiled energy-force model
    """
    # Create the base models
    model_charge = make_model(**model_config)

    # Name the outputs for better history tracking
    model_charge.output_names = ["charge"]

    optimizer = ks.optimizers.Adam()
    optimizer.clipnorm = 1.0 # Gradient clipping to avoid exploding gradients

    model_charge.compile(
        loss=["mean_squared_error"],
        optimizer=optimizer,
        metrics=None,
    )
    return model_charge

def train_single_fold(train_val_dataset: MemoryGraphDataset,
                        train_index: np.ndarray,
                        val_index: np.ndarray,
                        model_energy_force: EnergyForceModel,
                        model_config: Dict,
                        outputs: List[Dict],
                        train_config: Dict,
                        model_index: int,
                        **kwargs
                        ) -> tf.keras.callbacks.History:
    """Train a single fold of the cross-validation."""
    
    # Prepare data for this fold
    x_train = train_val_dataset[train_index].tensor(model_config["inputs"])
    x_val = train_val_dataset[val_index].tensor(model_config["inputs"])
    energy_force_train = train_val_dataset[train_index].tensor(outputs)
    energy_force_val = train_val_dataset[val_index].tensor(outputs)

    # Extract training parameters
    energy_initial_learning_rate = train_config["energy_initial_learning_rate"]
    energy_final_learning_rate = train_config["energy_final_learning_rate"]
    energy_epochs = train_config["energy_epochs"]
    energy_early_stopping = train_config["energy_early_stopping"]
    energy_batch_size = train_config["energy_batch_size"]

    # Train energy-force model
    callbacks = []
    scheduler = LinearLearningRateScheduler(
        learning_rate_start=energy_initial_learning_rate,
        learning_rate_stop=energy_final_learning_rate,
        epo_min=0,
        epo=energy_epochs)
    callbacks.append(scheduler)
    callbacks.append(TrainingTimeCallback())

    if energy_early_stopping > 0:
        earlystop = ks.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=energy_early_stopping,
            verbose=0,
            restore_best_weights=True
        )
        callbacks.append(earlystop)

    if train_config["use_wandb"]:
        wandb_wizard.init_wandb(train_config)
        callbacks.append(wandb_wizard.construct_wandb_callback(key_prefix="Charge"))

    start = time.time()
    kwargs["epochs"] = kwargs.get("epochs", train_config["energy_epochs"])
    kwargs["initial_epoch"] = kwargs.get("initial_epoch", train_config.get("initial_epoch", 0))
    kwargs["callbacks"] = kwargs.get("callbacks", []) + callbacks
    hist = model_energy_force.fit(
        x_train, energy_force_train,
        validation_data=(x_val, energy_force_val),
        batch_size=energy_batch_size,
        verbose=2,
        shuffle=True,
        **kwargs
    )
    stop = time.time()
    print("Energy-force model training time: ", str(timedelta(seconds=stop - start)))

    return hist

def train_models(dataset: MemoryGraphDataset,
                models: List[EnergyForceModel],
                model_config: Dict,
                outputs: List[Dict],
                train_config: Dict,
                **kwargs
                ) -> Tuple[
                    EnergyForceModel,
                    List[tf.keras.callbacks.History],
                    Optional[EnergyForceExtensiveLabelScaler]
                ]:
    """Train models using cross-validation."""

    n_splits = train_config["n_splits"]
    model_prefix = train_config["model_prefix"]
    do_search = train_config.get("do_search", False)

    scaler = None

    data_indices = np.arange(len(dataset))
    train_val_index, test_index = train_test_split(
        data_indices, test_size=0.10, random_state=42, shuffle=True
    )
    # Subsample training/validation set if specified
    if train_config.get("max_dataset_size") is not None:
        train_val_index = train_val_index[:min(train_config["max_dataset_size"], len(train_val_index))]
    train_val_dataset = dataset[train_val_index]

    if n_splits > 1:
        kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    else:
        kf = KFold(n_splits=3, random_state=42, shuffle=True)
    hists = []
    indices = [[],[],test_index]  # Store train, val, and test indices
    model_index = 0
    
    for train_index, val_index in kf.split(X=np.expand_dims(train_val_index, axis=-1)):
        train_index, val_index = val_index, train_index # Switched train and test indices to keep training data separate
        print(f"Training fold {model_index + 1}/{n_splits}")

        model_charge = models[model_index]
        
        # Train single fold
        hist = train_single_fold(
            train_val_dataset=train_val_dataset,
            train_index=train_index,
            val_index=val_index,
            model_energy_force=model_charge,
            model_config=model_config,
            outputs=outputs,
            train_config=train_config,
            model_index=model_index,
            **kwargs
        )

        # Evaluate the model after training
        # Convert relative indices of train_val_dataset to absolute indices in the full dataset
        # This is necessary because the indices from KFold are relative to the train_val_dataset
        # and we need to evaluate on the full dataset
        abs_train_index = train_val_index[train_index]
        abs_val_index = train_val_index[val_index]
        
        if not do_search:
            # Save the model
            model_charge.save(model_prefix + str(model_index))

            evaluate_model(
                dataset=dataset,
                model_energy_force=model_charge,
                indices=(abs_train_index, abs_val_index, test_index),
                model_config=model_config,
                train_config=train_config,
                scaler=scaler,
                model_index=model_index
            )

        # Store results (using absolute indices)
        hists.append(hist)
        indices[0].append(abs_train_index)
        indices[1].append(abs_val_index)
        model_index += 1
        if n_splits == 1:
            break

    if not do_search:
        # Save training history and indices
        save_load_utils.save_history(hists, filename="histories.pkl")
        save_load_utils.save_training_indices(*indices)

        plot_train_test_loss(hists,
            filepath="", data_unit="eV",
            model_name="HDNNP", dataset_name=dataset.dataset_name, file_name="loss.png", show_fig=False)

        model_charge.summary()
    
    return model_charge, indices, hists, scaler

def evaluate_model(dataset: MemoryGraphDataset,
                   model_energy_force: EnergyForceModel,
                   indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   model_config: Dict,
                   train_config: Dict,
                   scaler: Optional[EnergyForceExtensiveLabelScaler] = None,
                   model_index: Optional[int] = None) -> None:
    """Evaluate the trained model on train, validation, and test sets."""
    dataset_name = dataset.dataset_name
    model_suffix = f"_{model_index}" if model_index is not None else ""

    # Process each stage of the dataset, test must be the last stage
    stages = ["train", "val", "test"]
    error_dict = {}
    wandb_error_dict = {}
    for stage, stage_index in zip(stages, indices):
        if not stage_index.size:
            continue # Skip empty indices
        stage_dataset = dataset[stage_index]
        atomic_numbers_list = stage_dataset.get("node_number")
        true_charge = stage_dataset.get("charge")
        x_stage = stage_dataset.tensor(model_config["inputs"])
        predicted_charge = model_energy_force.predict(x_stage, batch_size=train_config["energy_batch_size"], verbose=0)

        flat_true_charge = np.concatenate([array.reshape(-1,1) for array in true_charge])
        flat_predicted_charge = np.concatenate([array.reshape(-1,1) for array in predicted_charge])


        # Calculate metrics
        for label, true_value, predicted_value in zip(
            ["charge"],
            [flat_true_charge],
            [flat_predicted_charge]
        ):
            rmse = mean_squared_error(true_value, predicted_value, squared=False)
            mae = mean_absolute_error(true_value, predicted_value)
            r2 = r2_score(true_value, predicted_value)
            error_dict[f"{stage.title()} RMSE {label.title()}"] = rmse
            error_dict[f"{stage.title()} MAE {label.title()}"] = mae
            error_dict[f"{stage.title()} R2 {label.title()}"] = r2
            wandb_error_dict[f"{stage.title()}/{label}_rmse"] = rmse
            wandb_error_dict[f"{stage.title()}/{label}_mae"] = mae
            wandb_error_dict[f"{stage.title()}/{label}_r2"] = r2

    print_error_dict(error_dict)

    with open(os.path.join("", f"errors{model_suffix}.json"), "w") as f:
        json.dump(error_dict, f, indent=2, sort_keys=True)

    if train_config["use_wandb"]:
        wandb_wizard.log_wandb_metrics(metrics=wandb_error_dict)
        wandb_wizard.finish_wandb()

    # Remaining evaluation only on the test set
    positions_list: List[np.ndarray] = stage_dataset.get("node_coordinates")
    positions_list = [array*constants.bohr_to_angstrom for array in positions_list]

    ref_infos: Dict[str, np.ndarray] = {}
    ref_arrays: Dict[str, List[np.ndarray]] = {}
    pred_infos: Dict[str, np.ndarray] = {}
    pred_arrays: Dict[str, List[np.ndarray]] = {}


    ref_arrays["charges"] = true_charge
    pred_arrays["charges"] = predicted_charge

    save_load_utils.save_extxyz(
        atomic_numbers_list=atomic_numbers_list,
        positions_list=positions_list,
        ref_infos=ref_infos,
        ref_arrays=ref_arrays,
        pred_infos=pred_infos,
        pred_arrays=pred_arrays,
        filename=f"HDNNP_geoms{model_suffix}.extxyz"
    )

    plot_predict_true(flat_predicted_charge, flat_true_charge,
        filepath="", data_unit="e",
        model_name="HDNNP", dataset_name=dataset_name, target_names="Charge",
        error="RMSE", file_name=f"predict_charge{model_suffix}.png", show_fig=False)


    charge_df = pd.DataFrame({"charge_reference": flat_true_charge.flatten(), "charge_prediction": flat_predicted_charge.flatten()})

    at_types_column = pd.Series(np.array(atomic_numbers_list).flatten(), name="at_types").replace(constants.atomic_number_to_element)
    charge_df["at_types"] = at_types_column

    rmse_charge, r2_charge = error_dict["Test RMSE Charge"], error_dict["Test R2 Charge"]

    plot_test_set_prediction(charge_df, "charge_reference", "charge_prediction",
        "Charge", "e", rmse_charge, r2_charge, f"charge_lmplot{model_suffix}.png")

def parse_arguments() -> Dict[str, Any]:
    """Parse command line arguments and return configuration data."""
    ap = argparse.ArgumentParser(description="Handle gpu_ids and training parameters")
    ap.add_argument("-g", "--gpuid", type=int, help="GPU ID to use for training")
    ap.add_argument("-c", "--conf", default=None, type=str, dest="config_path", 
                   action="store", required=False, help="Path to config file, default: None", 
                   metavar="config")
    args = ap.parse_args()
    
    if args.gpuid is not None:
        set_devices_gpu([args.gpuid])
        
    config_data = {key.lower(): value for key, value in CONFIG_DATA.items()}
    if args.config_path is not None:
        try:
            with open(args.config_path, 'r') as config_file:
                file_config_data = json.load(config_file)
            file_config_data = {key.lower(): value for key, value in file_config_data.items()}
        except FileNotFoundError:
            print(f"Config file {args.config_path} not found.")
            exit(1)

        for key in file_config_data.keys():
            if key not in config_data.keys():
                raise KeyError(f"Unknown configuration key: {key}")

        # Update config_data with values from config file
        config_data.update(file_config_data)
    
    config_data["git_commit_hash"] = get_git_commit_hash()

    for key, value in config_data.items():
        print(f"{key}: {value}")
    return config_data

def main(config: Dict[str, Any]) -> None:
    model_config = create_model_config(config)

    outputs = [
        {"name": "charge", "shape": (None, 1), "ragged": True},
    ]

    dataset = load_data(config)
    models = [create_model(config, model_config) for _ in range(config["n_splits"])]
    model_energy_force, indices, hists, scaler = train_models(dataset, models, model_config, outputs, config)
    print("Training and evaluation completed successfully.")

if __name__ == "__main__":
    config = parse_arguments()
    main(config)
