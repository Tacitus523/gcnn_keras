import argparse
from datetime import timedelta
import json
import os
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any

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
from kgcnn.training.scheduler import LinearLearningRateScheduler
from kgcnn.literature.PAiNN import make_model
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss, plot_test_set_prediction, print_error_dict
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.utils import constants, save_load_utils, activations, wandb_wizard
from kgcnn.utils.tools import get_git_commit_hash
from kgcnn.model.force import EnergyForceModel

# DEFAULT VALUES
# DATA READ AND SAVE
DATA_DIRECTORY = os.getcwd() # Folder containing DATASET_NAME.kgcnn.pickle
DATASET_NAME = "Alanindipeptide" # Used in naming plots and looking for data
MODEL_PREFIX = "model_energy_force" # Will be used to save the models

# ENERGY MODEL HYPER PARAMETERS
ENERGY_EPOCHS                = 1000 # Epochs during training
ENERGY_INITIAL_LEARNING_RATE = 1e-3 # Initial learning rate during training
ENERGY_FINAL_LEARNING_RATE   = 1e-8 # Final learning rate during training
ENERGY_BATCH_SIZE            = 64 # Batch size during training
ENERGY_EARLY_STOPPING        = 0 # Patience of Early Stopping. If 0, no Early Stopping
FORCE_LOSS_FACTOR            = 95

# PAINN MODEL HYPER PARAMETERS
INPUT_EMBEDDING_DIM          = 128 # Output dimension of node embedding
BESSEL_NUM_RADIAL            = 20 # Number of radial basis functions
BESSEL_CUTOFF                = 5.0 # Cutoff distance for radial basis
BESSEL_ENVELOPE_EXPONENT     = 5 # Envelope exponent for radial basis
CONV_UNITS                   = 128 # Units in convolution layers
UPDATE_UNITS                 = 128 # Units in update layers
MODEL_DEPTH                  = 6 # Number of interaction layers
OUTPUT_MLP_UNITS             = [128, 1] # Units in output MLP layers

N_SPLITS = 3 # Number of splits for cross-validation, used in KFold

MAX_DATASET_SIZE = None  # we will subsample the dataset to this size for speed

# SCALER PARAMETERS
USE_SCALER = True # If True, the scaler will be used
SCALER_PATH = "scaler.json" # None if no scaler is used
STANDARDIZE_SCALE = True # If True, the scaler will standardize the energy labels and force labels accordingly

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
    "energy_epochs": ENERGY_EPOCHS,
    "energy_initial_learning_rate": ENERGY_INITIAL_LEARNING_RATE,
    "energy_final_learning_rate": ENERGY_FINAL_LEARNING_RATE,
    "energy_batch_size": ENERGY_BATCH_SIZE,
    "energy_early_stopping": ENERGY_EARLY_STOPPING,
    "force_loss_factor": FORCE_LOSS_FACTOR,
    "input_embedding_dim": INPUT_EMBEDDING_DIM,
    "bessel_num_radial": BESSEL_NUM_RADIAL,
    "bessel_cutoff": BESSEL_CUTOFF,
    "bessel_envelope_exponent": BESSEL_ENVELOPE_EXPONENT,
    "conv_units": CONV_UNITS,
    "update_units": UPDATE_UNITS,
    "model_depth": MODEL_DEPTH,
    "output_mlp_units": OUTPUT_MLP_UNITS,
    "n_splits": N_SPLITS,
    "max_dataset_size": MAX_DATASET_SIZE,
    "use_scaler": USE_SCALER,
    "scaler_path": SCALER_PATH,
    "standardize_scale": STANDARDIZE_SCALE,
    "use_wandb": USE_WANDB,
    "wandb_project": WANDB_PROJECT,
    "wandb_entity": WANDB_ENTITY,
    "wandb_name": WANDB_NAME
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

def create_model(train_config: Dict, model_config: Dict) -> EnergyForceModel:
    """Create and return a PAiNN energy-force model."""
    force_loss_factor = train_config["force_loss_factor"]
    
    # Create energy model
    model_energy = make_model(**model_config)

    # Create energy-force model
    model_energy_force: EnergyForceModel = EnergyForceModel(
        model_energy=model_energy,
        energy_output=0,
        esp_input=None,
        esp_grad_input=None,
        output_to_tensor=True,
        output_as_dict=False,
        output_squeeze_states=True,
        is_physical_force=False
    )

    # energy_initial_learning_rate = train_config["energy_initial_learning_rate"]
    # energy_final_learning_rate = train_config["energy_final_learning_rate"]
    # lr_schedule = ks.optimizers.schedules.CosineDecayRestarts(
    #     initial_learning_rate=energy_initial_learning_rate, 
    #     first_decay_steps=1e3, 
    #     t_mul=1.5, 
    #     m_mul=0.7, 
    #     alpha=energy_final_learning_rate/energy_initial_learning_rate
    # )
    
    model_energy_force.compile(
        loss=["mean_squared_error", "mean_squared_error"],
        #optimizer=ks.optimizers.Adam(lr_schedule),
        optimizer=ks.optimizers.Adam(),
        metrics=None,
        loss_weights=[1/force_loss_factor, 1-1/force_loss_factor]
    )
    return model_energy_force

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
    # lrlog = callbacks.LearningRateLoggingCallback()
    # callbacks.append(lrlog)

    if energy_early_stopping > 0:
        earlystop = ks.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=energy_early_stopping,
            verbose=0
        )
        callbacks.append(earlystop)

    if train_config["use_wandb"]:
        wandb_wizard.init_wandb(train_config)
        callbacks.append(wandb_wizard.construct_wandb_callback(key_prefix="EnergyForce"))

    start = time.process_time()
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
    stop = time.process_time()
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
    print(model_config)

    n_splits = train_config["n_splits"]
    use_scaler = train_config["use_scaler"]
    scaler_path = train_config["scaler_path"]
    standardize_scale = train_config["standardize_scale"]
    model_prefix = train_config["model_prefix"]
    do_search = train_config.get("do_search", False)

    # Scaling energy and forces.
    if use_scaler:
        scaler = EnergyForceExtensiveLabelScaler(
            standardize_scale=standardize_scale,
            standardize_coordinates = False,
            energy= "graph_labels",
            force = "force",
            atomic_number = "node_number",
            sample_weight = None)
        scaler.fit_transform_dataset(dataset)
        scaler.save(scaler_path)
    else:
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
        
        # Train single fold
        model_energy_force = models[model_index]
        hist = train_single_fold(
            train_val_dataset=train_val_dataset,
            train_index=train_index,
            val_index=val_index,
            model_energy_force=model_energy_force,
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
            model_energy_force.save(model_prefix + str(model_index))

            evaluate_model(
                dataset=dataset,
                model_energy_force=model_energy_force,
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
        save_load_utils.save_history(hists, filename="histories.pkl")
        save_load_utils.save_training_indices(*indices)

        plot_train_test_loss(hists,
            filepath="", data_unit="eV",
            model_name="PaiNN", dataset_name=dataset.dataset_name, file_name="loss.png", show_fig=False)

        model_energy_force.summary()
        energy_model = model_energy_force._model_energy
        energy_model.summary()

    return model_energy_force, indices, hists, scaler

def evaluate_model(dataset: MemoryGraphDataset,
                   model_energy_force: EnergyForceModel,
                   indices: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   model_config: Dict,
                   train_config: Dict,
                   scaler: Optional[EnergyForceExtensiveLabelScaler] = None,
                   model_index: Optional[int] = None) -> None:
    dataset_name = dataset.dataset_name
    model_suffix = f"_{model_index}" if model_index is not None else ""

    # Process each stage of the dataset, test must be the last stage
    stages = ["train", "val", "test"]
    error_dict = {}
    wandb_error_dict = {}
    for stage, stage_index in zip(stages, indices):
        stage_dataset = dataset[stage_index]
        atomic_numbers_list = stage_dataset.get("node_number")
        true_energy = stage_dataset.get("graph_labels")
        true_force = stage_dataset.get("force")
        x_stage = stage_dataset.tensor(model_config["inputs"])
        predicted_energy, predicted_force = model_energy_force.predict(x_stage, batch_size=train_config["energy_batch_size"], verbose=0)

        if train_config["use_scaler"]:
            true_energy, true_force = scaler.inverse_transform(
                y=(true_energy, true_force), X=atomic_numbers_list)
            predicted_energy, predicted_force = scaler.inverse_transform(
            y=(predicted_energy.flatten(), predicted_force), X=atomic_numbers_list)

        true_energy = np.array(true_energy).reshape(-1,1)*constants.hartree_to_eV
        true_force = np.array(true_force).reshape(-1,1)*constants.hartree_bohr_to_eV_angstrom

        predicted_energy = np.array(predicted_energy).reshape(-1,1)*constants.hartree_to_eV
        predicted_force = np.array(predicted_force).reshape(-1,1)*constants.hartree_bohr_to_eV_angstrom
        
        # Calculate metrics
        for label, true_value, predicted_value in zip(
            ["energy", "force"],
            [true_energy, true_force],
            [predicted_energy, predicted_force]
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
    ref_infos: Dict[str, np.ndarray] = {}
    ref_arrays: Dict[str, List[np.ndarray]] = {}
    pred_infos: Dict[str, np.ndarray] = {}
    pred_arrays: Dict[str, List[np.ndarray]] = {}

    ref_infos["energy"] = true_energy
    ref_arrays["forces"] = true_force
    pred_infos["energy"] = predicted_energy
    pred_arrays["forces"] = predicted_force

    save_load_utils.save_extxyz(
        atomic_numbers_list=atomic_numbers_list,
        positions_list=positions_list,
        ref_infos=ref_infos,
        ref_arrays=ref_arrays,
        pred_infos=pred_infos,
        pred_arrays=pred_arrays,
        filename=f"PaiNN_geoms{model_suffix}.extxyz"
    )

    plot_predict_true(predicted_energy, true_energy,
        filepath="", data_unit="eV",
        model_name="PaiNN", dataset_name=dataset_name, target_names="Energy",
        error="RMSE", file_name=f"predict_energy{model_suffix}.png", show_fig=False)

    plot_predict_true(predicted_force, true_force,
        filepath="", data_unit=r"$\frac{eV}{\AA}$",
        model_name="PaiNN", dataset_name=dataset_name, target_names="Force",
        error="RMSE", file_name=f"predict_force{model_suffix}.png", show_fig=False)

    energy_df = pd.DataFrame({"energy_reference": true_energy.flatten(), "energy_prediction": predicted_energy.flatten()})
    force_df = pd.DataFrame({"force_reference": true_force.flatten(), "force_prediction": predicted_force.flatten()})

    at_types_column = pd.Series(np.array(atomic_numbers_list).flatten(), name="at_types").replace(constants.atomic_number_to_element)
    force_df["at_types"] = at_types_column.repeat(3).reset_index(drop=True)

    rmse_energy, r2_energy = error_dict["Test RMSE Energy"], error_dict["Test R2 Energy"]
    rmse_force, r2_force = error_dict["Test RMSE Force"], error_dict["Test R2 Force"]
    plot_test_set_prediction(energy_df, "energy_reference", "energy_prediction",
        "Energy", "eV", rmse_energy, r2_energy, f"energy_lmplot{model_suffix}.png")
    plot_test_set_prediction(force_df, "force_reference", "force_prediction",
        "Force", r"$\frac{eV}{\AA}$", rmse_force, r2_force, f"force_lmplot{model_suffix}.png")

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

        # Update config_data with values from config file
        config_data.update(file_config_data)
    
    config_data["git_commit_hash"] = get_git_commit_hash()

    for key, value in config_data.items():
        print(f"{key}: {value}")
    return config_data

def main(config: Dict[str, Any]) -> None:
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
        "input_embedding": {"node": {"input_dim": 95, "output_dim": config["input_embedding_dim"]}},
        "equiv_initialize_kwargs": {"dim": 3, "method": "eps"},# "units": 128},
        #"input_node_embedding": {"input_dim": 95, "output_dim": 128},
        "bessel_basis": {
            "num_radial": config["bessel_num_radial"],
            "cutoff": config["bessel_cutoff"], 
            "envelope_exponent": config["bessel_envelope_exponent"]
        },
        "pooling_args": {"pooling_method": "sum"},
        "conv_args": {"units": config["conv_units"], "cutoff": None},
        "update_args": {"units": config["update_units"]},
        "depth": config["model_depth"], "verbose": 10,
        "output_embedding": "graph",
        "output_mlp": {"use_bias": [True, True], "units": config["output_mlp_units"], "activation": ["swish", "linear"]},
    }

    outputs = [
        {"name": "graph_labels", "ragged": False},
        {"name": "force", "shape": (None, 3), "ragged": True}
    ]

    dataset = load_data(config)
    
    # Create models for cross-validation
    models = [create_model(config, model_config) for _ in range(config["n_splits"])]
    
    model_energy_force, indices, hists, scaler = train_models(dataset, models, model_config, outputs, config)
    print("Training and evaluation completed successfully.")

if __name__ == "__main__":
    config = parse_arguments()
    main(config)