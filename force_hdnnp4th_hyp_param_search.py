import argparse
from datetime import timedelta
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
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
import keras_tuner as kt
from tensorflow.keras.activations import relu, tanh, elu, selu # type: ignore

from kgcnn.graph.base import GraphDict
from kgcnn.data.base import MemoryGraphList, MemoryGraphDataset
from kgcnn.data.qm import QMDataset
from kgcnn.training.scheduler import LinearLearningRateScheduler
from kgcnn.literature.HDNNP4th import make_model_behler_charge_separat as make_model
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.utils import constants, callbacks, activations
from kgcnn.utils.data_splitter import idx_generator
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss, plot_test_set_prediction
from kgcnn.model.force import EnergyForceModel
from kgcnn.model.mlmm import MLMMEnergyForceModel
from kgcnn.metrics.loss import RaggedMeanAbsoluteError

from force_hdnnp4th import load_data, train_models, evaluate_model

DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
DATASET_NAME="ThiolDisulfidExchange"
# DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water"
# DATASET_NAME="Alanindipeptide"
MODEL_PREFIX = "model_energy_force" # Will be used to save the models

TRIAL_FOLDER_NAME = "trials"

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), os.path.normpath(DATA_DIRECTORY))

INPUT_CONFIG = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
                {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
                {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
                {"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}]

CHARGE_OUTPUT = {"name": "charge", "shape": (None, 1), "ragged": True}

OUTPUTS = [
    {"name": "charge", "shape": (None, 1), "ragged": True},
    {"name": "graph_labels", "ragged": False},
    {"name": "force", "shape": (None, 3), "ragged": True}
]

# Assignment of parameters to elements
MAX_ELEMENTS = 30 # Length of the array with the symmetry function parameters, the highest possible atomic number of the elements is determined by this
ELEMENTAL_MAPPING = [1, 6, 16] # Parameters can be given individually per element. This list maps the parameters to its element
#ELEMENTAL_MAPPING = [1, 6, 7, 8] # Parameters can be given individually per element. This list maps the parameters to its element

# CHARGE MODEL HYPER PARAMETERS
CHARGE_EPOCHS                = 50 # Epochs during training
CHARGE_INITIAL_LEARNING_RATE = 1e-4 # Initial learning rate during training
CHARGE_FINAL_LEARNING_RATE   = 1e-8 # Initial learning rate during training
CHARGE_BATCH_SIZE            = 128 # Batch size during training
CHARGE_EARLY_STOPPING        = 10 # Patience of Early Stopping. If 0, no Early Stopping, Early Stopping breaks loss history plot

# ENERGY MODEL HYPER PARAMETERS
ENERGY_EPOCHS                = 50 # Epochs during training
ENERGY_INITIAL_LEARNING_RATE = 1e-4 # Initial learning rate during training
ENERGY_FINAL_LEARNING_RATE   = 1e-8 # Initial learning rate during training
ENERGY_BATCH_SIZE            = 128 # Batch size during training
ENERGY_EARLY_STOPPING        = 10 # Patience of Early Stopping. If 0, no Early Stopping, Early Stopping breaks loss history plot
FORCE_LOSS_FACTOR            = 200 # Weight of the force loss relative to the energy loss, gets normalized

# Set default configuration from global constants
CONFIG_DATA = {
    "max_elements": MAX_ELEMENTS,
    "elemental_mapping": ELEMENTAL_MAPPING,
    "input_config": INPUT_CONFIG,
    "charge_output": CHARGE_OUTPUT,
    "outputs": OUTPUTS,
    "charge_epochs": CHARGE_EPOCHS,
    "charge_initial_learning_rate": CHARGE_INITIAL_LEARNING_RATE,
    "charge_final_learning_rate": CHARGE_FINAL_LEARNING_RATE,
    "charge_batch_size": CHARGE_BATCH_SIZE,
    "charge_early_stopping": CHARGE_EARLY_STOPPING,
    "energy_epochs": ENERGY_EPOCHS,
    "energy_initial_learning_rate": ENERGY_INITIAL_LEARNING_RATE,
    "energy_final_learning_rate": ENERGY_FINAL_LEARNING_RATE,
    "energy_batch_size": ENERGY_BATCH_SIZE,
    "energy_early_stopping": ENERGY_EARLY_STOPPING,
    "force_loss_factor": FORCE_LOSS_FACTOR,
    "n_splits": 1,
}

class BaseHDNNPTuner:
    """Base class for HDNNP tuners with common hyperparameter building logic."""
    
    def _build_hyperparameters(self, hp: kt.HyperParameters) -> Dict[str, Any]:
        """Build model configuration from hyperparameters."""
        # Radial parameters
        cutoff_rad = hp.Float("cutoff_rad", 8, 30, 8)
        rs_array_choice = hp.Choice("Rs_array", [
            "0.0 4.0 6.0 8.0",
            #"0.0 3.0 5.0 7.0 9.0",
            #"0.0 3.0 4.0 5.0 6.0 7.0 8.0",
            "0.0 4.0 6.0 8.0 10.0 12.0 16.0",
            #"0.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0"
        ])
        rs_array = [float(x) for x in rs_array_choice.split()]
        
        eta_array_choice = hp.Choice("eta_array", [
            "0.0 0.08 0.3",
            "0.03 0.16 0.5",
            "0.0 0.03 0.08 0.16 0.3 0.5",
            "0.0 0.06 0.16 0.32 0.6 0.8 1.0",
            "0.0 0.03 0.08 0.16 0.3 0.5 0.6 0.75 0.9 1.0"
        ])
        eta_array = [float(x) for x in eta_array_choice.split()]

        # Angular parameters
        cutoff_ang = hp.Float("cutoff_ang", 8, 30, 8)
        lambd_array_choice = hp.Choice("lamb_array", [
            "-1 1",
            "-1 0 1", 
            "-1 -0.5 0 0.5 1"
        ])
        lambd_array = [float(x) for x in lambd_array_choice.split()]
        
        zeta_array_choice = hp.Choice("zeta_array", [
            "2 8 16",
            "1 4 8 16",
            "1 2 4 8 16",
            "1 2 4 8 16 32"
        ])
        zeta_array = [float(x) for x in zeta_array_choice.split()]

        # Charge model architecture
        charge_n_layers = hp.Int("charge_n_layers", 1, 2, 1)
        charge_layers = []
        charge_max_neurons = 151
        for i in range(charge_n_layers):
            charge_neurons = hp.Int(f"charge_neurons_{i}", 25, charge_max_neurons, 25)
            charge_max_neurons = charge_neurons + 1
            charge_layers.append(charge_neurons)
        charge_layers.append(1)

        charge_activation = hp.Choice("charge_activation", 
                                    ["relu", "tanh", "elu", "selu", "swish", "leaky_relu"])
        charge_activations = ([lambda x: activations.custom_activation(x, charge_activation)] * 
                            charge_n_layers + ["linear"])

        # Energy model architecture
        energy_n_layers = hp.Int("energy_n_layers", 1, 3, 1)
        energy_layers = []
        energy_max_neurons = 251
        for i in range(energy_n_layers):
            energy_neurons = hp.Int(f"energy_neurons_{i}", 25, energy_max_neurons, 25)
            energy_max_neurons = energy_neurons + 1
            energy_layers.append(energy_neurons)
        energy_layers.append(1)

        energy_activation = hp.Choice("energy_activation", 
                                    ["relu", "tanh", "elu", "selu", "swish", "leaky_relu"])
        energy_activations = ([lambda x: activations.custom_activation(x, energy_activation)] * 
                            energy_n_layers + ["linear"])

        return {
            "cutoff_rad": cutoff_rad,
            "rs_array": rs_array,
            "eta_array": eta_array,
            "cutoff_ang": cutoff_ang,
            "lambd_array": lambd_array,
            "zeta_array": zeta_array,
            "charge_layers": charge_layers,
            "charge_activations": charge_activations,
            "energy_layers": energy_layers,
            "energy_activations": energy_activations
        }
    
    def _build_model_config(self, hp_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Build the complete model configuration."""
        return {
            "name": "HDNNP4th",
            "inputs": config["input_config"],
            "g2_kwargs": {
                "eta": hp_config["eta_array"], 
                "rs": hp_config["rs_array"], 
                "rc": hp_config["cutoff_rad"], 
                "elements": config["elemental_mapping"]
            },
            "g4_kwargs": {
                "eta": hp_config["eta_array"], 
                "zeta": hp_config["zeta_array"], 
                "lamda": hp_config["lambd_array"], 
                "rc": hp_config["cutoff_ang"], 
                "elements": config["elemental_mapping"], 
                "multiplicity": 2.0
            },
            "normalize_kwargs": {},
            "mlp_charge_kwargs": {
                "units": hp_config["charge_layers"],
                "num_relations": config["max_elements"],
                "activation": hp_config["charge_activations"]
            },
            "mlp_local_kwargs": {
                "units": hp_config["energy_layers"],
                "num_relations": config["max_elements"],
                "activation": hp_config["energy_activations"]
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
            "output_embedding": "charge+qm_energy", 
            "output_to_tensor": True,
            "use_output_mlp": False
        }

class MyRandomTuner(BaseHDNNPTuner, kt.RandomSearch):
    def run_trial(self, trial, **kwargs):
        ks.backend.clear_session() # RAM is apparently not released between trials. This should clear some of it, but probably not all. https://github.com/keras-team/keras-tuner/issues/395
        config = kwargs.get('hyp_search_config')
        if config is None:
            raise ValueError("Hyperparameter search config must be provided")

        hp = trial.hyperparameters
        hp_config = self._build_hyperparameters(hp)
        model_config = self._build_model_config(hp_config, config)

        dataset = load_data(config)
        model_energy_force, test_index, charge_hists, hists, scaler = train_models(dataset, model_config, config["charge_output"], config["outputs"], config)
        return hists[0]

class MyHyperbandTuner(BaseHDNNPTuner, kt.Hyperband):
    def run_trial(self, trial, **kwargs):
        ks.backend.clear_session() # RAM is apparently not released between trials. This should clear some of it, but probably not all. https://github.com/keras-team/keras-tuner/issues/395
        
        config = kwargs.get('hyp_search_config')
        if config is None:
            raise ValueError("Hyperparameter search config must be provided")

        hp = trial.hyperparameters
        if "tuner/epochs" in hp.values:
            config["energy_epochs"] = hp.get("tuner/epochs")
            config["charge_epochs"] = hp.get("tuner/epochs")
        else:
            raise ValueError("tuner/epochs must be in hyperparameters?")
        hp_config = self._build_hyperparameters(hp)
        model_config = self._build_model_config(hp_config, config)

        dataset = load_data(config)
        model_energy_force, test_index, charge_hists, hists, scaler = train_models(dataset, model_config, config["charge_output"], config["outputs"], config)
        return hists[0]

class MyGridSearchTuner(BaseHDNNPTuner, kt.GridSearch):
    def run_trial(self, trial, **kwargs):
        ks.backend.clear_session() # RAM is apparently not released between trials. This should clear some of it, but probably not all.
        
        config = kwargs.get('hyp_search_config')
        if config is None:
            raise ValueError("Hyperparameter search config must be provided")
        
        hp = trial.hyperparameters
        hp_config = self._build_hyperparameters(hp)
        model_config = self._build_model_config(hp_config, config)
        dataset = load_data(config)
        model_energy_force, test_index, charge_hists, hists, scaler = train_models(dataset, model_config, config["charge_output"], config["outputs"], config)
        return hists[0]

if __name__ == "__main__":
    # Ability to restrict the model to only use a certain GPU, which is passed with python -g gpu_id, or to use a config file
    ap = argparse.ArgumentParser(description="Handle gpu_ids and training parameters")
    ap.add_argument("-g", "--gpuid", type=int)
    ap.add_argument("-c", "--conf", default=None, type=str, dest="config_path", action="store", required=False, help="Path to config file, default: None", metavar="config")
    args = ap.parse_args()
    if args.gpuid is not None:
        set_devices_gpu([args.gpuid])

    config = CONFIG_DATA.copy()
    if args.config_path is not None:
        try:
            with open(args.config_path, 'r') as config_file:
                file_config = json.load(config_file)
        except FileNotFoundError:
            print(f"Config file {args.config_path} not found.")
            exit(1)
        config.update(file_config)

    for key, value in config.items():
        print(f"{key}: {value}")
    

    # random_tuner = MyRandomTuner(
    #     objective=kt.Objective("val_output_3_loss", direction="min"),
    #     max_trials=500,
    #     overwrite=False,
    #     directory=TRIAL_FOLDER_NAME,
    #     project_name="random_search"
    # )
    # random_tuner.search_space_summary()
    # random_tuner.search(hyp_search_config=config) 

    # random_tuner.results_summary(num_trials=10)

    # n_best_hps = random_tuner.get_best_hyperparameters(num_trials=10)

    # with open(os.path.join("best_hp.json"), "w") as f:
    #     json.dump(n_best_hps[0].values, f, indent=2)

    hyperband_tuner = MyHyperbandTuner(
        objective=kt.Objective("val_output_3_loss", direction="min"),
        max_epochs=400,
        factor=3,
        hyperband_iterations=1,
        overwrite=False,
        directory=TRIAL_FOLDER_NAME,
        project_name="hyperband_search",
        max_consecutive_failed_trials=1
    )
    hyperband_tuner.search_space_summary()
    hyperband_tuner.search(hyp_search_config=config) 
    hyperband_tuner.results_summary(num_trials=10)
    n_best_hps = hyperband_tuner.get_best_hyperparameters(num_trials=10)
    with open(os.path.join("best_hp_hyperband.json"), "w") as f:
        json.dump(n_best_hps[0].values, f, indent=2)