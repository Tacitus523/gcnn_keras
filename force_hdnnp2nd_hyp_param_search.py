import argparse
from datetime import timedelta
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
ks=tf.keras
print(tf.config.list_physical_devices('GPU'))
import keras_tuner as kt

from kgcnn.utils import constants, callbacks, activations
from kgcnn.utils.devices import set_devices_gpu

from force_hdnnp2nd import load_data, train_models, evaluate_model, CONFIG_DATA, create_model, create_model_config

TRIAL_FOLDER_NAME = "trials"
PROJECT_NAME = "hdnnp2nd_hyp_search"

MAX_DATASET_SIZE = 5000  # we will subsample the dataset to this size for speed

# MAX_EPOCHS = 25 # Less epochs for grid search
MAX_EPOCHS = 200 # Maximum epochs during search
HYPERBAND_FACTOR = 2 # Factor by which to increase hyperband epochs until MAX_EPOCHS is reached

# Set default configuration from global constants
HYP_PARAM_SEARCH_CONFIGS = {
    "project_name": PROJECT_NAME,
    "max_epochs": MAX_EPOCHS,
    "hyperband_factor": HYPERBAND_FACTOR
}
CONFIG_DATA.update(HYP_PARAM_SEARCH_CONFIGS)

# Temporary configuration for hyperparameter search, overwrites configs temporarily, original configs used for training of best model
HYP_PARAM_SEARCH_TEMP_CONFIGS = {
    "do_search": True,
    "n_splits": 1,
    "max_dataset_size": MAX_DATASET_SIZE,
    "use_wandb": False
}

def parse_args() -> Dict[str, Any]:
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
                file_config = {k.lower(): v for k, v in file_config.items()}
        except FileNotFoundError:
            print(f"Config file {args.config_path} not found.")
            exit(1)

        for key in file_config.keys():
            if key not in config.keys():
                raise KeyError(f"Unknown configuration key: {key}")
            
        config.update(file_config)

    for key, value in config.items():
        print(f"{key}: {value}")
    return config
    

class BaseHDNNP2ndTuner:
    """Base class for HDNNP2nd tuners with common hyperparameter building logic."""
    _outputs = [
        {"name": "graph_labels", "ragged": False},
        {"name": "force", "shape": (None, 3), "ragged": True}
    ]

    def _build_raw_hyperparameters(self, hp: kt.HyperParameters) -> Dict[str, Any]:
        """Build raw hyperparameters from the tuner."""

        # Symmetry function hyperparameters
        rs_array_choice = "0.0 2.0 4.0 6.0"
        # rs_array_choice = hp.Choice("Rs_array", [
        #     "0.0 1.0 2.0 3.0",
        #     "0.0 2.0 4.0 6.0",
        #     "0.0 3.0 5.0 7.0 9.0",
        #     "0.0 1.0 3.0 4.0 5.0 6.0 7.0 ",
        #     "0.0 2.0 4.0 6.0 8.0 10.0 12.0",
        #     "0.0 1.0 2.0 3.0 4.0 5.0 7.0 10.0 12.0"
        # ])
        rs_array_choice = hp.Choice("Rs_array", [
            "0.0 4.0 6.0 8.0",
            "0.0 3.0 5.0 7.0 9.0",
            "0.0 3.0 4.0 5.0 6.0 7.0 8.0",
            "0.0 4.0 6.0 8.0 10.0 12.0 16.0",
        ])
        eta_array_choice = "0.08 0.3"
        # eta_array_choice = hp.Choice("eta_array", [
        #     "0.03 0.08",
        #     "0.08 0.3",
        #     "0.3 0.8",
        #     "0.03 0.16 0.5",
        #     "0.03 0.3 0.8",
        #     "0.03 0.08 0.16 0.3 0.5",
        #     "0.06 0.16 0.32 0.6 0.8 1.0",
        #     "0.03 0.08 0.16 0.3 0.5 0.6 0.75 0.9 1.0"
        # ])
        eta_array_choice = hp.Choice("eta_array", [
            "0.08 0.3",
            "0.03 0.16 0.5",
            "0.03 0.08 0.16 0.3 0.5",
            "0.03 0.08 0.16 0.32 0.6 0.8 1.0",
        ])
        lambd_array_choice = "-1 1"
        lambd_array_choice = hp.Choice("lamb_array", [
            "-1 1",
            "-1 0 1", 
        ])
        zeta_array_choice = "2 8 16"
        # zeta_array_choice = hp.Choice("zeta_array", [
        #     "1 2",
        #     "1 2 4",
        #     "2 8 16",
        #     "1 4 8 16",
        #     "1 2 4 8 16",
        #     "1 2 4 8 16 32"
        # ])
        zeta_array_choice = hp.Choice("zeta_array", [
            "2 8 16",
            "1 4 8 16",
            "1 2 4 8 16",
            "1 2 4 8 16 32"
        ])

        # Energy model architecture hyperparameters
        #energy_max_layers = 2
        energy_max_neurons = 276
        energy_max_layers = 3
        energy_n_layers = 1
        energy_n_layers = hp.Int("energy_n_layers", 1, energy_max_layers, 1)
        energy_neurons = []
        for i in range(energy_max_layers):
            energy_neuron = 25
            energy_neuron = hp.Int(f"energy_neurons_{i}", 25, energy_max_neurons, 50)
            energy_neurons.append(energy_neuron)
            energy_max_neurons = energy_neuron + 1   # Ensure decreasing order
        
        energy_activation = "tanh"
        energy_activation = hp.Choice("energy_activation", 
            ["relu", "tanh", "elu", "swish", "leaky_relu", "shifted_softplus"])

        raw_hp = {
            "rs_array_choice": rs_array_choice,
            "eta_array_choice": eta_array_choice,
            "lambd_array_choice": lambd_array_choice,
            "zeta_array_choice": zeta_array_choice,
            "energy_n_layers": energy_n_layers,
            "energy_activation": energy_activation
        }
        for i in range(energy_max_layers):
            raw_hp[f"energy_neurons_{i}"] = energy_neurons[i]
        return raw_hp

    def _build_hyperparameters(self, hp: kt.HyperParameters) -> Dict[str, Any]:
        """Build model configuration from hyperparameters."""

        raw_hp = self._build_raw_hyperparameters(hp)

        # Radial parameters
        rs_array = [float(x) for x in raw_hp["rs_array_choice"].split()]
        eta_array = [float(x) for x in raw_hp["eta_array_choice"].split()]

        # Angular parameters
        lambd_array = [float(x) for x in raw_hp["lambd_array_choice"].split()]
        zeta_array = [float(x) for x in raw_hp["zeta_array_choice"].split()]

        # Energy model architecture
        energy_layers = [raw_hp[f"energy_neurons_{i}"] for i in range(raw_hp["energy_n_layers"])]
        energy_activations = [raw_hp["energy_activation"]] * raw_hp["energy_n_layers"]

        return {
            "rs_array": rs_array,
            "eta_array": eta_array,
            "eta_ang_array": eta_array,  # Use same eta for angular
            "lambd_array": lambd_array,
            "zeta_array": zeta_array,
            "energy_hidden_layers": energy_layers,
            "energy_hidden_activations": energy_activations
        }

class MyHyperModel(kt.HyperModel, BaseHDNNP2ndTuner):
    def __init__(self, hyp_search_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._hyp_search_config: Optional[Dict[str, Any]] = hyp_search_config.copy()
        self._hyp_search_config["energy_epochs"] = self._hyp_search_config["max_epochs"] 
        self._hp_config: Optional[Dict[str, Any]] = None
        self._model_config: Optional[Dict[str, Any]] = None

    def build(self, hp):
        self._hp_config = self._build_hyperparameters(hp)
        # Update the search config with hyperparameters
        combined_config = self._hyp_search_config.copy()
        combined_config.update(self._hp_config)
        self._model_config = create_model_config(combined_config)
        return create_model(self._hyp_search_config, self._model_config)
    
    def fit(self, hp, models, dataset, *args, **kwargs):
        ks.backend.clear_session() # RAM is apparently not released between trials. This should clear some of it, but probably not all. https://github.com/keras-team/keras-tuner/issues/395
        config = self._hyp_search_config.copy()
        hp_config = self._hp_config
        model_config = self._model_config
        outputs = self._outputs
        if not isinstance(models, List):
            models = [models]

        model_energy_force, indices, hists, scaler = train_models(dataset, models, model_config, outputs, config, **kwargs)

        return hists[0]
    
    def deactivate_search(self, train_config):
        self._hyp_search_config = train_config.copy()
        self._hyp_search_config["do_search"] = False
        self._hyp_search_config["max_dataset_size"] = None

if __name__ == "__main__":
    config: Dict[str, Any] = parse_args()
    hyp_param_search_params = config.copy()
    hyp_param_search_params.update(HYP_PARAM_SEARCH_TEMP_CONFIGS)

    hypermodel = MyHyperModel(hyp_search_config=hyp_param_search_params)
    tuner = kt.Hyperband(
        hypermodel=hypermodel,
        objective=kt.Objective("val_force_loss", direction="min"),
        max_epochs=hyp_param_search_params["max_epochs"],
        factor=hyp_param_search_params["hyperband_factor"],
        hyperband_iterations=1,
        overwrite=False,
        directory=TRIAL_FOLDER_NAME,
        project_name=config["project_name"],
        max_consecutive_failed_trials=1
    )
    # tuner = kt.GridSearch(
    #     hypermodel=hypermodel,
    #     objective=kt.Objective("val_force_loss", direction="min"),
    #     max_trials=25,
    #     overwrite=True,
    #     directory=TRIAL_FOLDER_NAME,
    #     project_name=config["project_name"],
    #     max_consecutive_failed_trials=1
    # )
    
    dataset = load_data(config)

    tuner.search_space_summary()
    tuner.search(dataset=dataset) 
    tuner.results_summary(num_trials=10)

    if isinstance(tuner, kt.Hyperband):
        n_best_hps = tuner.get_best_hyperparameters(num_trials=10)
        with open(os.path.join("best_hp_hdnnp2nd.json"), "w") as f:
            json.dump(n_best_hps[0].values, f, indent=2)

        hypermodel.deactivate_search(config)
        n_best_hps = tuner.get_best_hyperparameters(num_trials=config["n_splits"])
        best_models = [hypermodel.build(n_best_hps[i]) for i in range(len(n_best_hps))]
        hypermodel.fit(n_best_hps[0], best_models, dataset)
