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

from force_hdnnp4th import load_data, train_models, evaluate_model, CONFIG_DATA, create_model, create_model_config

TRIAL_FOLDER_NAME = "trials"
PROJECT_NAME = "hdnnp4th_hyp_search"

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
    

class BaseHDNNPTuner:
    """Base class for HDNNP tuners with common hyperparameter building logic."""
    _charge_output = {"name": "charge", "shape": (None, 1), "ragged": True}

    _outputs = [
            {"name": "charge", "shape": (None, 1), "ragged": True},
            {"name": "graph_labels", "ragged": False},
            {"name": "force", "shape": (None, 3), "ragged": True}
        ]

    def _build_raw_hyperparameters(self, hp: kt.HyperParameters) -> Dict[str, Any]:
        """Build raw hyperparameters from the tuner."""

        # cutoff_rad = 25
        # cutoff_ang = 16
        # cutoff_rad = hp.Float("cutoff_rad", 30, 30, 8)
        # cutoff_ang = hp.Float("cutoff_ang", 30, 30, 8)
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
            "0.06 0.16 0.32 0.6 0.8 1.0",
        ])
        lambd_array_choice = "-1 1"
        lambd_array_choice = hp.Choice("lamb_array", [
            "-1 1",
            #"-1 0 1", 
            #"-1 -0.5 0 0.5 1"
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
            "1 2 4 8 16 32"
        ])

        #charge_max_layers = 2
        #charge_max_neurons = 101
        charge_max_layers = 1
        charge_max_neurons = 51
        charge_n_layers = 1
        charge_n_layers = hp.Int("charge_n_layers", 1, charge_max_layers, 1)
        charge_neurons = []
        for i in range(charge_max_layers):
            charge_neuron = 25
            charge_neuron = hp.Int(f"charge_neurons_{i}", 25, charge_max_neurons, 25)
            charge_neurons.append(charge_neuron)
            charge_max_neurons = charge_neuron + 1   # Ensure decreasing order
        
        charge_activation = "tanh"
        charge_activation = hp.Choice("charge_activation", 
                                   ["relu", "tanh", "elu", "swish", "leaky_relu", "shifted_softplus"])        
        
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
            "charge_n_layers": charge_n_layers,
            "charge_activation": charge_activation,
            "energy_n_layers": energy_n_layers,
            "energy_activation": energy_activation
        }
        for i in range(charge_max_layers):
            raw_hp[f"charge_neurons_{i}"] = charge_neurons[i]
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

        # Charge model architecture - only hidden layers (output layer added by create_model_config)
        charge_hidden_layers = [raw_hp[f"charge_neurons_{i}"] for i in range(raw_hp["charge_n_layers"])]
        charge_hidden_activation = [raw_hp["charge_activation"]] * raw_hp["charge_n_layers"]

        # Energy model architecture - only hidden layers (output layer added by create_model_config)
        energy_hidden_layers = [raw_hp[f"energy_neurons_{i}"] for i in range(raw_hp["energy_n_layers"])]
        energy_hidden_activation = [raw_hp["energy_activation"]] * raw_hp["energy_n_layers"]

        clean_hyperparams = {
            "rs_array": rs_array,
            "eta_array": eta_array,
            "eta_ang_array": eta_array,  # Using same eta for angular
            "zeta_array": zeta_array,
            "lambd_array": lambd_array,
            "charge_hidden_layers": charge_hidden_layers,
            "charge_hidden_activation": charge_hidden_activation,
            "energy_hidden_layers": energy_hidden_layers,
            "energy_hidden_activation": energy_hidden_activation
        }
        assert clean_hyperparams.keys() in CONFIG_DATA.keys()
        return clean_hyperparams

class MyHyperModel(kt.HyperModel, BaseHDNNPTuner):
    def __init__(self, hyp_search_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._hyp_search_config: Optional[Dict[str, Any]] = hyp_search_config.copy()
        self._hyp_search_config["energy_epochs"] = self._hyp_search_config["max_epochs"] 
        self._hyp_search_config["charge_epochs"] = self._hyp_search_config["max_epochs"]
        self._hp_config: Optional[Dict[str, Any]] = None
        self._model_config: Optional[Dict[str, Any]] = None

    def build(self, hp):
        hp_config = self._build_hyperparameters(hp)
        # Combine hyperparameters with base configuration for create_model_config
        combined_config = self._hyp_search_config.copy()
        combined_config.update(hp_config)
        self._hp_config = hp_config
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

# class MyRandomTuner(BaseHDNNPTuner, kt.RandomSearch):
#     def run_trial(self, trial, **kwargs):
#         ks.backend.clear_session() # RAM is apparently not released between trials. This should clear some of it, but probably not all. https://github.com/keras-team/keras-tuner/issues/395
#         config = kwargs.get('hyp_search_config')
#         if config is None:
#             raise ValueError("Hyperparameter search config must be provided")

#         hp = trial.hyperparameters
#         hp_config = self._build_hyperparameters(hp)
#         model_config = self._build_model_config(hp_config, config)
#         charge_output = self._charge_output
#         outputs = self._outputs

#         if not hp_config["is_valid"]:
#             return 9999.0  # Invalid configuration, return a high loss

#         dataset = load_data(config)
#         dataset_name = dataset.dataset_name
#         np.random.seed(42)
#         subsample_indices = np.random.choice(len(dataset), size=min(config["max_dataset_size"], len(dataset)), replace=False)
#         dataset = dataset[subsample_indices]
#         dataset.dataset_name = dataset_name  # hack to keep the name after subsampling
#         model_energy_force, test_index, charge_hists, hists, scaler = train_models(dataset, model_config, charge_output, outputs, config)
#         return hists[0]

# class MyHyperbandTuner(kt.Hyperband, BaseHDNNPTuner):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def run_trial(self, trial, **kwargs):
#         ks.backend.clear_session() # RAM is apparently not released between trials. This should clear some of it, but probably not all. https://github.com/keras-team/keras-tuner/issues/395
        
#         config = kwargs.get('hyp_search_config')
#         if config is None:
#             raise ValueError("Hyperparameter search config must be provided")

#         hp = trial.hyperparameters
#         if "tuner/epochs" in hp.values:
#             print(f"Setting epochs to {hp.get('tuner/epochs')}")
#             config["energy_epochs"] = hp.get("tuner/epochs")
#             config["charge_epochs"] = hp.get("tuner/epochs")
#         else:
#             print("tuner/epochs not found in hyperparameters, using default epochs")
#             #raise ValueError("tuner/epochs must be in hyperparameters?")
#             config["energy_epochs"] = 25
#             config["charge_epochs"] = 25
#         hp_config = self._build_hyperparameters(hp)
#         model_config = self._build_model_config(hp_config, config)
#         charge_output = self._charge_output
#         outputs = self._outputs
#         print(model_config)

#         dataset = load_data(config)
#         dataset_name = dataset.dataset_name
#         np.random.seed(42)
#         subsample_indices = np.random.choice(len(dataset), size=min(config["max_dataset_size"], len(dataset)), replace=False)
#         dataset = dataset[subsample_indices]
#         dataset.dataset_name = dataset_name  # hack to keep the name after subsampling
#         assert config["n_splits"] == 1, "Hyperband tuner currently only supports n_splits=1"
#         models = [create_model(config, model_config) for _ in range(config["n_splits"])]
#         model_energy_force, test_index, hists, scaler = train_models(dataset, models, model_config, outputs, config)
#         return hists[0]

# class MyGridSearchTuner(BaseHDNNPTuner, kt.GridSearch):
#     def run_trial(self, trial, **kwargs):
#         ks.backend.clear_session() # RAM is apparently not released between trials. This should clear some of it, but probably not all.
        
#         config = kwargs.get('hyp_search_config')
#         if config is None:
#             raise ValueError("Hyperparameter search config must be provided")
        
#         hp = trial.hyperparameters
#         hp_config = self._build_hyperparameters(hp)
#         model_config = self._build_model_config(hp_config, config)
#         charge_output = self._charge_output
#         outputs = self._outputs

#         dataset = load_data(config)
#         dataset_name = dataset.dataset_name
#         np.random.seed(42)
#         subsample_indices = np.random.choice(len(dataset), size=min(config["max_dataset_size"], len(dataset)), replace=False)
#         dataset = dataset[subsample_indices]
#         dataset.dataset_name = dataset_name  # hack to keep the name after subsampling
#         model_energy_force, test_index, charge_hists, hists, scaler = train_models(dataset, model_config, charge_output, outputs, config)
#         return hists[0]

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

    ## Random Search
    # random_tuner = MyRandomTuner(
    #     objective=kt.Objective("val_force_loss", direction="min"),
    #     max_trials=500,
    #     overwrite=False,
    #     directory=TRIAL_FOLDER_NAME,
    #     project_name=config["project_name"],
    #     max_consecutive_failed_trials=1
    # )
    # random_tuner.search_space_summary()
    # random_tuner.search(hyp_search_config=config) 

    # random_tuner.results_summary(num_trials=10)

    # n_best_hps = random_tuner.get_best_hyperparameters(num_trials=10)

    # with open(os.path.join("best_hp.json"), "w") as f:
    #     json.dump(n_best_hps[0].values, f, indent=2)

    # # Grid Search
    # grid_tuner = MyGridSearchTuner(
    #     objective=kt.Objective("val_force_loss", direction="min"),
    #     max_trials=1000,
    #     overwrite=False,
    #     directory=TRIAL_FOLDER_NAME,
    #     project_name=config["project_name"],
    #     max_consecutive_failed_trials=1
    # )
    # grid_tuner.search_space_summary()
    # grid_tuner.search(hyp_search_config=config) 
    # grid_tuner.results_summary(num_trials=10)
    # n_best_hps = grid_tuner.get_best_hyperparameters(num_trials=10)
    # with open(os.path.join("best_hp_grid.json"), "w") as f:
    #     json.dump(n_best_hps[0].values, f, indent=2)

    if isinstance(tuner, kt.Hyperband):
        n_best_hps = tuner.get_best_hyperparameters(num_trials=10)
        with open(os.path.join("best_hp_hdnnp4th.json"), "w") as f:
            json.dump(n_best_hps[0].values, f, indent=2)
        hypermodel.deactivate_search(config)
        n_best_hps = tuner.get_best_hyperparameters(num_trials=config["n_splits"])
        best_models = [hypermodel.build(n_best_hps[i]) for i in range(len(n_best_hps))]
        hypermodel.fit(n_best_hps[0], best_models, dataset)
