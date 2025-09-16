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

from force_schnet import load_data, train_models, evaluate_model, CONFIG_DATA, create_model, create_model_config

TRIAL_FOLDER_NAME = "trials"
PROJECT_NAME = "schnet_hyp_search"

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
    

class BaseSchnetTuner:
    """Base class for SchNet tuners with common hyperparameter building logic."""
    _outputs = [
        {"name": "graph_labels", "ragged": False},
        {"name": "force", "shape": (None, 3), "ragged": True}
    ]

    def _build_raw_hyperparameters(self, hp: kt.HyperParameters) -> Dict[str, Any]:
        """Build raw hyperparameters from the tuner."""

        # Model architecture hyperparameters
        #input_embedding_dim = 32
        input_embedding_dim = hp.Choice("input_embedding_dim", [32, 64, 128, 256])
        #interaction_units = 32
        interaction_units = hp.Choice("interaction_units", [32, 64, 128, 256])
        #model_depth = 5
        model_depth = hp.Int("model_depth", 3, 8, 1)
        
        # Gaussian basis hyperparameters
        #gauss_bins = 20
        gauss_bins = hp.Int("gauss_bins", 20, 101, 10)
        #gauss_distance = 4.0
        gauss_distance = hp.Choice("gauss_distance", [4.0, 5.0, 6.0])
        #gauss_offset = 0.0
        gauss_offset = hp.Choice("gauss_offset", [0.0, 0.5, 1.0])
        #gauss_sigma = 0.2
        gauss_sigma = hp.Choice("gauss_sigma", [0.2, 0.4, 0.6])
        
        # Output MLP hyperparameters
        #output_mlp_choice = "128 64 1"
        output_mlp_choice = hp.Choice("output_mlp_layers", [
           "128 64 1",
           "256 128 1", 
           "128 128 64 1",
           "256 128 64 1"
        ])

        #activation = "shifted_softplus"
        activation = hp.Choice("energy_activation", 
            ["relu", "tanh", "elu", "swish", "leaky_relu", "shifted_softplus"])

        raw_hp = {
            "input_embedding_dim": input_embedding_dim,
            "interaction_units": interaction_units,
            "model_depth": model_depth,
            "gauss_bins": gauss_bins,
            "gauss_distance": gauss_distance,
            "gauss_offset": gauss_offset,
            "gauss_sigma": gauss_sigma,
            "output_mlp_choice": output_mlp_choice,
            "activation": activation
        }
        return raw_hp

    def _build_hyperparameters(self, hp: kt.HyperParameters) -> Dict[str, Any]:
        """Build model configuration from hyperparameters."""

        raw_hp = self._build_raw_hyperparameters(hp)

        # Output MLP units
        output_mlp_units = [int(x) for x in raw_hp["output_mlp_choice"].split()]

        # Activation function
        activation = lambda x: activations.custom_activation(x, raw_hp["activation"])

        return {
            "input_embedding_dim": raw_hp["input_embedding_dim"],
            "interaction_units": raw_hp["interaction_units"],
            "model_depth": raw_hp["model_depth"],
            "gauss_bins": raw_hp["gauss_bins"],
            "gauss_distance": raw_hp["gauss_distance"],
            "gauss_offset": raw_hp["gauss_offset"],
            "gauss_sigma": raw_hp["gauss_sigma"],
            "last_mlp_units": output_mlp_units,
            "activation": activation,
        }


class MyHyperModel(kt.HyperModel, BaseSchnetTuner):
    def __init__(self, hyp_search_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._hyp_search_config: Optional[Dict[str, Any]] = hyp_search_config.copy()
        self._hyp_search_config["energy_epochs"] = self._hyp_search_config["max_epochs"] 
        self._hp_config: Optional[Dict[str, Any]] = None
        self._model_config: Optional[Dict[str, Any]] = None

    def build(self, hp):
        self._hp_config = self._build_hyperparameters(hp)
        self._model_config = create_model_config(self._hp_config)
        return create_model(self._hyp_search_config, self._model_config)

class MyHyperModel(kt.HyperModel, BaseSchnetTuner):
    def __init__(self, hyp_search_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._hyp_search_config: Optional[Dict[str, Any]] = hyp_search_config.copy()
        self._hyp_search_config["energy_epochs"] = self._hyp_search_config["max_epochs"] 
        self._hp_config: Optional[Dict[str, Any]] = None
        self._model_config: Optional[Dict[str, Any]] = None

    def build(self, hp):
        hp_config = self._build_hyperparameters(hp)
        self._hp_config = hp_config
        model_config = self._build_model_config(hp_config, self._hyp_search_config)
        self._model_config = model_config
        return create_model(self._hyp_search_config, model_config)
    
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
        with open(os.path.join("best_hp_schnet.json"), "w") as f:
            json.dump(n_best_hps[0].values, f, indent=2)

        hypermodel.deactivate_search(config)
        n_best_hps = tuner.get_best_hyperparameters(num_trials=config["n_splits"])
        best_models = [hypermodel.build(n_best_hps[i]) for i in range(len(n_best_hps))]
        hypermodel.fit(n_best_hps[0], best_models, dataset)
