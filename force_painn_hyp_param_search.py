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

from force_painn import load_data, train_models, evaluate_model, CONFIG_DATA, create_model

TRIAL_FOLDER_NAME = "trials"
PROJECT_NAME = "painn_hyp_search"

MAX_DATASET_SIZE = 5000  # we will subsample the dataset to this size for speed

MAX_EPOCHS = 200 # Maximum epochs for Hyperband

# Set default configuration from global constants
CONFIG_DATA.update({
    "project_name": PROJECT_NAME,
    "energy_epochs": 25,
    "n_splits": 1,
    "max_dataset_size": MAX_DATASET_SIZE,
    "max_epochs": MAX_EPOCHS
})

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
        config.update(file_config)

    for key, value in config.items():
        print(f"{key}: {value}")
    return config
    

class BasePaiNNTuner:
    """Base class for PAiNN tuners with common hyperparameter building logic."""
    _outputs = [
        {"name": "graph_labels", "ragged": False},
        {"name": "force", "shape": (None, 3), "ragged": True}
    ]

    def _build_raw_hyperparameters(self, hp: kt.HyperParameters) -> Dict[str, Any]:
        """Build raw hyperparameters from the tuner."""

        # Model architecture hyperparameters
        input_embedding_dim = hp.Choice("input_embedding_dim", [64, 128, 256])
        conv_units = hp.Choice("conv_units", [64, 128, 256])
        update_units = hp.Choice("update_units", [64, 128, 256])
        model_depth = hp.Int("model_depth", 3, 8, 1)
        
        # Bessel basis hyperparameters
        bessel_num_radial = hp.Int("bessel_num_radial", 15, 30, 5)
        bessel_cutoff = hp.Choice("bessel_cutoff", [4.0, 5.0, 6.0])
        bessel_envelope_exponent = hp.Int("bessel_envelope_exponent", 4, 6, 1)
        
        # Output MLP hyperparameters
        output_mlp_choice = hp.Choice("output_mlp_layers", [
            "128 1",
            "256 1", 
            "128 64 1",
            "256 128 1"
        ])

        raw_hp = {
            "input_embedding_dim": input_embedding_dim,
            "conv_units": conv_units,
            "update_units": update_units,
            "model_depth": model_depth,
            "bessel_num_radial": bessel_num_radial,
            "bessel_cutoff": bessel_cutoff,
            "bessel_envelope_exponent": bessel_envelope_exponent,
            "output_mlp_choice": output_mlp_choice
        }
        return raw_hp

    def _build_hyperparameters(self, hp: kt.HyperParameters) -> Dict[str, Any]:
        """Build model configuration from hyperparameters."""

        raw_hp = self._build_raw_hyperparameters(hp)

        # Output MLP units
        output_mlp_units = [int(x) for x in raw_hp["output_mlp_choice"].split()]

        return {
            "input_embedding_dim": raw_hp["input_embedding_dim"],
            "conv_units": raw_hp["conv_units"],
            "update_units": raw_hp["update_units"],
            "model_depth": raw_hp["model_depth"],
            "bessel_num_radial": raw_hp["bessel_num_radial"],
            "bessel_cutoff": raw_hp["bessel_cutoff"],
            "bessel_envelope_exponent": raw_hp["bessel_envelope_exponent"],
            "output_mlp_units": output_mlp_units,
            "is_valid": True  # All PAiNN configurations are valid
        }
    
    def _build_model_config(self, hp_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Build the complete model configuration."""
        return {
            "name": "PAiNNEnergy",
            "inputs": [
                {"shape": [None], "name": "node_number", "dtype": "int64", "ragged": True},
                {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
            ],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": hp_config["input_embedding_dim"]}},
            "equiv_initialize_kwargs": {"dim": 3, "method": "eps"},
            "bessel_basis": {"num_radial": hp_config["bessel_num_radial"], "cutoff": hp_config["bessel_cutoff"], 
                            "envelope_exponent": hp_config["bessel_envelope_exponent"]},
            "pooling_args": {"pooling_method": "sum"},
            "conv_args": {"units": hp_config["conv_units"], "cutoff": None},
            "update_args": {"units": hp_config["update_units"]},
            "depth": hp_config["model_depth"], "verbose": 10,
            "output_embedding": "graph",
            "output_mlp": {"use_bias": [True] * len(hp_config["output_mlp_units"]), 
                          "units": hp_config["output_mlp_units"], 
                          "activation": ["swish"] * (len(hp_config["output_mlp_units"]) - 1) + ["linear"]},
        }

class MyHyperModel(kt.HyperModel, BasePaiNNTuner):
    def __init__(self, hyp_search_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._hyp_search_config: Optional[Dict[str, Any]] = hyp_search_config
        self._hp_config: Optional[Dict[str, Any]] = None
        self._model_config: Optional[Dict[str, Any]] = None

    def build(self, hp):
        hp_config = self._build_hyperparameters(hp)
        self._hp_config = hp_config
        model_config = self._build_model_config(hp_config, self._hyp_search_config)
        self._model_config = model_config
        return create_model(self._hyp_search_config, model_config)
    
    def fit(self, hp, model, *args, **kwargs):
        ks.backend.clear_session() # RAM is apparently not released between trials. This should clear some of it, but probably not all. https://github.com/keras-team/keras-tuner/issues/395
        config = self._hyp_search_config.copy()
        hp_config = self._hp_config
        model_config = self._model_config
        outputs = self._outputs

        if not hp_config["is_valid"]:
            return {"val_output_2_loss": 9999.0}  # Return dict for proper metric handling

        dataset = load_data(config)
        dataset_name = dataset.dataset_name
        np.random.seed(42)
        subsample_indices = np.random.choice(len(dataset), size=min(config["max_dataset_size"], len(dataset)), replace=False)
        dataset = dataset[subsample_indices]
        dataset.dataset_name = dataset_name  # hack to keep the name after subsampling
        
        model_energy_force, test_index, hists, scaler = train_models(dataset, [model], model_config, outputs, config, **kwargs)
        
        return hists[0]

if __name__ == "__main__":
    config: Dict[str, Any] = parse_args()
    
    hypermodel = MyHyperModel(hyp_search_config=config)
    # tuner = kt.Hyperband(
    #     hypermodel=hypermodel,
    #     objective=kt.Objective("val_output_2_loss", direction="min"),
    #     max_epochs=config["max_epochs"],
    #     factor=2,
    #     hyperband_iterations=1,
    #     overwrite=False,
    #     directory=TRIAL_FOLDER_NAME,
    #     project_name=config["project_name"],
    #     max_consecutive_failed_trials=1
    # )
    tuner = kt.GridSearch(
        hypermodel=hypermodel,
        objective=kt.Objective("val_output_2_loss", direction="min"),
        max_trials=25,
        overwrite=False,
        directory=TRIAL_FOLDER_NAME,
        project_name=config["project_name"],
        max_consecutive_failed_trials=1
    )

    if isinstance(tuner, kt.Hyperband):
        config["energy_epochs"] = tuner.hypermodel._hyp_search_config["max_epochs"] # For proper handling of LinearLearningRateScheduler
        config["charge_epochs"] = tuner.hypermodel._hyp_search_config["max_epochs"] # For proper handling of LinearLearningRateScheduler

    tuner.search_space_summary()
    tuner.search() 
    tuner.results_summary(num_trials=10)
    n_best_hps = tuner.get_best_hyperparameters(num_trials=10)
    with open(os.path.join("best_hp_painn.json"), "w") as f:
        json.dump(n_best_hps[0].values, f, indent=2)
