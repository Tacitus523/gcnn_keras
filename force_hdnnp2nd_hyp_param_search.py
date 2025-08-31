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

from force_hdnnp2nd import load_data, train_models, evaluate_model, CONFIG_DATA, create_model

TRIAL_FOLDER_NAME = "trials"
PROJECT_NAME = "hdnnp2nd_hyp_search"

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
    

class BaseHDNNP2ndTuner:
    """Base class for HDNNP2nd tuners with common hyperparameter building logic."""
    _outputs = [
        {"name": "graph_labels", "ragged": False},
        {"name": "node_forces", "ragged": True}
    ]

    def _build_raw_hyperparameters(self, hp: kt.HyperParameters) -> Dict[str, Any]:
        """Build raw hyperparameters from the tuner."""

        # Symmetry function hyperparameters
        rs_array_choice = hp.Choice("Rs_array", [
            "0.0 4.0 6.0 8.0",
            "0.0 3.0 5.0 7.0 9.0",
            "0.0 3.0 4.0 5.0 6.0 7.0 8.0",
        ])
        eta_array_choice = hp.Choice("eta_array", [
            "0.0 0.08 0.3",
            "0.03 0.16 0.5",
            "0.0 0.03 0.08 0.16 0.3 0.5",
        ])
        lambd_array_choice = hp.Choice("lamb_array", [
            "-1 1",
            "-1 0 1", 
        ])
        zeta_array_choice = hp.Choice("zeta_array", [
            "2 8 16",
            "1 4 8 16",
            "1 2 4 8 16",
        ])

        # Energy model architecture hyperparameters
        energy_max_layers = 3
        energy_n_layers = hp.Int("energy_n_layers", 1, energy_max_layers, 1)
        energy_neurons = []
        energy_max_neurons = 100
        for i in range(energy_max_layers):
            energy_neuron = hp.Int(f"energy_neurons_{i}", 25, energy_max_neurons, 25)
            energy_neurons.append(energy_neuron)
            energy_max_neurons = energy_neuron + 1   # Ensure decreasing order
            
        energy_activation = hp.Choice("energy_activation", 
                                    ["relu", "tanh", "elu", "swish"])

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

        # Symmetry function parameters
        rs_array = [float(x) for x in raw_hp["rs_array_choice"].split()]
        eta_array = [float(x) for x in raw_hp["eta_array_choice"].split()]
        lambd_array = [float(x) for x in raw_hp["lambd_array_choice"].split()]
        zeta_array = [float(x) for x in raw_hp["zeta_array_choice"].split()]

        # Energy model architecture
        energy_layers = [raw_hp[f"energy_neurons_{i}"] for i in range(raw_hp["energy_n_layers"])]
        energy_layers.append(1)
        energy_activations = ([lambda x: activations.custom_activation(x, raw_hp["energy_activation"])] *
            raw_hp["energy_n_layers"] + ["linear"])

        # Ensure that the layers are in decreasing order
        is_valid = True
        is_energy_layers_sorted = all(x >= y for x, y in zip(energy_layers, energy_layers[1:]))
        if not is_energy_layers_sorted:
            is_valid = False

        return {
            "rs_array": rs_array,
            "eta_array": eta_array,
            "lambd_array": lambd_array,
            "zeta_array": zeta_array,
            "energy_layers": energy_layers,
            "energy_activations": energy_activations,
            "is_valid": is_valid
        }
    
    def _build_model_config(self, hp_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Build the complete model configuration."""
        return {
            "name": "HDNNP2nd",
            "inputs": [
                {"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True}
            ],
            "g2_kwargs": {
                "eta": hp_config["eta_array"], 
                "rs": hp_config["rs_array"], 
                "rc": config["cutoff_rad"],
                "elements": config["elemental_mapping"]
            },
            "g4_kwargs": {
                "eta": hp_config["eta_array"], 
                "zeta": hp_config["zeta_array"], 
                "lamda": hp_config["lambd_array"], 
                "rc": config["cutoff_ang"], 
                "elements": config["elemental_mapping"], 
                "multiplicity": 2.0
            },
            "normalize_kwargs": {},
            "mlp_kwargs": {
                "units": hp_config["energy_layers"],
                "num_relations": config["max_elements"],
                "activation": hp_config["energy_activations"]
            },
            "node_pooling_args": {"pooling_method": "sum"},
            "verbose": 10,
            "output_embedding": "graph", 
            "output_to_tensor": True,
            "use_output_mlp": False,
            "output_mlp": None
        }

class MyHyperModel(kt.HyperModel, BaseHDNNP2ndTuner):
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
        
        model_energy_force, hists, scaler = train_models(dataset, [model], model_config, outputs, config, **kwargs)
        
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
    with open(os.path.join("best_hp_hdnnp2nd.json"), "w") as f:
        json.dump(n_best_hps[0].values, f, indent=2)
