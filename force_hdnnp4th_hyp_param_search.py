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
import keras_tuner as kt
from tensorflow.keras.activations import relu, tanh, elu, selu # type: ignore

from kgcnn.graph.base import GraphDict
from kgcnn.data.base import MemoryGraphList, MemoryGraphDataset
from kgcnn.data.qm import QMDataset
from kgcnn.training.scheduler import LinearLearningRateScheduler
from kgcnn.literature.HDNNP4th import make_model_behler_charge_separat as make_model
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.utils import constants, callbacks
from kgcnn.utils.data_splitter import idx_generator
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss, plot_test_set_prediction
from kgcnn.model.force import EnergyForceModel
from kgcnn.model.mlmm import MLMMEnergyForceModel
from kgcnn.metrics.loss import RaggedMeanAbsoluteError

from force_hdnnp4th import load_data, train_model, evaluate_model

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

# Define a custom Swish activation function, Tensorflow one has problems with saving custom gradients
def swish(x):
    return x * tf.sigmoid(x)

# Define Leaky ReLU as a custom activation function
def leaky_relu(x):
    return tf.keras.activations.relu(x, alpha=0.2)

# Wrapper function to select activation dynamically
def custom_activation(x, activation):
    if activation == 'swish':
        return swish(x)
    elif activation == 'leaky_relu':
        return leaky_relu(x)
    elif activation == 'relu':
        return relu(x)
    elif activation == 'tanh':
        return tanh(x)
    elif activation == 'elu':
        return elu(x)
    elif activation == 'selu':
        return selu(x)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

class MyRandomTuner(kt.RandomSearch):
    def run_trial(self, trial, **kwargs):
        dataset = kwargs.get('dataset')
        if dataset is None:
            raise ValueError("Dataset must be provided")
        
        build_config = kwargs.get('build_config')
        if build_config is None:
            raise ValueError("Build config must be provided")
        
        train_config = kwargs.get('train_config')
        if train_config is None:
            raise ValueError("Train config must be provided")

        hp = trial.hyperparameters
            
        # Radial parameters
        cutoff_rad = hp.Float("cutoff_rad", 8, 30, 8) # in Bohr
        Rs_array_choice = hp.Choice("Rs_array", [
            "0.0 4.0 6.0 8.0",
            "0.0 3.0 5.0 7.0 9.0",
            "0.0 3.0 4.0 5.0 6.0 7.0 8.0",
            "0.0 4.0 6.0 8.0 10.0 12.0 16.0",
            "0.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0"
        ])
        Rs_array = [float(x) for x in Rs_array_choice.split()]
        eta_array_choice  = hp.Choice("eta_array", [
            "0.0 0.08 0.3",
            "0.03 0.16 0.5",
            "0.0 0.03 0.08 0.16 0.3 0.5",
            "0.0 0.06 0.16 0.32 0.6 0.8 1.0",
            "0.0 0.03 0.08 0.16 0.3 0.5 0.6 0.75 0.9 1.0"
        ])
        eta_array = [float(x) for x in eta_array_choice.split()]

        # Angular parameters
        cutoff_ang = hp.Float("cutoff_ang", 8, 30, 8) # in Bohr
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
        eta_ang_array = eta_array

        charge_n_layers = hp.Int("charge_n_layers", 1, 2, 1)
        charge_layers = []
        charge_max_neurons = 151
        for i in range(charge_n_layers):
            charge_neurons = hp.Int(f"charge_neurons_{i}", 25, charge_max_neurons, 25)
            charge_max_neurons = charge_neurons+1
            charge_layers.append(charge_neurons)
        charge_layers.append(1)

        charge_activation = hp.Choice("charge_activation", ["relu", "tanh", "elu", "selu", "swish", "leaky_relu"])
        charge_activations = [lambda x: custom_activation(x, charge_activation)]*charge_n_layers + ["linear"]

        energy_n_layers = hp.Int("energy_n_layers", 1, 3, 1)
        energy_layers = []
        energy_max_neurons = 251
        for i in range(energy_n_layers):
            energy_neurons = hp.Int(f"energy_neurons_{i}", 25, energy_max_neurons, 25)
            energy_max_neurons = energy_neurons+1
            energy_layers.append(energy_neurons)
        energy_layers.append(1)

        energy_activation = hp.Choice("energy_activation", ["relu", "tanh", "elu", "selu", "swish", "leaky_relu"])
        energy_activations = [lambda x: custom_activation(x, energy_activation)]*energy_n_layers + ["linear"]

        max_elements = build_config["max_elements"]
        elemental_mapping = build_config["elemental_mapping"]
        model_config = {
            "name": "HDNNP4th",
            "inputs": build_config["input_config"],
            "g2_kwargs": {"eta": eta_array, "rs": Rs_array, "rc": cutoff_rad, "elements": elemental_mapping},
            "g4_kwargs": {"eta": eta_ang_array, "zeta": zeta_array, "lamda": lambd_array, "rc": cutoff_ang, 
                            "elements": elemental_mapping, "multiplicity": 2.0},
            "normalize_kwargs": {},
            "mlp_charge_kwargs": {"units": charge_layers,
                                "num_relations": max_elements,
                                "activation": charge_activations},
            "mlp_local_kwargs": {"units": energy_layers,
                                "num_relations": max_elements,
                                "activation": energy_activations},
            "cent_kwargs": {},
            "electrostatic_kwargs": {"name": "electrostatic_layer",
                                        "use_physical_params": True,
                                        "param_trainable": False},
            "qmmm_kwargs": {"name": "qmmm_layer"},
            "node_pooling_args": {"pooling_method": "sum"},
            "verbose": 10,
            "output_embedding": "charge+qm_energy", "output_to_tensor": True,
            "use_output_mlp": False
        }

        model_energy_force, test_index, charge_hists, hists, scaler = train_model(dataset, model_config, build_config["charge_output"], build_config["outputs"], train_config)
        return hists[0]

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        ks.backend.clear_session() # RAM is apparently not released between trials. This should clear some of it, but probably not all. https://github.com/keras-team/keras-tuner/issues/395

        # Radial parameters
        # cutoff_rad = hp.Float("cutoff_rad", 8, 30, 8) # in Bohr
        cutoff_rad = 20 # in Bohr
        Rs_array_choice = hp.Choice("Rs_array", [
            # "0.0 4.0 6.0 8.0",
            # "0.0 3.0 5.0 7.0 9.0",
            "0.0 3.0 4.0 5.0 6.0 7.0 8.0",
            #"0.0 4.0 6.0 8.0 10.0 12.0 16.0",
            #"0.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0"
        ])
        Rs_array = [float(x) for x in Rs_array_choice.split()]
        eta_array_choice  = hp.Choice("eta_array", [
            # "0.0 0.08 0.3",
            # "0.03 0.16 0.5",
            "0.0 0.03 0.08 0.16 0.3 0.5",
            #"0.0 0.06 0.16 0.32 0.6 0.8 1.0",
            #"0.0 0.03 0.08 0.16 0.3 0.5 0.6 0.75 0.9 1.0"
        ])
        eta_array = [float(x) for x in eta_array_choice.split()]
        # Angular parameters
        # cutoff_ang = hp.Float("cutoff_ang", 8, 30, 8) # in Bohr
        cutoff_ang = 20 # in Bohr
        lambd_array_choice = hp.Choice("lamb_array", [
            "-1 1",
            #"-1 0 1", 
            #"-1 -0.5 0 0.5 1"
        ])
        lambd_array = [float(x) for x in lambd_array_choice.split()]
        zeta_array_choice = hp.Choice("zeta_array", [
            #"2 8 16",
            #"1 4 8 16",
            "1 2 4 8 16",
            #"1 2 4 8 16 32"
        ])
        zeta_array = [float(x) for x in zeta_array_choice.split()]
        eta_ang_array = eta_array

        charge_n_layers = hp.Int("charge_n_layers", 1, 1, 1)
        charge_layers = []
        charge_max_neurons = 16
        for i in range(charge_n_layers):
            charge_neurons = hp.Int(f"charge_neurons_{i}", 15, charge_max_neurons, 15)
            charge_max_neurons = charge_neurons+1
            charge_layers.append(charge_neurons)
        charge_layers.append(1)

        charge_activation = hp.Choice("charge_activation", ["relu", "tanh", "elu", "selu", "swish", "leaky_relu"])
        charge_activations = [lambda x: custom_activation(x, charge_activation)]*charge_n_layers + ["linear"]

        energy_n_layers = hp.Int("energy_n_layers", 2, 2, 1)
        energy_layers = []
        energy_max_neurons = 36
        for i in range(energy_n_layers):
            energy_neurons = hp.Int(f"energy_neurons_{i}", 35, energy_max_neurons, 25)
            energy_max_neurons = energy_neurons+1
            energy_layers.append(energy_neurons)
        energy_layers.append(1)

        energy_activation = hp.Choice("energy_activation", ["relu", "tanh"])
        energy_activations = [lambda x: custom_activation(x, energy_activation)]*energy_n_layers + ["linear"]

        max_elements = 30
        elemental_mapping = [1, 6, 7, 8]
        model_config = {
            "name": "HDNNP4th",
            "inputs": INPUT_CONFIG,
            "g2_kwargs": {"eta": eta_array, "rs": Rs_array, "rc": cutoff_rad, "elements": elemental_mapping},
            "g4_kwargs": {"eta": eta_ang_array, "zeta": zeta_array, "lamda": lambd_array, "rc": cutoff_ang, 
                          "elements": elemental_mapping, "multiplicity": 2.0},
            "normalize_kwargs": {},
            "mlp_charge_kwargs": {"units": charge_layers,
                                "num_relations": max_elements,
                                "activation": charge_activations},
            "mlp_local_kwargs": {"units": energy_layers,
                                "num_relations": max_elements,
                                "activation": energy_activations},
            "cent_kwargs": {},
            "electrostatic_kwargs": {"name": "electrostatic_layer",
                                     "use_physical_params": True,
                                     "param_trainable": False},
            "qmmm_kwargs": {"name": "qmmm_layer"},
            "node_pooling_args": {"pooling_method": "sum"},
            "verbose": 10,
            "output_embedding": "charge+qm_energy", "output_to_tensor": True,
            "use_output_mlp": False
        }
        self.model_config = model_config

        model_charge, model_energy = make_model(**model_config)

        model_energy_force = EnergyForceModel(
            model_energy = model_energy,
            energy_output = 1,
            esp_input = 5,
            esp_grad_input = 6,
            output_to_tensor = True,
            output_as_dict = False,
            output_squeeze_states = True,
            is_physical_force = False
        )

        lr_schedule_charge = ks.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-4, first_decay_steps=1e4, t_mul=1.5, m_mul=0.3, alpha=1e-4)
        model_charge.compile(
            loss="mean_squared_error",
            optimizer=ks.optimizers.Adam(lr_schedule_charge),
            metrics=None
        )
        self.model_charge = model_charge

        # force_loss_factor = hp.Int("force_loss_factor", 1, 202, 50)
        force_loss_factor = 200
        lr_schedule = ks.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-4, first_decay_steps=1e4, t_mul=1.2, m_mul=0.3, alpha=1e-5)
        self.force_loss_factor = force_loss_factor
        self.lr_schedule = lr_schedule
        model_energy_force.compile(
            loss=["mean_squared_error", "mean_squared_error", "mean_squared_error"],
            optimizer=ks.optimizers.Adam(lr_schedule),
            loss_weights=[0, 1/force_loss_factor, 1-1/force_loss_factor],
            metrics=None
        )
        return model_energy_force

    def fit(self, hp, model, *args, **kwargs):
        x_train, y_train = args
        x_val, y_val = kwargs.get("validation_data")

        validation_data = [x_val, y_val[0]]
        charge_kwargs = kwargs.copy()
        charge_kwargs["validation_data"] = validation_data

        charge_hist = self.model_charge.fit(x_train, y_train[0], **charge_kwargs)
        self.charge_hist = charge_hist
        
        charge_mlp_layer_model_charge = self.model_charge.layers[10]
        charge_mlp_layer_model_force = model.layers[0].layers[10]
        assert "relational_mlp" in charge_mlp_layer_model_charge.name, "This is not a relational MLP, double check your model"
        assert "relational_mlp" in charge_mlp_layer_model_force.name, "This is not a relational MLP, double check your model"
        charge_mlp_layer_model_force.set_weights(charge_mlp_layer_model_charge.get_weights())
        charge_mlp_layer_model_force.trainable = False
        # for charge_weights, force_weights in zip(charge_mlp_layer_model_charge.get_weights(), charge_mlp_layer_model_force.get_weights()):
        #     print(np.allclose(charge_weights, force_weights))
        # print(charge_mlp_layer_model_force is model.layers[0].layers[10])

        electrostatic_layer_model_charge = self.model_charge.layers[13]
        electrostatic_layer_model_force = model.layers[0].layers[13]
        assert "electrostatic_layer" in electrostatic_layer_model_charge.name, "This is not an electrostatic_layer, double check your model"
        assert "electrostatic_layer" in electrostatic_layer_model_force.name, "This is not an electrostatic_layer, double check your model"
        electrostatic_layer_model_force.set_weights(electrostatic_layer_model_charge.get_weights())
        electrostatic_layer_model_force.trainable = False

        model.compile(
            loss=["mean_squared_error", "mean_squared_error", "mean_squared_error"],
            optimizer=ks.optimizers.Adam(self.lr_schedule),
            loss_weights=[0, 1/self.force_loss_factor, 1-1/self.force_loss_factor],
            metrics=None
        )

        hist = model.fit(*args, **kwargs)
        return hist 

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

        MAX_ELEMENTS = config_data.get("MAX_ELEMENTS", MAX_ELEMENTS)
        ELEMENTAL_MAPPING = config_data["ELEMENTAL_MAPPING"]

        CHARGE_EPOCHS                = config_data.get("CHARGE_EPOCHS", CHARGE_EPOCHS)
        CHARGE_INITIAL_LEARNING_RATE = config_data.get("CHARGE_INITIAL_LEARNING_RATE", CHARGE_INITIAL_LEARNING_RATE)
        CHARGE_FINAL_LEARNING_RATE   = config_data.get("CHARGE_FINAL_LEARNING_RATE", CHARGE_FINAL_LEARNING_RATE)
        CHARGE_BATCH_SIZE            = config_data.get("CHARGE_BATCH_SIZE", CHARGE_BATCH_SIZE)
        CHARGE_EARLY_STOPPING        = config_data.get("CHARGE_EARLY_STOPPING", CHARGE_EARLY_STOPPING)

        ENERGY_EPOCHS                = config_data.get("ENERGY_EPOCHS", ENERGY_EPOCHS)
        ENERGY_INITIAL_LEARNING_RATE = config_data.get("ENERGY_INITIAL_LEARNING_RATE", ENERGY_INITIAL_LEARNING_RATE)
        ENERGY_FINAL_LEARNING_RATE   = config_data.get("ENERGY_FINAL_LEARNING_RATE", ENERGY_FINAL_LEARNING_RATE)
        ENERGY_BATCH_SIZE            = config_data.get("ENERGY_BATCH_SIZE", ENERGY_BATCH_SIZE)
        ENERGY_EARLY_STOPPING        = config_data.get("ENERGY_EARLY_STOPPING", ENERGY_EARLY_STOPPING)
        FORCE_LOSS_FACTOR            = config_data.get("FORCE_LOSS_FACTOR", FORCE_LOSS_FACTOR)

    build_config = {
        "input_config": INPUT_CONFIG,
        "charge_output": CHARGE_OUTPUT,
        "outputs": OUTPUTS,
        "max_elements": MAX_ELEMENTS,
        "elemental_mapping": ELEMENTAL_MAPPING,
    }
    print(build_config)

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
    print(train_config)

    # Load dataset
    dataset = load_data(DATA_DIRECTORY, DATASET_NAME)
    print(DATA_DIRECTORY, DATASET_NAME)

    # # # Scaling energy and forces.
    # # scaler = EnergyForceExtensiveLabelScaler()
    # # scaler_mapping = {"atomic_number": "node_number", "y": ["graph_labels", "force"]}
    # # scaler.fit_transform_dataset(dataset, **scaler_mapping)
    # scaler = None

    # train_index, val_index, test_index = idx_generator(len(dataset), 0.3, 0.3)
    # y_train_charge = dataset[train_index].tensor(CHARGE_OUTPUT)
    # y_val_charge = dataset[val_index].tensor(CHARGE_OUTPUT)
    # y_test_charge = dataset[test_index].tensor(CHARGE_OUTPUT)
    # x_train, y_train = dataset[train_index].tensor(INPUT_CONFIG), dataset[train_index].tensor(OUTPUTS)
    # x_val, y_val = dataset[val_index].tensor(INPUT_CONFIG), dataset[val_index].tensor(OUTPUTS)
    # x_test, y_test = dataset[test_index].tensor(INPUT_CONFIG), dataset[test_index].tensor(OUTPUTS)

    # # Hyperparameter Search
    # max_epochs = 400
    # hp_factor = 3
    # hyperband_iterations = 1
    # batch_size = 16
    # patience = 100
    # earlystop = ks.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, verbose=0)
    # lrlog = callbacks.LearningRateLoggingCallback()
    # callbacks = [earlystop, lrlog]
    # my_hyper_model = MyHyperModel()
    # tuner = kt.Hyperband(my_hyper_model, objective=kt.Objective("val_loss", direction="min"),
    #                     max_epochs=max_epochs, factor=hp_factor, hyperband_iterations=hyperband_iterations, directory=TRIAL_FOLDER_NAME, 
    #                     max_consecutive_failed_trials=1)
    # tuner.search_space_summary()
    # tuner.search(x_train, y_train, batch_size=batch_size, epochs=max_epochs, callbacks=callbacks, verbose=2, validation_data=[x_val, y_val])
    # tuner.results_summary(num_trials=10)

    # n_best_hps = tuner.get_best_hyperparameters(num_trials=10)

    # with open(os.path.join("best_hp.json"), "w") as f:
    #     json.dump(n_best_hps[0].values, f, indent=2)

    # model_index = 0
    # # best_model_force = tuner.get_best_models(num_models=1)[model_index] # Pretrained during trial
    # best_model_force = tuner.hypermodel.build(n_best_hps[model_index]) # New initialized model
    # best_model_charge = tuner.hypermodel.model_charge

    # charge_hists = []
    # hists = []
    # epochs = 500

    # # train model again

    # # model_energy.summary()
    # best_model_force.save("chosen_force_model")

    random_tuner = MyRandomTuner(
        objective=kt.Objective("val_output_3_loss", direction="min"),
        max_trials=500,
        overwrite=False,
        directory=TRIAL_FOLDER_NAME,
        project_name="random_search"
    )
    random_tuner.search_space_summary()
    random_tuner.search(dataset=dataset, build_config=build_config, train_config=train_config)

    random_tuner.results_summary(num_trials=10)

    n_best_hps = random_tuner.get_best_hyperparameters(num_trials=10)

    with open(os.path.join("best_hp.json"), "w") as f:
        json.dump(n_best_hps[0].values, f, indent=2)

