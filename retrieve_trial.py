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
from tensorflow.keras.activations import relu, tanh, elu, selu

from kgcnn.graph.base import GraphDict
from kgcnn.data.base import MemoryGraphList, MemoryGraphDataset
from kgcnn.data.qm import QMDataset
from kgcnn.training.scheduler import LinearLearningRateScheduler
from kgcnn.literature.HDNNP4th import make_model_behler_charge_separat as make_model
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss, plot_test_set_prediction
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.utils import constants
from kgcnn.model.force import EnergyForceModel
from kgcnn.model.mlmm import MLMMEnergyForceModel
from kgcnn.metrics.loss import RaggedMeanAbsoluteError

from kgcnn.utils.data_splitter import idx_generator

DATA_DIRECTORY = "/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/"

DATASET_NAME = "ThiolDisulfidExchange"

TRIAL_DIRECTORY = "/data/lpetersen/Behler_training/thiol_disulfide/07_esp_derivative/B3LYP_aug-cc-pVTZ_water/force_hp_search_05/trials"
# Retrieve a specific trial ID from your tuner search, purpose changed, not used anymore
TRIAL_ID = "0427"  # Replace with the actual trial ID

# Ability to restrict the model to only use a certain GPU, which is passed with python -g gpu_id
ap = argparse.ArgumentParser(description="Handle gpu_ids")
ap.add_argument("-g", "--gpuid", type=int)
args = ap.parse_args()
if args.gpuid is not None:
    set_devices_gpu([args.gpuid])

data_directory = os.path.join(os.path.dirname(__file__), os.path.normpath(DATA_DIRECTORY))
dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=DATASET_NAME)
dataset.load()

input_config = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
                {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
                {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
                {"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}]

charge_output = {"name": "charge", "shape": (None, 1), "ragged": True}

outputs = [
    {"name": "charge", "shape": (None, 1), "ragged": True},
    {"name": "graph_labels", "ragged": False},
    {"name": "force", "shape": (None, 3), "ragged": True}
]

x_test, y_test = dataset[[0]].tensor(input_config), dataset[[0]].tensor(outputs)

# Define a custom Swish activation function, Tensorflow one has problems with saving custom gradients
def custom_swish(x):
    return x * tf.sigmoid(x)

# Define Leaky ReLU as a custom activation function
def leaky_relu(x):
    return tf.keras.activations.relu(x, alpha=0.2)

# Wrapper function to select activation dynamically
def custom_activation(x, activation):
    if activation == 'swish':
        return custom_swish(x)
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

def zero_loss_function(y_true, y_pred):
    return 0

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        ks.backend.clear_session() # RAM is apparently not released between trials. This should clear some of it, but probably not all. https://github.com/keras-team/keras-tuner/issues/395

        # Radial parameters
        cutoff_rad = hp.Float("cutoff_rad", 8, 30, 4)
        Rs_array_choice = hp.Choice("Rs_array", [
            #"0.0 4.0 6.0 8.0",
            #"0.0 3.0 5.0 7.0 9.0",
            "0.0 3.0 4.0 5.0 6.0 7.0 8.0",
            "0.0 4.0 6.0 8.0 10.0 12.0 16.0",
            "0.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0"
        ])
        Rs_array = [float(x) for x in Rs_array_choice.split()]
        eta_array_choice  = hp.Choice("eta_array", [
            #"0.0 0.08 0.3",
            #"0.03 0.16 0.5",
            "0.0 0.03 0.08 0.16 0.3 0.5",
            "0.0 0.06 0.16 0.32 0.6 0.8 1.0",
            "0.0 0.03 0.08 0.16 0.3 0.5 0.6 0.75 0.9 1.0"
        ])
        eta_array = [float(x) for x in eta_array_choice.split()]
        # Angular parameters
        cutoff_ang = hp.Float("cutoff_ang", 8, 30, 4)
        lambd_array_choice = hp.Choice("lamb_array", [
            "-1 1",
            "-1 0 1", 
            "-1 -0.5 0 0.5 1"
        ])
        lambd_array = [float(x) for x in lambd_array_choice.split()]
        zeta_array_choice = hp.Choice("zeta_array", [
            #"2 8 16",
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

        charge_activation = hp.Choice("charge_activation", ["relu", "tanh", "elu", "selu", "swish"])
        charge_activations = [lambda x: custom_activation(x, charge_activation)]*charge_n_layers + ["linear"]

        energy_n_layers = hp.Int("energy_n_layers", 1, 3, 1)
        energy_layers = []
        energy_max_neurons = 251
        for i in range(energy_n_layers):
            energy_neurons = hp.Int(f"energy_neurons_{i}", 25, energy_max_neurons, 25)
            energy_max_neurons = energy_neurons+1
            energy_layers.append(energy_neurons)
        energy_layers.append(1)

        energy_activation = hp.Choice("energy_activation", ["relu", "tanh", "elu", "selu", "swish"])
        energy_activations = [lambda x: custom_activation(x, energy_activation)]*energy_n_layers + ["linear"]

        elemental_mapping = [1, 6, 16]
        model_config = {
            "name": "HDNNP4th",
            "inputs": input_config,
            "g2_kwargs": {"eta": eta_array, "rs": Rs_array, "rc": cutoff_rad, "elements": elemental_mapping},
            "g4_kwargs": {"eta": eta_ang_array, "zeta": zeta_array, "lamda": lambd_array, "rc": cutoff_ang, 
                          "elements": elemental_mapping, "multiplicity": 2.0},
            "normalize_kwargs": {},
            "mlp_charge_kwargs": {"units": charge_layers,
                                "num_relations": 30,
                                "activation": charge_activations},
            "mlp_local_kwargs": {"units": energy_layers,
                                "num_relations": 30,
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
        lr_schedule = ks.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-4, first_decay_steps=1e4, t_mul=1.5, m_mul=0.3, alpha=1e-4)
        self.force_loss_factor = force_loss_factor
        self.lr_schedule = lr_schedule
        model_energy_force.compile(
            loss=["mean_squared_error", "mean_squared_error", "mean_squared_error"],
            optimizer=ks.optimizers.Adam(lr_schedule),
            loss_weights=[0, 1, force_loss_factor],
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
            loss_weights=[0, 1, self.force_loss_factor],
            metrics=None
        )

        hist = model.fit(*args, **kwargs)
        return hist 

class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            current_lr = optimizer.lr
        if logs is not None:
            logs["lr"] = current_lr

# Hyperparameter Search
max_epochs = 600
hp_factor = 3
hyperband_iterations = 2
batch_size = 32
patience = 100
earlystop = ks.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, verbose=0)
lrlog = LearningRateLoggingCallback()
callbacks = [earlystop, lrlog]
my_hyper_model = MyHyperModel()
tuner = kt.Hyperband(my_hyper_model, objective=kt.Objective("val_loss", direction="min"),
                     max_epochs=max_epochs, factor=hp_factor, hyperband_iterations=hyperband_iterations, directory=TRIAL_DIRECTORY, 
                     max_consecutive_failed_trials=1)


tuner.results_summary(20)
# # Build the model using the hyperparameters from the specified trial
# target_trial = tuner.oracle.get_trial(TRIAL_ID)
# target_trial.display_hyperparameters()
# target_hyperparameters = target_trial.hyperparameters
# target_model = tuner.hypermodel.build(target_hyperparameters)

# # Save best models anyway
# best_models = tuner.get_best_models(num_models=3)
# for idx, model in enumerate(best_models):
#     model.predict(x_test, verbose=0)
#     model.save("model_energy_force"+str(idx))

n_best_hps = tuner.get_best_hyperparameters(num_trials=20)

charge_hists = []
hists = []
epochs = 500

kf = KFold(n_splits=3, random_state=42, shuffle=True)
model_indexes = [0,0,0]
model_index = 0
for train_index, test_index in kf.split(X=np.expand_dims(np.array(dataset.get("graph_labels")), axis=-1)):
    model_hp = n_best_hps[model_indexes[model_index]]
    print("")
    print(f"Model {model_index}: Hyperparameters {model_indexes[model_index]}")
    for key, value in model_hp.values.items():
        print(f"{key}: {value}")
    print("")
    best_model_force = tuner.hypermodel.build(model_hp) # New initialized model
    best_model_charge = tuner.hypermodel.model_charge

    x_train = dataset[train_index].tensor(input_config)
    x_test = dataset[test_index].tensor(input_config)
    charge_train = dataset[train_index].tensor(charge_output)
    charge_test = dataset[test_index].tensor(charge_output)
    energy_force_train = dataset[train_index].tensor(outputs)
    energy_force_test = dataset[test_index].tensor(outputs)

    start = time.process_time()
    hist = tuner.hypermodel.fit(
        model_hp,
        best_model_force,
        x_train, energy_force_train,
        callbacks=callbacks,
        validation_data=(x_test, energy_force_test),
        epochs=epochs,
        batch_size=32,
        verbose=2
    )
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))
    charge_hists.append(tuner.hypermodel.charge_hist)
    hists.append(hist)
    best_model_force.save("model_energy_force"+str(model_index))
    model_index += 1

true_charge = np.array(dataset[test_index].get("charge")).reshape(-1,1)
true_energy = np.array(dataset[test_index].get("graph_labels")).reshape(-1,1)*constants.hartree_to_kcalmol
true_force = np.array(dataset[test_index].get("force")).reshape(-1,1)
predicted_charge, predicted_energy, predicted_force = best_model_force.predict(x_test, verbose=0)
predicted_charge = np.array(predicted_charge).reshape(-1,1)
predicted_energy = np.array(predicted_energy).reshape(-1,1)*constants.hartree_to_kcalmol
predicted_force = np.array(predicted_force).reshape(-1,1)

plot_predict_true(predicted_charge, true_charge,
    filepath="", data_unit="e",
    model_name="HDNNP", dataset_name=DATASET_NAME, target_names="Charge",
    error="RMSE", file_name=f"predict_charge.png", show_fig=False)

plot_predict_true(predicted_energy, true_energy,
    filepath="", data_unit=r"$\frac{kcal}{mol}$",
    model_name="HDNNP", dataset_name=DATASET_NAME, target_names="Energy",
    error="RMSE", file_name=f"predict_energy.png", show_fig=False)

plot_predict_true(predicted_force, true_force,
    filepath="", data_unit="Eh/B",
    model_name="HDNNP", dataset_name=DATASET_NAME, target_names="Force",
    error="RMSE", file_name=f"predict_force.png", show_fig=False)

plot_train_test_loss(charge_hists,
    filepath="", data_unit="e",
    model_name="HDNNP", dataset_name=DATASET_NAME, file_name="charge_loss.png", show_fig=False)

plot_train_test_loss(hists,
    filepath="", data_unit="Eh",
    model_name="HDNNP", dataset_name=DATASET_NAME, file_name="loss.png", show_fig=False)

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
force_df["at_types"] =  at_types_column

plot_test_set_prediction(charge_df, "charge_reference", "charge_prediction",
    "Charge", "e", rmse_charge, r2_charge, "")
plot_test_set_prediction(energy_df, "energy_reference", "energy_prediction",
    "Energy", r"$\frac{kcal}{mol}$", rmse_energy, r2_energy, "")
plot_test_set_prediction(force_df, "force_reference", "force_prediction",
    "Force", r"$\frac{E_h}{B}$", rmse_force, r2_force, "")