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

data_directory="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/"
dataset_name="ThiolDisulfidExchange"

file_name=f"{dataset_name}.csv"
print("Dataset:", data_directory+file_name)

# Ability to restrict the model to only use a certain GPU, which is passed with python -g gpu_id
ap = argparse.ArgumentParser(description="Handle gpu_ids")
ap.add_argument("-g", "--gpuid", type=int)
args = ap.parse_args()
if args.gpuid is not None:
    set_devices_gpu([args.gpuid])

data_directory = os.path.join(os.path.dirname(__file__), os.path.normpath(data_directory))
dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=dataset_name)
dataset.load()
#dataset = dataset[:10]
print(dataset[0].keys())

input_config = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
                {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
                {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
                {"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}]

charge_output = {"name": "charge", "shape": (None, 1), "ragged": True}

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        # Radial parameters
        cutoff_rad = 20
        Rs_array   = [0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        eta_array  = [0.0, 0.03, 0.08, 0.16, 0.3, 0.5]

        # Angular parameters
        cutoff_ang    = 12
        lambd_array   = [-1, 1]
        zeta_array    = [1, 2, 4, 8, 16]
        eta_ang_array = eta_array

        n_layers = hp.Int("n_layers", 1, 3, 1)
        layers = []
        for i in range(n_layers):
            neurons = hp.Int(f"neurons_{i}", 25, 126, 25)
            layers.append(neurons)
        layers.append(1)

        activation = hp.Choice("activation", ["relu", "tanh", "elu", "selu"])
        activations = [activation]*n_layers + ["linear"]

        elemental_mapping = [1, 6, 16]
        model_config = {
            "name": "HDNNP4th",
            "inputs": input_config,
            "g2_kwargs": {"eta": eta_array, "rs": Rs_array, "rc": cutoff_rad, "elements": elemental_mapping},
            "g4_kwargs": {"eta": eta_ang_array, "zeta": zeta_array, "lamda": lambd_array, "rc": cutoff_ang, 
                          "elements": elemental_mapping, "multiplicity": 2.0},
            "normalize_kwargs": {},
            "mlp_charge_kwargs": {"units": layers,
                                "num_relations": 30,
                                "activation": activations},
            "cent_kwargs": {},
            "electrostatic_kwargs": {"name": "electrostatic_layer",
                                     "use_physical_params": True,
                                     "param_trainable": True},
            "verbose": 10,
            "output_embedding": "charge", "output_to_tensor": True,
            "use_output_mlp": False
        }

        model_charge, model_energy = make_model(**model_config)

        lr_schedule = ks.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-4, first_decay_steps=1e4, t_mul=1.5, m_mul=0.3, alpha=1e-4)
        model_charge.compile(
            loss="mean_squared_error",
            optimizer=ks.optimizers.Adam(lr_schedule),
            metrics=None
        )
        return model_charge

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs
            )

inputs = dataset.tensor(input_config)
print("Amount of inputs:", len(inputs))
for i in range(len(inputs)):
    print(f"Shape {input_config[i]['name']}:", inputs[i].shape)

# # Scaling energy and forces.
# scaler = EnergyForceExtensiveLabelScaler()
# scaler_mapping = {"atomic_number": "node_number", "y": ["graph_labels", "force"]}
# scaler.fit_transform_dataset(dataset, **scaler_mapping)

train_index, val_index, test_index = idx_generator(len(dataset), 0.2, 0.2)
x_train, y_train = dataset[train_index].tensor(input_config), dataset[train_index].tensor(charge_output)
x_val, y_val = dataset[val_index].tensor(input_config), dataset[val_index].tensor(charge_output)
x_test, y_test = dataset[test_index].tensor(input_config), dataset[test_index].tensor(charge_output)

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
hp_factor = 21
hyperband_iterations = 2
batch_size = 64
patience = 100
earlystop = ks.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, verbose=0)
lrlog = LearningRateLoggingCallback()
callbacks = [earlystop, lrlog]
my_hyper_model = MyHyperModel()
tuner = kt.Hyperband(my_hyper_model, objective="val_loss",
                     max_epochs=max_epochs, factor=hp_factor, hyperband_iterations=hyperband_iterations, directory="trials", 
                     max_consecutive_failed_trials=2)
tuner.search_space_summary()
tuner.search(x_train, y_train, batch_size=batch_size, epochs=max_epochs, callbacks=callbacks, verbose=2, validation_data=[x_val, y_val])
tuner.results_summary()

n_best_hps = tuner.get_best_hyperparameters(num_trials=10)

with open(os.path.join("best_hp.json"), "w") as f:
    json.dump(n_best_hps[0].values, f, indent=2)

chosen_model_idx = 0
# best_model_charge = tuner.get_best_models(num_models=10)[chosen_model_idx] # Pretrained during trial
chosen_model_parameters = n_best_hps[chosen_model_idx]
best_model_charge = tuner.hypermodel.build(chosen_model_parameters) # New initialized model
print("")
print(f"Chosen model number: {chosen_model_idx}")
for key, value in chosen_model_parameters.values.items():
    print(f"{key}: {value}")

charge_hists = []

start = time.process_time()
charge_hist = best_model_charge.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    epochs=500,
    batch_size=64,
    verbose=2
)
best_model_charge.save("model_charge")
stop = time.process_time()
print("Print Time for training: ", str(timedelta(seconds=stop - start)))
charge_hists.append(charge_hist)

# scaler.inverse_transform_dataset(dataset, **scaler_mapping)
true_charge = y_test.to_tensor().numpy().reshape(-1,1)

predicted_charge = best_model_charge.predict(x_test, verbose=0)
predicted_charge = predicted_charge.reshape(-1,1)
del best_model_charge

plot_predict_true(predicted_charge, true_charge,
    filepath="", data_unit="e",
    model_name="HDNNP", dataset_name=dataset_name, target_names="Charge",
    error="RMSE", file_name=f"predict_charge.png", show_fig=False)

plot_train_test_loss(charge_hists,
    filepath="", data_unit="e",
    model_name="HDNNP", dataset_name=dataset_name, file_name="charge_loss.png", show_fig=False)

rmse_charge = mean_squared_error(true_charge, predicted_charge, squared=False)
mae_charge  = mean_absolute_error(true_charge, predicted_charge)
r2_charge   = r2_score(true_charge, predicted_charge)

error_dict = {
    "RMSE Charge": f"{rmse_charge:.3f}",
    "MAE Charge": f"{mae_charge:.3f}",
    "R2 Charge": f"{r2_charge:.2f}"
}

for key, value in error_dict.items():
    print(f"{key}: {value}")

with open(os.path.join("", "errors.json"), "w") as f:
    json.dump(error_dict, f, indent=2, sort_keys=True)

charge_df = pd.DataFrame({"charge_reference": true_charge.flatten(), "charge_prediction": predicted_charge.flatten()})

atomic_numbers = np.array(dataset[test_index].get("node_number")).flatten()
at_types_column = pd.Series(atomic_numbers, name="at_types").replace(constants.atomic_number_to_element)
charge_df["at_types"] = at_types_column

plot_test_set_prediction(charge_df, "charge_reference", "charge_prediction",
    "Charge", "e", rmse_charge, r2_charge, "")