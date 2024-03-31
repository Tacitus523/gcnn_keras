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

from kgcnn.data.base import MemoryGraphList, MemoryGraphDataset
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.model.force import EnergyForceModel
from kgcnn.training.scheduler import LinearLearningRateScheduler
from kgcnn.utils import constants
from kgcnn.utils.plots import plot_predict_true, plot_train_test_loss, plot_test_set_prediction
from kgcnn.utils.devices import set_devices_gpu

DATA_DIRECTORY = "/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/"
DATASET_NAME = "ThiolDisulfidExchange"
MODEL_PREFIX = "/data/lpetersen/tests/adaptive_sampling/model_energy_force"
EPOCHS = 200

# Ability to restrict the model to only use a certain GPU, which is passed with python -g gpu_id
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
    MODEL_PREFIX = config_data["MODEL_PREFIX"]
    EPOCHS = int(config_data.get("EPOCHS", EPOCHS))


file_name = f"{DATASET_NAME}.csv"
data_directory = os.path.join(os.path.dirname(__file__), os.path.normpath(DATA_DIRECTORY))
dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=DATASET_NAME)
dataset.load()
#dataset=dataset[:10]

inputs = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
          {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
          {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
          {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
          {"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
          {"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
          {"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}]

outputs = [
    {"name": "charge", "shape": (None, 1), "ragged": True},
    {"name": "graph_labels", "ragged": False},
    {"name": "force", "shape": (None, 3), "ragged": True}
]

class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            current_lr = optimizer.lr
        if logs is not None:
            logs["lr"] = current_lr

model_prefix = os.path.basename(MODEL_PREFIX)
model_folder = os.path.dirname(MODEL_PREFIX)
if not model_folder:
    model_folder = "."

# Use list comprehension to filter files based on the pattern
model_paths = [os.path.join(model_folder,item) for item in os.listdir(model_folder) if item.startswith(model_prefix)]
model_paths.sort()

@ks.utils.register_keras_serializable(package="kgcnn", name="zero_loss_function")
def zero_loss_function(y_true, y_pred):
    return 0
models = [tf.keras.models.load_model(model_path, custom_objects={'zero_loss_function': zero_loss_function}, compile=False)
            for model_path in model_paths]

kf = KFold(n_splits=len(models), shuffle=True)
hists = []
model_index = 0
for train_index, test_index in kf.split(X=np.expand_dims(np.array(dataset.get("graph_labels")), axis=-1)):
    x_train = dataset[train_index].tensor(inputs)
    x_test = dataset[test_index].tensor(inputs)
    energy_force_train = dataset[train_index].tensor(outputs)
    energy_force_test = dataset[test_index].tensor(outputs)

    model_energy_force = models[model_index]
    force_loss_factor = 200
    lr_schedule = ks.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-4, first_decay_steps=5e3, t_mul=1.2, m_mul=0.3, alpha=1e-4)
    model_energy_force.compile(
        loss=["mean_squared_error", "mean_squared_error", "mean_squared_error"],
        optimizer=ks.optimizers.Adam(lr_schedule),
        loss_weights=[0, 1/force_loss_factor, 1-1/force_loss_factor],
        metrics=None
    )

    lrlog = LearningRateLoggingCallback()
    
    start = time.process_time()
    hist = model_energy_force.fit(
        x_train, energy_force_train,
        callbacks=[lrlog
        ],
        validation_data=(x_test, energy_force_test),
        epochs=EPOCHS,
        batch_size=64,
        verbose=2
    )
    stop = time.process_time()
    print("Print Time for training: ", str(timedelta(seconds=stop - start)))
    hists.append(hist)
    model_energy_force.save(model_paths[model_index])
    model_index += 1

true_charge = np.array(dataset[test_index].get("charge")).reshape(-1,1)
true_energy = np.array(dataset[test_index].get("graph_labels")).reshape(-1,1)*constants.hartree_to_kcalmol
true_force = np.array(dataset[test_index].get("force")).reshape(-1,1)

predicted_charge, predicted_energy, predicted_force = model_energy_force.predict(x_test, verbose=0)
predicted_charge = np.array(predicted_charge).reshape(-1,1)
predicted_energy = np.array(predicted_energy).reshape(-1,1)*constants.hartree_to_kcalmol
predicted_force = np.array(predicted_force).reshape(-1,1)

# plot_predict_true(predicted_charge, true_charge,
#     filepath="", data_unit="e",
#     model_name="HDNNP", dataset_name=DATASET_NAME, target_names="Charge",
#     error="RMSE", file_name=f"predict_charge.png", show_fig=False)

plot_predict_true(predicted_energy, true_energy,
    filepath="", data_unit=r"$\frac{kcal}{mol}$",
    model_name="HDNNP", dataset_name=DATASET_NAME, target_names="Energy",
    error="RMSE", file_name=f"predict_energy.png", show_fig=False)

plot_predict_true(predicted_force, true_force,
    filepath="", data_unit="Eh/B",
    model_name="HDNNP", dataset_name=DATASET_NAME, target_names="Force",
    error="RMSE", file_name=f"predict_force.png", show_fig=False)

# plot_train_test_loss(charge_hists,
#     filepath="", data_unit="e",
#     model_name="HDNNP", dataset_name=DATASET_NAME, file_name="charge_loss.png", show_fig=False)

plot_train_test_loss(hists,
    filepath="", data_unit="Eh",
    model_name="HDNNP", dataset_name=DATASET_NAME, file_name="loss.png", show_fig=False)

# rmse_charge = mean_squared_error(true_charge, predicted_charge, squared=False)
# mae_charge  = mean_absolute_error(true_charge, predicted_charge)
# r2_charge   = r2_score(true_charge, predicted_charge)

rmse_energy = mean_squared_error(true_energy, predicted_energy, squared=False)
mae_energy  = mean_absolute_error(true_energy, predicted_energy)
r2_energy   = r2_score(true_energy, predicted_energy)

rmse_force = mean_squared_error(true_force, predicted_force, squared=False)
mae_force  = mean_absolute_error(true_force, predicted_force)
r2_force   = r2_score(true_force, predicted_force)

error_dict = {
    # "RMSE Charge": f"{rmse_charge:.3f}",
    # "MAE Charge": f"{mae_charge:.3f}",
    # "R2 Charge": f"{r2_charge:.2f}",
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

# charge_df = pd.DataFrame({"charge_reference": true_charge.flatten(), "charge_prediction": predicted_charge.flatten()})
energy_df = pd.DataFrame({"energy_reference": true_energy.flatten(), "energy_prediction": predicted_energy.flatten()})
force_df = pd.DataFrame({"force_reference": true_force.flatten(), "force_prediction": predicted_force.flatten()})

atomic_numbers = np.array(dataset[test_index].get("node_number")).flatten()
at_types_column = pd.Series(atomic_numbers, name="at_types").replace(constants.atomic_number_to_element)
# charge_df["at_types"] = at_types_column
force_df["at_types"] =  at_types_column

# plot_test_set_prediction(charge_df, "charge_reference", "charge_prediction",
#     "Charge", "e", rmse_charge, r2_charge, "")
plot_test_set_prediction(energy_df, "energy_reference", "energy_prediction",
    "Energy", r"$\frac{kcal}{mol}$", rmse_energy, r2_energy, "")
plot_test_set_prediction(force_df, "force_reference", "force_prediction",
    "Force", r"$\frac{E_h}{B}$", rmse_force, r2_force, "")

print(len(predicted_force))
print(len(force_df))