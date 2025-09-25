import argparse
from datetime import timedelta
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from ase import Atoms
from ase.io import read
from ase.data import chemical_symbols
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
ks=tf.keras

from HDNNPPlot import extract_data, PRED_ENERGY_KEY, PRED_FORCES_KEY, H_to_eV, H_B_to_ev_angstrom

# Data in .extxyz in eV and eV/Angstrom
# Predictions in .extxyz in Hartree and Hartree/Bohr
# Mean and Std decisions are made on the Hartree and Hartree/Bohr scale
ENERGY_UNIT: str = "H"
FORCES_UNIT: str = r"$\frac{H}{Bohr}$"

FIGSIZE = (8, 6)
DPI = 100

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plotting script for HDNNP data")
    parser.add_argument(
        "geoms",
        nargs="+",
        type=str,
        help="Path to a geometry files containing both energy and forces predictions."
    )
    return parser.parse_args()

def main():
    args: argparse.Namespace = parse_args()
    predictions: List[Dict[str, np.ndarray]] = []
    for geom in args.geoms:
        molecules: List[Atoms] = read(geom, index=":")
        prediction: Dict[str, np.ndarray] = extract_data(
            molecules, PRED_ENERGY_KEY, PRED_FORCES_KEY, None
        )
        predictions.append(prediction)

    energy_predictions = np.stack([pred["energy"] for pred in predictions], axis=1)
    force_predictions = np.stack([pred["forces"] for pred in predictions], axis=1)
    atom_types = np.concatenate([molecule.get_atomic_numbers() for molecule in molecules], axis=0)

    # Convert units for statistics
    energy_predictions /= H_to_eV # from eV to Hartree
    force_predictions /= H_B_to_ev_angstrom # from eV/Angstrom to Hartree/Bohr

    (energy_means, energy_stds, mean_energy_mean, mean_energy_std, std_energy_std, max_energy_std) = calc_statistics(energy_predictions)
    (force_means, force_stds, mean_force_mean, mean_force_std, std_force_std, max_force_std) = calc_statistics(force_predictions)


    print("Energy Statistics:")
    print(f"Mean of Means: {mean_energy_mean: .4f} {ENERGY_UNIT}")
    print(f"Mean of Stds: {mean_energy_std: .4f} {ENERGY_UNIT}")
    print(f"Std of Stds: {std_energy_std: .4f} {ENERGY_UNIT}")
    print(f"Max of Stds: {max_energy_std: .4f} {ENERGY_UNIT}")

    print("Force Statistics:")
    print(f"Mean of Means: {mean_force_mean: .4f} {FORCES_UNIT} ")
    print(f"Mean of Stds: {mean_force_std: .4f} {FORCES_UNIT} ")
    print(f"Std of Stds: {std_force_std: .4f} {FORCES_UNIT} ")
    print(f"Max of Stds: {max_force_std: .4f} {FORCES_UNIT} ")

    print(f"Suggested Energy Threshold: {(mean_energy_std + 10*std_energy_std): .4f} {ENERGY_UNIT}")
    print(f"Suggested Force Threshold: {(mean_force_std + 10*std_force_std): .4f} {FORCES_UNIT}")

    # Create plots
    plot_std_histogram(energy_stds, "Energy", unit=ENERGY_UNIT)
    plot_std_histogram(force_stds, "Force", atom_types=atom_types, unit=FORCES_UNIT)

def calc_statistics(stacked_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Calculate means, stds, mean of means, mean of stds, std of stds, max of stds."""
    # stacked_predictions shape (n_models, n_samples, ...)
    means = np.mean(stacked_predictions, axis=1)  # Mean for each sample
    stds = np.std(stacked_predictions, axis=1)  # Std for each sample, shape (n_samples, ...)
    mean_of_means = np.mean(means)  # Mean of means
    mean_of_stds = np.mean(stds)  # Mean of stds
    std_of_stds = np.std(stds)  # Std of stds
    max_of_stds = np.max(stds)  # Max of stds
    return means, stds, mean_of_means, mean_of_stds, std_of_stds, max_of_stds

def plot_std_histogram(
        stds: np.ndarray,
        data_type: str,
        atom_types: Optional[np.ndarray] = None,
        unit: str = ""
    ) -> None:
    """
    Plot histogram of standard deviations.
    
    Args:
        stds: Tensor of standard deviations
        data_type: Type of data ('Energy' or 'Force')
        atom_types: Array of atom types for coloring (optional)
    """
    
    # Prepare data
    std_column = f"{data_type} Std"
    df = pd.DataFrame()
    df[std_column] = stds.flatten()
    
    # Add atom types if provided (for force plots)
    if atom_types is not None:
        df["Atom Types"] = np.repeat(atom_types, stds.shape[0] // atom_types.shape[0])
        df["Atom Types"] = df["Atom Types"].replace({i: chemical_symbols[i] for i in range(len(chemical_symbols))})
        
    
    # Convert atomic numbers to element symbols if atom types provided
    if atom_types is not None:
        hue = "Atom Types"
        common_norm = False
    else:
        hue = None
        common_norm = True

    stat = 'probability'
    ylabel = 'Probability'

    # Create plot
    plt.figure(figsize=FIGSIZE)
    sns.histplot(data=df, x=std_column, hue=hue, stat=stat, common_norm=common_norm)
    plt.xlabel(f'{data_type} Standard Deviation ({unit})')
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # Textbox with statistics
    mean_std = np.mean(stds)
    std_std = np.std(stds)
    max_std = np.max(stds)
    textstr = '\n'.join((
        f'Mean: {mean_std:.4f} {unit}',
        f'Std: {std_std:.4f} {unit}',
        f'Max: {max_std:.4f} {unit}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.gca().text(0.95, 0.50, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    filename = f"{data_type.lower()}_std_histogram.png"
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {filename}")

def old_main():
    from kgcnn.data.base import MemoryGraphList, MemoryGraphDataset
    from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
    from kgcnn.model.force import EnergyForceModel
    from kgcnn.utils.plots import plot_predict_true
    from kgcnn.utils import constants
    from kgcnn.utils.devices import set_devices_gpu

    MODEL_PATHS = [
        "model_energy_force0",
        "model_energy_force1",
        "model_energy_force2"
    ]

    # DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water"
    # DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
    # DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/07_vacuum_sampling_retry/adaptive_sampling/current_training_data"
    DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
    # DATA_DIRECTORY="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water"
    # DATASET_NAME="ThiolDisulfidExchange"
    DATASET_NAME="Alanindipeptide"

    USE_SCALER = True
    SCALER_PATH = "scaler.json"

    file_name=f"{DATASET_NAME}.csv"
    print("Dataset:", os.path.join(DATA_DIRECTORY, file_name))

    # Ability to restrict the model to only use a certain GPU, which is passed with python -g gpu_id
    ap = argparse.ArgumentParser(description="Handle gpu_ids")
    ap.add_argument("-g", "--gpuid", type=int)
    args = ap.parse_args()
    if args.gpuid is not None:
        set_devices_gpu([args.gpuid])

    dataset = MemoryGraphDataset(data_directory=DATA_DIRECTORY, dataset_name=DATASET_NAME)
    dataset.load()
    #dataset=dataset[:10]

    inputs = [{"shape": (None,), "name": "node_number", "dtype": "int64", "ragged": True},
            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
            {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
            #{"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64", "ragged": True},
            #{"shape": (1,), "name": "total_charge", "dtype": "float32", "ragged": False},
            #{"shape": (None,), "name": "esp", "dtype": "float32", "ragged": True},
            #{"shape": (None, 3), "name": "esp_grad", "dtype": "float32", "ragged": True}
            ]

    outputs = [
        #{"name": "charge", "shape": (None, 1), "ragged": True},
        {"name": "graph_labels", "ragged": False},
        {"name": "force", "shape": (None, 3), "ragged": True}
    ]
    charge_output = {"name": "charge", "shape": (None, 1), "ragged": True}

    models = [tf.keras.models.load_model(model_path, compile=False) for model_path in MODEL_PATHS]

    #predicted_charges = []
    predicted_energies, predicted_forces = [], []

    for model in models:
        #predicted_charge, predicted_energy, predicted_force = model.predict(dataset.tensor(inputs), batch_size=128, verbose=2)
        predicted_energy, predicted_force = model.predict(dataset.tensor(inputs), batch_size=128, verbose=2)

        #predicted_charges.append(predicted_charge)
        predicted_energies.append(predicted_energy)
        predicted_forces.append(predicted_force)

    del models

    if USE_SCALER:
        scaler = EnergyForceExtensiveLabelScaler()
        scaler.load(SCALER_PATH)
        for index,(predicted_energy, predicted_force) in enumerate(zip(predicted_energies, predicted_forces)):
            predicted_energy, predicted_force = scaler.inverse_transform(
                y=(predicted_energy.flatten(), predicted_force), X=dataset.get("node_number"))
            predicted_energies[index] = predicted_energy
            predicted_forces[index] = predicted_force
    
    # Convert lists to arrays
    #predicted_charges = np.stack(predicted_charges, axis=1)
    predicted_energies = np.stack(predicted_energies, axis=1)
    predicted_forces = np.stack(predicted_forces, axis=1)

    #(charge_means, charge_stds, mean_charge_mean, mean_charge_std, std_charge_std, max_charge_std) = calc_statistics(predicted_charges)
    (energy_means, energy_stds, mean_energy_mean, mean_energy_std, std_energy_std, max_energy_std) = calc_statistics(predicted_energies)
    (force_means, force_stds, mean_force_mean, mean_force_std, std_force_std, max_force_std) = calc_statistics(predicted_forces)
    
    #print("Mean Charge Mean:", mean_charge_mean)
    print("Mean Energy Mean:", mean_energy_mean)
    print("Mean Force Mean:", mean_force_mean)

    #print("Mean Charge Std:", mean_charge_std)
    print("Mean Energy Std:", mean_energy_std)
    print("Mean Force Std:", mean_force_std)

    #print("Std Charge Std:", std_charge_std)
    print("Std Energy Std:", std_energy_std)
    print("Std Force Std:", std_force_std)

    #print("Max Charge Std:", max_charge_std)
    print("Max Energy Std:", max_energy_std)
    print("Max Force Std:", max_force_std)

    print("Suggested Energy Treshold:", (mean_energy_std + 10*std_energy_std))
    print("Suggested Force Threshold:", (mean_force_std + 10*std_force_std))

    # Create plots
    plot_std_histogram(energy_stds, "Energy", unit=ENERGY_UNIT)
    plot_std_histogram(force_stds, "Force", atom_types=dataset.get("node_number").numpy(), unit=FORCES_UNIT)
    #plot_std_histogram(charge_stds, "Charge", atom_types=dataset.get("node_number").numpy())

if __name__ == "__main__":
    main()