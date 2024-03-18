#!/usr/bin/python3
import argparse
import json
import numpy as np
import pandas as pd
import os
from os.path import join
import shutil
import warnings

OVERWRITE = True # Set to True to enforce the writing in TARGET_FOLDER possibly overwriting data

DATA_FOLDER = "/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_vacuum" # Folder that contains data the files
GEOMETRY_FILE = "geoms.xyz" # path to geometry-file, gromacs-format, in Angstrom
ENERGY_FILE = "energy_diff.txt" # path to energy-file, no header, separated by new lines, in Hartree
CHARGE_FILE = "charges.txt" # path to charge-file, one line per molecule geometry,  "" if not available, in elementary charges
ESP_FILE = "esps_by_mm.txt" # path to esp caused by mm atoms, one line per molecule geometry, "" if not available, in V
ESP_GRAD_FILE = "esp_gradients.txt" # path to the ESP gradients, "" if not available, in Eh/Bohr^2, I hope
AT_COUNT = 15 # atom count, used if no charges supplied
CUTOFF = 10.0 # Max distance for bonds and angles to be considered relevant, None if not available, in Angstrom, default 10, CONSIDER CUTOFF IN YOUR SYMMETRY FUNCTIONS
FORCE_FILE = "forces.xyz" # path to force-file, "" if not available, in Eh/Bohr, apparently given like that from Orca
TOTAL_CHARGE = -1 # total charge of molecule, None if not available, different charges not supported
PREFIX = "ThiolDisulfidExchange" # prefix to generated files, compulsary for kgcnn read-in
TARGET_FOLDER = "/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_vacuum" # target folder to save the data

BABEL_DATADIR = "/usr/local/run/openbabel-2.4.1" # local installation of openbabel
os.environ['BABEL_DATADIR'] = BABEL_DATADIR

ap = argparse.ArgumentParser(description="Give config file")
ap.add_argument("-c", "--conf", default=None, type=str, dest="config_path", action="store", required=False, help="Path to config file, default: None", metavar="config")
ap.add_argument("-g", "--gpuid", type=int) # Just here as a dummy, nothing actually uses a GPU
args = ap.parse_args()
config_path = args.config_path
if config_path is not None:
    try:
        with open(config_path, 'r') as config_file:
            config_data = json.load(config_file)
    except FileNotFoundError:
        print(f"Config file {config_path} not found.")
        exit(1)

    DATA_FOLDER = config_data.get("DATA_FOLDER", DATA_FOLDER)
    GEOMETRY_FILE = config_data.get("GEOMETRY_FILE", GEOMETRY_FILE)
    ENERGY_FILE = config_data.get("ENERGY_FILE", ENERGY_FILE)
    CHARGE_FILE = config_data.get("CHARGE_FILE", CHARGE_FILE)
    ESP_FILE = config_data.get("ESP_FILE", ESP_FILE)
    ESP_GRAD_FILE = config_data.get("ESP_GRAD_FILE", ESP_GRAD_FILE)
    AT_COUNT = int(config_data.get("AT_COUNT", AT_COUNT))
    CUTOFF = float(config_data.get("CUTOFF", CUTOFF))
    FORCE_FILE = config_data.get("FORCE_FILE", FORCE_FILE)
    TOTAL_CHARGE = int(config_data.get("TOTAL_CHARGE", TOTAL_CHARGE))
    PREFIX = config_data.get("PREFIX", PREFIX)
    TARGET_FOLDER = config_data.get("TARGET_FOLDER", TARGET_FOLDER)

# Supress tensorflow info-messages and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from kgcnn.data.base import MemoryGraphDataset
from kgcnn.data.qm import QMDataset

def copy_data(geometry_path: str, charge_path: str, esp_path: str, esp_grad_path: str, force_path: str, prefix: str, target_path: str) -> None:
    target_geometry_path = join(target_path, f"{prefix}.xyz")
    target_charge_path = join(target_path, f"charges.txt")
    target_esp_path = join(target_path, f"esps_by_mm.txt")
    target_esp_grad_path = join(target_path, f"esp_gradients.txt")
    target_force_path = join(target_path, f"forces.xyz")

    if os.path.abspath(geometry_path) != os.path.abspath(target_geometry_path):
        shutil.copyfile(geometry_path, target_geometry_path)
    if os.path.abspath(charge_path) != os.path.abspath(target_charge_path):
        shutil.copyfile(charge_path, target_charge_path)
    if os.path.abspath(esp_path) != os.path.abspath(target_esp_path):
        shutil.copyfile(esp_path, target_esp_path)
    if os.path.abspath(esp_grad_path) != os.path.abspath(target_esp_grad_path):
        shutil.copyfile(esp_grad_path, target_esp_grad_path)
    if os.path.abspath(force_path) != os.path.abspath(target_force_path):
        shutil.copyfile(force_path, target_force_path)


def make_and_write_csv(energy_path: str, total_charge: int, prefix: str, target_path: str) -> None:
    """prepares the csv for fourth generation HDNNP

    Args:
        energy_path (str): path to energy-file, no header, separated by new lines
        total_charge (int): total charge of molecule, different charges not supported
        target_path (str): target folder to save the data
    """
    df = pd.read_csv(energy_path, names=["energy"])
    if total_charge is not None:
        df["total_charge"] = np.ones_like(df["energy"], dtype=float)*total_charge
        df.to_csv(join(target_path,f"{prefix}.csv"), index=False, header=True, sep=',')
    
def prepare_kgcnn_dataset(data_directory: str, dataset_name: str, cutoff: float) -> None:
    file_name=f"{dataset_name}.csv"
    
    dataset = QMDataset(data_directory=data_directory, file_name=file_name, dataset_name=dataset_name)
    dataset.prepare_data(overwrite=True, make_sdf = True)
    dataset.read_in_memory(label_column_name="energy", additional_callbacks = {'total_charge': lambda mg, dd: dd['total_charge']})
    
    dataset.map_list(method="set_range", max_distance=cutoff+1.0)
    dataset.map_list(method="set_angle")
    
    # Distancs in a.u.
    angstrom_to_bohr = 1.8897259886
    for i in range(len(dataset)):
        dataset[i]["node_coordinates"] *= angstrom_to_bohr
    
    charge_path = os.path.join(os.path.normpath(os.path.dirname(dataset.file_path)), "charges.txt")
    try:
        charges = np.loadtxt(charge_path)
        for i in range(len(dataset)):
            dataset[i].set("charge", charges[i])
        at_count = charges.shape[1]
        print("Got Charges")
    except:
        print("No Charges")
        at_count = AT_COUNT

    #TODO: Indicator for molecule end in forces file
    force_path = os.path.join(os.path.normpath(os.path.dirname(dataset.file_path)), "forces.xyz")
    try:
        forces = np.loadtxt(force_path)
        forces = forces.reshape((-1, at_count, 3))
        for i in range(len(dataset)):
            dataset[i].set("force", forces[i])
        print("Got Forces")
    except:
        print("No Forces")    
    
    V_to_au = 1/27.211386245988
    esp_path = os.path.join(os.path.normpath(os.path.dirname(dataset.file_path)), "esps_by_mm.txt")
    try:
        esps = np.loadtxt(esp_path)*V_to_au
        for i in range(len(dataset)):
            dataset[i].set("esp", esps[i])
        print("Got ESP")
    except:
        for i in range(len(dataset)):
            dataset[i].set("esp", np.zeros_like(dataset[i]["node_number"], dtype=np.float64))
        print("Vacuum")

    esp_grad_path = os.path.join(os.path.normpath(os.path.dirname(dataset.file_path)), "esp_gradients.txt")
    try:
        esp_grads = np.loadtxt(esp_grad_path)
        esp_grads = esp_grads.reshape((-1, at_count, 3))
        for i in range(len(dataset)):
            dataset[i].set("esp_grad", esp_grads[i])
        print("Got ESP Gradient")
    except:
        for i in range(len(dataset)):
            dataset[i].set("esp_grad", np.zeros_like(dataset[i]["node_coordinates"], dtype=np.float64))
        print("No ESP Gradient")
        
    dataset.save()

if __name__ == "__main__":
    target_path = join(os.path.normpath(os.path.dirname(__file__)), join("data", TARGET_FOLDER))
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    elif OVERWRITE is False:
        print(f"{target_path} already exists and OVERWRITE is False. Aborting")
        exit(1)
    else:
        print(f"Warning: Existing data in {target_path} was overwritten")

    geometry_path = join(DATA_FOLDER, GEOMETRY_FILE)
    energy_path = join(DATA_FOLDER, ENERGY_FILE)
    charge_path = join(DATA_FOLDER, CHARGE_FILE)
    esp_path = join(DATA_FOLDER, ESP_FILE)
    esp_grad_path = join(DATA_FOLDER, ESP_GRAD_FILE)
    force_path = join(DATA_FOLDER, FORCE_FILE)

    if CUTOFF is None:
        CUTOFF = 10.0
    
    copy_data(geometry_path=geometry_path, charge_path=charge_path, esp_path=esp_path, esp_grad_path=esp_grad_path, force_path=force_path, prefix=PREFIX, target_path=target_path)
    make_and_write_csv(energy_path=energy_path, total_charge=TOTAL_CHARGE, prefix=PREFIX, target_path=target_path)
    
    prepare_kgcnn_dataset(data_directory=target_path, dataset_name=PREFIX, cutoff=CUTOFF)