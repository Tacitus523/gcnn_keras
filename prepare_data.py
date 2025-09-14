#!/usr/bin/python3
import argparse
import json
import numpy as np
import pandas as pd
import os
from os.path import join
import shutil
import warnings
from typing import Dict, Sequence, Optional, List, Tuple

from ase.io import read as read_molecule

BABEL_DATADIR = "/usr/local/run/openbabel-2.4.1" # local installation of openbabel
os.environ['BABEL_DATADIR'] = BABEL_DATADIR

# Supress tensorflow info-messages and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from kgcnn.data.base import MemoryGraphDataset
from kgcnn.data.qm import QMDataset
from kgcnn.utils import constants

OVERWRITE = True # Set to True to enforce the writing in TARGET_FOLDER possibly overwriting data

DATA_FOLDER = os.getcwd() # Folder that contains data the files
GEOMETRY_FILE = "ThiolDisulfidExchange.xyz" # path to geometry-file, gromacs-format, in Angstrom, converted to Bohr
ENERGY_FILE = "energy_diff.txt" # path to energy-file, no header, separated by new lines, in Hartree
FORCE_FILE = "forces.xyz" # path to force-file, "" if not available, in Eh/Bohr, apparently given like that from Orca
CHARGE_FILE = "charges.txt" # path to charge-file, one line per molecule geometry,  "" if not available, in elementary charges
ESP_FILE = "esps_by_mm.txt" # path to esp caused by mm atoms, one line per molecule geometry, "" if not available, in Volts, converted to Hartree/electron
ESP_GRAD_FILE = "esp_gradients.txt" # path to the ESP gradients, "" if not available, in Eh/Bohr^2, I hope
CUTOFF = 10.0 # Max distance for bonds and angles to be considered relevant, None if not available, in Angstrom, converted to Bohr, default 10, CONSIDER CUTOFF IN YOUR SYMMETRY FUNCTIONS
MAX_NEIGHBORS = 25 # Maximal neighbors per atom to be considered relevant, disregards neighbors within cutoff distance if too small
PREFIX = "ThiolDisulfidExchange" # prefix to generated files, compulsary for kgcnn read-in
TARGET_FOLDER = "kgcnn_inputs2" # target folder to save the data

# Alternative method of giving the data via extxyz-file
# Geometry: Angstrom --> Bohr
# Energy: eV --> Hartree
# Forces: eV/Angstrom --> Hartree/Bohr
# Charges: elementary charges --> elementary charges
# ESP: Volts (eV/electron) --> Hartree/electron
# ESP Gradients: Volts/Angstrom (eV/electron/Angstrom) --> Hartree/electron/Bohr
EXTXYZ_FILE = None # "geoms.extxyz" # Path to extxyz-file containing all data, None if not used
# Keys for extxyz-file, only used if EXTXYZ_FILE is not None
EXTXYZ_ENERGY_KEY = "ref_energy" # Key in extxyz-file for energy
EXTXYZ_FORCE_KEY = "ref_force" # Key in extxyz-file for forces
EXTXYZ_CHARGE_KEY = "ref_charge" # Key in extxyz-file for charges
EXTXYZ_TOTAL_CHARGE_KEY = "total_charge" # Key in extxyz-file for total charge
EXTXYZ_ESP_KEY = "esp" # Key in extxyz-file for ESP
EXTXYZ_ESP_GRAD_KEY = "esp_gradient" # Key in extxyz-file for

def parse_args() -> Dict:
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

    total_charge = config_data.get("TOTAL_CHARGE", None)
    if total_charge is not None:
        print("INFO: Giving total charge as directly as input is deprecated. Using charge-file instead.")
    at_count = config_data.get("AT_COUNT", None)
    if at_count is not None:
        print("INFO: Giving atom count as directly as input is redundant by now.")

    # Set default values if they are not present in the config file
    config_data.setdefault("DATA_FOLDER", DATA_FOLDER)
    config_data.setdefault("GEOMETRY_FILE", GEOMETRY_FILE)
    config_data.setdefault("ENERGY_FILE", ENERGY_FILE)
    config_data.setdefault("CHARGE_FILE", CHARGE_FILE)
    config_data.setdefault("ESP_FILE", ESP_FILE)
    config_data.setdefault("ESP_GRAD_FILE", ESP_GRAD_FILE)
    config_data.setdefault("CUTOFF", CUTOFF)
    config_data.setdefault("MAX_NEIGHBORS", MAX_NEIGHBORS)
    config_data.setdefault("FORCE_FILE", FORCE_FILE)
    config_data.setdefault("PREFIX", PREFIX)
    config_data.setdefault("TARGET_FOLDER", TARGET_FOLDER)

    config_data.setdefault("EXTXYZ_FILE", EXTXYZ_FILE)
    config_data.setdefault("EXTXYZ_ENERGY_KEY", EXTXYZ_ENERGY_KEY)
    config_data.setdefault("EXTXYZ_FORCE_KEY", EXTXYZ_FORCE_KEY)
    config_data.setdefault("EXTXYZ_CHARGE_KEY", EXTXYZ_CHARGE_KEY)
    config_data.setdefault("EXTXYZ_TOTAL_CHARGE_KEY", EXTXYZ_TOTAL_CHARGE_KEY)
    config_data.setdefault("EXTXYZ_ESP_KEY", EXTXYZ_ESP_KEY)
    config_data.setdefault("EXTXYZ_ESP_GRAD_KEY", EXTXYZ_ESP_GRAD_KEY)

    config_data["CUTOFF"] = float(config_data["CUTOFF"])
    config_data["MAX_NEIGHBORS"] = int(config_data["MAX_NEIGHBORS"])

    return config_data

def read_irregular_file(file_path, conversion_factor=1.0):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into values
            values = line.split()
            # Convert values to floats, filling missing values with NaN
            float_values = np.array([np.float64(value)*conversion_factor for value in values])
            data.append(float_values)

    return data

def read_forces_file(file_path: str) -> Sequence:
    """Reads forces from file, abusing .xyz format with ases Atoms-object"""
    all_molecules = read_molecule(file_path, index=":")
    all_forces = [molecule.positions for molecule in all_molecules]
    return all_forces

def get_charges(config: Dict[str, any]) -> Tuple[Optional[List[np.ndarray]], Optional[List[float]]]:
    charge_path = join(config["DATA_FOLDER"], config["CHARGE_FILE"])

    try:
        charges = read_irregular_file(charge_path)
        total_charges = []
        for i in range(len(charges)):
            total_charge = np.sum(charges[i])
            total_charges.append(total_charge)
        print("Got Charges")
    except FileNotFoundError:
        print("No Charges")
        charges = None
        total_charges = None

def get_forces(config: Dict[str, any]) -> Optional[List[np.ndarray]]:
    force_path = join(config["DATA_FOLDER"], config["FORCE_FILE"])
    if config["FORCE_FILE"] == "":
        print("No Forces")
        return None
    try:
        forces = read_forces_file(force_path)
        print("Got Forces")
        return forces
    except FileNotFoundError:
        print("No Forces")
        return None

def get_esps(config: Dict[str, any]) -> Optional[List[np.ndarray]]:
    esp_path = join(config["DATA_FOLDER"], config["ESP_FILE"])
    if config["ESP_FILE"] == "":
        print("No ESPs")
        return None
    try:
        V_to_au = constants.V_to_au
        esps = read_irregular_file(esp_path, conversion_factor=V_to_au)
        print("Got ESPs")
        return esps
    except FileNotFoundError:
        print("No ESPs")
        return None
    
def get_esp_grads(config: Dict[str, any]) -> Optional[List[np.ndarray]]:
    esp_grad_path = join(config["DATA_FOLDER"], config["ESP_GRAD_FILE"])
    if config["ESP_GRAD_FILE"] == "":
        print("No ESP Grads")
        return None
    try:
        esp_grads = read_irregular_file(esp_grad_path)
        print("Got ESP Grads")
        return esp_grads
    except FileNotFoundError:
        print("No ESP Grads")
        return None


def make_and_write_csv(energy_path: str, total_charge: np.ndarray | None, prefix: str, target_path: str) -> None:
    """Prepares the csv for fourth generation HDNNP

    Args:
        energy_path (str): path to energy-file, no header, separated by new lines
        total_charge (np.ndarray): total charge of molecule
        target_path (str): target folder to save the data
    """
    df = pd.read_csv(energy_path, names=["energy"])
    if total_charge is not None:
        df["total_charge"] = total_charge
    df.to_csv(join(target_path,f"{prefix}.csv"), index=False, header=True, sep=',')
    
def prepare_kgcnn_dataset(config: Dict[str, any]) -> None:
    data_folder = config["DATA_FOLDER"]
    geometry_file = config["GEOMETRY_FILE"]
    energy_file = config["ENERGY_FILE"]
    extyxz_file = config["EXTXYZ_FILE"]

    cutoff = config["CUTOFF"]
    max_neighbors = config["MAX_NEIGHBORS"]
    dataset_name = config["PREFIX"]
    target_folder = config["TARGET_FOLDER"]

    geometry_path = join(data_folder, geometry_file)
    energy_path = join(data_folder, energy_file)
    
    file_name=f"{dataset_name}.csv"
    
    charges, total_charges = get_charges(config) 
    forces = get_forces(config)
    esps = get_esps(config)
    esp_grads = get_esp_grads(config)
    
    make_and_write_csv(energy_path=energy_path, total_charge=total_charges, prefix=dataset_name, target_path=target_folder)

    dataset = QMDataset(
        data_directory=target_folder, 
        file_name=file_name, 
        dataset_name=dataset_name,
        file_name_xyz=geometry_path,
    )

    dataset.prepare_data(overwrite=True, make_sdf = True)
    #dataset.read_in_memory(label_column_name="energy", additional_callbacks = {'total_charge': lambda mg, dd: dd['total_charge']})
    dataset.read_in_memory(label_column_name="energy")

    print(f"Setting external properties for {len(dataset)} molecules")
    for i in range(len(dataset)):
        dataset[i]["node_coordinates"] *= constants.angstrom_to_bohr # Convert to Bohr

        if charges is not None:
            dataset[i].set("charge", charges[i])
            dataset[i].set("total_charge", total_charges[i])

        if forces is not None:
            dataset[i].set("forces", forces[i])  
        
        if esps is not None:
            for i in range(len(dataset)):
                dataset[i].set("esp", esps[i])
        else:
            for i in range(len(dataset)):
                dataset[i].set("esp", np.zeros_like(dataset[i]["node_number"], dtype=np.float64))

        if esp_grads is not None:
            for i in range(len(dataset)):
                dataset[i].set("esp_grad", esp_grads[i])
        else:
            for i in range(len(dataset)):
                dataset[i].set("esp_grad", np.zeros_like(dataset[i]["node_coordinates"], dtype=np.float64))

        if (i+1) % 1000 == 0:
            print(f"Set properties for {i+1} molecules")

    dataset.map_list(
        method="set_range", 
        max_distance=(cutoff+1.0)*constants.angstrom_to_bohr, 
        max_neighbours=max_neighbors
        ) # Add one Angstrom to cutoff to be sure
    dataset.map_list(method="set_angle")
    dataset.map_list(
        method="count_nodes_and_edges", 
        total_nodes="total_nodes", 
        total_edges="total_ranges", 
        count_nodes="node_number", 
        count_edges="range_indices"
    )
    dataset.save()

def main():
    config_data = parse_args()
    
    target_folder = config_data["TARGET_FOLDER"]
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    elif OVERWRITE is False:
        print(f"{target_folder} already exists and OVERWRITE is False. Aborting")
        exit(1)
    else:
        print(f"Warning: Existing data in {target_folder} was overwritten")
        
    prepare_kgcnn_dataset(
        config=config_data
    )

if __name__ == "__main__":
    main()