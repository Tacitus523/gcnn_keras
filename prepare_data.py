#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
import os
from os.path import join
import shutil
import warnings
from typing import Dict, Sequence, Optional, List, Tuple, Any

from ase import Atoms
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
GEOMETRY_FILE = None # path to geometry-file, xyz format, Angstrom --> Bohr
ENERGY_FILE = None # path to energy-file, no header, separated by new lines, Hartree --> Hartree
FORCE_FILE = None # path to force-file, xyz format, Hartree/Bohr --> Hartree/Bohr, not actually forces but gradients
CHARGE_FILE = None # path to charge-file, one line per molecule, elementary charges --> elementary charges
ESP_FILE = None # path to esp caused by mm atoms, one line per molecule, Volt (eV/electron) --> Hartree/electron
ESP_GRAD_FILE = None # path to the ESP gradients, xyz format, Volt/Angstrom (eV/electron/Angstrom) --> Hartree/electron/Bohr
CUTOFF = 10.0 # Max distance for bonds and angles to be considered relevant, Angstrom --> Bohr, CONSIDER CUTOFF IN YOUR SYMMETRY FUNCTIONS
MAX_NEIGHBORS = 25 # Maximal neighbors per atom to be considered relevant, disregards neighbors within cutoff distance if too small
PREFIX = "ThiolDisulfidExchange" # prefix to generated files, compulsary for kgcnn read-in
TARGET_FOLDER = "kgcnn_inputs" # target folder to save the data

# Alternative method of giving the data via extxyz-file
# Geometry: Angstrom --> Bohr
# Energy: eV --> Hartree
# Forces: eV/Angstrom --> Hartree/Bohr
# Charges: elementary charges --> elementary charges
# ESP: Volt (eV/electron) --> Hartree/electron
# ESP Gradients: Volt/Angstrom (eV/electron/Angstrom) --> Hartree/electron/Bohr
EXTXYZ_FILE = None # "geoms.extxyz" # Path to extxyz-file containing all data, None if not used
# Keys for extxyz-file, only used if EXTXYZ_FILE is not None
EXTXYZ_ENERGY_KEY = "ref_energy" # Key in extxyz-file for energy
EXTXYZ_FORCE_KEY = "ref_force" # Key in extxyz-file for forces, gets converted to gradient
EXTXYZ_CHARGE_KEY = "ref_charge" # Key in extxyz-file for charges
EXTXYZ_TOTAL_CHARGE_KEY = "total_charge" # Key in extxyz-file for total charge
EXTXYZ_ESP_KEY = "esp" # Key in extxyz-file for ESP
EXTXYZ_ESP_GRAD_KEY = "esp_gradient" # Key in extxyz-file for

# Conversion factors
ANGSTROM_TO_BOHR = constants.angstrom_to_bohr
ELECTRONVOLT_TO_ATOMIC_UNITS = constants.eV_to_hartree
ELECTRONVOLT_ANGSTROM_TO_ATOMIC_UNITS = constants.eV_angstrom_to_hartree_bohr
VOLT_TO_ATOMIC_UNITS = constants.V_to_au
VOLT_PER_ANGSTROM_TO_ATOMIC_UNITS = constants.eV_angstrom_to_hartree_bohr

def parse_args() -> Dict:
    ap = argparse.ArgumentParser(description="Give config file")
    ap.add_argument("-c", "--conf", default=None, type=str, dest="config_path", action="store", required=False, help="Path to config file, default: None", metavar="config")
    ap.add_argument("-g", "--gpuid", type=int) # Just here as a dummy, nothing actually uses a GPU
    args = ap.parse_args()
    config_path = args.config_path
    if config_path is not None:
        try:
            with open(config_path, 'r') as config_file:
                raw_config = json.load(config_file)
        except FileNotFoundError:
            print(f"Config file {config_path} not found.")
            exit(1)
    else:
        raw_config = {}
    return raw_config

def prepare_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare the config dictionary by setting default values and checking file structure."""

    config = raw_config.copy()
    # Set default values if they are not present in the config file
    config.setdefault("DATA_FOLDER", DATA_FOLDER)
    config.setdefault("GEOMETRY_FILE", GEOMETRY_FILE)
    config.setdefault("ENERGY_FILE", ENERGY_FILE)
    config.setdefault("CHARGE_FILE", CHARGE_FILE)
    config.setdefault("ESP_FILE", ESP_FILE)
    config.setdefault("ESP_GRAD_FILE", ESP_GRAD_FILE)
    config.setdefault("CUTOFF", CUTOFF)
    config.setdefault("MAX_NEIGHBORS", MAX_NEIGHBORS)
    config.setdefault("FORCE_FILE", FORCE_FILE)
    config.setdefault("PREFIX", PREFIX)
    config.setdefault("TARGET_FOLDER", TARGET_FOLDER)

    config.setdefault("EXTXYZ_FILE", EXTXYZ_FILE)
    config.setdefault("EXTXYZ_ENERGY_KEY", EXTXYZ_ENERGY_KEY)
    config.setdefault("EXTXYZ_FORCE_KEY", EXTXYZ_FORCE_KEY)
    config.setdefault("EXTXYZ_CHARGE_KEY", EXTXYZ_CHARGE_KEY)
    config.setdefault("EXTXYZ_TOTAL_CHARGE_KEY", EXTXYZ_TOTAL_CHARGE_KEY)
    config.setdefault("EXTXYZ_ESP_KEY", EXTXYZ_ESP_KEY)
    config.setdefault("EXTXYZ_ESP_GRAD_KEY", EXTXYZ_ESP_GRAD_KEY)

    # Convert types
    config["CUTOFF"] = float(config["CUTOFF"])
    config["MAX_NEIGHBORS"] = int(config["MAX_NEIGHBORS"])

    # File structure check
    target_folder = config["TARGET_FOLDER"]
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    elif OVERWRITE is False:
        print(f"{target_folder} already exists and OVERWRITE is False. Aborting")
        exit(1)
    else:
        print(f"Warning: Existing data in {target_folder} was overwritten")

    # Deprecated warnings
    total_charge = config.get("TOTAL_CHARGE", None)
    if total_charge is not None:
        print("INFO: Giving total charge as directly as input is deprecated. Using charge-file instead.")
    at_count = config.get("AT_COUNT", None)
    if at_count is not None:
        print("INFO: Giving atom count as directly as input is redundant by now.")

    return config

def read_irregular_file(file_path, conversion_factor=1.0) -> List[np.ndarray]:
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into values
            values = line.split()
            # Convert values to floats, filling missing values with NaN
            float_values = np.array([np.float64(value)*conversion_factor for value in values])
            data.append(float_values)

    return data

def read_forces_file(file_path: str, conversion_factor: float = 1.0) -> List[np.ndarray]:
    """Reads forces from file, abusing .xyz format with ases Atoms-object"""
    all_molecules = read_molecule(file_path, index=":", format="xyz")
    all_forces = [molecule.positions * conversion_factor for molecule in all_molecules]
    return all_forces

def get_charges(config: Dict[str, Any]) -> Tuple[Optional[List[np.ndarray]], Optional[List[float]]]:
    if not config["CHARGE_FILE"]:
        return None, None
    charge_path = join(config["DATA_FOLDER"], config["CHARGE_FILE"])
    if not os.path.exists(charge_path):
        raise FileNotFoundError(f"Given charge file {config['CHARGE_FILE']} not found in {config['DATA_FOLDER']}")

    charges = read_irregular_file(charge_path)
    total_charges = []
    for i in range(len(charges)):
        total_charge = np.round(np.sum(charges[i]),0) # Rounding to next integer
        total_charges.append(total_charge)
    return charges, total_charges

def get_energies(config: Dict[str, Any]) -> np.ndarray:
    if not config["ENERGY_FILE"]:
        return None
    if not os.path.exists(join(config["DATA_FOLDER"], config["ENERGY_FILE"])):
        raise FileNotFoundError(f"Given energy file {config['ENERGY_FILE']} not found in {config['DATA_FOLDER']}")
    
    energy_path = join(config["DATA_FOLDER"], config["ENERGY_FILE"])
    energies = np.loadtxt(energy_path, dtype=np.float64)
    return energies

def get_forces(config: Dict[str, Any]) -> Optional[List[np.ndarray]]:
    if not config["FORCE_FILE"]:
        return None
    force_path = join(config["DATA_FOLDER"], config["FORCE_FILE"])
    if not os.path.exists(force_path):
        raise FileNotFoundError(f"Given force file {config['FORCE_FILE']} not found in {config['DATA_FOLDER']}")

    forces = read_forces_file(force_path)
    return forces

def get_esps(config: Dict[str, Any]) -> Optional[List[np.ndarray]]:
    if not config["ESP_FILE"]:
        return None
    esp_path = join(config["DATA_FOLDER"], config["ESP_FILE"])
    if not os.path.exists(esp_path):
        raise FileNotFoundError(f"Given ESP file {config['ESP_FILE']} not found in {config['DATA_FOLDER']}")

    esps = read_irregular_file(esp_path, conversion_factor=VOLT_TO_ATOMIC_UNITS)
    return esps
    
def get_esp_grads(config: Dict[str, Any]) -> Optional[List[np.ndarray]]:
    if not config["ESP_GRAD_FILE"]:
        return None
    esp_grad_path = join(config["DATA_FOLDER"], config["ESP_GRAD_FILE"])
    if not os.path.exists(esp_grad_path):
        raise FileNotFoundError(f"Given ESP gradient file {config['ESP_GRAD_FILE']} not found in {config['DATA_FOLDER']}")

    esp_grads = read_forces_file(esp_grad_path, conversion_factor=VOLT_PER_ANGSTROM_TO_ATOMIC_UNITS)
    return esp_grads

def get_properties_from_extxyz(config: Dict[str, Any]) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """Reads properties from extxyz-file if given, otherwise returns None for each property"""
    if config["EXTXYZ_FILE"] is None:
        return None, None, None, None, None, None
    data_folder = config["DATA_FOLDER"]
    extyxz_file = config["EXTXYZ_FILE"]
    extyxz_path = join(data_folder, extyxz_file)
    if not os.path.exists(extyxz_path):
        raise FileNotFoundError(f"Given extxyz file {extyxz_file} not found in {data_folder}")

    all_molecules: List[Atoms] = read_molecule(extyxz_path, index=":", format="extxyz")

    charges = []
    total_charges = []
    energies = []
    forces = []
    esps = []
    esp_grads = []
    for molecule in all_molecules:
        if config["EXTXYZ_CHARGE_KEY"] in molecule.arrays:
            charge = molecule.arrays[config["EXTXYZ_CHARGE_KEY"]]
            charges.append(charge)
        
        if config["EXTXYZ_TOTAL_CHARGE_KEY"] in molecule.info:
            total_charge = molecule.info[config["EXTXYZ_TOTAL_CHARGE_KEY"]]
            total_charges.append(total_charge)

        if config["EXTXYZ_ENERGY_KEY"] in molecule.info:
            energy = molecule.info[config["EXTXYZ_ENERGY_KEY"]]
            energies.append(energy * ELECTRONVOLT_TO_ATOMIC_UNITS) # Convert eV to Hartree

        if config["EXTXYZ_FORCE_KEY"] in molecule.arrays:
            force = molecule.arrays[config["EXTXYZ_FORCE_KEY"]]
            forces.append(force * ELECTRONVOLT_ANGSTROM_TO_ATOMIC_UNITS * -1) # Convert eV/Angstrom to Hartree/Bohr and invert sign

        if config["EXTXYZ_ESP_KEY"] in molecule.arrays:
            esp = molecule.arrays[config["EXTXYZ_ESP_KEY"]]
            esps.append(esp * VOLT_TO_ATOMIC_UNITS) # Convert Volts to Hartree/electron

        if config["EXTXYZ_ESP_GRAD_KEY"] in molecule.arrays:
            esp_grad = molecule.arrays[config["EXTXYZ_ESP_GRAD_KEY"]]
            esp_grads.append(esp_grad * VOLT_PER_ANGSTROM_TO_ATOMIC_UNITS) # Convert Volts/Angstrom to Hartree/electron/Bohr

    if len(charges) == 0:
        charges = None
    if len(total_charges) == 0:
        total_charges = None
    else:
        total_charges = np.array(total_charges)
    if len(energies) == 0:
        energies = None
    else:
        energies = np.array(energies)
    if len(forces) == 0:
        forces = None
    if len(esps) == 0:
        esps = None
    if len(esp_grads) == 0:
        esp_grads = None

    return charges, total_charges, energies, forces, esps, esp_grads


def make_and_write_csv(energy: np.ndarray, total_charge: np.ndarray | None, prefix: str, target_path: str) -> None:
    """Prepares the csv for fourth generation HDNNP

    Args:
        energy (np.ndarray): energies of molecules
        total_charge (np.ndarray): total charge of molecule
        target_path (str): target folder to save the data
    """
    df = pd.DataFrame()
    df["energy"] = energy
    if total_charge is not None:
        df["total_charge"] = total_charge
    df.to_csv(join(target_path,f"{prefix}.csv"), index=False, header=True, sep=',')
    
def prepare_kgcnn_dataset(config: Dict[str, Any]) -> QMDataset:
    data_folder = config["DATA_FOLDER"]
    geometry_file = config["GEOMETRY_FILE"]
    extyxz_file = config["EXTXYZ_FILE"]

    cutoff = config["CUTOFF"]
    max_neighbors = config["MAX_NEIGHBORS"]
    dataset_name = config["PREFIX"]
    target_folder = config["TARGET_FOLDER"]

    if geometry_file is not None:
        geometry_path = join(data_folder, geometry_file)
        if not os.path.exists(geometry_path):
            raise FileNotFoundError(f"Given geometry-file {geometry_path} not found in {data_folder}")
    elif extyxz_file is not None:
        geometry_path = join(data_folder, extyxz_file)
        if not os.path.exists(geometry_path):
            raise FileNotFoundError(f"Given extxyz-file {geometry_path} not found in {data_folder}")
    else:
        raise ValueError("No geometry file or extxyz-file given in config")

    file_name=f"{dataset_name}.csv"

    # Read properties from files
    energies = get_energies(config)
    charges, total_charges = get_charges(config) 
    forces = get_forces(config)
    esps = get_esps(config)
    esp_grads = get_esp_grads(config)

    # Read data from extxyz-file if given
    charges_extxyz, total_charges_extxyz, energies_extxyz, forces_extxyz, esps_extxyz, esp_grads_extxyz = get_properties_from_extxyz(config)

    # Use individual files if given, otherwise use extxyz data
    charges = charges if charges is not None else charges_extxyz
    total_charges = total_charges if total_charges is not None else total_charges_extxyz
    energies = energies if energies is not None else energies_extxyz
    forces = forces if forces is not None else forces_extxyz
    esps = esps if esps is not None else esps_extxyz
    esp_grads = esp_grads if esp_grads is not None else esp_grads_extxyz

    if charges is None:
        print("No charges")
    if energies is None:
        raise ValueError("No energies given, cannot proceed")
    if forces is None:
        print("No forces")
    if esps is None:
        print("No ESPs")
    if esp_grads is None:
        print("No ESP Grads")

    make_and_write_csv(energy=energies, total_charge=total_charges, prefix=dataset_name, target_path=target_folder)

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
            dataset[i].set("force", forces[i])  
        
        if esps is not None:
            dataset[i].set("esp", esps[i])
        else:
            dataset[i].set("esp", np.zeros_like(dataset[i]["node_number"], dtype=np.float64))

        if esp_grads is not None:
                dataset[i].set("esp_grad", esp_grads[i])
        else:
            dataset[i].set("esp_grad", np.zeros_like(dataset[i]["node_coordinates"], dtype=np.float64))

        if (i+1) % 10000 == 0:
            print(f"Set properties for {i+1} molecules")

    print("Creating graph connections")
    dataset.map_list(
        method="set_range", 
        max_distance=(cutoff+1.0)*constants.angstrom_to_bohr, 
        max_neighbours=max_neighbors
        ) # Add one Angstrom to cutoff to be sure
    print("Creating angles")
    dataset.map_list(method="set_angle")
    dataset.map_list(
        method="count_nodes_and_edges", 
        total_nodes="total_nodes", 
        total_edges="total_ranges", 
        count_nodes="node_number", 
        count_edges="range_indices"
    )
    dataset.save()
    print(f"Dataset with {len(dataset)} molecules saved with prefix {dataset_name} in {target_folder}")
    return dataset

def main():
    raw_config = parse_args()
    config = prepare_config(raw_config)
    prepare_kgcnn_dataset(config=config)


if __name__ == "__main__":
    main()