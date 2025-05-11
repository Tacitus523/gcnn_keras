#!/usr/bin/env python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=plot
#SBATCH --output=plot.out
#SBATCH --error=plot.out

import argparse
from typing import Dict, List, Optional, Union
from ase.atoms import Atoms
from ase.io import read
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Default geometry file
HDNNP_GEOMS: str = "HDNNP_geoms.extxyz"
DATA_SOURCES_FILE: Optional[str] = None  # File containing the data source of each entry

# Keywords for extracting data
REF_ENERGY_KEY: str = "ref_energy"
REF_FORCES_KEY: str = "ref_forces"
REF_CHARGES_KEY: str = "ref_charges"
PRED_ENERGY_KEY: str = "pred_energy"
PRED_FORCES_KEY: str = "pred_forces"
PRED_CHARGES_KEY: str = "pred_charges"

# Units for plotting
ENERGY_UNIT: str = "eV"
FORCES_UNIT: str = r"$\frac{eV}{\AA}$"
CHARGES_UNIT: str = "e"

# Conversion factors
H_to_eV: float = 27.2114
angstrom_to_bohr: float = 1.88973
bohr_to_angstrom: float = 1 / angstrom_to_bohr
H_B_to_ev_angstrom: float = H_to_eV / bohr_to_angstrom

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plotting script for HDNNP data")
    parser.add_argument(
        "-g",
        "--geoms",
        type=str,
        default=HDNNP_GEOMS,
        help="Path to a geometry file containing both reference and predicted data. Run test_hdnnp.py to generate this file."
    )
    parser.add_argument(
        "-s",
        "--sources",
        type=str,
        default=DATA_SOURCES_FILE,
        help="Path to the data sources file, default: %s" % DATA_SOURCES_FILE,
    )
    return parser.parse_args()

def extract_data(
    mols: List[Atoms],
    energy_keyword: Optional[str] = None,
    forces_keyword: Optional[str] = None,
    charges_keyword: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    ref_energy: List[float] = []
    ref_forces: List[float] = []
    ref_charges: List[float] = []
    ref_elements: List[str] = []
    for m in mols:
        if charges_keyword is not None and charges_keyword in m.arrays:
            ref_charges.extend(m.arrays[charges_keyword].flatten())
        if energy_keyword is not None:
            if energy_keyword == "energy":
                ref_energy.append(m.get_potential_energy())
            else:
                ref_energy.append(m.info[energy_keyword])
        if forces_keyword is not None:
            if forces_keyword == "forces":
                ref_forces.extend(m.get_forces().flatten())
            else:
                ref_forces.extend(m.arrays[forces_keyword].flatten())
        ref_elements.extend(m.get_chemical_symbols())
    result = {}
    result["energy"] = np.array(ref_energy) * H_to_eV # Convert Hartree to eV
    result["forces"] = np.array(ref_forces) * H_B_to_ev_angstrom # Convert Hartree/Bohr to eV/Å
    result["charges"] = np.array(ref_charges)
    result["elements"] = np.array(ref_elements)
    return result

def plot_data(
    ref_data: Dict[str, np.ndarray],
    pred_data: Dict[str, np.ndarray],
    key: str,
    sources: Optional[np.ndarray],
    x_label: str,
    y_label: str,
    unit: str,
    filename: str,
) -> None:
    """
    Create a scatter plot comparing reference and HDNNP values.
    
    Args:
        ref_data: Dictionary containing reference data
        pred_data: Dictionary containing predicted data
        key: Key to extract the specific data from dictionaries
        sources: Data sources
        x_label: Label for x-axis
        y_label: Label for y-axis
        filename: Output filename for the plot
    """
    df: pd.DataFrame = create_dataframe(ref_data, pred_data, key, sources, x_label, y_label)
    rmse: float = np.sqrt(np.mean((df[x_label] - df[y_label]) ** 2))
    r2: float = df[x_label].corr(df[y_label], method="pearson") ** 2

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=x_label,
        y=y_label,
        hue="source" if sources is not None else None,
        palette="viridis",
        alpha=0.7,
        edgecolor=None,
    )
    plt.plot(ref_data[key], ref_data[key], color="black", label="_Identity Line")
    plt.xlabel(f"{x_label} ({unit})")
    plt.ylabel(f"{y_label} ({unit})")
    plt.text(
        0.70,
        0.25,
        f"RMSE: {rmse:.2f} {unit}\nR²: {r2:.4f}",
        transform=plt.gca().transAxes,
        fontsize=15,
        verticalalignment="top",
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
    )
    plt.legend(title=None, loc="upper left", fontsize="small")
    for legend_handle in ax.get_legend().legend_handles:
        legend_handle.set_alpha(1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def create_dataframe(
    ref_data: Dict[str, np.ndarray],
    pred_data: Dict[str, np.ndarray],
    key: str,
    sources: Optional[np.ndarray],
    x_label: str,
    y_label: str,
) -> pd.DataFrame:
    """
    Create a DataFrame from reference and predicted data.
    
    Args:
        ref_data: Dictionary containing reference data
        pred_data: Dictionary containing predicted data
        key: Key to extract the specific data from dictionaries
        sources: Optional data sources
        x_label: Label for x-axis
        y_label: Label for y-axis
    Returns:
        DataFrame containing the reference and predicted data
    """
    df = pd.DataFrame(
        {
            x_label: ref_data[key],
            y_label: pred_data[key],
        }
    )
    if sources is not None:
        assert len(ref_data[key]) % len(sources) == 0, "Number of sources does not match the number of data points"
        repetitions = len(ref_data[key]) // len(sources)
        df["source"] = np.repeat(sources, repetitions)
    return df

def main() -> None:
    args: argparse.Namespace = parse_args()
    hdnnp_mols: List[Atoms] = read(args.geoms, format="extxyz", index=":")
    ref_data: Dict[str, np.ndarray] = extract_data(
        hdnnp_mols, REF_ENERGY_KEY, REF_FORCES_KEY, REF_CHARGES_KEY
    )
    hdnnp_data: Dict[str, np.ndarray] = extract_data(
        hdnnp_mols, PRED_ENERGY_KEY, PRED_FORCES_KEY, PRED_CHARGES_KEY
    )
    assert len(ref_data["energy"]) == len(hdnnp_mols), "Number of reference data does not match the number of configurations"
    assert len(hdnnp_data["energy"]) == len(hdnnp_mols), "Number of HDNNP data does not match the number of configurations"

    if args.sources is not None:
        with open(args.sources, "r") as f:
            sources: np.ndarray = np.array([line.strip() for line in f.readlines()])
        assert len(sources) == len(hdnnp_mols), f"Number of sources does not match the number of configurations: {len(sources)} != {len(hdnnp_mols)}"
    else:
        sources = None

    for name, data in zip(["Ref", "HDNNP"], [ref_data, hdnnp_data]):
        for key, value in data.items():
            # Skip non-numeric data
            if value.dtype not in (np.float32, np.float64, np.int32, np.int64):
                continue
            print(value.dtype)
            print(f"{name} {key}: {value.shape} Min Max: {np.min(value): .1f} {np.max(value): .1f}")

    plot_data(
        ref_data,
        hdnnp_data,
        "energy",
        sources,
        "Ref Energy",
        "HDNNP Energy",
        ENERGY_UNIT,
        "HDNNPenergy.png"
    )

    plot_data(
        ref_data,
        hdnnp_data,
        "forces",
        sources if sources is not None else ref_data["elements"],
        "Ref Forces",
        "HDNNP Forces",
        FORCES_UNIT,
        "HDNNPforces.png"
    )

    if "charges" in ref_data and "charges" in hdnnp_data:
        plot_data(
            ref_data,
            hdnnp_data,
            "charges",
            sources if sources is not None else ref_data["elements"],
            "Ref charges",
            "HDNNP charges",
            CHARGES_UNIT,
            "HDNNPcharges.png",
        )

if __name__ == "__main__":
    main()